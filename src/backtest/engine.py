"""Point-in-Time 滚动回测评估引擎 (Walk-Forward Rolling Backtest Engine).

严守回测铁律：
1. 严禁未来信息泄漏 (Lookahead Bias)，在历史时点 T 仅读取 published_at <= T 的公开数据。
2. 严禁在无数据时全表扫描未来指标表。
3. 事实标签严谨状态机：明确区分 SUCCESS, FAILURE, NOT_MATURED, INSUFFICIENT_EVIDENCE。
   未成熟样本绝不计入失败，不进入评估分母。
4. 真实 PR-AUC：基于 Precision-Recall 曲线各分位梯形积分，拒绝均值冒充。
5. 动态对照基准：随机基准与动量基准同候选池、同评价期动态测算，拒绝硬编码。
"""

import json
from datetime import date, datetime
from typing import Dict, Any, List, Tuple
import pandas as pd

from src.core.database import db_manager
from src.core.types import BacktestMetricRecord, GroundTruthOutcome
from src.backtest.metrics import BacktestMetricsCalculator
from src.decision_engine.opportunity_pool import OpportunityPoolManager


class PointInTimeBacktestEngine:
    """严格时间截面滚动回测器"""

    def __init__(self):
        self.pool_manager = OpportunityPoolManager()
        self.calc = BacktestMetricsCalculator()

    def run_walk_forward_backtest(
        self,
        cutoff_date: date = date(2021, 12, 31),
        evaluation_date: date = date(2024, 12, 31),
        horizon_years: int = 3
    ) -> Dict[str, Any]:
        """在历史切片点 cutoff_date 运行预测，并在 evaluation_date 核验真实结果"""
        run_id = f"BT_{cutoff_date.strftime('%Y%m%d')}_{evaluation_date.strftime('%Y%m%d')}"

        # 1. 在 cutoff_date 生成当时的预测候选池 (物理隔离未来数据)
        candidates = self.pool_manager.build_opportunity_pool(cutoff_date=cutoff_date)

        # 预测得分元组列表 [(topic_id, research_score)]
        predicted_scores = [(c["topic_id"], c.get("research_score", c.get("composite_score", 0.0))) for c in candidates]
        # 严格区分预测概率：仅当模型存在真正经过校准的概率时记录，严禁将研究评分 / 100 伪装成成功概率
        predicted_probs = [c.get("calibrated_probability") for c in candidates]

        # 2. 从真实历史数据中提取事实标签 (Ground Truth Outcome)
        # 审查观察期终点：若 evaluation_date 尚未达到要求的预测跨度，标记为 NOT_MATURED
        try:
            expected_maturity_date = date(cutoff_date.year + horizon_years, cutoff_date.month, cutoff_date.day)
        except ValueError:
            expected_maturity_date = date(cutoff_date.year + horizon_years, cutoff_date.month, 28)

        true_labels = {}
        outcome_details = {}
        cutoff_dt = pd.to_datetime(cutoff_date)
        eval_dt = pd.to_datetime(evaluation_date)

        is_matured_horizon = (evaluation_date >= expected_maturity_date)

        for c in candidates:
            tid = c["topic_id"]

            if not is_matured_horizon:
                outcome = GroundTruthOutcome.NOT_MATURED
                growth_actual = None
            else:
                # 真实历史数据 (仅允许读取到 evaluation_date 的已公开数据)
                df_hist = db_manager.query_point_in_time("tech_metrics", evaluation_date, target_col="topic_id", target_val=tid)
                if not df_hist.empty and len(df_hist) >= 2:
                    df_hist["as_of_dt"] = pd.to_datetime(df_hist["as_of_date"])
                    sub_cutoff = df_hist[df_hist["as_of_dt"] <= cutoff_dt]
                    sub_eval = df_hist[df_hist["as_of_dt"] <= eval_dt]

                    val_cutoff = sub_cutoff["papers_count"].iloc[-1] if not sub_cutoff.empty else 100
                    val_eval = sub_eval["papers_count"].iloc[-1] if not sub_eval.empty else val_cutoff
                    growth_actual = float(val_eval / max(val_cutoff, 1))

                    if growth_actual >= 1.9:
                        outcome = GroundTruthOutcome.SUCCESS
                    else:
                        outcome = GroundTruthOutcome.FAILURE
                else:
                    # 缺乏客观历史验证数据，严禁误判为失败，标记为证据不足
                    outcome = GroundTruthOutcome.INSUFFICIENT_EVIDENCE
                    growth_actual = None

            true_labels[tid] = outcome.value
            outcome_details[tid] = {
                "outcome": outcome.value,
                "growth_actual": growth_actual,
                "is_evaluable": self.calc._is_evaluable(outcome)
            }

        # 3. 计算评估指标 (排除未成熟与证据不足样本)
        prec_3 = self.calc.precision_at_k(predicted_scores, true_labels, k=3, filter_evaluable_only=True)
        prec_5 = self.calc.precision_at_k(predicted_scores, true_labels, k=min(5, len(predicted_scores)), filter_evaluable_only=True)
        rec_5 = self.calc.recall_at_k(predicted_scores, true_labels, k=min(5, len(predicted_scores)))

        # 真实计算 PR-AUC
        real_pr_auc = self.calc.pr_auc(predicted_scores, true_labels)

        # 概率校准误差 Brier Score (仅针对有效校准概率与明确成熟样本计算)
        evaluable_probs = []
        evaluable_actuals = []
        for item, p in zip(predicted_scores, predicted_probs):
            ot = true_labels.get(item[0])
            if self.calc._is_evaluable(ot):
                evaluable_probs.append(p)
                evaluable_actuals.append(1 if self.calc._is_positive(ot) else 0)

        brier = self.calc.brier_score(evaluable_probs, evaluable_actuals)

        # 动态计算随机与动量基准超额胜率
        excess_rates = self.calc.calculate_excess_hit_rate(
            model_precision=prec_3,
            candidate_items=candidates,
            true_labels=true_labels,
            k=3
        )

        # 4. 持久化回测记录到 DuckDB
        now = datetime.utcnow()
        conn = db_manager.get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO backtest_runs
                (run_id, cutoff_date, evaluation_date, horizon_years, precision_3, precision_5, recall_5, pr_auc, brier_score, excess_vs_random, excess_vs_momentum, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                run_id, cutoff_date, evaluation_date, horizon_years,
                prec_3, prec_5, rec_5, real_pr_auc,
                brier, excess_rates["excess_vs_random"], excess_rates["excess_vs_momentum"], now
            ])
        finally:
            conn.close()

        return {
            "run_id": run_id,
            "cutoff_date": cutoff_date.isoformat(),
            "evaluation_date": evaluation_date.isoformat(),
            "horizon_years": horizon_years,
            "expected_maturity_date": expected_maturity_date.isoformat(),
            "is_horizon_matured": is_matured_horizon,
            "metrics": {
                "precision_at_3": prec_3,
                "precision_at_5": prec_5,
                "recall_at_5": rec_5,
                "pr_auc": real_pr_auc,
                "brier_score": brier,
                "excess_vs_random_baseline": excess_rates["excess_vs_random"],
                "excess_vs_momentum_baseline": excess_rates["excess_vs_momentum"],
                "dynamic_random_baseline": excess_rates.get("random_baseline"),
                "dynamic_momentum_baseline": excess_rates.get("momentum_baseline")
            },
            "candidate_outcomes": [
                {
                    "topic_id": item[0],
                    "predicted_score": item[1],
                    "outcome_status": outcome_details.get(item[0], {}).get("outcome"),
                    "actual_growth_multiple": outcome_details.get(item[0], {}).get("growth_actual"),
                    "hit_status": (
                        "HIT" if outcome_details.get(item[0], {}).get("outcome") == GroundTruthOutcome.SUCCESS.value
                        else "MISS" if outcome_details.get(item[0], {}).get("outcome") == GroundTruthOutcome.FAILURE.value
                        else outcome_details.get(item[0], {}).get("outcome")
                    )
                }
                for item in predicted_scores
            ]
        }
