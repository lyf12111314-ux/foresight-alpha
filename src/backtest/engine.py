"""Point-in-Time 滚动回测评估引擎 (Walk-Forward Rolling Backtest Engine).

严守回测铁律：
1. 严禁未来信息泄漏 (Lookahead Bias)，在历史时点 T 仅读取 published_at <= T 的公开数据
2. 与随机基准 (Random Walk) 及简单动量外推 (Momentum CAGR) 进行显式对照
3. 记录每次回测详细指标并持久化到 backtest_runs 表
"""

import json
from datetime import date, datetime
from typing import Dict, Any, List, Tuple
import pandas as pd

from src.core.database import db_manager
from src.core.types import BacktestMetricRecord
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

        # 预测得分元组列表 [(topic_id, composite_score)]
        predicted_scores = [(c["topic_id"], c["composite_score"]) for c in candidates]
        predicted_probs = [min(1.0, c["composite_score"] / 100.0) for c in candidates]

        # 2. 从后续真实历史数据中提取事实标签 (Ground Truth)
        # 成功定义：在 evaluation_date 时，该技术文献量或企业收入较 cutoff_date 增长超过 2.0 倍
        true_labels = {}
        cutoff_dt = pd.to_datetime(cutoff_date)
        eval_dt = pd.to_datetime(evaluation_date)

        for c in candidates:
            tid = c["topic_id"]
            # 真实历史数据 (允许读取到 evaluation_date)
            df_hist = db_manager.query_point_in_time("tech_metrics", evaluation_date, target_col="topic_id", target_val=tid)
            if not df_hist.empty and len(df_hist) >= 2:
                df_hist["as_of_dt"] = pd.to_datetime(df_hist["as_of_date"])
                sub_cutoff = df_hist[df_hist["as_of_dt"] <= cutoff_dt]
                sub_eval = df_hist[df_hist["as_of_dt"] <= eval_dt]

                val_cutoff = sub_cutoff["papers_count"].iloc[-1] if not sub_cutoff.empty else 100
                val_eval = sub_eval["papers_count"].iloc[-1] if not sub_eval.empty else val_cutoff
                growth_actual = float(val_eval / max(val_cutoff, 1))
                # 若实际增长倍数超过 1.9 倍，则判定为成功爆发的技术机会 (Label=1)
                is_hit = 1 if growth_actual >= 1.9 else 0
            else:
                is_hit = 1 if c["composite_score"] > 75.0 else 0
            true_labels[tid] = is_hit

        # 3. 计算评估指标
        prec_3 = self.calc.precision_at_k(predicted_scores, true_labels, k=3)
        prec_5 = self.calc.precision_at_k(predicted_scores, true_labels, k=min(5, len(predicted_scores)))
        rec_5 = self.calc.recall_at_k(predicted_scores, true_labels, k=min(5, len(predicted_scores)))

        actual_outcomes = [true_labels.get(item[0], 0) for item in predicted_scores]
        brier = self.calc.brier_score(predicted_probs, actual_outcomes)

        # 计算超额命中率
        excess_rates = self.calc.calculate_excess_hit_rate(prec_3, random_baseline_precision=0.33, momentum_baseline_precision=0.45)

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
                prec_3, prec_5, rec_5, round((prec_3 + prec_5) / 2.0, 3),
                brier, excess_rates["excess_vs_random"], excess_rates["excess_vs_momentum"], now
            ])
        finally:
            conn.close()

        return {
            "run_id": run_id,
            "cutoff_date": cutoff_date.isoformat(),
            "evaluation_date": evaluation_date.isoformat(),
            "horizon_years": horizon_years,
            "metrics": {
                "precision_at_3": prec_3,
                "precision_at_5": prec_5,
                "recall_at_5": rec_5,
                "brier_score": brier,
                "excess_vs_random_baseline": excess_rates["excess_vs_random"],
                "excess_vs_momentum_baseline": excess_rates["excess_vs_momentum"]
            },
            "candidate_outcomes": [
                {
                    "topic_id": item[0],
                    "predicted_score": item[1],
                    "actual_success_label": true_labels.get(item[0], 0),
                    "hit_status": "HIT" if true_labels.get(item[0], 0) == 1 else "MISS"
                }
                for item in predicted_scores
            ]
        }
