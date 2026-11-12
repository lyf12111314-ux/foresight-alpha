"""Point-in-Time 滚动回测评估引擎 (Walk-Forward Rolling Backtest Engine).

严守回测铁律：
1. 严禁未来信息泄漏 (Lookahead Bias)，在历史时点 T 仅读取 published_at <= T 的公开数据。
2. 历史回测必须按时点独立训练模型：在历史切片点 T，训练集仅包含 label_period_end <= T 的已成熟真实标签。
3. 严禁使用 2026 年训练好的模型穿越回测 2021 年。
4. 若历史时点可用成熟样本不足以训练模型，标记为 MODEL_UNAVAILABLE_AS_OF_DATE，不输出虚假预测。
5. 事实标签严谨状态机：明确区分 SUCCESS, FAILURE, NOT_MATURED, INSUFFICIENT_EVIDENCE。
   未成熟样本或评估日超过审计锚定日 (audit_as_of_date) 时绝不计入失败，标记为 NOT_MATURED。
6. 真实 PR-AUC 与 Brier Score 测算，同候选池动态对比随机基准与动量基准。
7. 回测记录不可篡改增量存储，杜绝静默覆盖。
"""

import json
from datetime import date, datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.core.database import db_manager
from src.core.types import (
    BacktestMetricRecord,
    GroundTruthOutcome,
    PredictionStatus,
    ModelTrainingStatus,
    ModelStage
)
from src.backtest.metrics import BacktestMetricsCalculator
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.diffusion.lgbm_forecaster import LightGBMGrowthForecaster


class PointInTimeBacktestEngine:
    """严格时间截面滚动回测器"""

    def __init__(self):
        self.calc = BacktestMetricsCalculator()

    def get_historical_training_samples(self, cutoff_date: date, horizon_years: int = 3) -> pd.DataFrame:
        """提取截至 cutoff_date 时点已公开且标签已成熟的训练数据 (彻底杜绝未来信息)"""
        try:
            df = db_manager.query_df("""
                SELECT * FROM tech_training_samples
                WHERE (feature_published_at IS NULL OR feature_published_at <= ?)
                  AND (base_date <= ?)
            """, [cutoff_date, cutoff_date])
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return df

        target_col = f"target_{horizon_years}y"
        period_end_col = f"label_{horizon_years}y_period_end"

        # 严格过滤成熟标签：若 label_Xy_period_end > cutoff_date，该标签在历史时点尚未发生，置为 None
        if period_end_col in df.columns:
            matured_mask = df[period_end_col].notna() & (pd.to_datetime(df[period_end_col]).dt.date <= cutoff_date)
            df.loc[~matured_mask, target_col] = None
        else:
            base_dts = pd.to_datetime(df["base_date"]).dt.date
            matured = [
                d is not None and (date(d.year + horizon_years, d.month, min(d.day, 28)) <= cutoff_date)
                for d in base_dts
            ]
            df.loc[~pd.Series(matured, index=df.index), target_col] = None

        return df

    def train_historical_forecaster(self, cutoff_date: date, horizon_years: int = 3) -> LightGBMGrowthForecaster:
        """在历史截面 cutoff_date 训练独立的历史预测模型 (历史模型独立性)"""
        forecaster = LightGBMGrowthForecaster()
        forecaster.model_version = f"hist_{cutoff_date.strftime('%Y%m%d')}"

        hist_df = self.get_historical_training_samples(cutoff_date, horizon_years)
        target_col = f"target_{horizon_years}y"

        if hist_df.empty or target_col not in hist_df.columns:
            forecaster.training_status = ModelTrainingStatus.MODEL_UNAVAILABLE_AS_OF_DATE
            forecaster.model_stage = ModelStage.UNTRAINED
            forecaster.rejection_reason = f"历史时点 {cutoff_date.isoformat()} 尚无可获得的成熟训练样本。"
            return forecaster

        valid_df = hist_df[hist_df[target_col].notna()].copy()
        if len(valid_df) < 5:
            forecaster.training_status = ModelTrainingStatus.MODEL_UNAVAILABLE_AS_OF_DATE
            forecaster.model_stage = ModelStage.UNTRAINED
            forecaster.rejection_reason = f"历史时点 {cutoff_date.isoformat()} 可用成熟样本不足 ({len(valid_df)} < 5)，无法拟合历史模型。"
            return forecaster

        # 提取特征与多期限标签
        for c in forecaster.feature_names:
            if c not in valid_df.columns:
                valid_df[c] = 0.0

        X = valid_df[forecaster.feature_names]
        labels = {horizon_years: valid_df[target_col].values}
        for h in [1, 3, 5]:
            if h != horizon_years:
                h_col = f"target_{h}y"
                if h_col in valid_df.columns and valid_df[h_col].notna().sum() == len(valid_df):
                    labels[h] = valid_df[h_col].values
                else:
                    labels[h] = valid_df[target_col].values

        forecaster.fit_experimental(
            features=X,
            labels_by_horizon=labels,
            dataset_version=f"pit_hist_{cutoff_date.strftime('%Y%m%d')}"
        )
        return forecaster

    def run_walk_forward_backtest(
        self,
        cutoff_date: date = date(2021, 12, 31),
        evaluation_date: date = date(2024, 12, 31),
        horizon_years: int = 3,
        audit_as_of_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """在历史切片点 cutoff_date 独立训练历史模型并运行预测，在 evaluation_date 核验真实结果"""
        run_id = f"BT_{cutoff_date.strftime('%Y%m%d')}_{evaluation_date.strftime('%Y%m%d')}"
        review_audit_date = audit_as_of_date or date(2026, 11, 11)

        # 1. 独立训练截至 cutoff_date 的历史时点模型 (物理隔离未来模型权重与未来数据)
        hist_forecaster = self.train_historical_forecaster(cutoff_date, horizon_years)
        pool_manager = OpportunityPoolManager(lgbm_forecaster=hist_forecaster)

        # 在 cutoff_date 生成当时的预测候选池
        candidates = pool_manager.build_opportunity_pool(cutoff_date=cutoff_date)

        # 预测得分元组列表 [(topic_id, research_score)]
        predicted_scores = [(c["topic_id"], c.get("research_score", c.get("composite_score", 0.0))) for c in candidates]
        predicted_probs = [c.get("calibrated_probability") for c in candidates]

        # 2. 从真实历史数据中提取事实标签 (Ground Truth Outcome)
        try:
            expected_maturity_date = date(cutoff_date.year + horizon_years, cutoff_date.month, cutoff_date.day)
        except ValueError:
            expected_maturity_date = date(cutoff_date.year + horizon_years, cutoff_date.month, 28)

        true_labels = {}
        outcome_details = {}
        cutoff_dt = pd.to_datetime(cutoff_date)
        eval_dt = pd.to_datetime(evaluation_date)

        # 必须满足：评估日已达到成熟观察期截止点，且评估日不超过当前历史数据审计锚定日 (杜绝未成熟与前视偏差)
        is_matured_horizon = (evaluation_date >= expected_maturity_date and evaluation_date <= review_audit_date)

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

                    if sub_cutoff.empty or sub_eval.empty:
                        outcome = GroundTruthOutcome.INSUFFICIENT_EVIDENCE
                        growth_actual = None
                    else:
                        val_cutoff = float(sub_cutoff["papers_count"].iloc[-1])
                        val_eval = float(sub_eval["papers_count"].iloc[-1])
                        if val_cutoff <= 0:
                            outcome = GroundTruthOutcome.INSUFFICIENT_EVIDENCE
                            growth_actual = None
                        else:
                            growth_actual = float(val_eval / val_cutoff)
                            if growth_actual >= 1.9:
                                outcome = GroundTruthOutcome.SUCCESS
                            else:
                                outcome = GroundTruthOutcome.FAILURE
                else:
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
        real_pr_auc = self.calc.pr_auc(predicted_scores, true_labels)

        evaluable_probs = []
        evaluable_actuals = []
        for item, p in zip(predicted_scores, predicted_probs):
            ot = true_labels.get(item[0])
            if self.calc._is_evaluable(ot):
                evaluable_probs.append(p)
                evaluable_actuals.append(1 if self.calc._is_positive(ot) else 0)

        brier = self.calc.brier_score(evaluable_probs, evaluable_actuals)

        excess_rates = self.calc.calculate_excess_hit_rate(
            model_precision=prec_3,
            candidate_items=candidates,
            true_labels=true_labels,
            k=3
        )

        # 4. 持久化回测记录到 DuckDB (版本化不可篡改存储)
        now = datetime.utcnow()
        conn = db_manager.get_connection()
        try:
            existing_runs = conn.execute(
                "SELECT run_id FROM backtest_runs WHERE run_id = ? OR run_id LIKE ?",
                [run_id, f"{run_id}_v%"]
            ).fetchall()
            if existing_runs:
                version_num = len(existing_runs) + 1
                actual_run_id = f"{run_id}_v{version_num}"
            else:
                actual_run_id = run_id

            conn.execute("""
                INSERT INTO backtest_runs
                (run_id, cutoff_date, evaluation_date, horizon_years, precision_3, precision_5, recall_5, pr_auc, brier_score, excess_vs_random, excess_vs_momentum, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                actual_run_id, cutoff_date, evaluation_date, horizon_years,
                prec_3, prec_5, rec_5, real_pr_auc,
                brier, excess_rates["excess_vs_random"], excess_rates["excess_vs_momentum"], now
            ])
        finally:
            conn.close()

        forecast_status = (
            PredictionStatus.MODEL_UNAVAILABLE_AS_OF_DATE.value
            if hist_forecaster.training_status == ModelTrainingStatus.MODEL_UNAVAILABLE_AS_OF_DATE
            else hist_forecaster.training_status.value
        )

        return {
            "run_id": actual_run_id,
            "cutoff_date": cutoff_date.isoformat(),
            "evaluation_date": evaluation_date.isoformat(),
            "horizon_years": horizon_years,
            "expected_maturity_date": expected_maturity_date.isoformat(),
            "is_horizon_matured": is_matured_horizon,
            "forecast_status": forecast_status,
            "historical_model_version": hist_forecaster.model_version,
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
