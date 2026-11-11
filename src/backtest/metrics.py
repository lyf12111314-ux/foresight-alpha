"""回测评估量化指标计算 (Backtest Evaluation Metrics: Precision@K, Brier Score, PR-AUC)."""

import numpy as np
from typing import Dict, Any, List, Tuple


class BacktestMetricsCalculator:
    """回测指标计算器"""

    @staticmethod
    def precision_at_k(predicted_scores: List[Tuple[str, float]], true_labels: Dict[str, int], k: int) -> float:
        """计算 Top-K 推荐预测精确率 Precision@K"""
        if not predicted_scores or k <= 0:
            return 0.0
        # 按预测得分降序排列
        sorted_preds = sorted(predicted_scores, key=lambda x: x[1], reverse=True)[:k]
        hits = sum(true_labels.get(item[0], 0) for item in sorted_preds)
        return round(float(hits / k), 3)

    @staticmethod
    def recall_at_k(predicted_scores: List[Tuple[str, float]], true_labels: Dict[str, int], k: int) -> float:
        """计算 Top-K 推荐预测召回率 Recall@K"""
        tot_positives = sum(true_labels.values())
        if tot_positives == 0 or not predicted_scores:
            return 0.0
        sorted_preds = sorted(predicted_scores, key=lambda x: x[1], reverse=True)[:k]
        hits = sum(true_labels.get(item[0], 0) for item in sorted_preds)
        return round(float(hits / tot_positives), 3)

    @staticmethod
    def brier_score(predicted_probs: List[float], actual_outcomes: List[int]) -> float:
        """计算 Brier Score (预测概率校准优度，取值 0.0-1.0，越低越好，0表示完美校准)
        
        B = (1/N) * sum((p_i - y_i)^2)
        """
        if len(predicted_probs) == 0:
            return 0.0
        p = np.array(predicted_probs, dtype=float)
        y = np.array(actual_outcomes, dtype=float)
        score = np.mean((p - y) ** 2)
        return round(float(score), 4)

    @staticmethod
    def calculate_excess_hit_rate(
        model_precision: float,
        random_baseline_precision: float = 0.33,
        momentum_baseline_precision: float = 0.45
    ) -> Dict[str, float]:
        """计算相对于随机挑选与简单动量外推的超额胜率"""
        return {
            "excess_vs_random": round(model_precision - random_baseline_precision, 3),
            "excess_vs_momentum": round(model_precision - momentum_baseline_precision, 3)
        }
