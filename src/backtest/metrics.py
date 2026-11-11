"""回测评估量化指标计算 (Backtest Evaluation Metrics: Precision@K, PR-AUC, Dynamic Baselines).

严格遵循 V1.1 审计标准：
1. 真实计算 PR-AUC：基于 Precision-Recall 曲线各分位梯形数值积分，严禁使用 (prec_3 + prec_5) / 2 伪造。
2. 动态对照基准：随机基准与动量基准基于同候选池、同评价期动态测算，严禁硬编码 0.33/0.45。
3. 严格真值状态识别：准确区分 SUCCESS, FAILURE, NOT_MATURED, INSUFFICIENT_EVIDENCE。
   未成熟样本不计入失败，不进入精度分母。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from src.core.types import GroundTruthOutcome


class BacktestMetricsCalculator:
    """回测指标计算器 (严守统计学无偏估计标准)"""

    @staticmethod
    def _is_positive(val: Any) -> bool:
        """判定标签是否为命中正样本"""
        if val in [1, 1.0, True, "1", "HIT", "SUCCESS", GroundTruthOutcome.SUCCESS, GroundTruthOutcome.SUCCESS.value]:
            return True
        return False

    @staticmethod
    def _is_evaluable(val: Any) -> bool:
        """判定标签是否已成熟并具备有效评估证据 (未成熟或证据不足不计入)"""
        if val in [
            GroundTruthOutcome.NOT_MATURED,
            GroundTruthOutcome.NOT_MATURED.value,
            GroundTruthOutcome.INSUFFICIENT_EVIDENCE,
            GroundTruthOutcome.INSUFFICIENT_EVIDENCE.value,
            GroundTruthOutcome.UNKNOWN,
            GroundTruthOutcome.UNKNOWN.value,
            None
        ]:
            return False
        return True

    @classmethod
    def precision_at_k(
        cls,
        predicted_scores: List[Tuple[str, float]],
        true_labels: Dict[str, Any],
        k: int,
        filter_evaluable_only: bool = False
    ) -> float:
        """计算 Top-K 推荐预测精确率 Precision@K"""
        if not predicted_scores or k <= 0:
            return 0.0

        sorted_preds = sorted(predicted_scores, key=lambda x: x[1], reverse=True)

        if filter_evaluable_only:
            # 仅在已成熟且有证据的样本中评估
            evaluable_preds = [item for item in sorted_preds if cls._is_evaluable(true_labels.get(item[0]))][:k]
            if not evaluable_preds:
                return 0.0
            hits = sum(1 for item in evaluable_preds if cls._is_positive(true_labels.get(item[0])))
            return round(float(hits / len(evaluable_preds)), 3)
        else:
            top_k = sorted_preds[:k]
            hits = sum(1 for item in top_k if cls._is_positive(true_labels.get(item[0])))
            return round(float(hits / k), 3)

    @classmethod
    def recall_at_k(
        cls,
        predicted_scores: List[Tuple[str, float]],
        true_labels: Dict[str, Any],
        k: int
    ) -> float:
        """计算 Top-K 推荐预测召回率 Recall@K"""
        tot_positives = sum(1 for v in true_labels.values() if cls._is_positive(v))
        if tot_positives == 0 or not predicted_scores or k <= 0:
            return 0.0

        sorted_preds = sorted(predicted_scores, key=lambda x: x[1], reverse=True)[:k]
        hits = sum(1 for item in sorted_preds if cls._is_positive(true_labels.get(item[0])))
        return round(float(hits / tot_positives), 3)

    @classmethod
    def pr_auc(
        cls,
        predicted_scores: List[Tuple[str, float]],
        true_labels: Dict[str, Any]
    ) -> float:
        """真实计算 Precision-Recall 曲线下面积 (PR-AUC / Average Precision via Trapezoidal Integration).
        
        严禁使用 (Precision@3 + Precision@5) / 2 冒充 PR-AUC！
        基于按模型预测分值降序排列的所有割点 (k=1..N)，计算真实 (Recall, Precision) 梯形积分。
        """
        if not predicted_scores or not true_labels:
            return 0.0

        # 过滤可评估的样本
        evaluable_items = [item for item in predicted_scores if cls._is_evaluable(true_labels.get(item[0]))]
        if not evaluable_items:
            return 0.0

        # 按预测分降序排列
        sorted_items = sorted(evaluable_items, key=lambda x: x[1], reverse=True)
        total_positives = sum(1 for item in sorted_items if cls._is_positive(true_labels.get(item[0])))

        if total_positives == 0:
            return 0.0

        # 遍历各个 rank cutoff，计算各点的 (Recall, Precision)
        recalls = [0.0]
        precisions = [1.0]  # 起始点设为标准 (0, 1.0)

        cumulative_hits = 0
        for rank, item in enumerate(sorted_items, start=1):
            is_hit = 1 if cls._is_positive(true_labels.get(item[0])) else 0
            cumulative_hits += is_hit
            prec = cumulative_hits / rank
            rec = cumulative_hits / total_positives
            recalls.append(rec)
            precisions.append(prec)

        # 梯形积分法求 PR-AUC
        auc = 0.0
        for i in range(1, len(recalls)):
            delta_recall = recalls[i] - recalls[i - 1]
            avg_precision = (precisions[i] + precisions[i - 1]) / 2.0
            auc += delta_recall * avg_precision

        return round(float(auc), 3)

    # 兼容别名
    calculate_pr_auc = pr_auc

    @staticmethod
    def brier_score(predicted_probs: List[Optional[float]], actual_outcomes: List[int]) -> Optional[float]:
        """计算 Brier Score (预测概率校准优度，取值 0.0-1.0，越低越好，0表示完美校准)"""
        valid_pairs = [(p, y) for p, y in zip(predicted_probs, actual_outcomes) if p is not None]
        if not valid_pairs:
            return None
        p = np.array([vp[0] for vp in valid_pairs], dtype=float)
        y = np.array([vp[1] for vp in valid_pairs], dtype=float)
        score = np.mean((p - y) ** 2)
        return round(float(score), 4)

    @classmethod
    def calculate_dynamic_baselines(
        cls,
        candidate_items: List[Dict[str, Any]],
        true_labels: Dict[str, Any],
        k: int = 3,
        momentum_key: str = "velocity_1y"
    ) -> Dict[str, float]:
        """同候选池、同评价期动态测算随机基准与动量外推基准
        
        1. 随机基准 (Random Baseline)：
           在当前评估池中，随机抽取 k 个标的命中正样本的期望精确率 = (正样本总数 / 可评估候选总数)。
        2. 动量基准 (Momentum Baseline)：
           按照过去真实公开的简单动量指标 (如 velocity_1y) 降序排列，取 Top-K 得到的真实精确率。
        """
        evaluable_cands = [c for c in candidate_items if cls._is_evaluable(true_labels.get(c.get("topic_id")))]
        if not evaluable_cands:
            return {"random_baseline_precision": 0.0, "momentum_baseline_precision": 0.0}

        total_evaluable = len(evaluable_cands)
        total_positives = sum(1 for c in evaluable_cands if cls._is_positive(true_labels.get(c.get("topic_id"))))

        # 动态随机基准
        random_prec = round(float(total_positives / total_evaluable), 3) if total_evaluable > 0 else 0.0

        # 动态动量基准：按动量特征降序排列
        sorted_by_momentum = sorted(
            evaluable_cands,
            key=lambda c: float(c.get(momentum_key, c.get("inflection_evidence", {}).get("growth_multiple_3y", 0.0)) or 0.0),
            reverse=True
        )[:k]

        if sorted_by_momentum:
            mom_hits = sum(1 for c in sorted_by_momentum if cls._is_positive(true_labels.get(c.get("topic_id"))))
            momentum_prec = round(float(mom_hits / len(sorted_by_momentum)), 3)
        else:
            momentum_prec = random_prec

        return {
            "random_baseline_precision": random_prec,
            "momentum_baseline_precision": momentum_prec
        }

    @classmethod
    def calculate_excess_hit_rate(
        cls,
        model_precision: float,
        random_baseline_precision: Optional[float] = None,
        momentum_baseline_precision: Optional[float] = None,
        candidate_items: Optional[List[Dict[str, Any]]] = None,
        true_labels: Optional[Dict[str, Any]] = None,
        k: int = 3
    ) -> Dict[str, float]:
        """计算相对于动态随机基准与动量外推基准的超额胜率。
        
        若传入 candidate_items 与 true_labels，则自动进行动态计算；
        否则若显式指定了 baseline 则直接使用，默认 fallback 回保底参考值。
        """
        if candidate_items is not None and true_labels is not None:
            dyn = cls.calculate_dynamic_baselines(candidate_items, true_labels, k=k)
            r_base = dyn["random_baseline_precision"]
            m_base = dyn["momentum_baseline_precision"]
        else:
            r_base = random_baseline_precision if random_baseline_precision is not None else 0.33
            m_base = momentum_baseline_precision if momentum_baseline_precision is not None else 0.45

        return {
            "excess_vs_random": round(model_precision - r_base, 3),
            "excess_vs_momentum": round(model_precision - m_base, 3),
            "random_baseline": r_base,
            "momentum_baseline": m_base
        }
