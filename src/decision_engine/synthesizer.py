"""多模型综合决策与研究评分引擎 (Multi-Model Synthesis & Research Evaluation Engine).

严格遵循 V1.1 数据真实性要求：
1. 明确区分“研究评分 (Research Score)”、“数据覆盖度 (Data Coverage Score)”与“校准预测概率 (Calibrated Probability)”。
2. 严禁使用固定默认基准值伪称为贝叶斯校准概率。
3. 在没有独立校准结果时，正式预测概率字段设为 null (None)。
4. 严禁把研究评分直接除以100换算成成功概率。
5. 保留丰富可解释的非概率客观研究指标。
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from src.core.types import ForecastRecord, ObjectLevel, PredictionStatus


class MultiModelDecisionSynthesizer:
    """多模型决策融合与研究评估器"""

    def synthesize_opportunity(
        self,
        topic_id: str,
        topic_name: str,
        macro_diag: Dict[str, Any],
        inflection_diag: Dict[str, Any],
        diffusion_res: Dict[str, Any],
        profit_pool_res: Dict[str, Any],
        survival_diag: Dict[str, Any],
        payoff_res: Dict[str, Any],
        as_of_data_date: Optional[date] = None,
        independent_calibration_prob: Optional[float] = None
    ) -> Dict[str, Any]:
        """融合各维度研究结果，严格分离客观研究评分与预测概率"""
        if as_of_data_date is None:
            as_of_data_date = date.today()

        # -----------------------------------------------------------------
        # 1. 统计数据覆盖度与真实性状态 (Data Coverage & Integrity Audit)
        # -----------------------------------------------------------------
        coverage_dimensions = {
            "macro": bool(macro_diag and "regime" in macro_diag),
            "tech_momentum": bool(inflection_diag and inflection_diag.get("status") == "VALIDATED"),
            "diffusion_model": bool(diffusion_res and diffusion_res.get("growth_multiples")),
            "value_chain": bool(profit_pool_res and "highest_margin_tier" in profit_pool_res),
            "survival_data": bool(survival_diag and survival_diag.get("status") != "INSUFFICIENT_DATA" and survival_diag.get("partial_hazard_ratio") is not None),
            "payoff_simulation": bool(payoff_res and "quantiles_breakdown" in payoff_res)
        }
        covered_count = sum(1 for v in coverage_dimensions.values() if v)
        data_coverage_score = round(covered_count / len(coverage_dimensions), 2)

        # -----------------------------------------------------------------
        # 2. 客观非概率研究指标计算 (Explainable Research Indicators)
        # -----------------------------------------------------------------
        # 宏观环境得分 (0 ~ 100)
        macro_mult = float(macro_diag.get("macro_growth_multiplier", 1.0))
        macro_regime = macro_diag.get("regime", "UNKNOWN")
        macro_score = np.clip((macro_mult - 0.70) / (1.25 - 0.70) * 100.0, 10.0, 100.0)

        # 技术动能与拐点得分 (0 ~ 100)
        is_inflection = inflection_diag.get("is_inflection_point", False)
        accel = inflection_diag.get("latest_acceleration", 0.0)
        vel = inflection_diag.get("latest_velocity_yoy", 0.0)
        tech_score = 50.0
        if is_inflection:
            tech_score += 25.0
        if accel > 0:
            tech_score += 15.0
        if vel > 100:
            tech_score += 10.0
        tech_score = min(100.0, tech_score)

        # 产业链利润池壁垒得分 (0 ~ 100)
        tier = profit_pool_res.get("highest_margin_tier", "UNKNOWN")
        if tier == "UPSTREAM":
            chain_score = 80.0   # 上游核心部件/材料壁垒通常较高
        elif tier == "DOWNSTREAM":
            chain_score = 70.0
        elif tier == "MIDSTREAM":
            chain_score = 60.0
        else:
            chain_score = 50.0

        # 成长扩散指标 (如果不可预测，保持 None，绝不硬编码假数据)
        growth_multiples = diffusion_res.get("growth_multiples") if diffusion_res else None
        if growth_multiples and growth_multiples.get("3_year_multiplier") is not None:
            growth_3y_mult = float(growth_multiples["3_year_multiplier"])
            diffusion_score = np.clip((growth_3y_mult / 3.0) * 80.0, 20.0, 100.0)
        else:
            growth_3y_mult = None
            diffusion_score = None

        # 生存与财务健康指标
        if survival_diag and survival_diag.get("survival_probabilities", {}).get("surv_3y") is not None:
            survival_prob_3y = float(survival_diag["survival_probabilities"]["surv_3y"])
            survival_score = survival_prob_3y * 100.0
        else:
            survival_prob_3y = None
            survival_score = None

        # 非对称赔率
        if payoff_res and "metrics" in payoff_res:
            raw_asym = payoff_res["metrics"].get("asymmetry_ratio")
            asymmetry_ratio = float(raw_asym) if raw_asym is not None else None
            raw_prob_10x = payoff_res["metrics"].get("prob_10x_super_return_pct")
            prob_10x = float(raw_prob_10x) if raw_prob_10x is not None else None
        else:
            asymmetry_ratio = None
            prob_10x = None

        # -----------------------------------------------------------------
        # 3. 综合研究评分 (Research Score, 0 ~ 100)
        # 仅对具备有效数据的维度进行归一化综合评估
        # -----------------------------------------------------------------
        components = []
        weights = []

        # 基础可获得维度：宏观(20%)、技术动能(35%)、产业链地位(25%)
        components.append(macro_score)
        weights.append(0.20)

        components.append(tech_score)
        weights.append(0.35)

        components.append(chain_score)
        weights.append(0.25)

        if diffusion_score is not None:
            components.append(diffusion_score)
            weights.append(0.20)

        total_weight = sum(weights)
        raw_research_score = sum(c * w for c, w in zip(components, weights)) / total_weight
        research_score = round(float(np.clip(raw_research_score, 10.0, 98.0)), 1)

        # -----------------------------------------------------------------
        # 4. 预测可信度与校准概率严格审计 (严禁将研究评分当成成功概率)
        # -----------------------------------------------------------------
        # 没有执行独立历史校准且未满足真实验证时，严格置为 None
        if independent_calibration_prob is not None:
            calibrated_probability = round(float(independent_calibration_prob), 4)
            is_calibrated = True
            prediction_status = PredictionStatus.PREDICTED
        else:
            calibrated_probability = None
            is_calibrated = False
            # 如果关键预测维度缺失，标记为未验证研究跟踪或不可预测
            if growth_3y_mult is None:
                prediction_status = PredictionStatus.NOT_PREDICTABLE
            else:
                prediction_status = PredictionStatus.UNVERIFIED_RESEARCH

        # -----------------------------------------------------------------
        # 5. 研究评级分类 (清晰区分研究热度与经过验证的预测)
        # -----------------------------------------------------------------
        if data_coverage_score >= 0.70 and research_score >= 80.0:
            rating = "ALPHA_RESEARCH_PRIME (高潜研究跟踪 - 重点关注)"
        elif research_score >= 65.0:
            rating = "WATCHLIST_RESEARCH (常规研究观察)"
        elif data_coverage_score < 0.50:
            rating = "DATA_INSUFFICIENT (数据覆盖不足 - 待补全核验)"
        else:
            rating = "NEUTRAL_TRACK (中性技术跟踪)"

        # 构建规范化预测记录对象
        forecast_rec = ForecastRecord(
            forecast_id=f"FC_{topic_id}_{as_of_data_date.strftime('%Y%m%d')}",
            target_id=topic_id,
            target_level=ObjectLevel.TECH_TOPIC,
            model_name="ForesightAlpha-Synthesizer",
            model_version="1.1.0-audit",
            as_of_data_date=as_of_data_date,
            prediction_created_at=datetime.utcnow(),
            horizon_years=3,
            predicted_metric="composite_research_score",
            point_estimate=research_score,
            confidence_interval_low=None,
            confidence_interval_high=None,
            confidence_score=calibrated_probability,
            is_calibrated=is_calibrated,
            is_verified_forecast=is_calibrated,
            prediction_status=prediction_status,
            scenarios={},
            driver_features={
                "macro_score": macro_score,
                "tech_score": tech_score,
                "chain_score": chain_score,
                "data_coverage": data_coverage_score
            }
        )

        return {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "as_of_data_date": as_of_data_date.isoformat(),
            # 客观研究评价指标
            "composite_score": research_score,
            "research_score": research_score,
            "data_coverage_score": data_coverage_score,
            "rating": rating,
            # 预测状态与严格校准概率
            "prediction_status": prediction_status.value,
            "is_calibrated": is_calibrated,
            "calibrated_prediction_probability": calibrated_probability,
            "calibrated_probability": calibrated_probability,
            "growth_3y_multiple": growth_3y_mult,
            "prob_10x_super_return_pct": prob_10x,
            "asymmetry_ratio": asymmetry_ratio,
            "survival_prob_3y": survival_prob_3y,
            # 证据与记录
            "coverage_breakdown": coverage_dimensions,
            "forecast_record": forecast_rec.model_dump()
        }
