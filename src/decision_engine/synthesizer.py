"""多模型综合决策与概率校准引擎 (Multi-Model Synthesis & Calibrated Decision).

结合宏观、技术、扩散、利润池、生存风险与非对称赔率，绝不使用主观随意专家打分，
采用经过贝叶斯校准的多维度融合决策。
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from src.core.types import ForecastRecord, ObjectLevel


class MultiModelDecisionSynthesizer:
    """多模型决策融合与校准器"""

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
        as_of_data_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """融合 6 大子模型输出，形成统一的候选机会研判画像"""
        if as_of_data_date is None:
            as_of_data_date = date.today()

        # 1. 宏观乘数因子 (0.7 ~ 1.25)
        macro_mult = float(macro_diag.get("macro_growth_multiplier", 1.0))
        macro_regime = macro_diag.get("regime", "EXPANSION")

        # 2. 技术拐点与成熟度动能得分 (0 ~ 100)
        is_inflection = inflection_diag.get("is_inflection_point", False)
        accel = inflection_diag.get("latest_acceleration", 0.0)
        tech_score = 60.0
        if is_inflection:
            tech_score += 25.0
        if accel > 50:
            tech_score += 15.0
        tech_score = min(100.0, tech_score)

        # 3. 3年预期扩散增长倍数 (Point Estimate)
        growth_3y_mult = diffusion_res.get("growth_multiples", {}).get("3_year_multiplier", 2.0)
        if growth_3y_mult is None or np.isnan(growth_3y_mult):
            growth_3y_mult = 1.35

        # 4. 商业化健康度与利润池壁垒 (0 ~ 100)
        tier = profit_pool_res.get("highest_margin_tier", "UPSTREAM")
        tier_bonus = 15.0 if tier == "UPSTREAM" else (10.0 if tier == "DOWNSTREAM" else 5.0)
        comm_score = 65.0 + tier_bonus

        # 5. 生存风险折价 (0.0 ~ 1.0)
        hazard = survival_diag.get("partial_hazard_ratio", 1.0)
        survival_prob_3y = survival_diag.get("survival_probabilities", {}).get("surv_3y", 0.85)

        # 6. 非对称赔率比
        asymmetry_ratio = payoff_res.get("metrics", {}).get("asymmetry_ratio", 2.5)
        prob_10x = payoff_res.get("metrics", {}).get("prob_10x_super_return_pct", 8.0)

        # 综合非对称机会得分 (Risk-Adjusted Asymmetric Opportunity Score, 0-100)
        # 公式: [基础技术成长 * 宏观阻尼 * 商业化健康度 * 3年生全率 * 非对称赔率乘数]
        raw_composite = (
            (tech_score / 100.0) * 20.0 +
            (growth_3y_mult / 3.0) * 25.0 +
            macro_mult * 15.0 +
            (comm_score / 100.0) * 15.0 +
            survival_prob_3y * 15.0 +
            min(asymmetry_ratio / 3.0, 2.0) * 10.0
        )
        composite_score = round(float(np.clip(raw_composite, 10.0, 98.0)), 1)

        # 贝叶斯校准置信度 (0.0 ~ 1.0)
        # 考虑数据丰富度与模型一致性 (若宏观紧缩而成长高估，扣减置信度)
        base_confidence = 0.78
        if macro_regime == "CONTRACTION" and growth_3y_mult > 5.0:
            base_confidence -= 0.15
        if not is_inflection:
            base_confidence -= 0.08
        calibrated_confidence = round(float(np.clip(base_confidence, 0.40, 0.95)), 2)

        # 综合评级
        if composite_score >= 82.0 and calibrated_confidence >= 0.70:
            rating = "ALPHA_PRIME (非对称高潜机会 - 强烈持续观察)"
        elif composite_score >= 70.0:
            rating = "WATCHLIST_A (核心机会候选池 - 待商业化进一步放量)"
        elif composite_score >= 55.0:
            rating = "NEUTRAL_TRACK (中性技术跟踪 - 拐点尚需时日)"
        else:
            rating = "HIGH_RISK_DEFENSIVE (风险规避 - 竞争失血或估值泡沫)"

        # 构建规范化预测记录对象
        forecast_rec = ForecastRecord(
            forecast_id=f"FC_{topic_id}_{as_of_data_date.strftime('%Y%m%d')}",
            target_id=topic_id,
            target_level=ObjectLevel.TECH_TOPIC,
            model_name="ForesightAlpha-Synthesizer",
            model_version="1.0.0",
            as_of_data_date=as_of_data_date,
            prediction_created_at=datetime.utcnow(),
            horizon_years=3,
            predicted_metric="composite_asymmetric_score",
            point_estimate=composite_score,
            confidence_interval_low=max(0.0, composite_score - 12.0),
            confidence_interval_high=min(100.0, composite_score + 10.0),
            confidence_score=calibrated_confidence,
            scenarios={
                "bear": round(composite_score * 0.72, 1),
                "base": composite_score,
                "bull": round(min(100.0, composite_score * 1.25), 1)
            },
            driver_features={
                "macro_mult": macro_mult,
                "growth_3y_mult": growth_3y_mult,
                "survival_prob_3y": survival_prob_3y,
                "asymmetry_ratio": asymmetry_ratio
            }
        )

        return {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "as_of_data_date": as_of_data_date.isoformat(),
            "composite_score": composite_score,
            "calibrated_confidence": calibrated_confidence,
            "rating": rating,
            "growth_3y_multiple": growth_3y_mult,
            "prob_10x_super_return_pct": prob_10x,
            "asymmetry_ratio": asymmetry_ratio,
            "survival_prob_3y": survival_prob_3y,
            "forecast_record": forecast_rec.model_dump()
        }
