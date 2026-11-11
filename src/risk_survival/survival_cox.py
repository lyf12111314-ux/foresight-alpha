"""企业生存与失败风险模型 (Kaplan-Meier & Cox Proportional Hazards Engine).

严格遵循 V1.1 数据真实性要求：
1. 隔离模拟数据生成，正式训练仅接受真实企业生存事件队列。
2. 明确定义破产、退市、重组和右删失。
3. 真实数据不足时明确返回 INSUFFICIENT_DATA，禁止输出虚构破产概率。
4. 严正免责声明：模型仅提供统计概率与敏感度归因，绝不断言企业必然失败。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from lifelines import KaplanMeierFitter, CoxPHFitter
from src.core.types import ModelTrainingStatus


class CompanySurvivalEngine:
    """企业经营生存与破产风险分析器"""

    # 严谨事件定义
    EVENT_DEFINITIONS = {
        0: "CENSORED_ACTIVE (正常存续/右删失，未发生终结性违约事件)",
        1: "DISTRESS_EVENT (包含：法院受理破产清算、财务指标连续违规被动退市、资不抵债进行司法重整)"
    }

    MIN_TOTAL_SAMPLES = 20    # 最低样本量门槛
    MIN_EVENTS_REQUIRED = 5   # 最低真实破产/违约事件样本门槛

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer
        self.kmf = KaplanMeierFitter()
        self.cph = CoxPHFitter(penalizer=penalizer)
        self.is_fitted: bool = False
        self.training_status: ModelTrainingStatus = ModelTrainingStatus.INSUFFICIENT_DATA
        self.is_synthetic: bool = False
        self.sample_count: int = 0
        self.event_count: int = 0

    def fit_verified(
        self,
        cohort_df: Optional[pd.DataFrame] = None,
        duration_col: str = "duration",
        event_col: str = "event"
    ) -> Dict[str, Any]:
        """正式训练接口：仅接受经过审计核实的真实企业生存事件队列"""
        self.is_synthetic = False
        if cohort_df is None or len(cohort_df) < self.MIN_TOTAL_SAMPLES:
            self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
            self.sample_count = len(cohort_df) if cohort_df is not None else 0
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "sample_count": self.sample_count,
                "min_required": self.MIN_TOTAL_SAMPLES,
                "message": f"真实企业生存样本不足 ({self.sample_count} < {self.MIN_TOTAL_SAMPLES})，模型保持未训练状态。"
            }

        if duration_col not in cohort_df.columns or event_col not in cohort_df.columns:
            self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "message": f"队列数据缺少必要列 [{duration_col}, {event_col}]。"
            }

        events_observed = int(cohort_df[event_col].sum())
        self.event_count = events_observed
        self.sample_count = len(cohort_df)

        if events_observed < self.MIN_EVENTS_REQUIRED:
            self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "events_observed": events_observed,
                "min_events_required": self.MIN_EVENTS_REQUIRED,
                "message": f"破产/重整真实事件数不足 ({events_observed} < {self.MIN_EVENTS_REQUIRED})，无法可靠估计 Cox 协变量系数。"
            }

        try:
            self.kmf.fit(durations=cohort_df[duration_col], event_observed=cohort_df[event_col], label="真实验证企业队列")
            self.cph.fit(cohort_df, duration_col=duration_col, event_col=event_col)
            self.is_fitted = True
            self.training_status = ModelTrainingStatus.TRAINED
            return {
                "status": ModelTrainingStatus.TRAINED.value,
                "sample_count": self.sample_count,
                "events_observed": self.event_count,
                "message": "已基于真实企业生存事件队列完成 Cox 比例风险拟合。"
            }
        except Exception as e:
            self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "error": str(e)
            }

    def fit(self, cohort_df: Optional[pd.DataFrame] = None):
        """兼容性接口 (拒绝无数据自动模拟训练)"""
        return self.fit_verified(cohort_df)

    def fit_synthetic_for_testing(self, n_samples: int = 250) -> Dict[str, Any]:
        """【仅供测试与算法演练使用】使用模拟数据测试 Cox 算法与 Kaplan-Meier 逻辑"""
        from tests.helpers.synthetic_survival import generate_synthetic_survival_cohort
        cohort_df = generate_synthetic_survival_cohort(n_samples=n_samples)
        self.kmf.fit(durations=cohort_df["duration"], event_observed=cohort_df["event"], label="合成测试队列")
        self.cph.fit(cohort_df, duration_col="duration", event_col="event")
        self.is_fitted = True
        self.training_status = ModelTrainingStatus.TRAINED
        self.is_synthetic = True
        self.sample_count = n_samples
        self.event_count = int(cohort_df["event"].sum())
        return {
            "status": "TRAINED_SYNTHETIC_TESTING_ONLY",
            "is_synthetic": True,
            "sample_count": n_samples,
            "warning": "使用合成模拟数据训练，结果不可用于实际投资/经营风险决策。"
        }

    def predict_company_survival(
        self,
        company_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """评估特定企业生存概率与危险比率。
        
        如果缺乏真实验证的生存模型，严禁虚构输出破产概率。
        """
        if self.training_status != ModelTrainingStatus.TRAINED:
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "is_verified": False,
                "is_synthetic": False,
                "partial_hazard_ratio": None,
                "risk_tier": "DATA_INSUFFICIENT",
                "survival_probabilities": {
                    "surv_1y": None,
                    "surv_3y": None,
                    "surv_5y": None
                },
                "risk_assessment": "缺乏足够真实企业违约/退市样本，暂不输出破产概率预测以防误导决策。",
                "hazard_factor_multipliers": {},
                "disclaimer": "本系统未在真实破产清算与违约事件样本上完成统计校准前，严格禁止虚构破产概率。"
            }

        sample_df = pd.DataFrame([{
            "cash_burn_months": company_metrics.get("cash_burn_months", 24.0),
            "debt_ratio": company_metrics.get("debt_ratio", 45.0),
            "gross_margin": company_metrics.get("gross_margin", 35.0),
            "competitor_count": company_metrics.get("competitor_count", 10.0)
        }])

        surv_func = self.cph.predict_survival_function(sample_df)

        times = [1.0, 3.0, 5.0]
        survival_probs = {}
        for t in times:
            closest_idx = np.argmin(np.abs(surv_func.index - t))
            p_surv = float(surv_func.iloc[closest_idx, 0])
            survival_probs[f"surv_{int(t)}y"] = round(p_surv, 3)

        partial_hazard = float(self.cph.predict_partial_hazard(sample_df).iloc[0])

        if partial_hazard > 2.0:
            risk_tier = "HIGH_DISTRESS_RISK"
            assessment = "企业现金消耗或偿债压力高企，抗行业内卷波动能力脆弱，处于高经营风险梯队。"
        elif partial_hazard > 1.0:
            risk_tier = "MEDIUM_RISK"
            assessment = "企业经营处于行业中游水平，需重点追踪自由现金流回正进度与应收账款周转。"
        else:
            risk_tier = "LOW_RISK_STABLE"
            assessment = "具备深厚毛利护城河与充裕现金安全垫，能够穿越产业出清周期。"

        hazard_ratios = {col: round(float(np.exp(coef)), 3) for col, coef in self.cph.params_.items()}

        return {
            "status": "PREDICTED",
            "is_verified": not self.is_synthetic,
            "is_synthetic": self.is_synthetic,
            "partial_hazard_ratio": round(partial_hazard, 2),
            "risk_tier": risk_tier,
            "survival_probabilities": survival_probs,
            "risk_assessment": assessment,
            "hazard_factor_multipliers": hazard_ratios,
            "disclaimer": "本预测仅反映基于公开财务指标的生存概率分布，不代表对企业存续性的绝对断言。"
        }
