"""企业生存与失败风险模型 (Kaplan-Meier & Cox Proportional Hazards Engine).

严格遵循 V1.1 数据真实性要求：
1. 建立正式训练准入门禁，fit_verified 仅接受经过审计核实的 VERIFIED 队列。
2. MANUAL_REFERENCE 历史队列仅允许进入 fit_experimental 实验模式，标记为 EXPERIMENTAL，绝不产生 is_verified=True。
3. 明确破产 (BANKRUPTCY)、退市 (DELISTING)、重组 (RESTRUCTURING) 与右删失 (CENSORED) 的定义。
4. 真实数据不足或未核验时，拒绝输出正式企业破产概率以防误导。
5. 区分四重验证状态：is_algorithm_fitted, is_data_verified, is_out_of_sample_validated, is_calibrated。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from lifelines import KaplanMeierFitter, CoxPHFitter
from src.core.types import (
    ModelTrainingStatus,
    ModelStage,
    DataQualityStatus,
    DistressEventType,
    ModelVerificationInfo
)


class CompanySurvivalEngine:
    """企业经营生存与破产风险分析器"""

    # 严谨事件定义
    EVENT_DEFINITIONS = {
        0: DistressEventType.CENSORED_ACTIVE.value + " (正常存续/右删失，未发生终结性违约事件)",
        1: "DISTRESS_EVENT (包含：法院受理破产清算、财务指标连续违规被动退市、资不抵债进行司法重整)"
    }

    MIN_TOTAL_SAMPLES = 20    # 最低样本量门槛
    MIN_EVENTS_REQUIRED = 5   # 最低真实破产/违约事件样本门槛

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer
        self.kmf = KaplanMeierFitter()
        self.cph = CoxPHFitter(penalizer=penalizer)
        # 多维验证状态
        self.is_fitted: bool = False
        self.training_status: ModelTrainingStatus = ModelTrainingStatus.INSUFFICIENT_DATA
        self.model_stage: ModelStage = ModelStage.UNTRAINED
        self.is_algorithm_fitted: bool = False
        self.is_data_verified: bool = False
        self.is_out_of_sample_validated: bool = False
        self.is_calibrated: bool = False
        self.is_synthetic: bool = False
        self.sample_count: int = 0
        self.event_count: int = 0
        self.model_version: str = "v1.1-p0-cox"
        self.dataset_version: Optional[str] = None
        self.rejection_reason: Optional[str] = None

    def get_verification_info(self) -> ModelVerificationInfo:
        """获取当前模型的多维验证状态详情"""
        return ModelVerificationInfo(
            stage=self.model_stage,
            is_algorithm_fitted=self.is_algorithm_fitted,
            is_data_verified=self.is_data_verified,
            is_out_of_sample_validated=self.is_out_of_sample_validated,
            is_calibrated=self.is_calibrated,
            dataset_version=self.dataset_version,
            model_version=self.model_version,
            rejection_reason=self.rejection_reason
        )

    def fit_verified(
        self,
        cohort_df: Optional[pd.DataFrame] = None,
        duration_col: str = "duration",
        event_col: str = "event",
        quality_status: DataQualityStatus = DataQualityStatus.UNVERIFIED,
        dataset_version: str = "v1.1-cohort-verified"
    ) -> Dict[str, Any]:
        """正式训练接口：仅接受经过审计核实的真实 VERIFIED 企业生存事件队列"""
        self.is_synthetic = False

        self.sample_count = len(cohort_df) if cohort_df is not None else 0
        if cohort_df is None or self.sample_count < self.MIN_TOTAL_SAMPLES:
            self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
            self.model_stage = ModelStage.UNTRAINED
            self.rejection_reason = f"真实验证企业生存样本不足 ({self.sample_count} < {self.MIN_TOTAL_SAMPLES})，模型保持未训练状态。"
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "sample_count": self.sample_count,
                "min_required": self.MIN_TOTAL_SAMPLES,
                "message": self.rejection_reason
            }

        if duration_col not in cohort_df.columns or event_col not in cohort_df.columns:
            self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
            self.model_stage = ModelStage.UNTRAINED
            self.rejection_reason = f"队列数据缺少必要列 [{duration_col}, {event_col}]。"
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "message": self.rejection_reason
            }

        events_observed = int(cohort_df[event_col].sum())
        self.event_count = events_observed

        if events_observed < self.MIN_EVENTS_REQUIRED:
            self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
            self.model_stage = ModelStage.UNTRAINED
            self.rejection_reason = f"破产/重整真实事件数不足 ({events_observed} < {self.MIN_EVENTS_REQUIRED})，无法可靠估计 Cox 协变量系数。"
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "events_observed": events_observed,
                "min_events_required": self.MIN_EVENTS_REQUIRED,
                "message": self.rejection_reason
            }

        # 1. 强制数据质量状态审计门禁
        if quality_status != DataQualityStatus.VERIFIED:
            self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
            self.model_stage = ModelStage.UNTRAINED
            self.is_data_verified = False
            self.is_algorithm_fitted = False
            self.rejection_reason = (
                f"正式生存模型准入门禁拒绝：队列数据质量状态为 {quality_status.value if hasattr(quality_status, 'value') else quality_status}，"
                f"非 VERIFIED 审计通过状态。MANUAL_REFERENCE 仅允许通过 fit_experimental 运行算法研究模式。"
            )
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "sample_count": self.sample_count,
                "events_observed": self.event_count,
                "min_events_required": self.MIN_EVENTS_REQUIRED,
                "rejection_reason": self.rejection_reason,
                "message": self.rejection_reason
            }

        try:
            self.kmf.fit(durations=cohort_df[duration_col], event_observed=cohort_df[event_col], label="真实验证企业队列")
            self.cph.fit(cohort_df, duration_col=duration_col, event_col=event_col)
            self.is_fitted = True
            self.is_algorithm_fitted = True
            self.is_data_verified = True
            self.model_stage = ModelStage.VERIFIED_PRODUCTION
            self.training_status = ModelTrainingStatus.TRAINED
            self.dataset_version = dataset_version
            self.rejection_reason = None
            return {
                "status": ModelTrainingStatus.TRAINED.value,
                "model_stage": self.model_stage.value,
                "is_verified": True,
                "sample_count": self.sample_count,
                "events_observed": self.event_count,
                "dataset_version": self.dataset_version,
                "message": "已基于逐项审计的真实企业生存事件队列完成正式拟合。"
            }
        except Exception as e:
            self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
            self.model_stage = ModelStage.UNTRAINED
            self.rejection_reason = str(e)
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "error": str(e)
            }

    def fit_experimental(
        self,
        cohort_df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
        dataset_version: str = "v1.1-cohort-ref-experimental"
    ) -> Dict[str, Any]:
        """实验模式拟合：允许使用人工参考队列 (MANUAL_REFERENCE) 检验 Cox 算法收敛性。
        
        严格约束：
        - 标记为 EXPERIMENTAL。
        - is_data_verified 严格为 False。
        - is_verified 严格为 False，绝不产生正式企业破产预测。
        """
        if cohort_df is None or len(cohort_df) < 5:
            self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "message": "实验队列样本不足。"
            }

        try:
            self.kmf.fit(durations=cohort_df[duration_col], event_observed=cohort_df[event_col], label="历史参考企业队列 (实验)")
            self.cph.fit(cohort_df, duration_col=duration_col, event_col=event_col)
            self.is_fitted = True
            self.is_algorithm_fitted = True
            self.is_data_verified = False  # MANUAL_REFERENCE
            self.is_synthetic = False
            self.model_stage = ModelStage.EXPERIMENTAL
            self.training_status = ModelTrainingStatus.EXPERIMENTAL
            self.sample_count = len(cohort_df)
            self.event_count = int(cohort_df[event_col].sum())
            self.dataset_version = dataset_version
            self.rejection_reason = "模型基于人工参考数据 (MANUAL_REFERENCE) 拟合，仅供实验研究，禁止用于正式企业破产决策。"
            return {
                "status": "TRAINED_EXPERIMENTAL_RESEARCH",
                "model_stage": ModelStage.EXPERIMENTAL.value,
                "is_verified": False,
                "sample_count": self.sample_count,
                "events_observed": self.event_count,
                "dataset_version": self.dataset_version,
                "warning": self.rejection_reason
            }
        except Exception as e:
            self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
            return {"status": ModelTrainingStatus.INSUFFICIENT_DATA.value, "error": str(e)}

    def fit(self, cohort_df: Optional[pd.DataFrame] = None):
        """兼容性接口 (默认执行严格准入)"""
        return self.fit_verified(cohort_df, quality_status=DataQualityStatus.UNVERIFIED)

    def fit_from_historical_cohort(self) -> Dict[str, Any]:
        """从历史持久化或采集的真实 A 股企业违约队列中加载并进行实验模式拟合 (严格标记为 EXPERIMENTAL)"""
        from src.core.database import db_manager
        df_cohort = None
        try:
            df_cohort = db_manager.query_df("SELECT * FROM company_distress_cohort")
        except Exception:
            pass

        if df_cohort is None or df_cohort.empty:
            from src.ingestion.enterprise import EnterpriseCollector
            collector = EnterpriseCollector()
            records = collector.get_historical_distress_cohort()
            df_cohort = pd.DataFrame(records)

        if "duration" not in df_cohort.columns and "duration_months" in df_cohort.columns:
            df_cohort["duration"] = df_cohort["duration_months"]
        if "event" not in df_cohort.columns and "is_distressed" in df_cohort.columns:
            df_cohort["event"] = df_cohort["is_distressed"]

        needed_cols = ["duration", "event", "cash_burn_months", "debt_ratio", "gross_margin", "competitor_count"]
        train_df = df_cohort[needed_cols].copy()
        # 由于企业违约历史队列属于 MANUAL_REFERENCE，必须路由至 fit_experimental
        return self.fit_experimental(train_df, duration_col="duration", event_col="event", dataset_version="v1.1-historical-cohort-ref")

    def fit_synthetic_for_testing(self, n_samples: int = 250) -> Dict[str, Any]:
        """【仅供测试与算法演练使用】使用模拟数据测试 Cox 算法与 Kaplan-Meier 逻辑"""
        from tests.helpers.synthetic_survival import generate_synthetic_survival_cohort
        cohort_df = generate_synthetic_survival_cohort(n_samples=n_samples)
        self.kmf.fit(durations=cohort_df["duration"], event_observed=cohort_df["event"], label="合成测试队列")
        self.cph.fit(cohort_df, duration_col="duration", event_col="event")
        self.is_fitted = True
        self.is_algorithm_fitted = True
        self.is_data_verified = False
        self.is_synthetic = True
        self.model_stage = ModelStage.EXPERIMENTAL
        self.training_status = ModelTrainingStatus.EXPERIMENTAL
        self.sample_count = n_samples
        self.event_count = int(cohort_df["event"].sum())
        self.dataset_version = "synthetic-test-v1.0"
        return {
            "status": "TRAINED_SYNTHETIC_TESTING_ONLY",
            "model_stage": ModelStage.EXPERIMENTAL.value,
            "is_verified": False,
            "is_synthetic": True,
            "sample_count": n_samples,
            "warning": "使用合成模拟数据训练，结果不可用于实际投资/经营风险决策。"
        }

    def predict_company_survival(
        self,
        company_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """评估特定企业生存概率与危险比率。
        
        若模型未训练或未通过真实验证，严禁输出虚构破产概率。
        若模型处于 EXPERIMENTAL 实验阶段，返回 UNVERIFIED_RESEARCH 且 is_verified 严格为 False。
        """
        if not self.is_algorithm_fitted or not self.is_fitted:
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": ModelStage.UNTRAINED.value,
                "is_verified": False,
                "is_synthetic": False,
                "dataset_version": self.dataset_version,
                "model_version": self.model_version,
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
        is_verified = (self.model_stage == ModelStage.VERIFIED_PRODUCTION and self.is_data_verified and not self.is_synthetic)
        status_str = "PREDICTED" if is_verified else "UNVERIFIED_RESEARCH"

        return {
            "status": status_str,
            "model_stage": self.model_stage.value,
            "is_verified": is_verified,
            "is_synthetic": self.is_synthetic,
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
            "partial_hazard_ratio": round(partial_hazard, 2),
            "risk_tier": risk_tier,
            "survival_probabilities": survival_probs,
            "risk_assessment": assessment,
            "hazard_factor_multipliers": hazard_ratios,
            "disclaimer": "本预测仅反映基于公开财务指标的生存概率分布，不代表对企业存续性的绝对断言。" if is_verified else "本估计基于实验性参考模型 (MANUAL_REFERENCE)，未通过真实数据审计门禁，非正式预测。"
        }
