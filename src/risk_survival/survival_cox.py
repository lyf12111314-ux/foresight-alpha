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
from datetime import date
from typing import Dict, Any, List, Optional
from lifelines import KaplanMeierFitter, CoxPHFitter
from src.core.types import (
    ModelTrainingStatus,
    ModelStage,
    DataQualityStatus,
    DistressEventType,
    ModelVerificationInfo,
    OOSValidationRecord,
    ValidationExecutionType,
    PredictionStatus
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
        self.oos_validation: Optional[OOSValidationRecord] = None

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
            rejection_reason=self.rejection_reason,
            oos_validation=self.oos_validation
        )

    def _reset_to_untrained(self, reason: str):
        """重置生存分析模型状态，清空拟合标志并记录拒绝原因"""
        self.is_fitted = False
        self.is_algorithm_fitted = False
        self.is_data_verified = False
        self.is_out_of_sample_validated = False
        self.oos_validation = None
        self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
        self.model_stage = ModelStage.UNTRAINED
        self.rejection_reason = reason

    def fit_verified(
        self,
        cohort_df: Optional[pd.DataFrame] = None,
        duration_col: str = "duration",
        event_col: str = "event",
        quality_status: DataQualityStatus = DataQualityStatus.UNVERIFIED,
        provenance_records: Optional[List[Any]] = None,
        dataset_version: str = "v1.1-cohort-verified"
    ) -> Dict[str, Any]:
        """正式训练接口：仅接受经过审计核实的真实 VERIFIED 企业生存事件队列及血缘证明"""
        self.is_synthetic = False

        self.sample_count = len(cohort_df) if cohort_df is not None else 0
        if cohort_df is None or self.sample_count < self.MIN_TOTAL_SAMPLES:
            reason = f"真实验证企业生存样本不足 ({self.sample_count} < {self.MIN_TOTAL_SAMPLES})，模型保持未训练状态。"
            self._reset_to_untrained(reason)
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "sample_count": self.sample_count,
                "min_required": self.MIN_TOTAL_SAMPLES,
                "message": reason
            }

        if duration_col not in cohort_df.columns or event_col not in cohort_df.columns:
            reason = f"队列数据缺少必要列 [{duration_col}, {event_col}]。"
            self._reset_to_untrained(reason)
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "message": reason
            }

        events_observed = int(cohort_df[event_col].sum())
        self.event_count = events_observed

        if events_observed < self.MIN_EVENTS_REQUIRED:
            reason = f"破产/重整真实事件数不足 ({events_observed} < {self.MIN_EVENTS_REQUIRED})，无法可靠估计 Cox 协变量系数。"
            self._reset_to_untrained(reason)
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "events_observed": events_observed,
                "min_events_required": self.MIN_EVENTS_REQUIRED,
                "message": reason
            }

        # 1. 强制数据质量状态审计门禁
        if quality_status != DataQualityStatus.VERIFIED:
            reason = (
                f"正式生存模型准入门禁拒绝：队列数据质量状态为 {quality_status.value if hasattr(quality_status, 'value') else quality_status}，"
                f"非 VERIFIED 审计通过状态。MANUAL_REFERENCE 仅允许通过 fit_experimental 运行算法研究模式。"
            )
            self._reset_to_untrained(reason)
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "sample_count": self.sample_count,
                "events_observed": self.event_count,
                "min_events_required": self.MIN_EVENTS_REQUIRED,
                "rejection_reason": reason,
                "message": reason
            }

        # 2. 强制真实数据血缘与有效核验记录验证
        if not provenance_records or len(provenance_records) < self.sample_count:
            reason = "正式生存模型准入门禁拒绝：未提供经核验的企业生存数据血缘记录 (provenance_records) 或血缘样本数量不足。"
            self._reset_to_untrained(reason)
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "sample_count": self.sample_count,
                "rejection_reason": reason,
                "message": reason
            }

        for idx, prov in enumerate(provenance_records):
            p_status = getattr(prov, "quality_status", None) or (prov.get("quality_status") if isinstance(prov, dict) else None)
            p_hash = getattr(prov, "raw_hash", None) or (prov.get("raw_hash") if isinstance(prov, dict) else None)
            if p_status not in [DataQualityStatus.VERIFIED, DataQualityStatus.VERIFIED.value, "VERIFIED"] or not p_hash:
                reason = f"正式生存模型准入门禁拒绝：血缘记录 #{idx} 质量状态非 VERIFIED 或缺失不可篡改哈希签名。"
                self._reset_to_untrained(reason)
                return {
                    "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                    "model_stage": self.model_stage.value,
                    "is_verified": False,
                    "rejection_reason": reason,
                    "message": reason
                }

        try:
            self.kmf.fit(durations=cohort_df[duration_col], event_observed=cohort_df[event_col], label="真实验证企业队列")
            self.cph.fit(cohort_df, duration_col=duration_col, event_col=event_col)
            self.is_fitted = True
            self.is_algorithm_fitted = True
            self.is_data_verified = True
            self.is_out_of_sample_validated = False
            self.model_stage = ModelStage.BENCHMARK
            self.training_status = ModelTrainingStatus.TRAINED
            self.dataset_version = dataset_version
            self.rejection_reason = None
            return {
                "status": ModelTrainingStatus.TRAINED.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "is_algorithm_fitted": True,
                "sample_count": self.sample_count,
                "events_observed": self.event_count,
                "dataset_version": self.dataset_version,
                "message": "已基于逐项审计的真实企业生存事件队列完成基准拟合，待样本外跨周期检验 (OOS) 后可晋升为生产模型。"
            }
        except Exception as e:
            self._reset_to_untrained(str(e))
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "error": str(e)
            }

    def validate_out_of_sample(
        self,
        oos_cohort_df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
        provenance_records: Optional[List[Any]] = None,
        min_c_index: Optional[float] = None,
        evaluation_date: Optional[date] = None,
        execution_type: ValidationExecutionType = ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
        registered_dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """样本外跨周期生存分析验证门禁 (OOS Concordance Index Validation - 审计级实施).
        
        要求独立样本外队列经过血缘审计且 C-index >= 0.60，方允许晋升为正式生产模型。
        """
        if not self.is_algorithm_fitted or not self.is_data_verified or not self.is_fitted:
            return {
                "status": "VALIDATION_FAILED",
                "model_stage": self.model_stage.value,
                "passed": False,
                "rejection_reason": "生存模型尚未完成真实验证基准拟合，无法执行样本外检验。",
                "message": "生存模型尚未完成真实验证基准拟合，无法执行样本外检验。"
            }

        from src.decision_engine.model_validator import ModelValidationService
        validator = ModelValidationService()
        audit_res = validator.audit_and_validate_cox(
            survival_engine=self,
            validation_cohort_df=oos_cohort_df,
            duration_col=duration_col,
            event_col=event_col,
            provenance_records=provenance_records,
            min_c_index=min_c_index,
            evaluation_date=evaluation_date,
            execution_type=execution_type,
            registered_dataset_id=registered_dataset_id
        )

        rec = audit_res.get("validation_record")
        self.oos_validation = rec

        if audit_res.get("passed") is True:
            self.is_out_of_sample_validated = True
            self.model_stage = ModelStage.VERIFIED_PRODUCTION
            return {
                "status": "PROMOTED_TO_PRODUCTION",
                "model_stage": self.model_stage.value,
                "passed": True,
                "c_index": audit_res.get("c_index"),
                "min_required": audit_res.get("threshold"),
                "message": f"样本外跨周期生存验证通过 (C-index: {audit_res.get('c_index')} >= {audit_res.get('threshold')})，模型晋升为正式生产验证状态。"
            }
        else:
            self.is_out_of_sample_validated = False
            return {
                "status": audit_res.get("status", "OOS_VALIDATION_FAILED"),
                "model_stage": self.model_stage.value,
                "passed": False,
                "c_index": audit_res.get("c_index"),
                "min_required": audit_res.get("threshold"),
                "rejection_reason": audit_res.get("rejection_reason"),
                "message": f"样本外跨周期生存验证未通过：{audit_res.get('rejection_reason')}，保持基准模型状态，禁止用于正式预测。"
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
        company_metrics: Dict[str, float],
        is_formal: bool = False
    ) -> Dict[str, Any]:
        """评估特定企业生存概率与危险比率。
        
        若模型未训练或未通过真实验证，严禁输出虚构破产概率。
        若模型处于 EXPERIMENTAL 实验阶段，返回 UNVERIFIED_RESEARCH 且 is_verified 严格为 False。
        若进行正式预测 (is_formal=True 或 VERIFIED_PRODUCTION)，所有协变量必须具备真实数据，严禁默认值填充。
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

        # 严格检查核心财务与经营协变量完整性 (严禁在财务数据缺失时使用默认数值填充替代)
        is_formal_prediction = bool(is_formal or self.model_stage == ModelStage.VERIFIED_PRODUCTION)
        if is_formal_prediction:
            required_covariates = ["cash_burn_months", "debt_ratio", "gross_margin", "competitor_count"]
        else:
            required_covariates = ["debt_ratio", "gross_margin"]

        missing_covs = [c for c in required_covariates if c not in company_metrics or company_metrics[c] is None]
        if missing_covs:
            return {
                "status": PredictionStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "is_synthetic": self.is_synthetic,
                "dataset_version": self.dataset_version,
                "model_version": self.model_version,
                "partial_hazard_ratio": None,
                "risk_tier": "DATA_INSUFFICIENT",
                "survival_probabilities": {
                    "surv_1y": None,
                    "surv_3y": None,
                    "surv_5y": None
                },
                "risk_assessment": f"企业关键财务与经营指标缺失 ({missing_covs})，正式预测拒绝使用默认数值填充以保证风险评估可信度。",
                "hazard_factor_multipliers": {},
                "disclaimer": "指标缺失时拒绝输出企业破产预测以维护决策可信度。"
            }

        sample_df = pd.DataFrame([{
            "cash_burn_months": float(company_metrics.get("cash_burn_months", 24.0)),
            "debt_ratio": float(company_metrics["debt_ratio"]),
            "gross_margin": float(company_metrics["gross_margin"]),
            "competitor_count": float(company_metrics.get("competitor_count", 1.0))
        }])

        surv_func = self.cph.predict_survival_function(sample_df)

        # 严格标准化时间单位为月份 (MONTHS: 1年=12月, 3年=36月, 5年=60月)，消除 >15 启发式引发的单位混淆
        min_observed_dur = float(surv_func.index.min()) if not surv_func.empty else 0.0
        max_observed_dur = float(surv_func.index.max()) if not surv_func.empty else 0.0
        time_unit = "MONTHS"

        horizon_months = {"surv_1y": 12.0, "surv_3y": 36.0, "surv_5y": 60.0}
        survival_probs = {}

        for key, t_eval in horizon_months.items():
            if t_eval < min_observed_dur:
                # 在首次违约/破产事件发生前，全队列存活概率为 1.0 (右连续阶梯函数初始平台)
                survival_probs[key] = 1.0
            elif t_eval > max_observed_dur:
                # 超过队列最大跟踪观察期，右截尾样本不支持不可信外推，置为 None
                survival_probs[key] = None
            else:
                # 阶梯函数：取满足 event_time <= t 的最后一个生存概率点
                eligible = surv_func.index[surv_func.index <= t_eval]
                if len(eligible) > 0:
                    p_surv = float(surv_func.loc[eligible[-1]].iloc[0])
                    survival_probs[key] = round(p_surv, 3)
                else:
                    survival_probs[key] = 1.0

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
        is_verified = (
            self.model_stage == ModelStage.VERIFIED_PRODUCTION
            and self.is_data_verified
            and self.is_out_of_sample_validated
            and not self.is_synthetic
        )
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
            "time_unit": "MONTHS",
            "horizon_months": horizon_months,
            "survival_probabilities": survival_probs,
            "risk_assessment": assessment,
            "hazard_factor_multipliers": hazard_ratios,
            "disclaimer": "本预测仅反映基于公开财务指标的生存概率分布，不代表对企业存续性的绝对断言。" if is_verified else "本估计基于实验性参考模型 (MANUAL_REFERENCE)，未通过真实数据审计门禁，非正式预测。"
        }
