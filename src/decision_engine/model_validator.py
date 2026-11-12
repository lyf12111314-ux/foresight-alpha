"""独立样本外验证与生产模型准入服务 (Independent OOS Model Validation Service).

严格遵循 V1.1 审计准则：
1. 样本外验证数据必须独立提供，严禁模型调用方使用自生预测值或合成数据伪造验证结果。
2. 严禁时序与标的信息重叠 (Disjointness & Window Overlap Check)。
3. 严格禁止未来信息泄漏：特征时间 <= 决策时间，标签成熟时间与公布时间 <= 评估截止时间。
4. 强制生产准入硬性门槛：LightGBM Max MAE <= 2.0，Cox C-index >= 0.60，调用方无法通过放宽参数绕过。
5. 必须超越基准模型 (Naive Baseline / Random Hazard)。
6. 测试模式 (TEST_ONLY/EXPERIMENTAL) 与合成数据只能产生测试记录，绝不晋升 VERIFIED_PRODUCTION。
"""

from datetime import date, datetime
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import json
import hashlib
import numpy as np
import pandas as pd

from src.core.types import (
    DataQualityStatus,
    ModelStage,
    ValidationExecutionType,
    OOSValidationSample,
    OOSValidationDataset,
    OOSValidationRecord,
    TrustedDatasetRegistration
)


def compute_samples_checksum(samples: List[OOSValidationSample]) -> str:
    """根据实际样本内容计算不可篡改的规范化 SHA-256 摘要 (防伪造、防替换样本)"""
    if not samples:
        return hashlib.sha256(b"EMPTY_DATASET").hexdigest()

    canonical_records = []
    for s in sorted(samples, key=lambda x: str(x.sample_id)):
        feat_items = sorted([(str(k), float(v)) for k, v in s.features.items()]) if s.features else []
        label_items = sorted([(int(k), float(v) if v is not None else None) for k, v in s.labels.items()]) if s.labels else []
        rec = {
            "sample_id": str(s.sample_id),
            "dataset_version": str(s.dataset_version),
            "decision_date": str(s.decision_date),
            "features": feat_items,
            "labels": label_items,
            "label_period_end": str(s.label_period_end),
            "is_synthetic": bool(s.is_synthetic),
            "is_model_generated": bool(s.is_model_generated)
        }
        canonical_records.append(rec)
    canonical_json = json.dumps(canonical_records, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class TrustedValidationRegistry:
    """可信样本外验证数据集注册中心 (P0-01 准入审计规范 - 权威注册与内容绑定)"""
    _registry: Dict[str, TrustedDatasetRegistration] = {}

    @classmethod
    def register(cls, registration: TrustedDatasetRegistration) -> None:
        """登记验证数据集，并记录样本内容摘要与防伪凭据"""
        actual_hash = compute_samples_checksum(registration.samples)
        
        # 若未填摘要，自动填充为根据实际样本计算的规范摘要
        if not registration.checksum_sha256 or not registration.checksum_sha256.strip():
            registration.checksum_sha256 = actual_hash
        
        registration.sample_count = len(registration.samples)

        # 数据集注册不能仅由业务调用方直接自设 audit_status=VERIFIED 而取得生产准入资格
        # 必须具备第三方权威机构审计凭证 (has_independent_audit_certificate)
        if registration.audit_status == DataQualityStatus.VERIFIED and not registration.has_independent_audit_certificate:
            registration.is_audited_production_ready = False
            registration.audit_status = DataQualityStatus.UNVERIFIED

        cls._registry[registration.dataset_id] = registration

    @classmethod
    def get(cls, dataset_id: str) -> Optional[TrustedDatasetRegistration]:
        """获取已登记的可信验证数据集"""
        return cls._registry.get(dataset_id)

    @classmethod
    def contains(cls, dataset_id: str) -> bool:
        """检查数据集是否已完成审计登记"""
        return dataset_id in cls._registry

    @classmethod
    def unregister(cls, dataset_id: str) -> None:
        """注销数据集"""
        if dataset_id in cls._registry:
            del cls._registry[dataset_id]

    @classmethod
    def clear(cls) -> None:
        """重置注册中心并恢复默认种子数据集"""
        cls._registry.clear()
        cls._seed_default_registries()

    @classmethod
    def _seed_default_registries(cls) -> None:
        """初始化预设示例数据集。
        
        【重要审计约束】：
        预置示例数据集严禁默认标记为经过真实独立生产审计！
        只能标记为 UNVERIFIED 或 TEST_ONLY，is_audited_production_ready 严格为 False。
        目前尚未建立真实生产审计数据来源时，正式生产验证入口保持关闭。
        """
        cls._registry["REG_OOS_TECH_V1_2023"] = TrustedDatasetRegistration(
            dataset_id="REG_OOS_TECH_V1_2023",
            dataset_version="v1.1-oos-2023-example",
            audit_status=DataQualityStatus.UNVERIFIED,
            description="系统内置示例技术发展样本外验证集（仅供测试与研究参考，未获真实生产审计）",
            checksum_sha256=compute_samples_checksum([]),
            sample_count=0,
            audit_authority="Internal_Research_Prototype",
            is_synthetic=False,
            is_audited_production_ready=False,
            has_independent_audit_certificate=False
        )
        cls._registry["REG_OOS_COHORT_V1_2023"] = TrustedDatasetRegistration(
            dataset_id="REG_OOS_COHORT_V1_2023",
            dataset_version="v1.1-oos-cohort-2023-example",
            audit_status=DataQualityStatus.UNVERIFIED,
            description="系统内置示例企业生存事件样本外验证集（仅供测试与研究参考，未获真实生产审计）",
            checksum_sha256=compute_samples_checksum([]),
            sample_count=0,
            audit_authority="Internal_Research_Prototype",
            is_synthetic=False,
            is_audited_production_ready=False,
            has_independent_audit_certificate=False
        )


# 模块加载时初始化注册中心
TrustedValidationRegistry._seed_default_registries()


class ModelValidationService:
    """独立样本外跨周期模型验证与生产准入审计服务"""

    # 生产准入硬性不可篡改门禁
    PRODUCTION_MAX_MAE_CEILING: float = 2.0
    PRODUCTION_MIN_C_INDEX_FLOOR: float = 0.60

    def __init__(self, service_version: str = "v1.1-audit-gate"):
        self.service_version = service_version

    def audit_and_validate_lgbm(
        self,
        forecaster: Any,
        validation_dataset: Union[OOSValidationDataset, List[OOSValidationSample], pd.DataFrame, Dict[str, Any], str, None] = None,
        training_sample_ids: Optional[Set[str]] = None,
        training_windows: Optional[Set[Tuple[str, date]]] = None,
        training_target_values: Optional[Dict[int, List[float]]] = None,
        max_allowed_mae: Optional[float] = None,
        evaluation_date: Optional[date] = None,
        execution_type: ValidationExecutionType = ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
        registered_dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """对 LightGBM 成长预测模型执行独立的 6 步生产准入审计与样本外验证"""
        eval_date = evaluation_date or date.today()

        # 计算生产准入硬性生效上限 (调用方放宽阈值自动被钳制在 2.0)
        if max_allowed_mae is not None:
            effective_threshold = min(float(max_allowed_mae), self.PRODUCTION_MAX_MAE_CEILING)
        else:
            effective_threshold = self.PRODUCTION_MAX_MAE_CEILING

        # 检查评估日期是否为未来日期
        if eval_date > date.today():
            rejection = f"评估时间违规：样本外验证评估日 {eval_date} 晚于当前真实日期 {date.today()}，禁止使用未来时点开展正式验证。"
            if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                forecaster.model_stage = ModelStage.BENCHMARK
            if hasattr(forecaster, "is_out_of_sample_validated"):
                forecaster.is_out_of_sample_validated = False
            return {
                "status": "VALIDATION_REJECTED",
                "passed": False,
                "threshold": effective_threshold,
                "rejection_reason": rejection,
                "model_stage": forecaster.model_stage.value if hasattr(forecaster, "model_stage") else ModelStage.UNTRAINED.value
            }

        # 检查是否传入显式 registered_dataset_id
        target_reg_id = registered_dataset_id or (validation_dataset if isinstance(validation_dataset, str) else None)
        if target_reg_id:
            if not TrustedValidationRegistry.contains(target_reg_id):
                rejection = f"正式生产验证准入门禁拒绝：数据集 ID {target_reg_id} 未在可信注册中心 (TrustedValidationRegistry) 登记，禁止进入正式生产认证。"
                if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    forecaster.model_stage = ModelStage.BENCHMARK
                if hasattr(forecaster, "is_out_of_sample_validated"):
                    forecaster.is_out_of_sample_validated = False
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "threshold": effective_threshold,
                    "rejection_reason": rejection,
                    "model_stage": forecaster.model_stage.value if hasattr(forecaster, "model_stage") else ModelStage.UNTRAINED.value
                }

        # 解析与标准化验证样本
        samples, dataset_meta = self._parse_validation_samples(validation_dataset, registered_dataset_id=target_reg_id)

        if not samples:
            if execution_type == ValidationExecutionType.FORMAL_PRODUCTION_AUDIT:
                # 检查是否注册
                if not dataset_meta.get("is_registered", False):
                    rejection = (
                        "正式生产验证准入门禁拒绝：验证数据集未在可信注册中心 (TrustedValidationRegistry) 登记。"
                        "调用方自报 is_registered 字段、自设 VERIFIED 或构造假哈希均无法通过真实性认证。"
                    )
                    rec = OOSValidationRecord(
                        metric_name="Max_MAE",
                        metric_value=999.0,
                        threshold=effective_threshold,
                        passed=False,
                        evaluation_date=eval_date,
                        test_sample_count=0,
                        validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                        validation_conclusion="REJECTED_UNREGISTERED_DATASET",
                        rejection_reason=rejection
                    )
                    return {
                        "status": "VALIDATION_REJECTED",
                        "passed": False,
                        "threshold": effective_threshold,
                        "rejection_reason": rejection,
                        "validation_record": rec,
                        "model_stage": forecaster.model_stage.value if hasattr(forecaster, "model_stage") else ModelStage.UNTRAINED.value
                    }
                # 检查是否获得独立生产审计证书
                if not dataset_meta.get("is_audited_production_ready", False) or dataset_meta.get("audit_status") not in [DataQualityStatus.VERIFIED, "VERIFIED"]:
                    rejection = (
                        "正式生产验证准入门禁拒绝：当前数据集为内置参考/示例数据集或未获得独立生产审计证书 (has_independent_audit_certificate=False)。"
                        "目前尚未建立权威真实生产审计数据来源时，正式生产验证入口保持关闭。"
                    )
                    rec = OOSValidationRecord(
                        metric_name="Max_MAE",
                        metric_value=999.0,
                        threshold=effective_threshold,
                        passed=False,
                        evaluation_date=eval_date,
                        test_sample_count=0,
                        validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                        validation_conclusion="REJECTED_DATASET_NOT_AUDITED_FOR_PRODUCTION",
                        rejection_reason=rejection
                    )
                    return {
                        "status": "VALIDATION_REJECTED",
                        "passed": False,
                        "threshold": effective_threshold,
                        "rejection_reason": rejection,
                        "validation_record": rec,
                        "model_stage": forecaster.model_stage.value if hasattr(forecaster, "model_stage") else ModelStage.UNTRAINED.value
                    }
                rec = OOSValidationRecord(
                    metric_name="Max_MAE",
                    metric_value=999.0,
                    threshold=effective_threshold,
                    passed=False,
                    evaluation_date=eval_date,
                    test_sample_count=0,
                    validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                    validation_conclusion="REJECTED_EMPTY_SAMPLES",
                    rejection_reason="未提供有效的样本外验证数据集或样本集为空。"
                )
                return {
                    "status": "VALIDATION_FAILED",
                    "passed": False,
                    "threshold": effective_threshold,
                    "rejection_reason": "未提供有效的样本外验证数据集或样本集为空。",
                    "validation_record": rec,
                    "execution_type": execution_type.value,
                    "model_stage": forecaster.model_stage.value if hasattr(forecaster, "model_stage") else ModelStage.UNTRAINED.value
                }

            if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                forecaster.model_stage = ModelStage.BENCHMARK
            if hasattr(forecaster, "is_out_of_sample_validated"):
                forecaster.is_out_of_sample_validated = False
            return {
                "status": "VALIDATION_FAILED",
                "passed": False,
                "threshold": effective_threshold,
                "rejection_reason": "未提供有效的样本外验证数据集或样本集为空。",
                "execution_type": execution_type.value,
                "model_stage": forecaster.model_stage.value if hasattr(forecaster, "model_stage") else ModelStage.UNTRAINED.value
            }

        # 第 1 步：时序与样本隔离检查 (Temporal & Window Isolation)
        if training_sample_ids:
            overlap_ids = {s.sample_id for s in samples}.intersection(training_sample_ids)
            if overlap_ids:
                rejection = f"样本外隔离校验失败：验证集与训练集存在重叠样本 ID {list(overlap_ids)[:5]}。"
                if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    forecaster.model_stage = ModelStage.BENCHMARK
                if hasattr(forecaster, "is_out_of_sample_validated"):
                    forecaster.is_out_of_sample_validated = False
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "threshold": effective_threshold,
                    "rejection_reason": rejection,
                    "model_stage": forecaster.model_stage.value
                }

        if training_windows:
            overlap_windows = {(s.topic_id, s.decision_date) for s in samples if s.topic_id}.intersection(training_windows)
            if overlap_windows:
                rejection = f"时序窗口重叠拒绝：验证集与训练集存在相同标的相同时序窗口 {list(overlap_windows)[:5]}。"
                if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    forecaster.model_stage = ModelStage.BENCHMARK
                if hasattr(forecaster, "is_out_of_sample_validated"):
                    forecaster.is_out_of_sample_validated = False
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "threshold": effective_threshold,
                    "rejection_reason": rejection,
                    "model_stage": forecaster.model_stage.value
                }

        # 检查时间泄露：feature_available_at <= decision_date，label_period_end <= eval_date, label_observed_at <= eval_date
        for s in samples:
            feat_avail_date = s.feature_available_at.date() if isinstance(s.feature_available_at, datetime) else s.feature_available_at
            if feat_avail_date > s.decision_date:
                rejection = f"前视偏差时间泄露：样本 {s.sample_id} 的特征公布时间 {feat_avail_date} 晚于决策时间 {s.decision_date}。"
                if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    forecaster.model_stage = ModelStage.BENCHMARK
                if hasattr(forecaster, "is_out_of_sample_validated"):
                    forecaster.is_out_of_sample_validated = False
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "threshold": effective_threshold,
                    "rejection_reason": rejection,
                    "model_stage": forecaster.model_stage.value
                }

            if s.label_period_end > eval_date:
                rejection = f"未成熟标签前视泄漏：样本 {s.sample_id} 的标签成熟期 {s.label_period_end} 晚于评估日 {eval_date}。"
                if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    forecaster.model_stage = ModelStage.BENCHMARK
                if hasattr(forecaster, "is_out_of_sample_validated"):
                    forecaster.is_out_of_sample_validated = False
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "threshold": effective_threshold,
                    "rejection_reason": rejection,
                    "model_stage": forecaster.model_stage.value
                }

            label_obs_date = s.label_observed_at.date() if isinstance(s.label_observed_at, datetime) else s.label_observed_at
            if label_obs_date > eval_date:
                rejection = f"标签公开时间未来泄露：样本 {s.sample_id} 的真实公开时间 {label_obs_date} 晚于评估日 {eval_date}。"
                if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    forecaster.model_stage = ModelStage.BENCHMARK
                if hasattr(forecaster, "is_out_of_sample_validated"):
                    forecaster.is_out_of_sample_validated = False
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "threshold": effective_threshold,
                    "rejection_reason": rejection,
                    "model_stage": forecaster.model_stage.value
                }

        # 第 2 步：检查可信注册中心审计注册与防作弊门禁 (P0-01 准入审计规范)
        if execution_type == ValidationExecutionType.FORMAL_PRODUCTION_AUDIT:
            # 1. 验证数据集必须在服务端可信注册中心登记，调用方自报 is_registered 字段无效
            if not dataset_meta.get("is_registered", False):
                rejection = (
                    "正式生产验证准入门禁拒绝：验证数据集未在可信注册中心 (TrustedValidationRegistry) 登记。"
                    "调用方自报 is_registered 字段、自设 VERIFIED 或构造假哈希均无法通过真实性认证。"
                )
                if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    forecaster.model_stage = ModelStage.BENCHMARK
                if hasattr(forecaster, "is_out_of_sample_validated"):
                    forecaster.is_out_of_sample_validated = False
                rec = OOSValidationRecord(
                    metric_name="Max_MAE",
                    metric_value=999.0,
                    threshold=effective_threshold,
                    passed=False,
                    evaluation_date=eval_date,
                    test_sample_count=len(samples),
                    validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                    validation_conclusion="REJECTED_UNREGISTERED_DATASET",
                    rejection_reason=rejection
                )
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "threshold": effective_threshold,
                    "rejection_reason": rejection,
                    "validation_record": rec,
                    "model_stage": forecaster.model_stage.value if hasattr(forecaster, "model_stage") else ModelStage.UNTRAINED.value
                }

            # 2. 检查调用方是否替换了样本内容 (Sample Substitution Check)
            if dataset_meta.get("sample_substituted", False):
                rejection = (
                    "正式生产验证准入门禁拒绝：检测到验证样本与已注册数据集摘要不一致，"
                    "调用方存在样本替换行为 (Sample Substitution Detected)，拒绝认证。"
                )
                if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    forecaster.model_stage = ModelStage.BENCHMARK
                if hasattr(forecaster, "is_out_of_sample_validated"):
                    forecaster.is_out_of_sample_validated = False
                rec = OOSValidationRecord(
                    metric_name="Max_MAE",
                    metric_value=999.0,
                    threshold=effective_threshold,
                    passed=False,
                    evaluation_date=eval_date,
                    test_sample_count=len(samples),
                    validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                    validation_conclusion="REJECTED_SAMPLE_SUBSTITUTION",
                    rejection_reason=rejection
                )
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "threshold": effective_threshold,
                    "rejection_reason": rejection,
                    "validation_record": rec,
                    "model_stage": forecaster.model_stage.value if hasattr(forecaster, "model_stage") else ModelStage.UNTRAINED.value
                }

            # 3. 检查注册数据集摘要篡改 (Tamper Check)
            if dataset_meta.get("tampered", False):
                rejection = (
                    "正式生产验证准入门禁拒绝：已注册数据集内容发生变化但摘要未更新，"
                    "数据真实性核准失效，拒绝认证。"
                )
                if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    forecaster.model_stage = ModelStage.BENCHMARK
                if hasattr(forecaster, "is_out_of_sample_validated"):
                    forecaster.is_out_of_sample_validated = False
                rec = OOSValidationRecord(
                    metric_name="Max_MAE",
                    metric_value=999.0,
                    threshold=effective_threshold,
                    passed=False,
                    evaluation_date=eval_date,
                    test_sample_count=len(samples),
                    validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                    validation_conclusion="REJECTED_CHECKSUM_TAMPERED",
                    rejection_reason=rejection
                )
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "threshold": effective_threshold,
                    "rejection_reason": rejection,
                    "validation_record": rec,
                    "model_stage": forecaster.model_stage.value if hasattr(forecaster, "model_stage") else ModelStage.UNTRAINED.value
                }

            # 4. 检查是否具备正式独立生产审计资质 (示例数据集或未审计数据集一律阻断)
            if not dataset_meta.get("is_audited_production_ready", False) or dataset_meta.get("audit_status") not in [DataQualityStatus.VERIFIED, "VERIFIED"]:
                rejection = (
                    "正式生产验证准入门禁拒绝：当前数据集为内置参考/示例数据集或未获得独立生产审计证书 (has_independent_audit_certificate=False)。"
                    "目前尚未建立权威真实生产审计数据来源时，正式生产验证入口保持关闭。"
                )
                if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    forecaster.model_stage = ModelStage.BENCHMARK
                if hasattr(forecaster, "is_out_of_sample_validated"):
                    forecaster.is_out_of_sample_validated = False
                rec = OOSValidationRecord(
                    metric_name="Max_MAE",
                    metric_value=999.0,
                    threshold=effective_threshold,
                    passed=False,
                    evaluation_date=eval_date,
                    test_sample_count=len(samples),
                    validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                    validation_conclusion="REJECTED_DATASET_NOT_AUDITED_FOR_PRODUCTION",
                    rejection_reason=rejection
                )
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "threshold": effective_threshold,
                    "rejection_reason": rejection,
                    "validation_record": rec,
                    "model_stage": forecaster.model_stage.value if hasattr(forecaster, "model_stage") else ModelStage.UNTRAINED.value
                }

        # 检查是否包含合成或模型自生标签
        has_synthetic = any(s.is_synthetic for s in samples) or dataset_meta.get("is_synthetic", False)
        has_model_generated = any(s.is_model_generated for s in samples)
        has_unverified_quality = any(
            s.data_quality_status not in [DataQualityStatus.VERIFIED, DataQualityStatus.VERIFIED.value]
            or s.label_quality_status not in [DataQualityStatus.VERIFIED, DataQualityStatus.VERIFIED.value]
            for s in samples
        ) or (dataset_meta.get("audit_status") not in [DataQualityStatus.VERIFIED, DataQualityStatus.VERIFIED.value, "VERIFIED"])

        # 若包含合成、自生或未核验数据，强制降级为 TEST_ONLY / 无法晋升
        effective_execution_type = execution_type
        if has_synthetic or has_model_generated or has_unverified_quality:
            if execution_type == ValidationExecutionType.FORMAL_PRODUCTION_AUDIT:
                # 正式生产准入要求真实核验数据，含有伪造/合成/自生样本直接拒绝生产晋升
                rejection = (
                    "正式生产验证准入门禁拒绝：验证集包含模型自生标签 (is_model_generated=True)、"
                    "合成模拟数据 (is_synthetic=True) 或未在可信注册中心登记的普通数组/未逐项核验数据 (quality_status != VERIFIED)。"
                )
                if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    forecaster.model_stage = ModelStage.BENCHMARK
                if hasattr(forecaster, "is_out_of_sample_validated"):
                    forecaster.is_out_of_sample_validated = False
                rec = OOSValidationRecord(
                    metric_name="Max_MAE",
                    metric_value=999.0,
                    threshold=effective_threshold,
                    passed=False,
                    evaluation_date=eval_date,
                    test_sample_count=len(samples),
                    validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                    validation_conclusion="REJECTED_SYNTHETIC_OR_SELF_GENERATED",
                    rejection_reason=rejection
                )
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "threshold": effective_threshold,
                    "rejection_reason": rejection,
                    "validation_record": rec,
                    "model_stage": forecaster.model_stage.value
                }
            else:
                effective_execution_type = ValidationExecutionType.TEST_ONLY

        # 第 3 步：独立样本外评测 (Independent OOS Testing)
        # 提取特征矩阵与标签
        feat_df = pd.DataFrame([s.features for s in samples])
        for col in forecaster.feature_names:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        X_test = feat_df[forecaster.feature_names].values

        # 检查是否使用模型自身预测值构造验证标签 (防作弊反向审计)
        if effective_execution_type == ValidationExecutionType.FORMAL_PRODUCTION_AUDIT:
            for horizon in [1, 3, 5]:
                if horizon in forecaster.models:
                    y_true = np.array([s.labels.get(horizon) for s in samples])
                    valid_mask = pd.notna(y_true) & [y is not None for y in y_true]
                    if np.sum(valid_mask) >= 3:
                        y_true_valid = y_true[valid_mask].astype(float)
                        y_pred_check = forecaster.models[horizon].predict(X_test[valid_mask])

                        # 1. 绝对数值近乎完全吻合检查 (如 y_pred + 0.1 类似构造)
                        if np.allclose(y_pred_check, y_true_valid, atol=0.25):
                            rejection = "正式生产验证准入门禁拒绝：检测到验证标签与模型自身预测值高度贴合，涉嫌使用模型自生预测值伪造验证标签 (Self-generated labels cheat detected)。"
                            if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                                forecaster.model_stage = ModelStage.BENCHMARK
                            if hasattr(forecaster, "is_out_of_sample_validated"):
                                forecaster.is_out_of_sample_validated = False
                            rec = OOSValidationRecord(
                                metric_name="Max_MAE",
                                metric_value=float(np.mean(np.abs(y_pred_check - y_true_valid))),
                                threshold=self.PRODUCTION_MAX_MAE_CEILING,
                                passed=False,
                                evaluation_date=eval_date,
                                test_sample_count=len(samples),
                                validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                                validation_conclusion="REJECTED_SELF_GENERATED_CHEAT",
                                rejection_reason=rejection
                            )
                            return {
                                "status": "VALIDATION_REJECTED",
                                "passed": False,
                                "rejection_reason": rejection,
                                "validation_record": rec,
                                "model_stage": forecaster.model_stage.value
                            }

                        # 2. 相关系数近乎完全一致 (corr > 0.999)
                        if np.std(y_pred_check) > 1e-4 and np.std(y_true_valid) > 1e-4:
                            corr = float(np.corrcoef(y_pred_check, y_true_valid)[0, 1])
                            if corr > 0.999:
                                rejection = f"正式生产验证准入门禁拒绝：验证标签与模型预测值相关系数高达 {corr:.4f} > 0.999，检测到模型自生标签作弊。"
                                if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                                    forecaster.model_stage = ModelStage.BENCHMARK
                                if hasattr(forecaster, "is_out_of_sample_validated"):
                                    forecaster.is_out_of_sample_validated = False
                                rec = OOSValidationRecord(
                                    metric_name="Max_MAE",
                                    metric_value=float(np.mean(np.abs(y_pred_check - y_true_valid))),
                                    threshold=self.PRODUCTION_MAX_MAE_CEILING,
                                    passed=False,
                                    evaluation_date=eval_date,
                                    test_sample_count=len(samples),
                                    validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                                    validation_conclusion="REJECTED_SELF_GENERATED_CHEAT",
                                    rejection_reason=rejection
                                )
                                return {
                                    "status": "VALIDATION_REJECTED",
                                    "passed": False,
                                    "rejection_reason": rejection,
                                    "validation_record": rec,
                                    "model_stage": forecaster.model_stage.value
                                }

        mae_by_horizon = {}
        naive_mae_by_horizon = {}
        for horizon in [1, 3, 5]:
            if horizon in forecaster.models:
                y_true = np.array([s.labels.get(horizon) for s in samples])
                valid_mask = pd.notna(y_true) & [y is not None for y in y_true]
                if np.sum(valid_mask) > 0:
                    y_true_valid = y_true[valid_mask].astype(float)
                    y_pred = forecaster.models[horizon].predict(X_test[valid_mask])
                    mae = float(np.mean(np.abs(y_pred - y_true_valid)))
                    mae_by_horizon[horizon] = round(mae, 3)

                    # Naive Baseline: 用训练集标签均值/中位数预测
                    if training_target_values and horizon in training_target_values and len(training_target_values[horizon]) > 0:
                        naive_val = float(np.median(training_target_values[horizon]))
                    else:
                        naive_val = float(np.mean(y_true_valid))
                    naive_mae = float(np.mean(np.abs(naive_val - y_true_valid)))
                    naive_mae_by_horizon[horizon] = round(naive_mae, 3)

        if not mae_by_horizon:
            rejection = "验证样本中无任何对应预测期限的有效成熟标签。"
            if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                forecaster.model_stage = ModelStage.BENCHMARK
            if hasattr(forecaster, "is_out_of_sample_validated"):
                forecaster.is_out_of_sample_validated = False
            return {
                "status": "VALIDATION_FAILED",
                "passed": False,
                "rejection_reason": rejection,
                "model_stage": forecaster.model_stage.value
            }

        max_mae = max(mae_by_horizon.values())
        naive_max_mae = max(naive_mae_by_horizon.values()) if naive_mae_by_horizon else 999.0

        # 第 4 步：生产准入硬性上限与基准对照 (Fixed Production Ceiling & Benchmark Beat)
        # 正式生产准入时，调用方即使传入更大的 max_allowed_mae，也不得放宽生产上限 PRODUCTION_MAX_MAE_CEILING
        if effective_execution_type == ValidationExecutionType.FORMAL_PRODUCTION_AUDIT:
            if max_allowed_mae is not None:
                effective_threshold = min(float(max_allowed_mae), self.PRODUCTION_MAX_MAE_CEILING)
            else:
                effective_threshold = self.PRODUCTION_MAX_MAE_CEILING
        else:
            effective_threshold = float(max_allowed_mae) if max_allowed_mae is not None else self.PRODUCTION_MAX_MAE_CEILING

        beats_benchmark = bool(max_mae < naive_max_mae)
        passed_threshold = bool(max_mae <= effective_threshold)

        if effective_execution_type == ValidationExecutionType.FORMAL_PRODUCTION_AUDIT:
            passed = bool(passed_threshold and beats_benchmark)
            conclusion = "PROMOTED_TO_PRODUCTION" if passed else "VALIDATION_FAILED"
        else:
            # TEST_ONLY 工程测试模式：指标达标即视为工程测试通过，但结论为 PASSED_TEST_ONLY，禁止晋升生产
            passed = bool(passed_threshold)
            conclusion = "PASSED_TEST_ONLY" if passed else "VALIDATION_FAILED"

        rejection_reason = None
        if not passed_threshold:
            rejection_reason = f"Max MAE {max_mae} 超过生产准入硬性上限 {effective_threshold}。"
        elif not beats_benchmark:
            rejection_reason = f"模型效果 (MAE: {max_mae}) 未能显著超越简单基准 (Naive MAE: {naive_max_mae})。"
        elif effective_execution_type != ValidationExecutionType.FORMAL_PRODUCTION_AUDIT:
            rejection_reason = "当前验证为工程测试或实验验证模式，禁止晋升生产阶段。"

        # 若未通过，清空可能遗留的生产状态
        if not passed:
            if hasattr(forecaster, "model_stage") and forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION:
                forecaster.model_stage = ModelStage.BENCHMARK
            if hasattr(forecaster, "is_out_of_sample_validated"):
                forecaster.is_out_of_sample_validated = False

        rec = OOSValidationRecord(
            metric_name="Max_MAE",
            metric_value=max_mae,
            threshold=effective_threshold,
            passed=passed,
            evaluation_date=eval_date,
            test_sample_count=len(samples),
            validation_type=effective_execution_type,
            benchmark_metric_value=naive_max_mae,
            benchmark_name="Naive_Historical_Median",
            beats_benchmark=beats_benchmark,
            validation_conclusion=conclusion,
            rejection_reason=rejection_reason,
            details={
                "mae_by_horizon": mae_by_horizon,
                "naive_mae_by_horizon": naive_mae_by_horizon,
                "effective_threshold": effective_threshold
            }
        )

        return {
            "status": conclusion,
            "passed": passed,
            "validation_record": rec,
            "max_mae": max_mae,
            "threshold": effective_threshold,
            "beats_benchmark": beats_benchmark,
            "naive_max_mae": naive_max_mae,
            "details": mae_by_horizon,
            "rejection_reason": rejection_reason,
            "execution_type": effective_execution_type.value
        }

    def audit_and_validate_cox(
        self,
        survival_engine: Any,
        validation_cohort_df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
        provenance_records: Optional[List[Any]] = None,
        min_c_index: Optional[float] = None,
        evaluation_date: Optional[date] = None,
        execution_type: ValidationExecutionType = ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
        registered_dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """对 Cox 企业生存风险分析模型执行独立的生产准入审计与样本外 C-index 验证"""
        eval_date = evaluation_date or date.today()

        if validation_cohort_df is None or (isinstance(validation_cohort_df, pd.DataFrame) and len(validation_cohort_df) == 0):
            return {
                "status": "VALIDATION_FAILED",
                "passed": False,
                "rejection_reason": "未提供有效的样本外验证企业队列。",
                "model_stage": survival_engine.model_stage.value
            }

        # 检查是否传入显式 registered_dataset_id
        target_reg_id = registered_dataset_id or (validation_cohort_df if isinstance(validation_cohort_df, str) else None)

        if duration_col not in validation_cohort_df.columns or event_col not in validation_cohort_df.columns:
            return {
                "status": "VALIDATION_FAILED",
                "passed": False,
                "rejection_reason": f"验证队列缺失必要字段 [{duration_col}, {event_col}]。",
                "model_stage": survival_engine.model_stage.value
            }

        # 独立样本外评测 C-index
        from lifelines.utils import concordance_index
        risk_scores = survival_engine.cph.predict_partial_hazard(validation_cohort_df)
        c_val = float(concordance_index(
            event_times=validation_cohort_df[duration_col],
            predicted_scores=-risk_scores,
            event_observed=validation_cohort_df[event_col]
        ))
        c_val = round(c_val, 4)

        # 生产准入硬性下限：min_c_index 即使传入更低数值，也不得低于 PRODUCTION_MIN_C_INDEX_FLOOR
        if min_c_index is not None:
            effective_min_c = max(float(min_c_index), self.PRODUCTION_MIN_C_INDEX_FLOOR)
        else:
            effective_min_c = self.PRODUCTION_MIN_C_INDEX_FLOOR

        # 检查真实血缘记录 (P1-01 规范：未提供经核验血缘记录直接阻断)
        has_provenance = bool(provenance_records and len(provenance_records) >= len(validation_cohort_df))
        if has_provenance:
            for idx, prov in enumerate(provenance_records):
                p_status = getattr(prov, "quality_status", None) or (prov.get("quality_status") if isinstance(prov, dict) else None)
                p_hash = getattr(prov, "raw_hash", None) or (prov.get("raw_hash") if isinstance(prov, dict) else None)
                if p_status not in [DataQualityStatus.VERIFIED, DataQualityStatus.VERIFIED.value, "VERIFIED"] or not p_hash:
                    has_provenance = False
                    break

        if execution_type == ValidationExecutionType.FORMAL_PRODUCTION_AUDIT and not has_provenance:
            rejection = "正式生产准入门禁拒绝：未提供经核验的样本外企业队列数据血缘记录 (provenance) 或质量状态非 VERIFIED。"
            if hasattr(survival_engine, "model_stage") and survival_engine.model_stage == ModelStage.VERIFIED_PRODUCTION:
                survival_engine.model_stage = ModelStage.BENCHMARK
            if hasattr(survival_engine, "is_out_of_sample_validated"):
                survival_engine.is_out_of_sample_validated = False
            rec = OOSValidationRecord(
                metric_name="Concordance_Index",
                metric_value=c_val,
                threshold=effective_min_c,
                passed=False,
                evaluation_date=eval_date,
                test_sample_count=len(validation_cohort_df),
                validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                benchmark_metric_value=0.50,
                benchmark_name="Random_Concordance",
                beats_benchmark=bool(c_val > 0.50),
                validation_conclusion="REJECTED_MISSING_PROVENANCE",
                rejection_reason=rejection,
                details={"c_index": c_val}
            )
            return {
                "status": "VALIDATION_REJECTED",
                "passed": False,
                "c_index": c_val,
                "threshold": effective_min_c,
                "validation_record": rec,
                "rejection_reason": rejection,
                "model_stage": survival_engine.model_stage.value
            }

        # 正式生产准入门禁检查 (P0-01 / P1-01 准入审计规范)
        if execution_type == ValidationExecutionType.FORMAL_PRODUCTION_AUDIT:
            if not target_reg_id or not TrustedValidationRegistry.contains(target_reg_id):
                rejection = (
                    "正式生产准入门禁拒绝：验证企业队列未在可信注册中心登记。"
                    "调用方自报 VERIFIED 状态、普通 DataFrame 或自制血缘记录无法取得生产准入资格。"
                )
                if hasattr(survival_engine, "model_stage") and survival_engine.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    survival_engine.model_stage = ModelStage.BENCHMARK
                if hasattr(survival_engine, "is_out_of_sample_validated"):
                    survival_engine.is_out_of_sample_validated = False
                rec = OOSValidationRecord(
                    metric_name="Concordance_Index",
                    metric_value=c_val,
                    threshold=effective_min_c,
                    passed=False,
                    evaluation_date=eval_date,
                    test_sample_count=len(validation_cohort_df),
                    validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                    benchmark_metric_value=0.50,
                    benchmark_name="Random_Concordance",
                    beats_benchmark=bool(c_val > 0.50),
                    validation_conclusion="REJECTED_UNREGISTERED_DATASET",
                    rejection_reason=rejection,
                    details={"c_index": c_val}
                )
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "c_index": c_val,
                    "threshold": effective_min_c,
                    "validation_record": rec,
                    "rejection_reason": rejection,
                    "model_stage": survival_engine.model_stage.value
                }

            reg = TrustedValidationRegistry.get(target_reg_id)
            if not getattr(reg, "is_audited_production_ready", False) or getattr(reg, "audit_status", None) not in [DataQualityStatus.VERIFIED, "VERIFIED"]:
                rejection = (
                    "正式生产准入门禁拒绝：当前企业队列为系统内置示例数据集或未获得独立生产审计证书 (has_independent_audit_certificate=False)。"
                    "目前尚未建立权威真实生产审计数据来源时，正式生产验证入口保持关闭。"
                )
                if hasattr(survival_engine, "model_stage") and survival_engine.model_stage == ModelStage.VERIFIED_PRODUCTION:
                    survival_engine.model_stage = ModelStage.BENCHMARK
                if hasattr(survival_engine, "is_out_of_sample_validated"):
                    survival_engine.is_out_of_sample_validated = False
                rec = OOSValidationRecord(
                    metric_name="Concordance_Index",
                    metric_value=c_val,
                    threshold=effective_min_c,
                    passed=False,
                    evaluation_date=eval_date,
                    test_sample_count=len(validation_cohort_df),
                    validation_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT,
                    benchmark_metric_value=0.50,
                    benchmark_name="Random_Concordance",
                    beats_benchmark=bool(c_val > 0.50),
                    validation_conclusion="REJECTED_DATASET_NOT_AUDITED_FOR_PRODUCTION",
                    rejection_reason=rejection,
                    details={"c_index": c_val}
                )
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "c_index": c_val,
                    "threshold": effective_min_c,
                    "validation_record": rec,
                    "rejection_reason": rejection,
                    "model_stage": survival_engine.model_stage.value
                }

        beats_random = bool(c_val > 0.50)
        passed_threshold = bool(c_val >= effective_min_c)
        passed = bool(passed_threshold and beats_random and (execution_type == ValidationExecutionType.FORMAL_PRODUCTION_AUDIT))

        conclusion = "PROMOTED_TO_PRODUCTION" if passed else (
            "PASSED_TEST_ONLY" if (passed_threshold and execution_type == ValidationExecutionType.TEST_ONLY)
            else "VALIDATION_FAILED"
        )

        rejection_reason = None
        if not passed_threshold:
            rejection_reason = f"C-index {c_val} 未达到生产准入硬性下限 {effective_min_c}。"
        elif not beats_random:
            rejection_reason = f"C-index {c_val} 未能超越随机概率 (0.50)。"
        elif execution_type != ValidationExecutionType.FORMAL_PRODUCTION_AUDIT:
            rejection_reason = "当前验证为工程测试模式，保持基准模型，不晋升正式生产。"

        # 若未通过，清空可能遗留的生产状态
        if not passed:
            if hasattr(survival_engine, "model_stage") and survival_engine.model_stage == ModelStage.VERIFIED_PRODUCTION:
                survival_engine.model_stage = ModelStage.BENCHMARK
            if hasattr(survival_engine, "is_out_of_sample_validated"):
                survival_engine.is_out_of_sample_validated = False

        rec = OOSValidationRecord(
            metric_name="Concordance_Index",
            metric_value=c_val,
            threshold=effective_min_c,
            passed=passed,
            evaluation_date=eval_date,
            test_sample_count=len(validation_cohort_df),
            validation_type=execution_type,
            benchmark_metric_value=0.50,
            benchmark_name="Random_Concordance",
            beats_benchmark=beats_random,
            validation_conclusion=conclusion,
            rejection_reason=rejection_reason,
            details={"c_index": c_val, "min_required": effective_min_c}
        )

        return {
            "status": conclusion,
            "passed": passed,
            "validation_record": rec,
            "c_index": c_val,
            "threshold": effective_min_c,
            "beats_benchmark": beats_random,
            "rejection_reason": rejection_reason,
            "execution_type": execution_type.value
        }

    def _parse_validation_samples(
        self,
        raw_data: Union[OOSValidationDataset, List[OOSValidationSample], pd.DataFrame, Dict[str, Any], str, None],
        registered_dataset_id: Optional[str] = None
    ) -> Tuple[List[OOSValidationSample], Dict[str, Any]]:
        """标准化解析样本外验证数据，优先通过可信注册中心受控读取"""
        meta = {}
        target_id = registered_dataset_id or (raw_data if isinstance(raw_data, str) else None)
        if target_id:
            reg = TrustedValidationRegistry.get(target_id)
            if reg is not None:
                actual_hash = compute_samples_checksum(reg.samples)
                meta["dataset_id"] = reg.dataset_id
                meta["dataset_version"] = reg.dataset_version
                meta["audit_status"] = reg.audit_status
                meta["is_registered"] = True
                meta["is_synthetic"] = reg.is_synthetic
                meta["checksum_sha256"] = reg.checksum_sha256
                meta["sample_count"] = len(reg.samples)
                meta["is_audited_production_ready"] = bool(getattr(reg, "is_audited_production_ready", False))
                meta["tampered"] = bool(reg.checksum_sha256 and actual_hash.lower() != reg.checksum_sha256.lower())
                return reg.samples, meta
            meta["is_registered"] = False
            meta["audit_status"] = DataQualityStatus.UNVERIFIED
            meta["is_audited_production_ready"] = False
            return [], meta

        if isinstance(raw_data, OOSValidationDataset):
            meta["dataset_id"] = raw_data.dataset_id
            meta["dataset_version"] = raw_data.dataset_version
            meta["is_synthetic"] = raw_data.is_synthetic
            
            # 安全准则：严禁以 raw_data.is_registered 作为已注册判断依据！
            # 只有当 TrustedValidationRegistry 真正登记了该 dataset_id 时才视为已注册
            is_reg = TrustedValidationRegistry.contains(raw_data.dataset_id)
            meta["is_registered"] = is_reg
            if is_reg:
                reg = TrustedValidationRegistry.get(raw_data.dataset_id)
                meta["audit_status"] = reg.audit_status
                meta["is_audited_production_ready"] = bool(getattr(reg, "is_audited_production_ready", False))
                # 检查调用方是否偷换了样本内容
                caller_hash = compute_samples_checksum(raw_data.samples)
                if reg.checksum_sha256 and caller_hash.lower() != reg.checksum_sha256.lower():
                    meta["sample_substituted"] = True
                # 正式验证必须读取注册记录中的受控真实样本，绝不接受调用方替换后的样本对象
                actual_hash = compute_samples_checksum(reg.samples)
                meta["tampered"] = bool(reg.checksum_sha256 and actual_hash.lower() != reg.checksum_sha256.lower())
                meta["checksum_sha256"] = reg.checksum_sha256
                meta["sample_count"] = len(reg.samples)
                return reg.samples, meta
            else:
                meta["audit_status"] = DataQualityStatus.UNVERIFIED
                meta["is_audited_production_ready"] = False
                meta["checksum_sha256"] = compute_samples_checksum(raw_data.samples)
                meta["sample_count"] = len(raw_data.samples)
                return raw_data.samples, meta

        if isinstance(raw_data, list):
            meta["is_registered"] = False
            meta["audit_status"] = DataQualityStatus.UNVERIFIED
            meta["is_audited_production_ready"] = False
            meta["is_synthetic"] = False
            samples = []
            for item in raw_data:
                if isinstance(item, OOSValidationSample):
                    samples.append(item)
                elif isinstance(item, dict):
                    samples.append(OOSValidationSample(**item))
            meta["checksum_sha256"] = compute_samples_checksum(samples)
            meta["sample_count"] = len(samples)
            return samples, meta

        if isinstance(raw_data, pd.DataFrame):
            meta["is_registered"] = False
            meta["audit_status"] = DataQualityStatus.UNVERIFIED
            meta["is_synthetic"] = False
            samples = []
            for idx, row in raw_data.iterrows():
                feat_dict = {k: float(v) for k, v in row.items() if k not in [
                    "sample_id", "dataset_version", "feature_snapshot_id", "decision_date",
                    "feature_available_at", "label_period_end", "label_observed_at",
                    "label_source", "label_quality_status", "source_record_id",
                    "data_quality_status", "topic_id", "target_1y", "target_3y", "target_5y",
                    "is_synthetic", "is_model_generated"
                ]}
                labels_dict = {
                    1: float(row["target_1y"]) if "target_1y" in row and pd.notna(row["target_1y"]) else None,
                    3: float(row["target_3y"]) if "target_3y" in row and pd.notna(row["target_3y"]) else None,
                    5: float(row["target_5y"]) if "target_5y" in row and pd.notna(row["target_5y"]) else None,
                }
                s = OOSValidationSample(
                    sample_id=str(row.get("sample_id", f"oos_sample_{idx}")),
                    dataset_version=str(row.get("dataset_version", "v1.1-oos-dyn")),
                    feature_snapshot_id=str(row.get("feature_snapshot_id", f"snap_{idx}")),
                    decision_date=row.get("decision_date", date.today()),
                    feature_available_at=row.get("feature_available_at", datetime.utcnow()),
                    label_period_end=row.get("label_period_end", date.today()),
                    label_observed_at=row.get("label_observed_at", datetime.utcnow()),
                    label_source=str(row.get("label_source", "Raw_DataFrame_Ingestion")),
                    label_quality_status=DataQualityStatus.UNVERIFIED,
                    source_record_id=str(row.get("source_record_id", f"rec_{idx}")),
                    data_quality_status=DataQualityStatus.UNVERIFIED,
                    topic_id=row.get("topic_id"),
                    features=feat_dict,
                    labels=labels_dict,
                    is_synthetic=bool(row.get("is_synthetic", False)),
                    is_model_generated=bool(row.get("is_model_generated", False))
                )
                samples.append(s)
            return samples, meta

        if isinstance(raw_data, dict):
            features = raw_data.get("features", {})
            labels = raw_data.get("labels", {})
            meta["is_registered"] = False
            meta["is_synthetic"] = raw_data.get("is_synthetic", False)
            meta["audit_status"] = DataQualityStatus.UNVERIFIED
            samples = []
            if isinstance(features, pd.DataFrame):
                for idx, row in features.iterrows():
                    lab_dict = {h: labels[h][idx] if (h in labels and idx < len(labels[h])) else None for h in [1, 3, 5]}
                    samples.append(OOSValidationSample(
                        sample_id=f"oos_{idx}",
                        dataset_version="v1.1-oos-dict",
                        feature_snapshot_id=f"snap_{idx}",
                        decision_date=raw_data.get("decision_date", date.today()),
                        feature_available_at=raw_data.get("feature_available_at", datetime.utcnow()),
                        label_period_end=raw_data.get("label_period_end", date.today()),
                        label_observed_at=raw_data.get("label_observed_at", datetime.utcnow()),
                        label_source=raw_data.get("label_source", "Raw_Dict_Ingestion"),
                        label_quality_status=DataQualityStatus.UNVERIFIED,
                        source_record_id=f"rec_{idx}",
                        data_quality_status=DataQualityStatus.UNVERIFIED,
                        features=row.to_dict(),
                        labels=lab_dict,
                        is_synthetic=raw_data.get("is_synthetic", False),
                        is_model_generated=raw_data.get("is_model_generated", False)
                    ))
            return samples, meta

        return [], meta
