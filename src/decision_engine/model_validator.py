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
import numpy as np
import pandas as pd

from src.core.types import (
    DataQualityStatus,
    ModelStage,
    ValidationExecutionType,
    OOSValidationSample,
    OOSValidationDataset,
    OOSValidationRecord
)


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
        validation_dataset: Union[OOSValidationDataset, List[OOSValidationSample], pd.DataFrame, Dict[str, Any]],
        training_sample_ids: Optional[Set[str]] = None,
        training_windows: Optional[Set[Tuple[str, date]]] = None,
        training_target_values: Optional[Dict[int, List[float]]] = None,
        max_allowed_mae: Optional[float] = None,
        evaluation_date: Optional[date] = None,
        execution_type: ValidationExecutionType = ValidationExecutionType.FORMAL_PRODUCTION_AUDIT
    ) -> Dict[str, Any]:
        """对 LightGBM 成长预测模型执行独立的 6 步生产准入审计与样本外验证"""
        eval_date = evaluation_date or date.today()

        # 解析与标准化验证样本
        samples, dataset_meta = self._parse_validation_samples(validation_dataset)

        if not samples:
            return {
                "status": "VALIDATION_FAILED",
                "passed": False,
                "rejection_reason": "未提供有效的样本外验证数据集或样本集为空。",
                "execution_type": execution_type.value,
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
                    "合成模拟数据 (is_synthetic=True) 或未逐项核验数据 (quality_status != VERIFIED)。"
                )
                rec = OOSValidationRecord(
                    metric_name="Max_MAE",
                    metric_value=999.0,
                    threshold=self.PRODUCTION_MAX_MAE_CEILING,
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
                    "rejection_reason": rejection,
                    "validation_record": rec,
                    "model_stage": forecaster.model_stage.value
                }
            else:
                effective_execution_type = ValidationExecutionType.TEST_ONLY

        # 第 2 步：时序与样本隔离检查 (Temporal & Window Isolation)
        if training_sample_ids:
            overlap_ids = {s.sample_id for s in samples}.intersection(training_sample_ids)
            if overlap_ids:
                rejection = f"样本外隔离校验失败：验证集与训练集存在重叠样本 ID {list(overlap_ids)[:5]}。"
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "rejection_reason": rejection,
                    "model_stage": forecaster.model_stage.value
                }

        if training_windows:
            overlap_windows = {(s.topic_id, s.decision_date) for s in samples if s.topic_id}.intersection(training_windows)
            if overlap_windows:
                rejection = f"时序窗口重叠拒绝：验证集与训练集存在相同标的相同时序窗口 {list(overlap_windows)[:5]}。"
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "rejection_reason": rejection,
                    "model_stage": forecaster.model_stage.value
                }

        # 检查时间泄露：feature_available_at <= decision_date，label_period_end <= eval_date, label_observed_at <= eval_date
        for s in samples:
            feat_avail_date = s.feature_available_at.date() if isinstance(s.feature_available_at, datetime) else s.feature_available_at
            if feat_avail_date > s.decision_date:
                rejection = f"前视偏差时间泄露：样本 {s.sample_id} 的特征公布时间 {feat_avail_date} 晚于决策时间 {s.decision_date}。"
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "rejection_reason": rejection,
                    "model_stage": forecaster.model_stage.value
                }

            if s.label_period_end > eval_date:
                rejection = f"未成熟标签前视泄漏：样本 {s.sample_id} 的标签成熟期 {s.label_period_end} 晚于评估日 {eval_date}。"
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "rejection_reason": rejection,
                    "model_stage": forecaster.model_stage.value
                }

            label_obs_date = s.label_observed_at.date() if isinstance(s.label_observed_at, datetime) else s.label_observed_at
            if label_obs_date > eval_date:
                rejection = f"标签公开时间未来泄露：样本 {s.sample_id} 的真实公开时间 {label_obs_date} 晚于评估日 {eval_date}。"
                return {
                    "status": "VALIDATION_REJECTED",
                    "passed": False,
                    "rejection_reason": rejection,
                    "model_stage": forecaster.model_stage.value
                }

        # 第 3 步：独立样本外评测 (Independent OOS Testing)
        # 提取特征矩阵与标签
        feat_df = pd.DataFrame([s.features for s in samples])
        for col in forecaster.feature_names:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        X_test = feat_df[forecaster.feature_names].values

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
            return {
                "status": "VALIDATION_FAILED",
                "passed": False,
                "rejection_reason": rejection,
                "model_stage": forecaster.model_stage.value
            }

        max_mae = max(mae_by_horizon.values())
        naive_max_mae = max(naive_mae_by_horizon.values()) if naive_mae_by_horizon else 999.0

        # 第 4 步：生产准入硬性上限与基准对照 (Fixed Production Ceiling & Benchmark Beat)
        # 调用方即使传入更大的 max_allowed_mae，也不得放宽生产上限 PRODUCTION_MAX_MAE_CEILING
        if max_allowed_mae is not None:
            effective_threshold = min(float(max_allowed_mae), self.PRODUCTION_MAX_MAE_CEILING)
        else:
            effective_threshold = self.PRODUCTION_MAX_MAE_CEILING

        beats_benchmark = bool(max_mae < naive_max_mae)
        passed_threshold = bool(max_mae <= effective_threshold)
        passed = bool(passed_threshold and beats_benchmark and (effective_execution_type == ValidationExecutionType.FORMAL_PRODUCTION_AUDIT))

        conclusion = "PROMOTED_TO_PRODUCTION" if passed else (
            "PASSED_TEST_ONLY" if (passed_threshold and effective_execution_type == ValidationExecutionType.TEST_ONLY)
            else "VALIDATION_FAILED"
        )

        rejection_reason = None
        if not passed_threshold:
            rejection_reason = f"Max MAE {max_mae} 超过生产准入硬性上限 {effective_threshold}。"
        elif not beats_benchmark:
            rejection_reason = f"模型效果 (MAE: {max_mae}) 未能显著超越简单基准 (Naive MAE: {naive_max_mae})。"
        elif effective_execution_type != ValidationExecutionType.FORMAL_PRODUCTION_AUDIT:
            rejection_reason = "当前验证为工程测试或实验验证模式，禁止晋升生产阶段。"

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
        execution_type: ValidationExecutionType = ValidationExecutionType.FORMAL_PRODUCTION_AUDIT
    ) -> Dict[str, Any]:
        """对 Cox 企业生存风险分析模型执行独立的生产准入审计与样本外 C-index 验证"""
        eval_date = evaluation_date or date.today()

        if validation_cohort_df is None or len(validation_cohort_df) == 0:
            return {
                "status": "VALIDATION_FAILED",
                "passed": False,
                "rejection_reason": "未提供有效的样本外验证企业队列。",
                "model_stage": survival_engine.model_stage.value
            }

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

        # 检查真实血缘记录
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
            rec = OOSValidationRecord(
                metric_name="Concordance_Index",
                metric_value=c_val,
                threshold=self.PRODUCTION_MIN_C_INDEX_FLOOR,
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
                "validation_record": rec,
                "rejection_reason": rejection,
                "model_stage": survival_engine.model_stage.value
            }

        # 生产准入硬性下限：min_c_index 即使传入更低数值，也不得低于 PRODUCTION_MIN_C_INDEX_FLOOR
        if min_c_index is not None:
            effective_min_c = max(float(min_c_index), self.PRODUCTION_MIN_C_INDEX_FLOOR)
        else:
            effective_min_c = self.PRODUCTION_MIN_C_INDEX_FLOOR

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
        raw_data: Union[OOSValidationDataset, List[OOSValidationSample], pd.DataFrame, Dict[str, Any]]
    ) -> Tuple[List[OOSValidationSample], Dict[str, Any]]:
        """标准化解析样本外验证数据"""
        meta = {}
        if isinstance(raw_data, OOSValidationDataset):
            meta["is_synthetic"] = raw_data.is_synthetic
            meta["audit_status"] = raw_data.audit_status
            return raw_data.samples, meta

        if isinstance(raw_data, list):
            samples = []
            for item in raw_data:
                if isinstance(item, OOSValidationSample):
                    samples.append(item)
                elif isinstance(item, dict):
                    samples.append(OOSValidationSample(**item))
            return samples, meta

        if isinstance(raw_data, pd.DataFrame):
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
                    dataset_version=str(row.get("dataset_version", "v1.1-oos")),
                    feature_snapshot_id=str(row.get("feature_snapshot_id", f"snap_{idx}")),
                    decision_date=row.get("decision_date", date.today()),
                    feature_available_at=row.get("feature_available_at", datetime.utcnow()),
                    label_period_end=row.get("label_period_end", date.today()),
                    label_observed_at=row.get("label_observed_at", datetime.utcnow()),
                    label_source=str(row.get("label_source", "Independent_Audit_Vault")),
                    label_quality_status=DataQualityStatus(row.get("label_quality_status", DataQualityStatus.VERIFIED)),
                    source_record_id=str(row.get("source_record_id", f"rec_{idx}")),
                    data_quality_status=DataQualityStatus(row.get("data_quality_status", DataQualityStatus.VERIFIED)),
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
            meta["is_synthetic"] = raw_data.get("is_synthetic", False)
            meta["audit_status"] = raw_data.get("audit_status", DataQualityStatus.VERIFIED)
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
                        label_source=raw_data.get("label_source", "Independent_Audit_Vault"),
                        label_quality_status=DataQualityStatus(raw_data.get("label_quality_status", DataQualityStatus.VERIFIED)),
                        source_record_id=f"rec_{idx}",
                        data_quality_status=DataQualityStatus(raw_data.get("data_quality_status", DataQualityStatus.VERIFIED)),
                        features=row.to_dict(),
                        labels=lab_dict,
                        is_synthetic=raw_data.get("is_synthetic", False),
                        is_model_generated=raw_data.get("is_model_generated", False)
                    ))
            return samples, meta

        return [], meta
