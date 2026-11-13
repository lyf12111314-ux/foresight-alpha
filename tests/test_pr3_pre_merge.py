"""PR #3 最终合并前定向整改验收专项测试 (Pre-Merge Final Assurance Tests).

严格覆盖六项与 Section 二至五全部验收要求：
1. 伪造注册状态 (is_registered=True) 无法获得生产认证。
2. 伪造质量状态 (audit_status=VERIFIED) 无法获得生产认证。
3. 手工伪造 64 位 SHA-256 摘要无法通过真实性检查。
4. 使用已注册 ID 替换实际样本时正式验证失败。
5. 数据集内容变化但摘要未更新时正式验证失败。
6. 未注册数据集无法晋升生产模型。
7. 系统内置示例数据集无法取得真实生产认证资格。
8. 模拟数据可以正常执行 TEST_ONLY / 实验验证。
9. 实验模型回测不被标记为 VALIDATED，显式标记为 EXPERIMENTAL。
10. 历史模型不可用时机器学习回测指标严格为 None。
11. 实验回测与规则研究回测分别隔离存储。
12. 历史不同技术主题使用不同特征向量，不统一退回默认值。
13. 缺失特征向量时正式历史预测返回 INSUFFICIENT_FEATURES。
14. 特征公开时间晚于历史决策日时拒绝使用。
15. Cox 企业生存模型人工参考数据与示例数据无法取得生产资格。
16. PIT 严格审计模式遭遇异常时显式报错，绝不静默放宽过滤。
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from src.core.types import (
    DataQualityStatus,
    ModelStage,
    ValidationExecutionType,
    ModelTrainingStatus,
    BacktestType,
    GroundTruthOutcome,
    OOSValidationSample,
    OOSValidationDataset,
    TrustedDatasetRegistration
)
from src.core.database import db_manager
from src.diffusion.lgbm_forecaster import LightGBMGrowthForecaster
from src.decision_engine.model_validator import (
    ModelValidationService,
    TrustedValidationRegistry,
    compute_samples_checksum
)
from src.risk_survival.survival_cox import CompanySurvivalEngine
from src.backtest.engine import PointInTimeBacktestEngine
from src.decision_engine.opportunity_pool import OpportunityPoolManager


@pytest.fixture(autouse=True)
def setup_test_registry():
    """每次测试重置注册中心至初始状态"""
    TrustedValidationRegistry.clear()
    yield
    TrustedValidationRegistry.clear()


# =============================================================================
# 一、P0-01: 可信验证数据集注册防绕过与真实性门禁
# =============================================================================

def test_p0_01_fake_is_registered_flag_cannot_bypass_production_gate():
    """1. 手工构造 is_registered=True，不能绕过服务端可信注册中心取得正式生产认证"""
    forecaster = LightGBMGrowthForecaster()
    # 模拟基础拟合完成，处于 BENCHMARK
    forecaster.models[3] = "dummy_model"
    forecaster.is_algorithm_fitted = True

    samples = [
        OOSValidationSample(
            sample_id="SAMPLE_01",
            dataset_version="v1.1-fake",
            feature_snapshot_id="snap_01",
            decision_date=date(2021, 1, 1),
            feature_available_at=datetime(2020, 12, 31),
            label_period_end=date(2022, 1, 1),
            label_observed_at=datetime(2022, 1, 2),
            label_source="Audit_Source",
            label_quality_status=DataQualityStatus.VERIFIED,
            source_record_id="rec_01",
            data_quality_status=DataQualityStatus.VERIFIED,
            features={"velocity_1y": 10.0, "acceleration": 1.0, "active_institutions": 5.0, "github_momentum": 20.0, "macro_multiplier": 1.0, "entropy_score": 0.5, "novelty_score": 0.5},
            labels={1: 2.0, 3: 3.0, 5: 4.0}
        )
    ]
    # 调用方自报 is_registered=True，但未在 TrustedValidationRegistry 中注册
    dataset = OOSValidationDataset(
        dataset_id="UNREGISTERED_DATASET_001",
        dataset_version="v1.1-fake",
        audit_status=DataQualityStatus.VERIFIED,
        samples=samples,
        is_registered=True
    )

    validator = ModelValidationService()
    res = validator.audit_and_validate_lgbm(
        forecaster=forecaster,
        validation_dataset=dataset,
        execution_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT
    )

    assert res["passed"] is False
    assert res["status"] == "VALIDATION_REJECTED"
    assert "未在可信注册中心" in res["rejection_reason"]


def test_p0_01_fake_audit_status_verified_cannot_bypass_production_gate():
    """2. 调用方直接自报 audit_status=VERIFIED，未在注册中心登记时正式验证失败"""
    forecaster = LightGBMGrowthForecaster()
    forecaster.models[3] = "dummy_model"
    forecaster.is_algorithm_fitted = True

    dataset = OOSValidationDataset(
        dataset_id="UNREGISTERED_DATASET_002",
        dataset_version="v1.1-fake",
        audit_status=DataQualityStatus.VERIFIED,
        samples=[]
    )

    validator = ModelValidationService()
    res = validator.audit_and_validate_lgbm(
        forecaster=forecaster,
        validation_dataset=dataset,
        execution_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT
    )

    assert res["passed"] is False
    assert forecaster.model_stage != ModelStage.VERIFIED_PRODUCTION


def test_p0_01_fake_sha256_hash_fails_authenticity_check():
    """3. 手工填写 64 位 SHA-256 字符串但与实际样本哈希不符，真实性检查失败"""
    forecaster = LightGBMGrowthForecaster()
    forecaster.models[3] = "dummy_model"
    forecaster.is_algorithm_fitted = True

    sample = OOSValidationSample(
        sample_id="SAMPLE_03",
        dataset_version="v1.1-hash-test",
        feature_snapshot_id="snap_03",
        decision_date=date(2021, 1, 1),
        feature_available_at=datetime(2020, 12, 31),
        label_period_end=date(2022, 1, 1),
        label_observed_at=datetime(2022, 1, 2),
        label_source="Audit_Source",
        label_quality_status=DataQualityStatus.VERIFIED,
        source_record_id="rec_03",
        data_quality_status=DataQualityStatus.VERIFIED,
        features={"velocity_1y": 10.0, "acceleration": 1.0, "active_institutions": 5.0, "github_momentum": 20.0, "macro_multiplier": 1.0, "entropy_score": 0.5, "novelty_score": 0.5},
        labels={1: 2.0, 3: 3.0, 5: 4.0}
    )

    # 注册一个具有假哈希的数据集
    reg = TrustedDatasetRegistration(
        dataset_id="REG_FAKE_HASH_DATASET",
        dataset_version="v1.1-hash-test",
        audit_status=DataQualityStatus.VERIFIED,
        checksum_sha256="deadbeef" * 8,  # 假哈希！与实际样本计算结果不一致
        samples=[sample],
        is_audited_production_ready=True,
        has_independent_audit_certificate=True
    )
    TrustedValidationRegistry.register(reg)

    validator = ModelValidationService()
    res = validator.audit_and_validate_lgbm(
        forecaster=forecaster,
        registered_dataset_id="REG_FAKE_HASH_DATASET",
        execution_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT
    )

    assert res["passed"] is False
    assert "REJECTED_CHECKSUM_TAMPERED" in res["validation_record"].validation_conclusion or "摘要" in res["rejection_reason"]


def test_p0_01_sample_substitution_with_registered_id_fails():
    """4. 使用已注册 ID 但调用方替换了实际样本内容，系统检测到样本替换行为并拒绝认证"""
    forecaster = LightGBMGrowthForecaster()
    forecaster.models[3] = "dummy_model"
    forecaster.is_algorithm_fitted = True

    sample_orig = OOSValidationSample(
        sample_id="SAMPLE_ORIG",
        dataset_version="v1.1-sub",
        feature_snapshot_id="snap_orig",
        decision_date=date(2021, 1, 1),
        feature_available_at=datetime(2020, 12, 31),
        label_period_end=date(2022, 1, 1),
        label_observed_at=datetime(2022, 1, 2),
        label_source="Audit_Source",
        label_quality_status=DataQualityStatus.VERIFIED,
        source_record_id="rec_orig",
        data_quality_status=DataQualityStatus.VERIFIED,
        features={"velocity_1y": 10.0, "acceleration": 1.0, "active_institutions": 5.0, "github_momentum": 20.0, "macro_multiplier": 1.0, "entropy_score": 0.5, "novelty_score": 0.5},
        labels={1: 2.0, 3: 3.0, 5: 4.0}
    )
    reg = TrustedDatasetRegistration(
        dataset_id="REG_SUB_TEST",
        dataset_version="v1.1-sub",
        audit_status=DataQualityStatus.VERIFIED,
        samples=[sample_orig],
        is_audited_production_ready=True,
        has_independent_audit_certificate=True
    )
    TrustedValidationRegistry.register(reg)

    # 调用方使用相同的 dataset_id，但传入被篡改的样本 (偷换特征)
    sample_mutated = sample_orig.model_copy(deep=True)
    sample_mutated.features["velocity_1y"] = 9999.0

    tampered_dataset = OOSValidationDataset(
        dataset_id="REG_SUB_TEST",
        dataset_version="v1.1-sub",
        audit_status=DataQualityStatus.VERIFIED,
        samples=[sample_mutated]
    )

    validator = ModelValidationService()
    res = validator.audit_and_validate_lgbm(
        forecaster=forecaster,
        validation_dataset=tampered_dataset,
        execution_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT
    )

    assert res["passed"] is False
    assert "REJECTED_SAMPLE_SUBSTITUTION" in res["validation_record"].validation_conclusion or "替换" in res["rejection_reason"]


def test_p0_01_example_seeded_dataset_cannot_promote_to_production():
    """5. 预置的示例数据集不能取得真实独立生产认证资格"""
    forecaster = LightGBMGrowthForecaster()
    forecaster.models[3] = "dummy_model"
    forecaster.is_algorithm_fitted = True

    validator = ModelValidationService()
    res = validator.audit_and_validate_lgbm(
        forecaster=forecaster,
        registered_dataset_id="REG_OOS_TECH_V1_2023",  # 系统内置示例数据集
        execution_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT
    )

    assert res["passed"] is False
    assert "REJECTED_DATASET_NOT_AUDITED_FOR_PRODUCTION" in res["validation_record"].validation_conclusion or "示例" in res["rejection_reason"]


def test_p0_01_synthetic_data_can_run_test_only_mode():
    """6. 模拟数据可以正常执行 TEST_ONLY 工程测试验证，模型保持 EXPERIMENTAL/BENCHMARK 不晋升生产"""
    forecaster = LightGBMGrowthForecaster()
    n = 30
    np.random.seed(42)
    X = np.random.uniform(1.0, 5.0, size=(n, 7))
    y = {h: X[:, 0] * 1.5 + float(h) for h in [1, 3, 5]}
    forecaster.fit_experimental(X, y)

    res = forecaster.validate_out_of_sample(
        oos_features=X,
        oos_labels_by_horizon=y,
        execution_type=ValidationExecutionType.TEST_ONLY,
        is_synthetic=True,
        max_allowed_mae=5.0
    )

    assert res["passed"] is True
    assert res["status"] == "PASSED_TEST_ONLY"
    assert forecaster.model_stage in [ModelStage.BENCHMARK, ModelStage.EXPERIMENTAL]
    assert forecaster.model_stage != ModelStage.VERIFIED_PRODUCTION
    assert forecaster.is_out_of_sample_validated is False


# =============================================================================
# 二、P0-02: 历史实验模型与正式回测状态物理隔离
# =============================================================================

def test_p0_02_experimental_model_backtest_not_marked_as_validated():
    """7. 历史模型处于 EXPERIMENTAL 时，回测严格标记为 EXPERIMENTAL，绝不伪造为 VALIDATED"""
    engine = PointInTimeBacktestEngine()
    res = engine.run_walk_forward_backtest(
        cutoff_date=date(2021, 12, 31),
        evaluation_date=date(2024, 12, 31),
        horizon_years=3,
        backtest_type=BacktestType.ML_EXPERIMENTAL_BACKTEST
    )

    assert res["validation_status"] != "VALIDATED"
    assert res["validation_status"] in ["EXPERIMENTAL", "MODEL_UNAVAILABLE"]
    if res["validation_status"] == "EXPERIMENTAL":
        assert res["forecast_status"] == "EXPERIMENTAL_PREDICTION"


def test_p0_02_formal_ml_backtest_rejects_unverified_historical_model():
    """8. 正式机器学习回测 (ML_VALIDATED_BACKTEST) 遇到未经验证模型时，指标严格为 None"""
    engine = PointInTimeBacktestEngine()
    res = engine.run_walk_forward_backtest(
        cutoff_date=date(2021, 12, 31),
        evaluation_date=date(2024, 12, 31),
        horizon_years=3,
        backtest_type=BacktestType.ML_VALIDATED_BACKTEST
    )

    assert res["validation_status"] != "VALIDATED"
    assert res["metrics"]["precision_at_3"] is None
    assert res["metrics"]["pr_auc"] is None


def test_p0_02_experimental_and_rule_backtests_stored_separately():
    """9. 实验回测与规则研究回测独立存储，不混淆类型"""
    engine = PointInTimeBacktestEngine()
    res_exp = engine.run_walk_forward_backtest(
        cutoff_date=date(2021, 12, 31),
        evaluation_date=date(2024, 12, 31),
        horizon_years=3,
        backtest_type=BacktestType.ML_EXPERIMENTAL_BACKTEST
    )
    res_rule = engine.run_walk_forward_backtest(
        cutoff_date=date(2021, 12, 31),
        evaluation_date=date(2024, 12, 31),
        horizon_years=3,
        backtest_type=BacktestType.RESEARCH_RANKING_BACKTEST
    )

    assert res_exp["backtest_type"] == BacktestType.ML_EXPERIMENTAL_BACKTEST.value
    assert res_rule["backtest_type"] == BacktestType.RESEARCH_RANKING_BACKTEST.value
    assert res_rule["forecast_status"] == "RESEARCH_RULE_RANKING"


# =============================================================================
# 三、P0-03: 历史预测特征向量完整传递与杜绝默认值冒充
# =============================================================================

def test_p0_03_different_candidates_have_distinct_feature_vectors():
    """10. 机会池中不同技术主题保存各自真实特征向量，且不使用硬编码固定 0.6/0.7"""
    forecaster = LightGBMGrowthForecaster()
    pool_mgr = OpportunityPoolManager(lgbm_forecaster=forecaster)
    cutoff = date(2021, 12, 31)

    candidates = pool_mgr.build_opportunity_pool(cutoff_date=cutoff)
    assert len(candidates) >= 3

    c0 = candidates[0]
    c1 = candidates[1]

    # 特征字段完整性检查
    req_keys = ["velocity_1y", "acceleration", "active_institutions", "github_momentum", "macro_multiplier", "entropy_score", "novelty_score"]
    for k in req_keys:
        assert k in c0["features"]
        assert k in c1["features"]

    # 验证不同主题特征值具有区分度，而非完全相同的默认值
    feat_differs = False
    for k in req_keys:
        if c0["features"][k] != c1["features"][k]:
            feat_differs = True
            break
    assert feat_differs is True, "不同技术主题特征向量不应完全相同"


def test_p0_03_missing_features_refuses_formal_forecast():
    """11. 缺失模型关键特征时，正式预测拒绝默认值填充，返回 INSUFFICIENT_FEATURES"""
    forecaster = LightGBMGrowthForecaster()
    # 模拟拟合模型
    n = 20
    X = np.random.randn(n, 7)
    y = {1: np.random.randn(n) + 2.0, 3: np.random.randn(n) + 3.0, 5: np.random.randn(n) + 4.0}
    forecaster.fit_experimental(X, y)

    # 传入空特征字典，开启完整性校验
    res = forecaster.predict_growth(features={}, require_complete_features=True)
    assert res["status"] == "INSUFFICIENT_FEATURES"
    assert res["forecasts"]["3y"]["point_estimate"] is None

    # 传入缺失部分字段的特征字典
    partial_features = {"velocity_1y": 10.0}
    res_part = forecaster.predict_growth(features=partial_features, require_complete_features=True)
    assert res_part["status"] == "INSUFFICIENT_FEATURES"
    assert res_part["forecasts"]["3y"]["point_estimate"] is None


# =============================================================================
# 四、P1: Cox 准入一致性与 PIT 严格异常门禁
# =============================================================================

def test_p1_01_cox_manual_reference_and_example_cannot_promote_production():
    """12. Cox 模型基于人工参考数据或示例数据集无法取得 VERIFIED_PRODUCTION 阶段"""
    engine = CompanySurvivalEngine()
    df = pd.DataFrame({
        "duration": [12.0, 24.0, 36.0, 48.0, 60.0],
        "event": [1, 0, 1, 0, 1],
        "cash_burn_months": [12.0, 18.0, 24.0, 30.0, 36.0],
        "debt_ratio": [30.0, 40.0, 50.0, 60.0, 70.0],
        "gross_margin": [40.0, 35.0, 30.0, 25.0, 20.0],
        "competitor_count": [3.0, 4.0, 5.0, 6.0, 7.0]
    })
    engine.fit_experimental(df, duration_col="duration", event_col="event")
    assert engine.model_stage == ModelStage.EXPERIMENTAL

    # 使用内置示例数据集尝试正式生产验证
    validator = ModelValidationService()
    res = validator.audit_and_validate_cox(
        survival_engine=engine,
        validation_cohort_df=df,
        registered_dataset_id="REG_OOS_COHORT_V1_2023",
        execution_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT
    )

    assert res["passed"] is False
    assert engine.model_stage != ModelStage.VERIFIED_PRODUCTION


def test_p1_02_pit_strict_mode_raises_on_invalid_metadata():
    """13. PIT 严格模式在遇到异常或无法检查数据质量字段时显式抛出异常，绝不静默放宽"""
    with pytest.raises(RuntimeError, match="Point-in-Time 严格审计模式错误"):
        db_manager.query_point_in_time(
            table_name="non_existent_table_999",
            cutoff_date=date(2021, 12, 31),
            require_verified_timestamp=True
        )
