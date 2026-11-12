"""PR #3 最终收口整改测试套件 (PR #3 Final Closure Verification Suite).

涵盖 16 项关键准入与反向审计验收测试：
P0-01:
1. 普通数组未标记模拟数据时，不能取得正式验证资格。
2. 模型预测结果作为标签但未设置 is_model_generated=True 时，反向作弊审计拦截拒绝。
3. 手工构造 VERIFIED 字段和哈希字符串，不能假冒注册中心审计记录。
4. 未注册的数据集 ID 不能进入正式验证。
5. 验证集与训练集样本重叠必须拒绝。
6. 验证标签公开时间晚于验证截止日必须拒绝。
7. 验证失败清空生产状态 (VERIFIED_PRODUCTION -> BENCHMARK)。
8. 使用模拟数据可以完成 TEST_ONLY 工程验证，但不晋升生产。

P0-02:
9. 独立期限训练：1y 有 30 条、3y 有 12 条、5y 有 0 条成熟标签。
10. 禁止跨期限标签复制：5y 保持不可预测 (None)，数据库 NULL 保持不变。

P0-03:
11. 历史模型缺失时，机器学习回测指标必须全部为 None。
12. 规则研究排序回测 (RESEARCH_RANKING_BACKTEST) 独立执行，显式标记规则排序。
13. 回测记录完整持久化 backtest_type, forecast_status, 样本统计与版本号。

P0-04:
14. 正式历史模型训练拒绝 feature_published_at IS NULL。
15. query_point_in_time(require_verified_timestamp=True) 严格过滤非 VERIFIED 记录。

P1:
16. 正式企业生存预测 (is_formal=True) 协变量缺失时返回 INSUFFICIENT_DATA，禁止默认值填充。
"""

from datetime import date, datetime
import pytest
import numpy as np
import pandas as pd

from src.core.database import DatabaseManager, db_manager
from src.core.types import (
    DataQualityStatus,
    ModelStage,
    ModelTrainingStatus,
    PredictionStatus,
    ValidationExecutionType,
    ProvenanceMetadata,
    OOSValidationSample,
    OOSValidationDataset,
    TrustedDatasetRegistration,
    BacktestType
)
from src.decision_engine.model_validator import (
    ModelValidationService,
    TrustedValidationRegistry
)
from src.diffusion.lgbm_forecaster import LightGBMGrowthForecaster
from src.risk_survival.survival_cox import CompanySurvivalEngine
from src.backtest.engine import PointInTimeBacktestEngine
from src.decision_engine.opportunity_pool import OpportunityPoolManager


@pytest.fixture
def temp_db(tmp_path):
    """创建隔离的测试临时 DuckDB 实例"""
    db_file = str(tmp_path / "test_closure.duckdb")
    mgr = DatabaseManager(db_path=db_file)
    mgr.init_schema()
    return mgr


# =============================================================================
# P0-01 验证数据真实性与可信注册中心审计门禁测试
# =============================================================================

def test_raw_array_without_simulation_flag_cannot_achieve_formal_verification():
    """1. 普通数组未标记模拟数据时，不能取得正式生产验证资格，模型保持 BENCHMARK"""
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) + 2.0, 3: np.random.randn(n) + 3.0, 5: np.random.randn(n) + 4.0}
    provs = [
        ProvenanceMetadata(
            record_id=f"TR_{i}",
            source_name="Audited_Vault",
            as_of_date=date(2020, 1, 1),
            published_at=datetime(2020, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="a" * 64
        )
        for i in range(n)
    ]
    forecaster.fit_verified(features, labels, quality_status=DataQualityStatus.VERIFIED, provenance_records=provs)
    assert forecaster.model_stage == ModelStage.BENCHMARK

    # 传入普通 numpy 数组且未提供注册数据集
    oos_feat = np.random.randn(10, 7)
    oos_labels = {1: np.random.randn(10) + 2.0, 3: np.random.randn(10) + 3.0, 5: np.random.randn(10) + 4.0}
    res = forecaster.validate_out_of_sample(oos_features=oos_feat, oos_labels_by_horizon=oos_labels)

    assert res["passed"] is False
    assert forecaster.model_stage == ModelStage.BENCHMARK
    assert forecaster.is_out_of_sample_validated is False
    assert "未在可信注册中心" in res["rejection_reason"]


def test_model_predicted_labels_without_is_model_generated_flag_rejected():
    """2. 模型预测结果作为标签但未设置 is_model_generated=True 时，反向作弊审计拦截拒绝"""
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) + 2.0, 3: np.random.randn(n) + 3.0, 5: np.random.randn(n) + 4.0}
    provs = [
        ProvenanceMetadata(
            record_id=f"TR_{i}",
            source_name="Audited_Vault",
            as_of_date=date(2020, 1, 1),
            published_at=datetime(2020, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="b" * 64
        )
        for i in range(n)
    ]
    forecaster.fit_verified(features, labels, quality_status=DataQualityStatus.VERIFIED, provenance_records=provs)

    # 注册一个数据集，但标签故意采用模型预测值构造 (未设置 is_model_generated=True)
    oos_feat = np.random.randn(10, 7)
    cheat_samples = [
        OOSValidationSample(
            sample_id=f"CHEAT_{i}",
            dataset_version="v1.1-cheat",
            feature_snapshot_id=f"snap_{i}",
            decision_date=date(2021, 1, 1),
            feature_available_at=datetime(2020, 12, 31),
            label_period_end=date(2022, 1, 1),
            label_observed_at=datetime(2022, 1, 2),
            label_source="Independent_Auditor",
            label_quality_status=DataQualityStatus.VERIFIED,
            source_record_id=f"rec_{i}",
            data_quality_status=DataQualityStatus.VERIFIED,
            features={forecaster.feature_names[j]: float(oos_feat[i][j]) for j in range(7)},
            labels={
                1: float(forecaster.models[1].predict(oos_feat)[i] + 0.05),  # 贴合 < 0.25 模拟作弊
                3: float(forecaster.models[3].predict(oos_feat)[i] + 0.05),
                5: float(forecaster.models[5].predict(oos_feat)[i] + 0.05)
            },
            is_model_generated=False  # 隐瞒自生标签事实！
        )
        for i in range(10)
    ]
    TrustedValidationRegistry.register(TrustedDatasetRegistration(
        dataset_id="REG_CHEAT_TEST_01",
        dataset_version="v1.1-cheat",
        audit_status=DataQualityStatus.VERIFIED,
        checksum_sha256="c" * 64,
        samples=cheat_samples
    ))

    res = forecaster.validate_formal_production_oos(registered_dataset_id="REG_CHEAT_TEST_01")
    assert res["passed"] is False
    assert forecaster.model_stage == ModelStage.BENCHMARK
    assert "自生" in res["rejection_reason"] or "cheat" in res["rejection_reason"].lower()


def test_manually_crafted_verified_and_fake_hash_cannot_fake_registry_record():
    """3. 手工构造 VERIFIED 字段和哈希字符串，不能假冒注册中心审计记录"""
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) + 2.0, 3: np.random.randn(n) + 3.0, 5: np.random.randn(n) + 4.0}
    provs = [
        ProvenanceMetadata(
            record_id=f"TR_{i}",
            source_name="Audited_Vault",
            as_of_date=date(2020, 1, 1),
            published_at=datetime(2020, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="d" * 64
        )
        for i in range(n)
    ]
    forecaster.fit_verified(features, labels, quality_status=DataQualityStatus.VERIFIED, provenance_records=provs)

    # 手工构造携带 VERIFIED 质量与 64 位哈希的假凭证
    fake_provs = [
        ProvenanceMetadata(
            record_id=f"FAKE_{i}",
            source_name="Fake_Audit_Archive",
            as_of_date=date(2021, 1, 1),
            published_at=datetime(2021, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="9" * 64
        )
        for i in range(10)
    ]
    oos_feat = np.random.randn(10, 7)
    oos_labels = {1: np.random.randn(10) + 2.0, 3: np.random.randn(10) + 3.0, 5: np.random.randn(10) + 4.0}

    res = forecaster.validate_out_of_sample(
        oos_features=oos_feat,
        oos_labels_by_horizon=oos_labels,
        provenance_records=fake_provs,
        execution_type=ValidationExecutionType.FORMAL_PRODUCTION_AUDIT
    )
    assert res["passed"] is False
    assert forecaster.model_stage == ModelStage.BENCHMARK
    assert "未在可信注册中心" in res["rejection_reason"]


def test_unregistered_dataset_id_cannot_enter_formal_validation():
    """4. 未注册的数据集 ID 不能进入正式验证"""
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) + 2.0, 3: np.random.randn(n) + 3.0, 5: np.random.randn(n) + 4.0}
    provs = [
        ProvenanceMetadata(
            record_id=f"TR_{i}",
            source_name="Audited_Vault",
            as_of_date=date(2020, 1, 1),
            published_at=datetime(2020, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="e" * 64
        )
        for i in range(n)
    ]
    forecaster.fit_verified(features, labels, quality_status=DataQualityStatus.VERIFIED, provenance_records=provs)

    # 尝试使用未在 TrustedValidationRegistry 中注册的 ID
    res = forecaster.validate_formal_production_oos(registered_dataset_id="UNREGISTERED_DATASET_XYZ")
    assert res["passed"] is False
    assert "未在可信注册中心" in res["rejection_reason"]
    assert forecaster.model_stage == ModelStage.BENCHMARK


def test_validation_set_overlapping_with_training_set_rejected():
    """5. 验证集与训练集样本 ID 重叠必须拒绝"""
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) + 2.0, 3: np.random.randn(n) + 3.0, 5: np.random.randn(n) + 4.0}
    provs = [
        ProvenanceMetadata(
            record_id=f"SAMP_ID_{i}",
            source_name="Audited_Vault",
            as_of_date=date(2020, 1, 1),
            published_at=datetime(2020, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="f" * 64
        )
        for i in range(n)
    ]
    forecaster.fit_verified(features, labels, quality_status=DataQualityStatus.VERIFIED, provenance_records=provs)

    # 构造包含训练集 SAMP_ID_0 的验证集并注册
    overlap_samples = [
        OOSValidationSample(
            sample_id="SAMP_ID_0",  # 存在重叠！
            dataset_version="v1.1-overlap",
            feature_snapshot_id="snap_0",
            decision_date=date(2021, 1, 1),
            feature_available_at=datetime(2020, 12, 31),
            label_period_end=date(2022, 1, 1),
            label_observed_at=datetime(2022, 1, 2),
            label_source="Independent_Auditor",
            label_quality_status=DataQualityStatus.VERIFIED,
            source_record_id="rec_0",
            data_quality_status=DataQualityStatus.VERIFIED,
            features={forecaster.feature_names[j]: float(features[0][j]) for j in range(7)},
            labels={1: 2.5, 3: 3.5, 5: 4.5}
        )
    ]
    TrustedValidationRegistry.register(TrustedDatasetRegistration(
        dataset_id="REG_OVERLAP_TEST",
        dataset_version="v1.1-overlap",
        audit_status=DataQualityStatus.VERIFIED,
        checksum_sha256="0" * 64,
        samples=overlap_samples
    ))

    res = forecaster.validate_formal_production_oos(registered_dataset_id="REG_OVERLAP_TEST")
    assert res["passed"] is False
    assert "重叠" in res["rejection_reason"]


def test_validation_label_published_after_validation_date_rejected():
    """6. 验证标签公开时间晚于验证截止日必须拒绝 (前视偏差检查)"""
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) + 2.0, 3: np.random.randn(n) + 3.0, 5: np.random.randn(n) + 4.0}
    provs = [
        ProvenanceMetadata(
            record_id=f"LEAK_TR_{i}",
            source_name="Audited_Vault",
            as_of_date=date(2020, 1, 1),
            published_at=datetime(2020, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="1" * 64
        )
        for i in range(n)
    ]
    forecaster.fit_verified(features, labels, quality_status=DataQualityStatus.VERIFIED, provenance_records=provs)

    eval_date = date(2021, 12, 31)
    future_samples = [
        OOSValidationSample(
            sample_id="LEAK_OOS_0",
            dataset_version="v1.1-leak",
            feature_snapshot_id="snap_leak",
            decision_date=date(2021, 6, 1),
            feature_available_at=datetime(2021, 5, 30),
            label_period_end=date(2021, 12, 1),
            label_observed_at=datetime(2022, 6, 1),  # 晚于评估日 2021-12-31！
            label_source="Audit_Archive",
            label_quality_status=DataQualityStatus.VERIFIED,
            source_record_id="rec_leak",
            data_quality_status=DataQualityStatus.VERIFIED,
            features={forecaster.feature_names[j]: 1.0 for j in range(7)},
            labels={1: 2.0, 3: 3.0, 5: 4.0}
        )
    ]
    TrustedValidationRegistry.register(TrustedDatasetRegistration(
        dataset_id="REG_LEAK_TEST",
        dataset_version="v1.1-leak",
        audit_status=DataQualityStatus.VERIFIED,
        checksum_sha256="2" * 64,
        samples=future_samples
    ))

    res = forecaster.validate_formal_production_oos(registered_dataset_id="REG_LEAK_TEST", evaluation_date=eval_date)
    assert res["passed"] is False
    assert "晚于评估日" in res["rejection_reason"] or "未来" in res["rejection_reason"]


def test_validation_failure_clears_production_status():
    """7. 验证失败后模型不得保留生产状态 (VERIFIED_PRODUCTION 降级为 BENCHMARK)"""
    forecaster = LightGBMGrowthForecaster()
    # 模拟先前已被标记为 VERIFIED_PRODUCTION
    forecaster.is_algorithm_fitted = True
    forecaster.is_data_verified = True
    forecaster.is_out_of_sample_validated = True
    forecaster.model_stage = ModelStage.VERIFIED_PRODUCTION

    # 使用不合法或未注册的数据集触发失败
    res = forecaster.validate_formal_production_oos(registered_dataset_id="UNREGISTERED_DATASET_FOR_RESET")
    assert res["passed"] is False
    assert forecaster.is_out_of_sample_validated is False
    assert forecaster.model_stage == ModelStage.BENCHMARK


def test_synthetic_data_can_perform_test_only_engineering_validation():
    """8. 使用模拟数据可以完成 TEST_ONLY 工程验证，但绝不晋升生产阶段"""
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) + 2.0, 3: np.random.randn(n) + 3.0, 5: np.random.randn(n) + 4.0}
    provs = [
        ProvenanceMetadata(
            record_id=f"SYNTH_TR_{i}",
            source_name="Audited_Vault",
            as_of_date=date(2020, 1, 1),
            published_at=datetime(2020, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="3" * 64
        )
        for i in range(n)
    ]
    forecaster.fit_verified(features, labels, quality_status=DataQualityStatus.VERIFIED, provenance_records=provs)

    oos_feat = np.random.randn(10, 7)
    # 传入合成数据
    synth_labels = {
        1: forecaster.models[1].predict(oos_feat) + 0.5,
        3: forecaster.models[3].predict(oos_feat) + 0.5,
        5: forecaster.models[5].predict(oos_feat) + 0.5
    }
    res = forecaster.validate_out_of_sample(
        oos_features=oos_feat,
        oos_labels_by_horizon=synth_labels,
        is_synthetic=True,
        execution_type=ValidationExecutionType.TEST_ONLY,
        max_allowed_mae=2.0
    )
    # 工程测试验证通过
    assert res["status"] == "PASSED_TEST_ONLY"
    assert res["passed"] is True
    # 关键：模型阶段绝不晋升为 VERIFIED_PRODUCTION，is_out_of_sample_validated 保持 False
    assert forecaster.model_stage == ModelStage.BENCHMARK
    assert forecaster.is_out_of_sample_validated is False


# =============================================================================
# P0-02 独立预测期限训练与禁止跨期限标签复制测试
# =============================================================================

def test_independent_horizon_training_30_12_0_samples():
    """9. 构建 1y: 30 条, 3y: 12 条, 5y: 0 条成熟标签：1y/3y 成功拟合，5y 不得拟合"""
    forecaster = LightGBMGrowthForecaster()
    n = 30
    features = np.random.randn(n, 7)
    labels_30_12_0 = {
        1: np.random.uniform(1.2, 2.5, 30),                         # 30 条成熟
        3: [np.random.uniform(2.0, 4.0) if i < 12 else None for i in range(30)],  # 12 条成熟
        5: [None for _ in range(30)]                                # 0 条成熟 (全未成熟)
    }

    res = forecaster.fit_experimental(features, labels_30_12_0)
    assert res["status"] == "TRAINED_EXPERIMENTAL_RESEARCH"
    # 1y 达到 >= 5 拟合，3y 达到 >= 5 拟合
    assert 1 in forecaster.models
    assert 3 in forecaster.models
    # 5y 样本为 0，绝对不拟合模型
    assert 5 not in forecaster.models


def test_no_cross_horizon_label_copying_and_5y_unavailable():
    """10. 禁止跨期限复制标签：5y 预测返回 None，状态标明不可用"""
    forecaster = LightGBMGrowthForecaster()
    n = 30
    features = np.random.randn(n, 7)
    labels_30_12_0 = {
        1: np.random.uniform(1.2, 2.5, 30),
        3: [np.random.uniform(2.0, 4.0) if i < 12 else None for i in range(30)],
        5: [None for _ in range(30)]
    }
    forecaster.fit_experimental(features, labels_30_12_0)

    pred = forecaster.predict_growth({"velocity_1y": 10.0})
    forecasts = pred["forecasts"]
    assert forecasts["1y"]["point_estimate"] is not None
    assert forecasts["3y"]["point_estimate"] is not None
    # 5y 模型未训练，绝对返回 None，禁止填充或复制 3y 数值
    assert forecasts["5y"]["point_estimate"] is None
    assert forecasts["5y"]["ci_low"] is None
    assert forecasts["5y"]["ci_high"] is None


# =============================================================================
# P0-03 严格隔离机器学习预测回测与规则研究排序回测测试
# =============================================================================

def test_missing_historical_model_ml_backtest_metrics_are_none(temp_db, monkeypatch):
    """11. 历史模型缺失时，机器学习回测指标必须全部为 None，严禁使用 research_score 充当 ML 成绩"""
    import src.core.database as db_mod
    import src.backtest.engine as bt_mod
    import src.decision_engine.opportunity_pool as op_mod
    monkeypatch.setattr(db_mod, "db_manager", temp_db)
    monkeypatch.setattr(bt_mod, "db_manager", temp_db)
    monkeypatch.setattr(op_mod, "db_manager", temp_db)

    # 清空样本库，模拟在无可用样本的历史时点运行机器学习回测
    temp_db.execute("DELETE FROM tech_training_samples;")

    engine = PointInTimeBacktestEngine()
    res = engine.run_walk_forward_backtest(
        cutoff_date=date(2021, 12, 31),
        evaluation_date=date(2024, 12, 31),
        horizon_years=3,
        backtest_type=BacktestType.ML_PREDICTION_BACKTEST
    )

    assert res["backtest_type"] == BacktestType.ML_PREDICTION_BACKTEST.value
    assert res["forecast_status"] == PredictionStatus.MODEL_UNAVAILABLE_AS_OF_DATE.value
    # 机器学习关键评估指标必须严格全部为 None
    metrics = res["metrics"]
    assert metrics["precision_at_3"] is None
    assert metrics["precision_at_5"] is None
    assert metrics["recall_at_5"] is None
    assert metrics["pr_auc"] is None
    assert metrics["brier_score"] is None
    assert metrics["excess_vs_random_baseline"] is None
    assert metrics["excess_vs_momentum_baseline"] is None


def test_rule_research_ranking_backtest_runs_separately(temp_db, monkeypatch):
    """12. 规则研究排序回测 (RESEARCH_RANKING_BACKTEST) 独立执行并显式标识，不冒充机器学习准确率"""
    import src.core.database as db_mod
    import src.backtest.engine as bt_mod
    import src.decision_engine.opportunity_pool as op_mod
    monkeypatch.setattr(db_mod, "db_manager", temp_db)
    monkeypatch.setattr(bt_mod, "db_manager", temp_db)
    monkeypatch.setattr(op_mod, "db_manager", temp_db)

    engine = PointInTimeBacktestEngine()
    res = engine.run_walk_forward_backtest(
        cutoff_date=date(2021, 12, 31),
        evaluation_date=date(2024, 12, 31),
        horizon_years=3,
        backtest_type=BacktestType.RESEARCH_RANKING_BACKTEST
    )

    assert res["backtest_type"] == BacktestType.RESEARCH_RANKING_BACKTEST.value
    assert res["forecast_status"] == "RESEARCH_RULE_RANKING"
    assert res["validation_status"] == "RESEARCH_HEURISTIC"


def test_backtest_runs_records_audit_columns_and_versioning(temp_db, monkeypatch):
    """13. 回测记录完整持久化 backtest_type, forecast_status, 样本统计与版本号"""
    import src.core.database as db_mod
    import src.backtest.engine as bt_mod
    import src.decision_engine.opportunity_pool as op_mod
    monkeypatch.setattr(db_mod, "db_manager", temp_db)
    monkeypatch.setattr(bt_mod, "db_manager", temp_db)
    monkeypatch.setattr(op_mod, "db_manager", temp_db)

    engine = PointInTimeBacktestEngine()
    # 运行第一次
    res1 = engine.run_walk_forward_backtest(
        cutoff_date=date(2021, 12, 31),
        evaluation_date=date(2024, 12, 31),
        horizon_years=3,
        backtest_type=BacktestType.ML_PREDICTION_BACKTEST
    )
    # 相同参数运行第二次，验证版本号自增
    res2 = engine.run_walk_forward_backtest(
        cutoff_date=date(2021, 12, 31),
        evaluation_date=date(2024, 12, 31),
        horizon_years=3,
        backtest_type=BacktestType.ML_PREDICTION_BACKTEST
    )

    assert res1["run_id"] != res2["run_id"]
    assert "_v2" in res2["run_id"]

    df_runs = temp_db.query_df("SELECT * FROM backtest_runs WHERE run_id = ?", [res2["run_id"]])
    assert len(df_runs) == 1
    row = df_runs.iloc[0]
    assert row["backtest_type"] == BacktestType.ML_PREDICTION_BACKTEST.value
    assert "forecast_status" in df_runs.columns
    assert "evaluable_sample_count" in df_runs.columns
    assert "unevaluable_sample_count" in df_runs.columns


# =============================================================================
# P0-04 严格历史时点数据隔离测试
# =============================================================================

def test_formal_historical_training_rejects_null_feature_published_at(temp_db, monkeypatch):
    """14. 正式历史模型训练拒绝 feature_published_at IS NULL (无发布时间凭证不可进入 PIT 训练)"""
    import src.core.database as db_mod
    import src.backtest.engine as bt_mod
    monkeypatch.setattr(db_mod, "db_manager", temp_db)
    monkeypatch.setattr(bt_mod, "db_manager", temp_db)

    # 插入一条没有 feature_published_at 的记录
    temp_db.execute("""
        INSERT INTO tech_training_samples
        (sample_id, topic_id, base_date, published_at, feature_published_at, velocity_1y, acceleration, active_institutions, github_momentum, macro_multiplier, entropy_score, novelty_score, target_1y, target_3y, target_5y, label_3y_period_end, quality_status)
        VALUES ('SAMP_NULL_PUB', 'TOPIC_AI', '2021-01-01', '2021-01-01 00:00:00', NULL, 10, 2, 20, 50, 1.0, 0.5, 0.5, 1.2, 2.5, NULL, '2021-12-31', 'VERIFIED');
    """)

    engine = PointInTimeBacktestEngine()
    df_pit = engine.get_historical_training_samples(cutoff_date=date(2021, 12, 31), horizon_years=3)
    # feature_published_at 为 NULL 的记录必须被严格过滤排除
    assert "SAMP_NULL_PUB" not in df_pit["sample_id"].values


def test_query_point_in_time_with_require_verified_timestamp_filters_unverified(temp_db):
    """15. query_point_in_time(require_verified_timestamp=True) 严格过滤非 VERIFIED 记录"""
    # 插入一条 UNVERIFIED 记录和一条 VERIFIED 记录
    temp_db.execute("""
        INSERT INTO tech_training_samples
        (sample_id, topic_id, base_date, published_at, feature_published_at, velocity_1y, acceleration, active_institutions, github_momentum, macro_multiplier, entropy_score, novelty_score, quality_status)
        VALUES ('SAMP_VERIFIED', 'TOPIC_VERIFIED', '2021-01-01', '2021-01-02 00:00:00', '2021-01-02 00:00:00', 10, 2, 20, 50, 1.0, 0.5, 0.5, 'VERIFIED');
    """)
    temp_db.execute("""
        INSERT INTO tech_training_samples
        (sample_id, topic_id, base_date, published_at, feature_published_at, velocity_1y, acceleration, active_institutions, github_momentum, macro_multiplier, entropy_score, novelty_score, quality_status)
        VALUES ('SAMP_UNVERIFIED', 'TOPIC_UNVERIFIED', '2021-01-01', '2021-01-02 00:00:00', '2021-01-02 00:00:00', 10, 2, 20, 50, 1.0, 0.5, 0.5, 'UNVERIFIED');
    """)

    df_normal = temp_db.query_point_in_time("tech_training_samples", cutoff_date=date(2021, 12, 31), require_verified_timestamp=False)
    assert "SAMP_VERIFIED" in df_normal["sample_id"].values
    assert "SAMP_UNVERIFIED" in df_normal["sample_id"].values

    df_verified = temp_db.query_point_in_time("tech_training_samples", cutoff_date=date(2021, 12, 31), require_verified_timestamp=True)
    # 严格模式下，只有 quality_status == 'VERIFIED' 的记录被检索
    assert "SAMP_VERIFIED" in df_verified["sample_id"].values
    assert "SAMP_UNVERIFIED" not in df_verified["sample_id"].values


# =============================================================================
# P1 协变量完整性与真实收益模拟状态隔离测试
# =============================================================================

def test_formal_prediction_with_missing_cox_covariates_returns_insufficient_data():
    """16. 正式企业生存预测 (is_formal=True) 协变量缺失时返回 INSUFFICIENT_DATA，禁止默认值填充"""
    engine = CompanySurvivalEngine()
    engine.is_fitted = True
    engine.is_algorithm_fitted = True
    engine.is_data_verified = True
    engine.model_stage = ModelStage.BENCHMARK

    # 缺失 cash_burn_months 和 competitor_count
    metrics_partial = {
        "debt_ratio": 45.0,
        "gross_margin": 30.0
    }

    # 正式预测调用
    res_formal = engine.predict_company_survival(metrics_partial, is_formal=True)
    assert res_formal["status"] == PredictionStatus.INSUFFICIENT_DATA.value
    assert res_formal["partial_hazard_ratio"] is None
    assert "缺失" in res_formal["risk_assessment"]
    assert "cash_burn_months" in res_formal["risk_assessment"]
