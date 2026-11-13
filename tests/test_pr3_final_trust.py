"""PR #3 最终可信度整改自动化测试套件 (PR #3 Final Trust Remediation Test Suite).

覆盖 15 项核心审计与可信度测试场景：
1. test_model_self_generated_labels_cannot_grant_oos_production_status
2. test_overlapping_test_and_training_sets_fail_validation
3. test_test_label_published_after_validation_cutoff_fails_validation
4. test_missing_provenance_fails_formal_model_promotion
5. test_caller_modifying_mae_threshold_cannot_bypass_production_ceiling
6. test_experimental_model_growth_multiple_cannot_enter_formal_10x_payoff_probability
7. test_simulated_payoff_results_marked_as_hypothetical_scenario_with_disclaimer
8. test_2021_backtest_cannot_use_2026_trained_model
9. test_2021_backtest_cannot_read_labels_matured_in_2024
10. test_backtest_returns_model_unavailable_as_of_date_when_insufficient_samples
11. test_immature_observation_period_produces_not_matured
12. test_historical_forecasts_protected_from_unannounced_overwrite
13. test_cox_short_duration_cohort_not_misinterpreted_as_years
14. test_missing_enterprise_financials_returns_insufficient_data_without_default_substitution
15. test_database_migration_raises_error_on_failure
"""

import os
import shutil
import tempfile
from datetime import date, datetime
import numpy as np
import pandas as pd
import pytest

from src.core.database import DatabaseManager, db_manager
from src.core.types import (
    DataQualityStatus,
    ModelStage,
    ModelTrainingStatus,
    PredictionStatus,
    GroundTruthOutcome,
    ValidationExecutionType,
    SimulationType,
    ProvenanceMetadata,
    OOSValidationSample,
    OOSValidationDataset
)
from src.diffusion.lgbm_forecaster import LightGBMGrowthForecaster
from src.risk_survival.survival_cox import CompanySurvivalEngine
from src.decision_engine.model_validator import ModelValidationService
from src.payoff_engine.monte_carlo import AsymmetricPayoffEngine
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.backtest.engine import PointInTimeBacktestEngine


@pytest.fixture
def temp_db():
    """提供独立的临时 DuckDB 数据库环境"""
    tmp_dir = tempfile.mkdtemp()
    db_file = os.path.join(tmp_dir, "test_final_trust.duckdb")
    db = DatabaseManager(db_path=db_file)
    db.init_schema()
    yield db
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# 1. 模型自生预测标签禁止晋升生产状态
# -----------------------------------------------------------------------------
def test_model_self_generated_labels_cannot_grant_oos_production_status():
    """模型使用自身预测值或人工合成标签作为 OOS 验证集，验证服务必须拒绝，模型阶段不得晋升为 VERIFIED_PRODUCTION"""
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) * 2.0 + 3.0, 3: np.random.randn(n) * 3.0 + 5.0, 5: np.random.randn(n) * 4.0 + 8.0}
    provenances = [
        ProvenanceMetadata(
            record_id=f"TR_{i}",
            source_name="Audit_Vault",
            as_of_date=date(2020, 1, 1),
            published_at=datetime(2020, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="a" * 64
        )
        for i in range(n)
    ]
    forecaster.fit_verified(features, labels, quality_status=DataQualityStatus.VERIFIED, provenance_records=provenances)
    assert forecaster.model_stage == ModelStage.BENCHMARK

    # 使用模型自身预测值构造标签并标记 is_model_generated=True
    oos_feat = np.random.randn(10, 7)
    oos_self_labels = {
        1: forecaster.models[1].predict(oos_feat),
        3: forecaster.models[3].predict(oos_feat),
        5: forecaster.models[5].predict(oos_feat)
    }
    res = forecaster.validate_out_of_sample(
        oos_features=oos_feat,
        oos_labels_by_horizon=oos_self_labels,
        is_model_generated=True
    )
    assert res["passed"] is False
    assert forecaster.model_stage == ModelStage.BENCHMARK
    assert forecaster.is_out_of_sample_validated is False
    assert "自生" in res["rejection_reason"] or "REJECTED" in res["status"]


# -----------------------------------------------------------------------------
# 2. 验证集与训练集样本重叠导致验证失败
# -----------------------------------------------------------------------------
def test_overlapping_test_and_training_sets_fail_validation():
    """OOS 验证集样本与训练集样本存在 ID 重叠或相同主题相同时间窗口，验证服务拒绝晋升"""
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) + 2.0, 3: np.random.randn(n) + 3.0, 5: np.random.randn(n) + 4.0}
    provenances = [
        ProvenanceMetadata(
            record_id=f"OVERLAP_{i}",
            source_name="Audit_Vault",
            as_of_date=date(2020, 1, 1),
            published_at=datetime(2020, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="b" * 64
        )
        for i in range(n)
    ]
    forecaster.fit_verified(features, labels, quality_status=DataQualityStatus.VERIFIED, provenance_records=provenances)

    # 构造包含重叠 sample_id 的验证样本
    overlap_samples = [
        OOSValidationSample(
            sample_id="OVERLAP_0",  # 与训练集重叠！
            dataset_version="v1.1-oos-overlap",
            feature_snapshot_id="snap_0",
            decision_date=date(2021, 1, 1),
            feature_available_at=datetime(2020, 12, 31),
            label_period_end=date(2022, 1, 1),
            label_observed_at=datetime(2022, 1, 2),
            label_source="Audit_Vault",
            label_quality_status=DataQualityStatus.VERIFIED,
            source_record_id="rec_0",
            data_quality_status=DataQualityStatus.VERIFIED,
            features={forecaster.feature_names[j]: float(features[0][j]) for j in range(7)},
            labels={1: 2.5, 3: 3.5, 5: 4.5}
        )
    ]
    dataset = OOSValidationDataset(
        dataset_id="overlap_ds",
        dataset_version="v1.1-oos",
        audit_status=DataQualityStatus.VERIFIED,
        samples=overlap_samples
    )
    res = forecaster.validate_out_of_sample(validation_dataset=dataset)
    assert res["passed"] is False
    assert forecaster.model_stage == ModelStage.BENCHMARK
    assert "重叠" in res["rejection_reason"]


# -----------------------------------------------------------------------------
# 3. 验证样本标签公布时间晚于评估截止日导致验证失败
# -----------------------------------------------------------------------------
def test_test_label_published_after_validation_cutoff_fails_validation():
    """验证样本标签公开时间晚于评估截止日，验证服务识别为未来信息泄漏，验证失败"""
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) + 2.0, 3: np.random.randn(n) + 3.0, 5: np.random.randn(n) + 4.0}
    provenances = [
        ProvenanceMetadata(
            record_id=f"TR_{i}",
            source_name="Audit_Vault",
            as_of_date=date(2020, 1, 1),
            published_at=datetime(2020, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="c" * 64
        )
        for i in range(n)
    ]
    forecaster.fit_verified(features, labels, quality_status=DataQualityStatus.VERIFIED, provenance_records=provenances)

    # 评估日为 2023-12-31，但标签真实公开时间为 2024-06-01 (未来泄漏)
    eval_date = date(2023, 12, 31)
    future_samples = [
        OOSValidationSample(
            sample_id=f"FUTURE_{i}",
            dataset_version="v1.1-oos-future",
            feature_snapshot_id=f"snap_{i}",
            decision_date=date(2022, 1, 1),
            feature_available_at=datetime(2021, 12, 31),
            label_period_end=date(2023, 1, 1),
            label_observed_at=datetime(2024, 6, 1),  # 晚于 eval_date!
            label_source="Audit_Vault",
            label_quality_status=DataQualityStatus.VERIFIED,
            source_record_id=f"rec_{i}",
            data_quality_status=DataQualityStatus.VERIFIED,
            features={forecaster.feature_names[j]: float(np.random.randn()) for j in range(7)},
            labels={1: 2.0, 3: 3.0, 5: 4.0}
        )
        for i in range(5)
    ]
    dataset = OOSValidationDataset(
        dataset_id="future_ds",
        dataset_version="v1.1-oos",
        audit_status=DataQualityStatus.VERIFIED,
        samples=future_samples
    )
    res = forecaster.validate_out_of_sample(validation_dataset=dataset, evaluation_date=eval_date)
    assert res["passed"] is False
    assert forecaster.model_stage == ModelStage.BENCHMARK
    assert "未来" in res["rejection_reason"] or "前视" in res["rejection_reason"]


# -----------------------------------------------------------------------------
# 4. 缺失血缘记录禁止晋升正式生产模型
# -----------------------------------------------------------------------------
def test_missing_provenance_fails_formal_model_promotion():
    """缺少血缘记录 (provenance) 或质量状态非 VERIFIED 的样本外数据，不能使模型晋升为正式生产状态"""
    engine = CompanySurvivalEngine()
    n = 25
    train_df = pd.DataFrame({
        "duration": np.random.uniform(12.0, 60.0, size=n),
        "event": [1 if i < 8 else 0 for i in range(n)],
        "cash_burn_months": np.random.uniform(6.0, 36.0, size=n),
        "debt_ratio": np.random.uniform(20.0, 80.0, size=n),
        "gross_margin": np.random.uniform(15.0, 50.0, size=n),
        "competitor_count": np.random.uniform(2.0, 20.0, size=n)
    })
    provenances = [
        ProvenanceMetadata(
            record_id=f"COH_{i}",
            source_name="Audited_Vault",
            as_of_date=date(2021, 1, 1),
            published_at=datetime(2021, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="d" * 64
        )
        for i in range(n)
    ]
    engine.fit_verified(train_df, quality_status=DataQualityStatus.VERIFIED, provenance_records=provenances)

    test_df = train_df.copy()
    # 验证时不提供 provenance_records
    res = engine.validate_out_of_sample(test_df, provenance_records=None)
    assert res["passed"] is False
    assert engine.model_stage == ModelStage.BENCHMARK
    assert "未提供经核验" in res["rejection_reason"]


# -----------------------------------------------------------------------------
# 5. 调用方修改宽松阈值无法突破生产硬性门禁上限
# -----------------------------------------------------------------------------
def test_caller_modifying_mae_threshold_cannot_bypass_production_ceiling():
    """调用方传入宽松阈值 (如 MAE=10.0)，系统强制应用生产硬性上限 (MAE <= 2.0)，不达标仍拒绝晋升"""
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) + 2.0, 3: np.random.randn(n) + 3.0, 5: np.random.randn(n) + 4.0}
    provenances = [
        ProvenanceMetadata(
            record_id=f"TR_{i}",
            source_name="Audit_Vault",
            as_of_date=date(2020, 1, 1),
            published_at=datetime(2020, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="e" * 64
        )
        for i in range(n)
    ]
    forecaster.fit_verified(features, labels, quality_status=DataQualityStatus.VERIFIED, provenance_records=provenances)

    # 构造测试集，MAE 约为 3.5 (大于 2.0 但小于 10.0)
    oos_feat = np.random.randn(10, 7)
    oos_labels = {
        1: np.array([forecaster.models[1].predict(oos_feat)[i] + 3.5 for i in range(10)]),
        3: np.array([forecaster.models[3].predict(oos_feat)[i] + 3.5 for i in range(10)]),
        5: np.array([forecaster.models[5].predict(oos_feat)[i] + 3.5 for i in range(10)])
    }
    oos_provs = [
        ProvenanceMetadata(
            record_id=f"OOS_{i}",
            source_name="Audited_Vault",
            as_of_date=date(2021, 1, 1),
            published_at=datetime(2021, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="f" * 64
        )
        for i in range(10)
    ]

    # 调用方传入宽松阈值 10.0
    res = forecaster.validate_out_of_sample(
        oos_features=oos_feat,
        oos_labels_by_horizon=oos_labels,
        max_allowed_mae=10.0,
        provenance_records=oos_provs
    )
    # 系统必须强制生效上限 2.0，进而拒绝晋升
    assert res["threshold"] <= 2.0
    assert res["passed"] is False
    assert forecaster.model_stage == ModelStage.BENCHMARK


# -----------------------------------------------------------------------------
# 6. 实验模型增长倍数不得进入正式 10 倍超额收益概率
# -----------------------------------------------------------------------------
def test_experimental_model_growth_multiple_cannot_enter_formal_10x_payoff_probability(temp_db, monkeypatch):
    """实验模型输出的 growth_multiplier 传入机会池时，正式 prob_10x_super_return_pct 必须为 None"""
    import src.core.database as db_mod
    monkeypatch.setattr(db_mod, "db_manager", temp_db)

    forecaster = LightGBMGrowthForecaster()
    # 拟合实验模型 (EXPERIMENTAL)
    X = np.random.randn(25, 7)
    labels = {1: np.random.uniform(1.2, 2.5, 25), 3: np.random.uniform(2.0, 5.0, 25), 5: np.random.uniform(3.0, 10.0, 25)}
    forecaster.fit_experimental(X, labels)
    assert forecaster.model_stage == ModelStage.EXPERIMENTAL

    pool_mgr = OpportunityPoolManager(lgbm_forecaster=forecaster)
    candidates = pool_mgr.build_opportunity_pool()
    assert len(candidates) > 0

    for c in candidates:
        # 正式 10 倍超额收益概率必须为 None
        assert c.get("prob_10x_super_return_pct") is None
        # 校准概率必须为 None
        assert c.get("calibrated_probability") is None


# -----------------------------------------------------------------------------
# 7. 情景模拟结果明确标注为 HYPOTHETICAL_SCENARIO 并带免责声明
# -----------------------------------------------------------------------------
def test_simulated_payoff_results_marked_as_hypothetical_scenario_with_disclaimer():
    """情景模拟结果必须标记为 HYPOTHETICAL_SCENARIO 并包含清晰免责声明"""
    engine = AsymmetricPayoffEngine()
    res = engine.simulate_payoff(
        base_growth_multiplier=3.5,
        current_valuation_multiple=25.0,
        target_valuation_multiple_range=(20.0, 45.0),
        simulation_type=SimulationType.HYPOTHETICAL_SCENARIO,
        is_verified=False
    )
    assert res["simulation_type"] == SimulationType.HYPOTHETICAL_SCENARIO.value
    assert res["is_verified"] is False
    assert "情景模拟结果仅供宏观与产业弹性敏感性参考" in res["disclaimer"]
    assert res["metrics"]["prob_10x_super_return_pct"] is None
    assert res["metrics"]["scenario_prob_10x_pct"] is not None


# -----------------------------------------------------------------------------
# 8. 2021 年回测禁止使用 2026 年训练好的模型
# -----------------------------------------------------------------------------
def test_2021_backtest_cannot_use_2026_trained_model():
    """2021 年的回测必须使用截至 2021 年可用数据训练的模型，不能使用 2026 年训练好的模型对象"""
    engine = PointInTimeBacktestEngine()
    cutoff_2021 = date(2021, 12, 31)

    hist_forecaster = engine.train_historical_forecaster(cutoff_2021, horizon_years=3)
    assert hist_forecaster.model_version == "hist_20211231"
    assert hist_forecaster.model_version != "v1.1-p0-lgbm"


# -----------------------------------------------------------------------------
# 9. 2021 年回测不能读取 2024 年才成熟的标签
# -----------------------------------------------------------------------------
def test_2021_backtest_cannot_read_labels_matured_in_2024(temp_db, monkeypatch):
    """2021 年时点的训练集构建中，2024 年才成熟的标签必须为 None 或被排除在训练集之外"""
    import src.core.database as db_mod
    import src.backtest.engine as bt_mod
    monkeypatch.setattr(db_mod, "db_manager", temp_db)
    monkeypatch.setattr(bt_mod, "db_manager", temp_db)

    conn = temp_db.get_connection()
    try:
        # 插入样本：2021-12-31 做决策，其 3 年标签成熟期为 2024-12-31
        conn.execute("""
            INSERT INTO tech_training_samples
            (sample_id, topic_id, base_date, published_at, feature_published_at, velocity_1y, acceleration, active_institutions, github_momentum, macro_multiplier, entropy_score, novelty_score, target_1y, target_3y, target_5y, label_3y_period_end, quality_status)
            VALUES ('SAMP_2021_TEST', 'TOPIC_AI', '2021-12-31', '2021-12-31 00:00:00', '2021-12-31 00:00:00', 10, 2, 20, 50, 1.0, 0.5, 0.5, 1.2, 3.8, NULL, '2024-12-31', 'VERIFIED');
        """)
    finally:
        conn.close()

    engine = PointInTimeBacktestEngine()
    df_2021 = engine.get_historical_training_samples(cutoff_date=date(2021, 12, 31), horizon_years=3)
    df_target = df_2021[df_2021["sample_id"] == "SAMP_2021_TEST"]
    assert len(df_target) == 1
    # 截至 2021-12-31，2024-12-31 才成熟的 3 年标签必须为 None (未成熟)
    assert pd.isna(df_target["target_3y"].iloc[0])


# -----------------------------------------------------------------------------
# 10. 历史样本不足时回测标记 MODEL_UNAVAILABLE_AS_OF_DATE
# -----------------------------------------------------------------------------
def test_backtest_returns_model_unavailable_as_of_date_when_insufficient_samples(temp_db, monkeypatch):
    """当历史时点可用真实样本不足以训练模型时，回测记录模型状态为 MODEL_UNAVAILABLE_AS_OF_DATE"""
    import src.core.database as db_mod
    import src.backtest.engine as bt_mod
    import src.decision_engine.opportunity_pool as op_mod
    monkeypatch.setattr(db_mod, "db_manager", temp_db)
    monkeypatch.setattr(bt_mod, "db_manager", temp_db)
    monkeypatch.setattr(op_mod, "db_manager", temp_db)

    # 清空样本库，模拟在无成熟样本的历史极早期时点运行回测
    temp_db.execute("DELETE FROM tech_training_samples;")

    engine = PointInTimeBacktestEngine()
    res = engine.run_walk_forward_backtest(
        cutoff_date=date(2010, 12, 31),
        evaluation_date=date(2013, 12, 31),
        horizon_years=3
    )
    assert res["forecast_status"] == PredictionStatus.MODEL_UNAVAILABLE_AS_OF_DATE.value


# -----------------------------------------------------------------------------
# 11. 未成熟观察期样本产生 NOT_MATURED
# -----------------------------------------------------------------------------
def test_immature_observation_period_produces_not_matured():
    """评估日未达到成熟期或超过审计锚定日的样本，结果状态必须为 NOT_MATURED，不能计算为成功或失败"""
    engine = PointInTimeBacktestEngine()
    # 审查基准锚定日为 2026-11-11，评估日 2028-12-31 尚未发生
    res = engine.run_walk_forward_backtest(
        cutoff_date=date(2024, 12, 31),
        evaluation_date=date(2028, 12, 31),
        horizon_years=3,
        audit_as_of_date=date(2026, 11, 11)
    )
    assert res["is_horizon_matured"] is False
    for cand in res["candidate_outcomes"]:
        assert cand["outcome_status"] == GroundTruthOutcome.NOT_MATURED.value
        assert cand["hit_status"] == GroundTruthOutcome.NOT_MATURED.value


# -----------------------------------------------------------------------------
# 12. 历史预测记录防止未声明静默覆盖
# -----------------------------------------------------------------------------
def test_historical_forecasts_protected_from_unannounced_overwrite(temp_db, monkeypatch):
    """对同一时点同一标的重复生成预测时，必须以版本化增量形式存储，不能通过 INSERT OR REPLACE 静默覆盖原始预测"""
    import src.core.database as db_mod
    import src.decision_engine.opportunity_pool as op_mod
    monkeypatch.setattr(db_mod, "db_manager", temp_db)
    monkeypatch.setattr(op_mod, "db_manager", temp_db)

    pool_mgr = OpportunityPoolManager()
    # 第 1 次构建机会池生成预测
    pool_mgr.build_opportunity_pool(cutoff_date=date(2024, 1, 1))

    df_first = temp_db.query_df("SELECT * FROM model_forecasts WHERE target_id = 'TOPIC_EMBODIED_AI'")
    assert len(df_first) == 1
    first_id = df_first["forecast_id"].iloc[0]

    # 第 2 次重复调用
    pool_mgr.build_opportunity_pool(cutoff_date=date(2024, 1, 1))
    df_second = temp_db.query_df("SELECT * FROM model_forecasts WHERE target_id = 'TOPIC_EMBODIED_AI'")
    assert len(df_second) == 2  # 版本化增量保存，保留历史！
    forecast_ids = list(df_second["forecast_id"])
    assert first_id in forecast_ids
    assert any("_v2" in fid for fid in forecast_ids)


# -----------------------------------------------------------------------------
# 13. Cox 短期队列不被误判为年份单位
# -----------------------------------------------------------------------------
def test_cox_short_duration_cohort_not_misinterpreted_as_years():
    """最大观察期小于等于 12 个月的队列，生存模型必须严格按月度解析，超过观察期的外推返回 None"""
    engine = CompanySurvivalEngine()
    n = 20
    # 构造最长观察期仅为 12 个月的月度短队列
    short_df = pd.DataFrame({
        "duration": np.random.uniform(2.0, 12.0, size=n),
        "event": [1 if i < 6 else 0 for i in range(n)],
        "cash_burn_months": np.random.uniform(3.0, 12.0, size=n),
        "debt_ratio": np.random.uniform(30.0, 70.0, size=n),
        "gross_margin": np.random.uniform(20.0, 40.0, size=n),
        "competitor_count": np.random.uniform(1.0, 5.0, size=n)
    })
    engine.fit_experimental(short_df)

    pred = engine.predict_company_survival({
        "cash_burn_months": 6.0,
        "debt_ratio": 45.0,
        "gross_margin": 30.0,
        "competitor_count": 2.0
    })
    assert pred["time_unit"] == "MONTHS"
    probs = pred["survival_probabilities"]
    # 36个月 (3年) 和 60个月 (5年) 明显超过该队列最大跟踪期 12 个月，右截尾外推必须为 None
    assert probs["surv_3y"] is None
    assert probs["surv_5y"] is None


# -----------------------------------------------------------------------------
# 14. 缺失企业财务数据返回 INSUFFICIENT_DATA 绝不自动填充默认值
# -----------------------------------------------------------------------------
def test_missing_enterprise_financials_returns_insufficient_data_without_default_substitution():
    """企业财务数据缺失关键协变量时，生存预测直接返回 INSUFFICIENT_DATA，不得自动填入 24/45/35 等默认数值"""
    engine = CompanySurvivalEngine()
    engine.fit_from_historical_cohort()

    # 1. 传入空财务字典
    pred_empty = engine.predict_company_survival({})
    assert pred_empty["status"] == PredictionStatus.INSUFFICIENT_DATA.value
    assert pred_empty["risk_tier"] == "DATA_INSUFFICIENT"
    assert pred_empty["survival_probabilities"]["surv_1y"] is None

    # 2. 仅传入非关键指标，缺失 debt_ratio 和 gross_margin
    pred_partial = engine.predict_company_survival({"competitor_count": 10.0})
    assert pred_partial["status"] == PredictionStatus.INSUFFICIENT_DATA.value
    assert pred_partial["risk_tier"] == "DATA_INSUFFICIENT"


# -----------------------------------------------------------------------------
# 15. 数据库迁移失败时抛出异常并记录 schema_migrations
# -----------------------------------------------------------------------------
def test_database_migration_raises_error_on_failure(temp_db):
    """当数据库迁移遇到不可恢复的 SQL 错误时，必须捕获并记录到 schema_migrations 表并抛出异常，而不是静默 pass"""
    with pytest.raises(RuntimeError) as exc_info:
        temp_db.execute_custom_migration_step(
            migration_id="failing_migration_test",
            sql="ALTER TABLE non_existent_table ADD COLUMN test_col VARCHAR;",
            description="Intentionally invalid migration"
        )
    assert "Database migration step 'failing_migration_test' failed" in str(exc_info.value)

    # 校验 schema_migrations 中包含失败记录
    df_mig = temp_db.query_df("SELECT * FROM schema_migrations WHERE migration_id = 'failing_migration_test'")
    assert len(df_mig) == 1
    assert df_mig["status"].iloc[0] == "FAILED"
    assert df_mig["error_message"].iloc[0] is not None
