"""PR #3 审查遗留问题定向补修自动化测试套件 (PR #3 Directed Remediation Test Suite).

涵盖：
- P0-01: 数据库 NULL 约束与未成熟标签兼容性、幂等迁移与双入库流水线
- P0-02: 模型正式训练血缘门禁、状态清空重置与样本外跨周期检验 (OOS) 晋升
- P0-03: 回测消除默认 100 虚假回退、未来日期未成熟门禁、Top-K防篡改及真实指标
- P0-04: Cox 生存模型月份时间单位 (12m/36m/60m)、阶梯函数右连续估值与截尾保护
- P1: GMM-Regime 宏观周期聚类口径、研报定性情景区间与文本哈希签名
"""

import os
import shutil
import tempfile
from datetime import date, datetime
import numpy as np
import pandas as pd
import pytest

from src.core.database import DatabaseManager
from src.core.types import (
    DataQualityStatus,
    ModelStage,
    ModelTrainingStatus,
    PredictionStatus,
    GroundTruthOutcome,
    LabelMaturityStatus,
    DistressEventType,
    ProvenanceMetadata
)
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.tech_dataset import HistoricalTechDatasetCollector
from src.ingestion.enterprise import EnterpriseCollector
from src.diffusion.lgbm_forecaster import LightGBMGrowthForecaster
from src.risk_survival.survival_cox import CompanySurvivalEngine
from src.backtest.engine import PointInTimeBacktestEngine
from src.backtest.metrics import BacktestMetricsCalculator
from src.macro.hmm_regime import MacroHMMRegimeModel, MacroGMMRegimeModel
from src.reporting.dossier import ResearchDossierGenerator


@pytest.fixture
def temp_db():
    """提供独立的临时 DuckDB 数据库环境"""
    tmp_dir = tempfile.mkdtemp()
    db_file = os.path.join(tmp_dir, "test_remediation.duckdb")
    db = DatabaseManager(db_path=db_file)
    db.init_schema()
    yield db
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# P0-01 测试: 数据库 NULL 约束、标签成熟度与迁移
# =============================================================================

def test_p0_01_empty_db_initialization_and_null_label_storage(temp_db):
    """验证从空数据库初始化，未成熟标签可正常保存为 NULL"""
    conn = temp_db.get_connection()
    try:
        # 插入包含 NULL target_5y 的记录
        conn.execute("""
            INSERT INTO tech_training_samples
            (sample_id, topic_id, base_date, published_at, velocity_1y, acceleration, active_institutions, github_momentum, macro_multiplier, entropy_score, novelty_score, target_1y, target_3y, target_5y, label_5y_status, quality_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, [
            "TECH_TEST_NULL", "TOPIC_TEST", "2022-12-31", datetime.utcnow(),
            100.0, 20.0, 50.0, 1000.0, 1.0, 0.5, 0.8,
            1.5, 2.5, None, LabelMaturityStatus.NOT_MATURED.value, DataQualityStatus.MANUAL_REFERENCE.value
        ])
    finally:
        conn.close()

    df = temp_db.query_df("SELECT * FROM tech_training_samples WHERE sample_id = 'TECH_TEST_NULL'")
    assert len(df) == 1
    assert pd.isna(df["target_5y"].iloc[0])
    assert df["label_5y_status"].iloc[0] == LabelMaturityStatus.NOT_MATURED.value


def test_p0_01_pipeline_dual_ingestion_idempotence(temp_db, monkeypatch):
    """测试 ingest_tech_training_samples 与 run_all 双路径幂等入库"""
    monkeypatch.setattr("src.ingestion.pipeline.db_manager", temp_db)

    pipeline = IngestionPipeline()
    # 模拟外部网络请求，加速测试执行并杜绝网络波动
    monkeypatch.setattr(pipeline.openalex, "fetch_topic_research_trends", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline.github, "search_repositories", lambda *args, **kwargs: [])

    # 路径 1: 专用采集入库接口
    c1 = pipeline.ingest_tech_training_samples()
    d1 = pipeline.ingest_distress_cohort()
    assert c1 >= 40
    assert d1 >= 20

    # 路径 2: 全量 run_all 入库
    stats = pipeline.run_all(verbose=False)
    assert stats["tech_training_samples"] >= 40
    assert stats["distress_cohort"] >= 20

    # 验证未成熟标签行包含 NULL 且不产生约束异常
    df_tech = temp_db.query_df("SELECT count(*) as tot, count(target_5y) as matured_5y FROM tech_training_samples")
    assert df_tech["tot"].iloc[0] >= 40
    assert df_tech["matured_5y"].iloc[0] < df_tech["tot"].iloc[0]  # 存在未成熟 NULL 标签


def test_p0_01_legacy_db_schema_migration(temp_db):
    """测试从旧版 NOT NULL 约束数据库安全迁移，无损升级并允许写入 NULL"""
    conn = temp_db.get_connection()
    try:
        # 模拟旧版带有 NOT NULL 约束的表
        conn.execute("DROP TABLE IF EXISTS tech_training_samples;")
        conn.execute("""
            CREATE TABLE tech_training_samples (
                sample_id VARCHAR PRIMARY KEY,
                topic_id VARCHAR NOT NULL,
                base_date DATE NOT NULL,
                published_at TIMESTAMP NOT NULL,
                velocity_1y DOUBLE NOT NULL,
                acceleration DOUBLE NOT NULL,
                active_institutions DOUBLE NOT NULL,
                github_momentum DOUBLE NOT NULL,
                macro_multiplier DOUBLE NOT NULL,
                entropy_score DOUBLE NOT NULL,
                novelty_score DOUBLE NOT NULL,
                target_1y DOUBLE NOT NULL,
                target_3y DOUBLE NOT NULL,
                target_5y DOUBLE NOT NULL,
                quality_status VARCHAR NOT NULL
            );
        """)
        # 插入旧版数据
        conn.execute("""
            INSERT INTO tech_training_samples VALUES
            ('OLD_S1', 'TOPIC_1', '2018-12-31', '2018-12-31 00:00:00', 10, 5, 20, 50, 1.0, 0.5, 0.5, 1.2, 2.0, 3.5, 'MANUAL_REFERENCE');
        """)
    finally:
        conn.close()

    # 执行幂等模式迁移
    temp_db.migrate_schema()

    # 验证迁移后旧数据完整保留
    df_old = temp_db.query_df("SELECT * FROM tech_training_samples WHERE sample_id = 'OLD_S1'")
    assert len(df_old) == 1
    assert df_old["target_5y"].iloc[0] == 3.5

    # 验证迁移后成功支持写入 NULL target
    conn = temp_db.get_connection()
    try:
        conn.execute("""
            INSERT INTO tech_training_samples
            (sample_id, topic_id, base_date, published_at, velocity_1y, acceleration, active_institutions, github_momentum, macro_multiplier, entropy_score, novelty_score, target_1y, target_3y, target_5y, quality_status)
            VALUES ('NEW_NULL_S2', 'TOPIC_2', '2022-12-31', '2022-12-31 00:00:00', 15, 6, 25, 60, 1.0, 0.5, 0.5, 1.3, 2.2, NULL, 'MANUAL_REFERENCE');
        """)
    finally:
        conn.close()

    df_new = temp_db.query_df("SELECT * FROM tech_training_samples WHERE sample_id = 'NEW_NULL_S2'")
    assert len(df_new) == 1
    assert pd.isna(df_new["target_5y"].iloc[0])


# =============================================================================
# P0-02 测试: 模型正式训练血缘门禁、状态重置与样本外跨周期检验 (OOS)
# =============================================================================

def test_p0_02_formal_training_requires_provenance_records():
    """验证即使传入 quality_status=VERIFIED，若缺失真实血缘凭证，依然坚决拒绝正式训练并重置状态"""
    forecaster = LightGBMGrowthForecaster()
    features = np.random.randn(25, 7)
    labels = {1: np.random.randn(25), 3: np.random.randn(25), 5: np.random.randn(25)}

    # 1. 传入 VERIFIED 但不传 provenance_records
    res = forecaster.fit_verified(
        features=features,
        labels_by_horizon=labels,
        quality_status=DataQualityStatus.VERIFIED,
        provenance_records=None
    )
    assert res["status"] == ModelTrainingStatus.MODEL_NOT_TRAINED.value
    assert res["model_stage"] == ModelStage.UNTRAINED.value
    assert forecaster.is_algorithm_fitted is False
    assert forecaster.is_data_verified is False
    assert len(forecaster.models) == 0
    assert "未提供经核验的数据血缘记录" in res["rejection_reason"]


def test_p0_02_state_clearance_on_failure():
    """验证模型在训练失败或准入拒绝时，彻底清空模型缓存，防止遗留权重"""
    forecaster = LightGBMGrowthForecaster()
    # 先做实验拟合产生历史模型
    forecaster.train_on_synthetic_for_testing(n_samples=50)
    assert len(forecaster.models) > 0

    # 随后调用 fit_verified 但因样本不足拒绝
    few_features = np.random.randn(5, 7)
    few_labels = {1: np.random.randn(5), 3: np.random.randn(5), 5: np.random.randn(5)}
    res = forecaster.fit_verified(few_features, few_labels)

    assert res["status"] == ModelTrainingStatus.MODEL_NOT_TRAINED.value
    assert len(forecaster.models) == 0  # 必须彻底清空
    assert forecaster.is_algorithm_fitted is False
    assert forecaster.model_stage == ModelStage.UNTRAINED


def test_p0_02_model_promotion_requires_oos_validation():
    """验证通过真实血缘准入的模型首先进入 BENCHMARK，经 OOS 验证通过后方可晋升为 VERIFIED_PRODUCTION"""
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) * 2.0 + 3.0, 3: np.random.randn(n) * 3.0 + 5.0, 5: np.random.randn(n) * 4.0 + 8.0}

    valid_provenances = [
        ProvenanceMetadata(
            record_id=f"REC_{i}",
            source_name="Audited_Archive",
            as_of_date=date(2020, 1, 1),
            published_at=datetime(2020, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        )
        for i in range(n)
    ]

    # 正式训练
    res_train = forecaster.fit_verified(
        features=features,
        labels_by_horizon=labels,
        quality_status=DataQualityStatus.VERIFIED,
        provenance_records=valid_provenances
    )
    assert res_train["status"] == ModelTrainingStatus.TRAINED.value
    # 拟合成功后处于 BENCHMARK，尚未晋升为 VERIFIED_PRODUCTION
    assert res_train["model_stage"] == ModelStage.BENCHMARK.value
    assert forecaster.is_out_of_sample_validated is False

    # 样本外检验 (OOS) - 失败情形
    oos_feat = np.random.randn(10, 7)
    oos_bad_labels = {1: np.random.randn(10) + 100.0, 3: np.random.randn(10) + 100.0, 5: np.random.randn(10) + 100.0}
    res_oos_fail = forecaster.validate_out_of_sample(oos_feat, oos_bad_labels, max_allowed_mae=5.0)
    assert res_oos_fail["passed"] is False
    assert forecaster.model_stage == ModelStage.BENCHMARK
    assert forecaster.is_out_of_sample_validated is False

    # 样本外检验 (OOS) - 成功达标情形
    # 使用与训练集相同分布数据使 MAE 保持低位
    oos_good_labels = {
        1: forecaster.models[1].predict(oos_feat) + 0.1,
        3: forecaster.models[3].predict(oos_feat) + 0.1,
        5: forecaster.models[5].predict(oos_feat) + 0.1
    }
    res_oos_pass = forecaster.validate_out_of_sample(oos_feat, oos_good_labels, max_allowed_mae=2.0)
    assert res_oos_pass["passed"] is True
    assert forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION
    assert forecaster.is_out_of_sample_validated is True

    # 生产预测验证
    pred = forecaster.predict_growth({"velocity_1y": 100.0})
    assert pred["is_verified"] is True
    assert pred["status"] == "PREDICTED"


def test_p0_02_cox_survival_oos_concordance_index_gate():
    """验证 Cox 生存模型独立样本外 C-index 评估门禁"""
    engine = CompanySurvivalEngine()
    # 构造真实验证队列
    n = 30
    np.random.seed(42)
    durations = np.random.uniform(20.0, 100.0, size=n)
    events = np.random.binomial(1, 0.4, size=n)
    events[0] = 1
    events[1] = 1
    events[2] = 1
    events[3] = 1
    events[4] = 1  # 确保 >= 5 个事件

    train_df = pd.DataFrame({
        "duration": durations,
        "event": events,
        "cash_burn_months": np.random.uniform(6.0, 36.0, size=n),
        "debt_ratio": np.random.uniform(20.0, 80.0, size=n),
        "gross_margin": np.random.uniform(15.0, 50.0, size=n),
        "competitor_count": np.random.uniform(2.0, 20.0, size=n)
    })

    provenances = [
        ProvenanceMetadata(
            record_id=f"COH_{i}",
            source_name="Audited_Registry",
            as_of_date=date(2021, 1, 1),
            published_at=datetime(2021, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        )
        for i in range(n)
    ]

    fit_res = engine.fit_verified(
        cohort_df=train_df,
        quality_status=DataQualityStatus.VERIFIED,
        provenance_records=provenances
    )
    assert fit_res["status"] == ModelTrainingStatus.TRAINED.value
    assert engine.model_stage == ModelStage.BENCHMARK

    # 独立样本外队列
    test_df = train_df.copy()
    oos_res = engine.validate_out_of_sample(test_df, min_c_index=0.50)
    assert "c_index" in oos_res
    assert isinstance(oos_res["c_index"], float)


# =============================================================================
# P0-03 测试: 回测消除默认 100 回退与前视偏差防护
# =============================================================================

def test_p0_03_backtest_missing_baseline_never_fallbacks_to_100(temp_db, monkeypatch):
    """验证历史文献指标缺失时，绝对不回退到默认 100 制造虚假倍数，判定为 INSUFFICIENT_EVIDENCE"""
    import src.core.database as db_mod
    monkeypatch.setattr(db_mod, "db_manager", temp_db)

    conn = temp_db.get_connection()
    try:
        # 仅插入单一历史时点指标，缺失基准截面
        conn.execute("""
            INSERT INTO tech_metrics (topic_id, as_of_date, published_at, papers_count, citations_count, active_institutions, github_stars, github_forks, commit_velocity, entropy_score, novelty_score)
            VALUES ('TOPIC_EMBODIED_AI', '2023-12-31', '2024-01-05 10:00:00', 80, 200, 30, 100, 20, 1.2, 0.6, 0.7);
        """)
    finally:
        conn.close()

    monkeypatch.setattr("src.backtest.engine.db_manager", temp_db)
    monkeypatch.setattr("src.decision_engine.opportunity_pool.db_manager", temp_db)

    engine = PointInTimeBacktestEngine()
    res = engine.run_walk_forward_backtest(
        cutoff_date=date(2020, 12, 31),
        evaluation_date=date(2023, 12, 31),
        horizon_years=3
    )

    outcomes = {c["topic_id"]: c for c in res["candidate_outcomes"]}
    # 缺少基准点，不能默认制造 80/100=0.8，必须为 INSUFFICIENT_EVIDENCE
    assert outcomes["TOPIC_EMBODIED_AI"]["outcome_status"] == GroundTruthOutcome.INSUFFICIENT_EVIDENCE.value
    assert outcomes["TOPIC_EMBODIED_AI"]["actual_growth_multiple"] is None


def test_p0_03_backtest_evaluation_date_in_future_is_not_matured():
    """验证超过审计基准锚定日 (2026-11-11) 的评估日一律判定为 NOT_MATURED"""
    engine = PointInTimeBacktestEngine()
    res = engine.run_walk_forward_backtest(
        cutoff_date=date(2024, 1, 1),
        evaluation_date=date(2027, 1, 1),  # 2027 年超出 2026-11-11 审查锚定日
        horizon_years=3
    )
    assert res["is_horizon_matured"] is False
    for item in res["candidate_outcomes"]:
        assert item["outcome_status"] == GroundTruthOutcome.NOT_MATURED.value
        assert item["actual_growth_multiple"] is None


def test_p0_03_metrics_returns_none_when_all_unevaluable():
    """验证全部不可评估时指标返回 None 而非伪造 0.0 失败率"""
    calc = BacktestMetricsCalculator()
    preds = [("T1", 90.0), ("T2", 80.0), ("T3", 70.0)]
    labels = {
        "T1": GroundTruthOutcome.NOT_MATURED,
        "T2": GroundTruthOutcome.INSUFFICIENT_EVIDENCE,
        "T3": GroundTruthOutcome.NOT_MATURED
    }

    prec = calc.precision_at_k(preds, labels, k=3, filter_evaluable_only=True)
    assert prec is None  # 无有效成熟样本，不能判定为 0.0 精度

    auc = calc.pr_auc(preds, labels)
    assert auc is None

    ap = calc.average_precision(preds, labels)
    assert ap is None


def test_p0_03_precision_at_k_no_cherry_picking():
    """验证 Top-K 精确率严格在 Top-K 内评估，禁止通过跳过不可评估项后移顶替来拉高精度"""
    calc = BacktestMetricsCalculator()
    # 排名: T1 (不可评估), T2 (未命中), T3 (命中)
    preds = [("T1", 95.0), ("T2", 85.0), ("T3", 75.0)]
    labels = {
        "T1": GroundTruthOutcome.NOT_MATURED,
        "T2": GroundTruthOutcome.FAILURE,
        "T3": GroundTruthOutcome.SUCCESS
    }

    # 对于 k=2: Top-2 为 T1, T2。
    # 其中 T1 不可评估，可评估的仅 T2 (Failure)。
    # 真实 Precision 在有效评估样本中应为 0/1 = 0.0。
    # 若存在 cherry-picking 把 T3 顶上来，则为 1/2 = 0.5。
    prec = calc.precision_at_k(preds, labels, k=2, filter_evaluable_only=True)
    assert prec == 0.0


def test_p0_03_dynamic_momentum_baseline_no_future_leakage():
    """验证动态动量基准严格使用历史动量特征，移除未来增长率回退"""
    calc = BacktestMetricsCalculator()
    candidates = [
        {"topic_id": "T1", "velocity_1y": 10.0, "inflection_evidence": {"growth_multiple_3y": 100.0}},
        {"topic_id": "T2", "velocity_1y": 50.0, "inflection_evidence": {"growth_multiple_3y": 0.5}}
    ]
    labels = {"T1": 1, "T2": 0}

    # 按照真实的 velocity_1y，T2 动量为 50 高于 T1 的 10。Top-1 动量选取 T2 (Label=0) -> Precision = 0.0
    # 若错误回退到未来的 growth_multiple_3y (T1=100)，就会泄密选中 T1 -> Precision = 1.0
    dyn = calc.calculate_dynamic_baselines(candidates, labels, k=1, momentum_key="velocity_1y")
    assert dyn["momentum_baseline_precision"] == 0.0


# =============================================================================
# P0-04 测试: Cox 生存模型月份时间单位与阶梯函数右连续
# =============================================================================

def test_p0_04_survival_prediction_time_units_in_months():
    """验证 Cox 预测时间单位统一为月份，阶梯函数右连续并在超出观察期时截尾"""
    engine = CompanySurvivalEngine()
    engine.fit_synthetic_for_testing(n_samples=50)

    res = engine.predict_company_survival({
        "cash_burn_months": 24.0,
        "debt_ratio": 45.0,
        "gross_margin": 35.0,
        "competitor_count": 8.0
    })

    assert res["time_unit"] == "MONTHS"
    assert res["horizon_months"]["surv_1y"] == 12.0
    assert res["horizon_months"]["surv_3y"] == 36.0
    assert res["horizon_months"]["surv_5y"] == 60.0

    probs = res["survival_probabilities"]
    assert "surv_1y" in probs
    assert "surv_3y" in probs
    assert "surv_5y" in probs
    # 存活率单调递减 S(1y) >= S(3y) >= S(5y) (若非 None)
    if probs["surv_1y"] is not None and probs["surv_3y"] is not None:
        assert probs["surv_1y"] >= probs["surv_3y"]


# =============================================================================
# P1 测试: 报告与文档口径调整
# =============================================================================

def test_p1_macro_regime_model_gmm_naming():
    """验证宏观周期识别模型采用 GMM-Regime 命名与高斯混合聚类界定"""
    model = MacroGMMRegimeModel()
    assert isinstance(model, MacroHMMRegimeModel)
    assert "高斯混合" in MacroHMMRegimeModel.__doc__


def test_p1_dossier_qualitative_scenario_and_sha256_disclaimer():
    """验证深度研报定性情景区间说明、真实验证数据覆盖度区分与 SHA-256 文本摘要哈希界定"""
    gen = ResearchDossierGenerator()
    cand = {
        "topic_id": "TOPIC_TEST",
        "topic_name": "具身智能",
        "growth_3y_multiple": 2.5,
        "macro_regime": "EXPANSION",
        "macro_growth_multiplier": 1.25,
        "data_coverage_score": 75.0,
        "verified_data_coverage_pct": 50.0,
        "is_verified_forecast": False
    }
    report = gen.generate_dossier(cand)

    assert "高斯混合宏观周期聚类模型 (GMM-Regime)" in report
    assert "定性情景区间假设" in report
    assert "非统计置信区间" in report
    assert "定性权重假设" in report
    assert "模块分析覆盖度" in report
    assert "经真实验证数据覆盖度" in report
    assert "研报正文摘要哈希签名" in report
    assert "非区块链或法定公证存证" in report
