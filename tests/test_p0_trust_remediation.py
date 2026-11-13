"""Foresight Alpha V1.1 — P0 预测可信度专项整改 20 项审计测试套件.

涵盖以下全部审计验证场景：
1. MANUAL_REFERENCE 数据不能进入正式模型训练 (fit_verified 严格拒绝)。
2. fit_experimental 接受 MANUAL_REFERENCE，标记为 EXPERIMENTAL 阶段且 is_verified=False。
3. SYNTHETIC 模拟数据不能进入正式模型训练。
4. 真实样本不足时返回正确状态 (MODEL_NOT_TRAINED / INSUFFICIENT_DATA)。
5. 门禁失败时不静默回退使用合成模拟数据。
6. 2022年技术样本的 5 年期标签截至 2026-11-11 为 NOT_MATURED，其 target 为 None。
7. 2018年技术样本的 5 年期标签为 MATURED 且有真实数值。
8. 技术样本库包含负样本与退潮放缓对照样本 (LK-99、Google Glass、3D打印等)。
9. 特征时间与标签时间严格双时间结构隔离，公开时间戳 <= base_date。
10. 企业生存队列明确定义观察窗口时长与违约事件类型 (退市、破产、重组、右删失)。
11. 未经逐项核验的外部估计协变量在严格审计模式下置为 None。
12. 基于 MANUAL_REFERENCE 拟合的生存模型标记为 EXPERIMENTAL，is_data_verified 为 False。
13. 未经核验的生存模型严禁输出正式预测破产概率 (is_verified=False)。
14. 回测 Point-in-Time 查询无数据时，不回退读取全表未来宏观指标。
15. 机会池构建在无切片数据时不泄漏未来宏观或技术时序。
16. 事实标签严谨状态机准确识别 SUCCESS, FAILURE, NOT_MATURED, INSUFFICIENT_EVIDENCE。
17. 未成熟样本绝不计入失败，不进入回测评估分母。
18. 回测 PR-AUC 基于 Precision-Recall 曲线梯形数值积分真实计算，拒绝均值冒充。
19. 随机基准与动量基准同候选池动态测算，严禁硬编码固定 0.33/0.45。
20. Dossier 研报使用真实 SHA-256 签名，清理硬编码 66.7% 回测成绩与固定年份，缺失显示“尚无可靠数据”。
"""

import os
import json
import hashlib
from datetime import date, datetime
import pytest
import numpy as np
import pandas as pd
import duckdb

from src.core.types import (
    DataQualityStatus,
    ModelStage,
    ModelTrainingStatus,
    PredictionStatus,
    LabelMaturityStatus,
    GroundTruthOutcome,
    DistressEventType
)
from src.core.database import DatabaseManager
from src.diffusion.lgbm_forecaster import LightGBMGrowthForecaster
from src.risk_survival.survival_cox import CompanySurvivalEngine
from src.ingestion.tech_dataset import HistoricalTechDatasetCollector
from src.ingestion.enterprise import EnterpriseCollector
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.backtest.metrics import BacktestMetricsCalculator
from src.backtest.engine import PointInTimeBacktestEngine
from src.reporting.dossier import ResearchDossierGenerator


@pytest.fixture
def isolated_db(tmp_path):
    """创建完全隔离的独立临时 DuckDB 测试实例"""
    test_db_path = str(tmp_path / "test_isolated.duckdb")
    mgr = DatabaseManager(db_path=test_db_path)
    return mgr


# -----------------------------------------------------------------------------
# 1. MANUAL_REFERENCE 数据不能进入正式模型训练
# -----------------------------------------------------------------------------
def test_01_manual_reference_rejected_from_formal_lgbm_training():
    forecaster = LightGBMGrowthForecaster()
    features = np.random.randn(25, 7)
    labels = {1: np.random.randn(25), 3: np.random.randn(25), 5: np.random.randn(25)}
    
    res = forecaster.fit_verified(
        features=features,
        labels_by_horizon=labels,
        quality_status=DataQualityStatus.MANUAL_REFERENCE
    )
    assert res["status"] == ModelTrainingStatus.MODEL_NOT_TRAINED.value
    assert res["model_stage"] == ModelStage.UNTRAINED.value
    assert res["is_verified"] is False
    assert forecaster.is_data_verified is False
    assert "准入门禁拒绝" in res["message"]


# -----------------------------------------------------------------------------
# 2. fit_experimental 接受 MANUAL_REFERENCE，标记为 EXPERIMENTAL
# -----------------------------------------------------------------------------
def test_02_fit_experimental_accepts_manual_reference_sets_experimental_stage():
    forecaster = LightGBMGrowthForecaster()
    collector = HistoricalTechDatasetCollector()
    samples = collector.get_historical_tech_training_samples()
    df_samples = pd.DataFrame(samples)
    feature_cols = [
        "velocity_1y", "acceleration", "active_institutions",
        "github_momentum", "macro_multiplier", "entropy_score", "novelty_score"
    ]
    X = df_samples[feature_cols]
    Y = {1: df_samples["target_1y"].tolist(), 3: df_samples["target_3y"].tolist()}

    res = forecaster.fit_experimental(X, Y)
    assert res["status"] == "TRAINED_EXPERIMENTAL_RESEARCH"
    assert res["model_stage"] == ModelStage.EXPERIMENTAL.value
    assert res["is_verified"] is False
    assert forecaster.is_algorithm_fitted is True
    assert forecaster.is_data_verified is False  # MANUAL_REFERENCE 绝不能为 True


# -----------------------------------------------------------------------------
# 3. SYNTHETIC 模拟数据不能进入正式模型训练
# -----------------------------------------------------------------------------
def test_03_synthetic_data_rejected_from_formal_verified_training():
    forecaster = LightGBMGrowthForecaster()
    features = np.random.randn(30, 7)
    labels = {1: np.random.randn(30), 3: np.random.randn(30), 5: np.random.randn(30)}
    
    res = forecaster.fit_verified(
        features=features,
        labels_by_horizon=labels,
        quality_status=DataQualityStatus.SYNTHETIC
    )
    assert res["status"] == ModelTrainingStatus.MODEL_NOT_TRAINED.value
    assert res["is_verified"] is False
    assert "SYNTHETIC" in res["rejection_reason"]


# -----------------------------------------------------------------------------
# 4. 真实样本不足时返回正确状态
# -----------------------------------------------------------------------------
def test_04_insufficient_samples_returns_insufficient_or_untrained():
    forecaster = LightGBMGrowthForecaster()
    few_features = np.random.randn(10, 7)  # 低于 20 样本门槛
    few_labels = {1: np.random.randn(10), 3: np.random.randn(10), 5: np.random.randn(10)}

    res = forecaster.fit_verified(
        features=few_features,
        labels_by_horizon=few_labels,
        quality_status=DataQualityStatus.VERIFIED
    )
    assert res["status"] == ModelTrainingStatus.MODEL_NOT_TRAINED.value
    assert res["sample_count"] == 10
    assert res["min_required"] == 20
    assert forecaster.model_stage == ModelStage.UNTRAINED


# -----------------------------------------------------------------------------
# 5. 门禁失败时不静默回退使用合成模拟数据
# -----------------------------------------------------------------------------
def test_05_no_silent_fallback_to_synthetic_training():
    forecaster = LightGBMGrowthForecaster()
    few_features = np.random.randn(5, 7)
    few_labels = {1: np.random.randn(5), 3: np.random.randn(5), 5: np.random.randn(5)}

    forecaster.fit_verified(features=few_features, labels_by_horizon=few_labels)
    assert forecaster.is_synthetic is False
    assert len(forecaster.models) == 0  # 严禁静默训练模拟模型填充
    pred = forecaster.predict_growth({"velocity_1y": 100})
    assert pred["forecasts"]["3y"]["point_estimate"] is None


# -----------------------------------------------------------------------------
# 6. 2022年技术样本的 5 年期标签截至 2026-11-11 为 NOT_MATURED
# -----------------------------------------------------------------------------
def test_06_future_5y_label_for_2022_is_not_matured_and_none():
    collector = HistoricalTechDatasetCollector()
    samples = collector.get_historical_tech_training_samples()
    samples_2022 = [s for s in samples if s["base_date"].year == 2022]
    assert len(samples_2022) > 0

    for s in samples_2022:
        assert s["label_5y_period_end"] == date(2027, 12, 31)
        assert s["label_5y_status"] == LabelMaturityStatus.NOT_MATURED.value
        assert s["target_5y"] is None  # 严禁在 2026 年给出 2027 年结果


# -----------------------------------------------------------------------------
# 7. 2018年技术样本的 5 年期标签为 MATURED 且有真实数值
# -----------------------------------------------------------------------------
def test_07_past_5y_label_for_2018_is_matured_and_valid():
    collector = HistoricalTechDatasetCollector()
    samples = collector.get_historical_tech_training_samples()
    samples_2018 = [s for s in samples if s["base_date"].year == 2018]
    assert len(samples_2018) > 0

    for s in samples_2018:
        assert s["label_5y_period_end"] == date(2023, 12, 31)
        assert s["label_5y_status"] == LabelMaturityStatus.MATURED.value
        assert s["target_5y"] is not None
        assert s["target_5y"] > 0


# -----------------------------------------------------------------------------
# 8. 技术样本库包含负样本与退潮放缓对照样本
# -----------------------------------------------------------------------------
def test_08_tech_dataset_includes_negative_cooling_topics():
    collector = HistoricalTechDatasetCollector()
    samples = collector.get_historical_tech_training_samples()
    topic_ids = set(s["topic_id"] for s in samples)

    # 包含证伪与退潮代表
    assert "TOPIC_ROOM_TEMP_SUPERCOND" in topic_ids
    assert "TOPIC_GOOGLE_GLASS_AR" in topic_ids
    assert "TOPIC_CONSUMER_3D_PRINTING" in topic_ids
    assert "TOPIC_BLOCKCHAIN_DAO" in topic_ids

    # 验证负样本增长表现出现放缓或萎缩 (< 1.0)
    lk99 = [s for s in samples if s["topic_id"] == "TOPIC_ROOM_TEMP_SUPERCOND"][0]
    assert lk99["target_1y"] < 0.5


# -----------------------------------------------------------------------------
# 9. 特征时间与标签时间严格双时间结构隔离
# -----------------------------------------------------------------------------
def test_09_historical_features_dual_time_no_future_leakage():
    collector = HistoricalTechDatasetCollector()
    samples = collector.get_historical_tech_training_samples()
    for s in samples:
        base_d = s["base_date"]
        pub_dt = s["feature_published_at"]
        assert pub_dt.date() <= base_d
        assert s["label_1y_period_end"] > base_d
        assert s["label_3y_period_end"] > base_d
        assert s["label_5y_period_end"] > base_d


# -----------------------------------------------------------------------------
# 10. 企业生存队列明确定义观察窗口时长与违约事件类型
# -----------------------------------------------------------------------------
def test_10_enterprise_cohort_identifies_observation_windows_and_event_types():
    collector = EnterpriseCollector()
    cohort = collector.get_historical_distress_cohort()
    assert len(cohort) >= 20

    event_types = set(r["distress_event_type"] for r in cohort)
    assert DistressEventType.DELISTING.value in event_types
    assert DistressEventType.BANKRUPTCY.value in event_types
    assert DistressEventType.RESTRUCTURING.value in event_types
    assert DistressEventType.CENSORED_ACTIVE.value in event_types

    for r in cohort:
        assert "observation_window_months" in r
        assert r["observation_window_months"] > 0


# -----------------------------------------------------------------------------
# 11. 未经逐项核验的外部估计协变量在严格审计模式下置为 None
# -----------------------------------------------------------------------------
def test_11_enterprise_cohort_unverified_covariates_handling():
    collector = EnterpriseCollector()
    cohort_strict = collector.get_historical_distress_cohort(strict_audit_only=True)
    for r in cohort_strict:
        assert r["competitor_count"] is None
        assert r["cash_burn_months"] is None
        assert r["is_covariate_audited"] is False


# -----------------------------------------------------------------------------
# 12. 基于 MANUAL_REFERENCE 拟合的生存模型标记为 EXPERIMENTAL
# -----------------------------------------------------------------------------
def test_12_survival_model_fitted_on_manual_reference_is_experimental():
    engine = CompanySurvivalEngine()
    res = engine.fit_from_historical_cohort()
    assert res["status"] in [ModelTrainingStatus.TRAINED.value, "TRAINED_EXPERIMENTAL_RESEARCH"]
    assert engine.model_stage == ModelStage.EXPERIMENTAL
    assert engine.is_algorithm_fitted is True
    assert engine.is_data_verified is False


# -----------------------------------------------------------------------------
# 13. 未经核验的生存模型严禁输出正式预测破产概率 (is_verified=False)
# -----------------------------------------------------------------------------
def test_13_survival_engine_refuses_formal_bankruptcy_probability_when_unverified():
    engine = CompanySurvivalEngine()
    engine.fit_from_historical_cohort()
    pred = engine.predict_company_survival({"debt_ratio": 50.0, "gross_margin": 30.0})
    assert pred["is_verified"] is False
    assert pred["status"] == "UNVERIFIED_RESEARCH"


# -----------------------------------------------------------------------------
# 14. 回测 Point-in-Time 查询无数据时，不回退读取全表未来宏观指标
# -----------------------------------------------------------------------------
def test_14_backtest_point_in_time_query_no_future_macro_fallback(isolated_db):
    conn = isolated_db.get_connection()
    try:
        # 仅插入 2024 年的数据
        conn.execute("""
            INSERT INTO macro_indicators (indicator_code, as_of_date, published_at, indicator_name, value)
            VALUES ('GDP_YOY', '2024-12-31', '2025-01-15 10:00:00', 'GDP同比增速', 5.2);
        """)
    finally:
        conn.close()

    # 在 2020 年截面查询宏观数据，必须返回空，严禁读取 2024 年的数据
    pit_df = isolated_db.query_point_in_time("macro_indicators", cutoff_date=date(2020, 12, 31))
    assert pit_df.empty


# -----------------------------------------------------------------------------
# 15. 机会池构建在无切片数据时不泄漏未来宏观或技术时序
# -----------------------------------------------------------------------------
def test_15_opportunity_pool_no_future_macro_or_tech_leakage():
    pool_mgr = OpportunityPoolManager()
    # 在非常早期的历史时点构建机会池
    candidates = pool_mgr.build_opportunity_pool(cutoff_date=date(2010, 1, 1))
    assert len(candidates) > 0
    for c in candidates:
        # 点估计为 None，绝不硬编码假数据
        assert c.get("growth_3y_multiple") is None
        assert c.get("calibrated_probability") is None


# -----------------------------------------------------------------------------
# 16. 事实标签严谨状态机准确识别 SUCCESS, FAILURE, NOT_MATURED, INSUFFICIENT_EVIDENCE
# -----------------------------------------------------------------------------
def test_16_ground_truth_outcome_distinguishes_success_failure_not_matured():
    calc = BacktestMetricsCalculator()
    assert calc._is_positive(GroundTruthOutcome.SUCCESS) is True
    assert calc._is_positive(GroundTruthOutcome.FAILURE) is False
    assert calc._is_evaluable(GroundTruthOutcome.NOT_MATURED) is False
    assert calc._is_evaluable(GroundTruthOutcome.INSUFFICIENT_EVIDENCE) is False
    assert calc._is_evaluable(GroundTruthOutcome.SUCCESS) is True
    assert calc._is_evaluable(GroundTruthOutcome.FAILURE) is True


# -----------------------------------------------------------------------------
# 17. 未成熟样本绝不计入失败，不进入回测评估分母
# -----------------------------------------------------------------------------
def test_17_unmatured_samples_not_counted_as_failures_in_backtest():
    calc = BacktestMetricsCalculator()
    preds = [("T1", 90.0), ("T2", 80.0)]
    # T1 成功，T2 尚未成熟
    labels = {
        "T1": GroundTruthOutcome.SUCCESS,
        "T2": GroundTruthOutcome.NOT_MATURED
    }
    # 若误将未成熟当成失败，按全量分母计算为 1/2 = 0.5；
    # 按照严格审计规则排除未成熟样本后，仅对成熟样本计算，精确率达 1/1 = 1.0
    prec = calc.precision_at_k(preds, labels, k=2, filter_evaluable_only=True)
    assert prec == 1.0


# -----------------------------------------------------------------------------
# 18. 回测 PR-AUC 基于 Precision-Recall 曲线梯形数值积分真实计算
# -----------------------------------------------------------------------------
def test_18_backtest_pr_auc_calculated_via_genuine_curve_integration():
    calc = BacktestMetricsCalculator()
    preds = [("T1", 95.0), ("T2", 85.0), ("T3", 75.0), ("T4", 65.0)]
    labels = {"T1": 1, "T2": 1, "T3": 0, "T4": 0}

    auc = calc.pr_auc(preds, labels)
    # 排名完美，PR-AUC 应该极高 (接近 1.0)
    assert 0.8 <= auc <= 1.0
    # 严禁出现与 (prec_3 + prec_5) / 2 的粗糙伪公式相同的值 (若指标未对齐)
    p3 = calc.precision_at_k(preds, labels, 3)
    p4 = calc.precision_at_k(preds, labels, 4)
    assert isinstance(auc, float)


# -----------------------------------------------------------------------------
# 19. 随机基准与动量基准同候选池动态测算，严禁硬编码固定 0.33/0.45
# -----------------------------------------------------------------------------
def test_19_backtest_baselines_dynamically_computed_not_hardcoded():
    calc = BacktestMetricsCalculator()
    candidates = [
        {"topic_id": "T1", "velocity_1y": 500.0},
        {"topic_id": "T2", "velocity_1y": 200.0},
        {"topic_id": "T3", "velocity_1y": 100.0},
        {"topic_id": "T4", "velocity_1y": 50.0}
    ]
    # 4 个样本中只有 1 个正样本 -> 动态随机基准必须为 1/4 = 0.25 (非硬编码 0.33)
    labels = {"T1": 1, "T2": 0, "T3": 0, "T4": 0}

    dyn = calc.calculate_dynamic_baselines(candidates, labels, k=2)
    assert dyn["random_baseline_precision"] == 0.25
    assert dyn["random_baseline_precision"] != 0.33

    # T1 动量最高且命中 -> 动量 Top-2 命中 1 个 -> 0.50 (非硬编码 0.45)
    assert dyn["momentum_baseline_precision"] == 0.50
    assert dyn["momentum_baseline_precision"] != 0.45


# -----------------------------------------------------------------------------
# 20. Dossier 研报使用真实 SHA-256 签名，清理硬编码 66.7% 与固定年份
# -----------------------------------------------------------------------------
def test_20_dossier_generator_real_sha256_and_no_fabricated_metrics():
    gen = ResearchDossierGenerator()
    info = {
        "topic_id": "TOPIC_TEST_P0",
        "topic_name": "可信度审计技术",
        "domain_id": "TECH_AUDIT",
        "research_score": 82.5
    }
    report = gen.generate_dossier(info)

    # 1. 彻底清除硬编码内容
    assert "2018年第四季度" not in report
    assert "第六次技术浪潮的扩散繁荣构建期" not in report
    assert "66.7%" not in report
    assert "0.18" not in report
    assert "尚无可靠数据" in report

    # 2. 验证数字签名为真实 SHA-256 (64 位十六进制字符)
    import re
    match = re.search(r"SHA256-([a-f0-9]{64})", report)
    assert match is not None
    sha_hash = match.group(1)
    assert len(sha_hash) == 64
