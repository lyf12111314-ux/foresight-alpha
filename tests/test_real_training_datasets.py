"""P1-01 真实企业违约与细粒度技术训练数据集测试 (P1-01 Real Training Dataset Tests).

验证目标：
1. A股真实企业违约队列样本量 >= 20，违约事件数 >= 5，且具备真实公告出处。
2. 历史前沿技术跨周期时序样本量 >= 20，包含真实的 1y/3y/5y 前瞻增长标签。
3. 企业生存分析 Cox 模型使用真实队列成功完成训练，非合成数据标记，且预测输出真实生存概率。
4. LightGBM 预测器使用真实技术时序成功完成训练，非合成数据标记，且输出真实 1y/3y/5y 成长倍数。
5. OpportunityPoolManager 真实训练管道正常运行，端到端打通真实特征到预测点估计。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from src.core.types import (
    DataQualityStatus,
    ModelTrainingStatus,
    PredictionStatus
)
from src.core.database import db_manager
from src.ingestion.enterprise import EnterpriseCollector
from src.ingestion.tech_dataset import HistoricalTechDatasetCollector
from src.risk_survival.survival_cox import CompanySurvivalEngine
from src.diffusion.lgbm_forecaster import LightGBMGrowthForecaster
from src.decision_engine.opportunity_pool import OpportunityPoolManager


def test_distress_cohort_dataset_integrity():
    """验证真实企业违约与存续队列的数据完整性与准入门槛"""
    collector = EnterpriseCollector()
    cohort = collector.get_historical_distress_cohort()

    # 1. 验证样本总量与事件数达标 (门槛：总样本 >= 20, 违约事件 >= 5)
    assert len(cohort) >= 20
    distress_events = [r for r in cohort if r["is_distressed"] == 1]
    censored_active = [r for r in cohort if r["is_distressed"] == 0]
    assert len(distress_events) >= 5
    assert len(censored_active) >= 10

    # 2. 验证关键协变量字段与官方公告信息完整性
    for r in cohort:
        assert r["duration"] > 0
        assert r["event"] in [0, 1]
        assert r["cash_burn_months"] > 0
        assert r["debt_ratio"] > 0
        assert "gross_margin" in r
        assert r["competitor_count"] > 0
        assert "official_announcement_id" in r
        assert r["provenance"]["quality_status"] == DataQualityStatus.MANUAL_REFERENCE
        assert r["provenance"]["quality_status"] != DataQualityStatus.SYNTHETIC


def test_historical_tech_dataset_integrity():
    """验证历史前沿技术多期特征库与真实标签的数据完整性"""
    collector = HistoricalTechDatasetCollector()
    samples = collector.get_historical_tech_training_samples()

    # 1. 验证真实监督学习样本量达标 (门槛：>= 20 期)
    assert len(samples) >= 20

    # 2. 验证特征与 1y/3y/5y 标签
    feature_keys = [
        "velocity_1y", "acceleration", "active_institutions",
        "github_momentum", "macro_multiplier", "entropy_score", "novelty_score"
    ]
    for s in samples:
        for fk in feature_keys:
            assert fk in s
            assert isinstance(s[fk], (int, float))
        assert s["target_1y"] > 0
        assert s["target_3y"] > 0
        assert s["target_5y"] > 0
        assert s["provenance"]["quality_status"] == DataQualityStatus.MANUAL_REFERENCE
        assert s["provenance"]["quality_status"] != DataQualityStatus.SYNTHETIC


def test_cox_survival_engine_real_training():
    """验证 Cox 生存分析模型基于真实企业队列完成正式训练"""
    engine = CompanySurvivalEngine()
    assert engine.training_status == ModelTrainingStatus.INSUFFICIENT_DATA
    assert engine.is_fitted is False

    # 执行基于真实队列的正式训练
    res = engine.fit_from_historical_cohort()
    assert res["status"] == ModelTrainingStatus.TRAINED.value
    assert engine.is_fitted is True
    assert engine.training_status == ModelTrainingStatus.TRAINED
    assert engine.is_synthetic is False  # 严格确认并非合成数据！
    assert engine.sample_count >= 20
    assert engine.event_count >= 5

    # 预测测试：输入某企业的真实经营指标，应成功输出真实估计值，且不再为 None
    test_metrics = {
        "cash_burn_months": 36.0,
        "debt_ratio": 35.0,
        "gross_margin": 40.0,
        "competitor_count": 6.0
    }
    pred = engine.predict_company_survival(test_metrics)
    assert pred["status"] in [PredictionStatus.PREDICTED.value, "PREDICTED"]
    assert pred["is_verified"] is True
    assert pred["is_synthetic"] is False
    assert pred["partial_hazard_ratio"] is not None
    assert pred["partial_hazard_ratio"] > 0
    assert pred["survival_probabilities"]["surv_1y"] is not None
    assert pred["survival_probabilities"]["surv_3y"] is not None
    assert 0.0 < pred["survival_probabilities"]["surv_3y"] <= 1.0


def test_lgbm_growth_forecaster_real_training():
    """验证 LightGBM 预测器基于真实技术时序完成多期正式训练"""
    forecaster = LightGBMGrowthForecaster()
    assert forecaster.training_status == ModelTrainingStatus.MODEL_NOT_TRAINED

    # 执行基于真实前沿技术样本的正式训练
    res = forecaster.fit_from_historical_tech_samples()
    assert res["status"] == ModelTrainingStatus.TRAINED.value
    assert forecaster.training_status == ModelTrainingStatus.TRAINED
    assert forecaster.is_synthetic is False  # 严格确认并非合成数据！
    assert forecaster.sample_count >= 20
    assert len(forecaster.models) == 3  # 1y, 3y, 5y 三个正式训练的模型

    # 预测测试：输入特征向量，应产生真实非空预测倍数
    features = {
        "velocity_1y": 300.0,
        "acceleration": 80.0,
        "active_institutions": 450.0,
        "github_momentum": 8500.0,
        "macro_multiplier": 1.15,
        "entropy_score": 0.72,
        "novelty_score": 0.85
    }
    pred = forecaster.predict_growth(features)
    assert pred["status"] == PredictionStatus.PREDICTED.value
    assert pred["is_synthetic"] is False
    assert "forecasts" in pred
    for horizon in ["1y", "3y", "5y"]:
        pt_est = pred["forecasts"][horizon]["point_estimate"]
        assert pt_est is not None
        assert pt_est > 0.5  # 存在基于真实规律拟合的有效点估计
        assert pred["forecasts"][horizon]["ci_low"] is not None
        assert pred["forecasts"][horizon]["ci_high"] is not None


def test_opportunity_pool_training_pipeline():
    """验证 OpportunityPoolManager 真实训练管道激活后，各候选机会输出真实预测点估计"""
    pool_mgr = OpportunityPoolManager()

    # 1. 训练前：未训练状态，点估计必须为 None
    pool_untrained = pool_mgr.build_opportunity_pool()
    for cand in pool_untrained:
        assert cand.get("growth_3y_multiple") is None

    # 2. 运行真实历史数据训练管道
    train_res = pool_mgr.train_models_on_historical_data()
    assert train_res["lgbm_growth_forecaster"]["status"] == ModelTrainingStatus.TRAINED.value
    assert train_res["company_survival_engine"]["status"] == ModelTrainingStatus.TRAINED.value

    # 3. 训练后：基于真实训练完成的模型输出有效点估计
    pool_trained = pool_mgr.build_opportunity_pool()
    assert len(pool_trained) > 0
    for cand in pool_trained:
        assert cand.get("growth_3y_multiple") is not None
        assert cand.get("growth_3y_multiple") > 0.5
        assert cand.get("survival_prob_3y") is not None
        assert 0.0 < cand.get("survival_prob_3y") <= 1.0
