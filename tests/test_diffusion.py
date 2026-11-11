"""扩散与成长预测模型测试 (Diffusion & Forecasting Tests)."""

import pytest
import numpy as np
from src.diffusion.s_curve import SCurveDiffusionModel
from src.diffusion.lgbm_forecaster import LightGBMGrowthForecaster
from src.diffusion.baseline import BaselineForecaster


from src.core.types import ModelTrainingStatus


def test_s_curve_diffusion_fit():
    s_curve = SCurveDiffusionModel()
    years = np.array([2019, 2020, 2021, 2022, 2023, 2024])
    adoptions = np.array([50, 120, 310, 780, 1600, 2900])
    res = s_curve.fit_and_forecast(years, adoptions, forecast_horizon_years=5)

    assert "growth_multiples" in res
    assert "current_penetration_pct" in res
    assert res["growth_multiples"]["3_year_multiplier"] >= 1.0


def test_lightgbm_growth_forecaster():
    # 1. 真实训练样本不足时，禁止自动用模拟数据训练，返回 MODEL_NOT_TRAINED，预测点估计为 None
    forecaster = LightGBMGrowthForecaster()
    res_train = forecaster.train()
    assert res_train["status"] == ModelTrainingStatus.MODEL_NOT_TRAINED.value

    features = {
        "velocity_1y": 250.0,
        "acceleration": 50.0,
        "active_institutions": 400.0,
        "github_momentum": 800.0,
        "macro_multiplier": 1.15,
        "entropy_score": 0.75,
        "novelty_score": 0.80
    }
    pred_untrained = forecaster.predict_growth(features)
    assert pred_untrained["status"] == "MODEL_NOT_TRAINED"
    assert pred_untrained["forecasts"]["3y"]["point_estimate"] is None

    # 2. 隔离的合成数据算法测试接口：在测试环境下验证算法训练与预测管线
    forecaster.train_on_synthetic_for_testing(n_samples=100)
    pred_trained = forecaster.predict_growth(features)
    assert "forecasts" in pred_trained
    assert "3y" in pred_trained["forecasts"]
    assert pred_trained["forecasts"]["3y"]["point_estimate"] > 1.0


def test_baseline_forecaster():
    random_samples = BaselineForecaster.random_baseline(10)
    assert len(random_samples) == 10
    cagr_mult = BaselineForecaster.cagr_momentum_baseline([100, 150, 225], horizon_years=3)
    assert cagr_mult > 1.0
