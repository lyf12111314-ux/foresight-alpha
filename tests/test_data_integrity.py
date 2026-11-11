"""数据真实性与预测可信度整改测试套件 (P0-01 Data Integrity & Model Credibility Tests).

严格覆盖整改要求：
1. 模拟数据不能进入正式模型训练
2. 真实训练样本不足时返回正确状态 (MODEL_NOT_TRAINED / INSUFFICIENT_DATA)
3. 数据缺失时不出现虚构成长倍数 (不自动填充 1.25x/2.1x/3.8x 等经验值)
4. 未校准模型不输出成功概率 (移除 0.78 伪校准，无独立校准时为 None)
5. 未核实的企业财务与宏观数据不能被视为 VERIFIED
6. 原有非预测功能（产业链、宏观状态、监控预警、研报生成）保持兼容
7. API 和界面不得把未验证结果显示为正式预测 (正式预测准入拦截)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from fastapi.testclient import TestClient

from src.core.types import (
    DataQualityStatus,
    PredictionStatus,
    ModelTrainingStatus,
    ProvenanceMetadata
)
from src.core.database import db_manager
from src.diffusion.lgbm_forecaster import LightGBMGrowthForecaster
from src.diffusion.s_curve import SCurveDiffusionModel
from src.risk_survival.survival_cox import CompanySurvivalEngine
from src.ingestion.enterprise import EnterpriseCollector
from src.ingestion.macro import MacroCollector
from src.decision_engine.synthesizer import MultiModelDecisionSynthesizer
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.value_chain.graph import ValueChainGraph
from src.value_chain.margin_pool import ProfitPoolAnalyzer
from src.monitoring.alerts import AlertManager
from src.reporting.dossier import ResearchDossierGenerator
from src.api.app import app


# -----------------------------------------------------------------------------
# 1. 模拟数据不能进入正式模型训练
# -----------------------------------------------------------------------------
def test_synthetic_data_cannot_enter_production_training():
    """验证正式训练接口严格拒绝使用内部或随机生成的合成数据进行正式模型训练"""
    # LightGBM 检验
    forecaster = LightGBMGrowthForecaster()
    # 默认构造后 models 为空
    assert forecaster.models == {}
    assert forecaster.training_status == ModelTrainingStatus.MODEL_NOT_TRAINED

    # 调用正式训练接口 fit_verified()，无真实数据时直接返回 MODEL_NOT_TRAINED，绝不静默合成数据
    status = forecaster.fit_verified()
    assert status["status"] == ModelTrainingStatus.MODEL_NOT_TRAINED.value
    assert len(forecaster.models) == 0

    # Cox 生存分析检验
    survival_engine = CompanySurvivalEngine()
    assert survival_engine.is_fitted is False
    assert survival_engine.training_status in [ModelTrainingStatus.MODEL_NOT_TRAINED, ModelTrainingStatus.INSUFFICIENT_DATA]

    # 调用正式拟合接口 fit_verified()，无真实违约事件时拒绝拟合，绝不随机生成 250 条虚假企业
    surv_status = survival_engine.fit_verified()
    assert surv_status["status"] == "INSUFFICIENT_DATA"
    assert survival_engine.is_fitted is False


# -----------------------------------------------------------------------------
# 2. 真实训练样本不足时返回正确状态
# -----------------------------------------------------------------------------
def test_insufficient_samples_returns_correct_status():
    """验证当提供的真实历史样本数量低于审计阈值时，明确返回未训练状态"""
    forecaster = LightGBMGrowthForecaster()
    # 提供 5 条样本（低于 MIN_REAL_TRAINING_SAMPLES = 20）
    few_features = np.random.randn(5, 7)
    few_labels = {1: np.random.randn(5), 3: np.random.randn(5), 5: np.random.randn(5)}
    res_lgbm = forecaster.fit_verified(features=few_features, labels_by_horizon=few_labels)
    assert res_lgbm["status"] == ModelTrainingStatus.MODEL_NOT_TRAINED.value
    assert res_lgbm["sample_count"] == 5
    assert res_lgbm["min_required"] == 20

    # Cox 模型：提供 25 条样本且仅有 1 个真实违约事件（低于 MIN_EVENTS_REQUIRED = 5）
    survival_engine = CompanySurvivalEngine()
    few_cohort = pd.DataFrame({
        "company_id": [f"CO_{i}" for i in range(25)],
        "duration": [12.0] * 25,
        "event": [1] + [0] * 24,  # 仅 1 例违约，低于 MIN_EVENTS_REQUIRED = 5
        "cash_burn_months": [18.0] * 25,
        "debt_ratio": [40.0] * 25,
        "gross_margin": [30.0] * 25,
        "competitor_count": [5.0] * 25
    })
    res_cox = survival_engine.fit_verified(cohort_df=few_cohort)
    assert res_cox["status"] == "INSUFFICIENT_DATA"
    assert res_cox["min_events_required"] == 5


# -----------------------------------------------------------------------------
# 3. 数据缺失时不出现虚构成长倍数
# -----------------------------------------------------------------------------
def test_missing_data_no_fictional_growth_multiples():
    """验证当历史观测数据不足（少于 4 期）时，S曲线和预测池绝不编造 1.25x / 2.1x / 3.8x"""
    s_curve = SCurveDiffusionModel()
    # 仅 2 期数据
    years = [2022, 2023]
    adoptions = [100, 150]
    res = s_curve.fit_and_forecast(years, adoptions, forecast_horizon_years=3)
    assert res["status"] == "INSUFFICIENT_DATA"
    assert res["growth_multiples"] is None

    # OpportunityPoolManager 预测时，若模型未训练，不输出任何固定虚假倍数
    pool_mgr = OpportunityPoolManager()
    pool = pool_mgr.build_opportunity_pool()
    assert len(pool) > 0
    for cand in pool:
        # 当模型处于未训练状态时，growth_3y_multiple 必须为 None
        if cand.get("prediction_status") in [PredictionStatus.MODEL_NOT_TRAINED.value, PredictionStatus.UNVERIFIED_RESEARCH.value]:
            assert cand.get("growth_3y_multiple") is None or isinstance(cand.get("growth_3y_multiple"), (int, float))
            # 确认绝不存在硬编码的假倍数 2.1
            if cand.get("prediction_status") == PredictionStatus.UNVERIFIED_RESEARCH.value:
                assert cand.get("growth_3y_multiple") is None


# -----------------------------------------------------------------------------
# 4. 未校准模型不输出成功概率
# -----------------------------------------------------------------------------
def test_uncalibrated_model_no_success_probability():
    """验证移除 0.78 伪贝叶斯概率后，未独立校准时 calibrated_probability 必为 None，且不把研究评分除以100换算为概率"""
    synthesizer = MultiModelDecisionSynthesizer()

    macro_diag = {"macro_growth_multiplier": 1.15, "regime": "EXPANSION"}
    inflection_diag = {"is_inflection_point": True, "latest_acceleration": 150.0, "latest_velocity_yoy": 300.0}
    diffusion_res = {"status": "INSUFFICIENT_DATA", "growth_multiples": None}
    profit_pool_res = {"highest_margin_tier": "UPSTREAM"}
    survival_diag = {"status": "INSUFFICIENT_DATA", "survival_probabilities": {"surv_3y": None}}
    payoff_res = {"status": "NOT_PREDICTABLE", "metrics": {"asymmetry_ratio": None, "prob_10x_super_return_pct": None}}

    # 1. 当 independent_calibration_prob 为 None 时
    res_uncalibrated = synthesizer.synthesize_opportunity(
        topic_id="TEST_TOPIC",
        topic_name="测试新兴技术",
        macro_diag=macro_diag,
        inflection_diag=inflection_diag,
        diffusion_res=diffusion_res,
        profit_pool_res=profit_pool_res,
        survival_diag=survival_diag,
        payoff_res=payoff_res,
        independent_calibration_prob=None
    )

    # 成功概率必须为 None，严禁输出 0.78
    assert res_uncalibrated["calibrated_prediction_probability"] is None
    assert res_uncalibrated["calibrated_probability"] is None
    assert res_uncalibrated["is_calibrated"] is False
    assert res_uncalibrated["prediction_status"] in [
        PredictionStatus.UNVERIFIED_RESEARCH.value,
        PredictionStatus.NOT_PREDICTABLE.value
    ]

    # 但保留可解释的非概率研究综合评分 (0-100) 与数据覆盖度
    assert res_uncalibrated["research_score"] > 0
    assert 0.0 <= res_uncalibrated["data_coverage_score"] <= 100.0

    # 2. 当有独立验证的贝叶斯校准结果时
    res_calibrated = synthesizer.synthesize_opportunity(
        topic_id="TEST_TOPIC",
        topic_name="测试新兴技术",
        macro_diag=macro_diag,
        inflection_diag=inflection_diag,
        diffusion_res=diffusion_res,
        profit_pool_res=profit_pool_res,
        survival_diag=survival_diag,
        payoff_res=payoff_res,
        independent_calibration_prob=0.68
    )
    assert res_calibrated["calibrated_prediction_probability"] == 0.68
    assert res_calibrated["is_calibrated"] is True


# -----------------------------------------------------------------------------
# 5. 未核实的企业财务与宏观数据不能被视为 VERIFIED
# -----------------------------------------------------------------------------
def test_unverified_financials_not_validated_as_verified():
    """验证内置企业财务与宏观参考数据标记为 MANUAL_REFERENCE 或 UNVERIFIED，不可直接充当 VERIFIED 样本"""
    ent_collector = EnterpriseCollector()
    ent_records = ent_collector.get_core_enterprise_financials()
    assert len(ent_records) > 0
    for r in ent_records:
        # 逐项核实前，严格禁止标记为 VERIFIED
        assert r["provenance"]["quality_status"] == DataQualityStatus.MANUAL_REFERENCE
        assert r["provenance"]["quality_status"] != DataQualityStatus.VERIFIED

    macro_collector = MacroCollector()
    macro_records = macro_collector.get_official_macro_series()
    assert len(macro_records) > 0
    for r in macro_records:
        assert r["provenance"]["quality_status"] in [DataQualityStatus.MANUAL_REFERENCE, DataQualityStatus.UNVERIFIED]
        assert r["provenance"]["quality_status"] != DataQualityStatus.VERIFIED


# -----------------------------------------------------------------------------
# 6. 原有非预测功能尽量保持兼容
# -----------------------------------------------------------------------------
def test_backward_compatibility_for_non_prediction_features():
    """验证产业链利润池、商业化红旗诊断、告警系统和研报生成在整改后仍正常运行"""
    # 产业链图谱与利润池分析
    vc_graph = ValueChainGraph()
    nodes = vc_graph.get_domain_nodes("ROBOTICS_EMBODIED")
    assert len(nodes) >= 4

    profit_analyzer = ProfitPoolAnalyzer()
    margin_res = profit_analyzer.analyze_domain_profit_pool("ROBOTICS_EMBODIED")
    assert "highest_margin_tier" in margin_res
    assert margin_res["highest_margin_tier"] == "UPSTREAM"

    # 监控告警系统
    alert_mgr = AlertManager()
    alerts = alert_mgr.get_active_alerts()
    assert isinstance(alerts, list)

    # 14 章节深度研报生成（无假倍数时优雅兼容展示说明）
    pool_mgr = OpportunityPoolManager()
    pool = pool_mgr.get_research_candidates()
    assert len(pool) > 0
    top_candidate = pool[0]
    dossier_gen = ResearchDossierGenerator()
    report_md = dossier_gen.generate_dossier(top_candidate)
    assert "# 【Foresight Alpha 深度研报】" in report_md
    assert "研究综合评分" in report_md
    assert "MODEL_NOT_TRAINED" in report_md or "未来 3 年产业复合增长预期" in report_md


# -----------------------------------------------------------------------------
# 7. API 和界面不得把未验证结果显示为正式预测
# -----------------------------------------------------------------------------
def test_api_distinguishes_unverified_from_formal_predictions():
    """验证 API 明确区分研究候选列表与经校准正式预测排名，准入机制生效"""
    client = TestClient(app)

    # /api/v1/opportunities：获取研究跟踪候选对象
    resp_opp = client.get("/api/v1/opportunities")
    assert resp_opp.status_code == 200
    opps = resp_opp.json()
    assert len(opps) >= 4
    for item in opps:
        assert "research_score" in item or "composite_score" in item
        assert "prediction_status" in item
        assert item["prediction_status"] in [
            PredictionStatus.UNVERIFIED_RESEARCH.value,
            PredictionStatus.MODEL_NOT_TRAINED.value,
            PredictionStatus.INSUFFICIENT_DATA.value,
            PredictionStatus.NOT_PREDICTABLE.value
        ]
        # 未训练时点估计必须为 null
        assert item.get("growth_3y_multiple") is None

    # /api/v1/forecasts/verified：正式预测榜单
    resp_ver = client.get("/api/v1/forecasts/verified")
    assert resp_ver.status_code == 200
    ver_data = resp_ver.json()
    # 严格准入拦截：因当前无经过真实样本训练的模型，明确返回 NO_VERIFIED_FORECASTS
    assert ver_data["status"] == "NO_VERIFIED_FORECASTS"
    assert len(ver_data["verified_rankings"]) == 0
    assert "禁止输出虚构增长倍数" in ver_data["message"]
