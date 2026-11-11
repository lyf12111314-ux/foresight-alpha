"""非对称机会候选池构建与准入治理引擎 (Opportunity Pool Governance & Dual Rankings).

严格遵循 V1.1 数据真实性要求：
1. 正式预测前检查数据质量、模型训练状态与时序充分性。
2. 数据不足时保留研究对象作为基础跟踪，但标记为不可预测 (NOT_PREDICTABLE)。
3. 严禁使用固定默认增长倍数替代缺失的真实预测。
4. 研究候选排序 (Research Ranking) 与真实验证预测排名 (Verified Forecast Ranking) 严格分离。
"""

import json
from datetime import date
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.database import db_manager
from src.core.types import PredictionStatus, ModelTrainingStatus
from src.macro.hmm_regime import MacroHMMRegimeModel
from src.tech_discovery.inflection import InflectionDetector
from src.diffusion.s_curve import SCurveDiffusionModel
from src.diffusion.lgbm_forecaster import LightGBMGrowthForecaster
from src.value_chain.margin_pool import ProfitPoolAnalyzer
from src.risk_survival.survival_cox import CompanySurvivalEngine
from src.payoff_engine.monte_carlo import AsymmetricPayoffEngine
from src.payoff_engine.asset_mapping import LegalAssetMapper
from src.decision_engine.synthesizer import MultiModelDecisionSynthesizer


class OpportunityPoolManager:
    """非对称机会候选池管理器 (研究与预测双轨制)"""

    def __init__(self):
        self.macro_model = MacroHMMRegimeModel()
        self.inflection_detector = InflectionDetector()
        self.diffusion_model = SCurveDiffusionModel()
        self.lgbm_forecaster = LightGBMGrowthForecaster()
        self.profit_analyzer = ProfitPoolAnalyzer()
        self.survival_engine = CompanySurvivalEngine()
        self.payoff_engine = AsymmetricPayoffEngine()
        self.asset_mapper = LegalAssetMapper()
        self.synthesizer = MultiModelDecisionSynthesizer()

    def train_models_on_historical_data(self) -> Dict[str, Any]:
        """使用真实历史数据集训练预测与生存模型 (P1-01 训练管道)
        
        执行真实多期前沿技术时序与真实A股违约/存续队列训练，
        使 LightGBMGrowthForecaster 与 CompanySurvivalEngine 达到 TRAINED 状态。
        """
        lgbm_res = self.lgbm_forecaster.fit_from_historical_tech_samples()
        surv_res = self.survival_engine.fit_from_historical_cohort()
        return {
            "lgbm_growth_forecaster": lgbm_res,
            "company_survival_engine": surv_res
        }

    def build_opportunity_pool(self, cutoff_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """构建机会候选池，执行真实性准入审计并持久化到 model_forecasts"""
        if cutoff_date is None:
            cutoff_date = date.today()

        # 1. 查询宏观数据 (严格遵守 Point-in-Time 物理隔离，严禁在无数据时回退查询全量未来表)
        macro_df = db_manager.query_point_in_time("macro_indicators", cutoff_date)
        if not macro_df.empty:
            latest_macro = self.macro_model.get_latest_regime(macro_df)
        else:
            latest_macro = {
                "regime": "UNKNOWN",
                "macro_growth_multiplier": 1.0,
                "is_point_in_time_available": False,
                "notes": f"截至时点 {cutoff_date.isoformat()} 尚无可获得的公开宏观指标。"
            }

        # 2. 核心跟踪赛道
        topics = [
            ("TOPIC_EMBODIED_AI", "具身智能与人形机器人", "ROBOTICS_EMBODIED"),
            ("TOPIC_SOLID_STATE_BATT", "全固态电池与先进电解质", "ENERGY_STORAGE"),
            ("TOPIC_AGENTIC_AI", "AI智能体与自主交易基础设施", "AI_INFRA"),
            ("TOPIC_NEUROMORPHIC", "神经拟态计算与类脑芯片", "AI_INFRA"),
            ("TOPIC_EVTOL_AERO", "低空经济与eVTOL飞行器", "LOW_ALTITUDE_AERO"),
            ("TOPIC_SYNTHETIC_BIO", "合成生物学与生物智造", "BIOTECH_SYNTHETIC")
        ]

        pool_candidates = []
        conn = db_manager.get_connection()

        try:
            for tid, tname, domain_id in topics:
                # 获取该主题技术时序指标 (严禁在无数据时全表扫描泄漏未来信息)
                topic_metrics = db_manager.query_point_in_time(
                    "tech_metrics", cutoff_date, target_col="topic_id", target_val=tid
                )

                # 拐点检测
                inflection_res = self.inflection_detector.analyze_tech_acceleration(topic_metrics)

                # 扩散S曲线拟合 (用于学术文献扩散拐点研判，严禁直接等同于产业/投资成长倍数)
                if not topic_metrics.empty and len(topic_metrics) >= 4 and topic_metrics["papers_count"].iloc[-1] > 0:
                    years = np.array([d.year if hasattr(d, "year") else int(str(d)[:4]) for d in topic_metrics["as_of_date"]])
                    papers = topic_metrics["papers_count"].values
                    try:
                        s_curve_res = self.diffusion_model.fit_and_forecast(years, papers, forecast_horizon_years=5)
                    except Exception:
                        s_curve_res = {
                            "status": "FIT_FAILED",
                            "growth_multiples": None,
                            "current_penetration_pct": None,
                            "stage_description": "时序波动过大，S曲线拟合未收敛"
                        }
                else:
                    s_curve_res = {
                        "status": "INSUFFICIENT_DATA",
                        "growth_multiples": None,
                        "current_penetration_pct": None,
                        "stage_description": "历史样本不足4期或最新文献量为0，禁止虚构增长倍数"
                    }

                # LightGBM 多维特征成长预测器 (准入控制：未用真实样本训练前严禁输出预测倍数)
                lgbm_features = {
                    "velocity_1y": inflection_res.get("latest_velocity_yoy", 0.0),
                    "acceleration": inflection_res.get("latest_acceleration", 0.0),
                    "active_institutions": float(topic_metrics["active_institutions"].iloc[-1]) if not topic_metrics.empty else 0.0,
                    "github_momentum": float(topic_metrics["github_stars"].iloc[-1]) if not topic_metrics.empty else 0.0,
                    "macro_multiplier": float(latest_macro.get("macro_growth_multiplier", 1.0)),
                    "entropy_score": 0.60,
                    "novelty_score": 0.70
                }
                forecaster_pred = self.lgbm_forecaster.predict_growth(lgbm_features)

                # 提取多期点估计 (若属于 UNVERIFIED_RESEARCH 实验模式，仅作为研究估计保留，绝不标记为正式已验证)
                growth_mult = None
                growth_multiples = None
                if forecaster_pred.get("status") in [PredictionStatus.PREDICTED.value, "PREDICTED", "UNVERIFIED_RESEARCH"] and forecaster_pred.get("forecasts"):
                    f3y = forecaster_pred["forecasts"].get("3y", {})
                    growth_mult = f3y.get("point_estimate")
                    if growth_mult is not None:
                        growth_multiples = {
                            "1_year_multiplier": forecaster_pred["forecasts"].get("1y", {}).get("point_estimate"),
                            "3_year_multiplier": growth_mult,
                            "5_year_multiplier": forecaster_pred["forecasts"].get("5y", {}).get("point_estimate")
                        }

                diffusion_res = {
                    "status": forecaster_pred.get("status", "MODEL_NOT_TRAINED"),
                    "growth_multiples": growth_multiples,
                    "current_penetration_pct": s_curve_res.get("current_penetration_pct"),
                    "stage_description": s_curve_res.get("stage_description")
                }

                # 产业链利润池分析
                profit_res = self.profit_analyzer.analyze_domain_profit_pool(domain_id)

                # 生存与失败风险 (严格调用生存引擎，若无真实验证队列则返回 INSUFFICIENT_DATA)
                survival_res = self.survival_engine.predict_company_survival({})

                # 非对称蒙特卡洛情景推演 (仅在具备有效增长倍数时进行推演，否则标记不可预测)
                growth_mult = None
                if diffusion_res and diffusion_res.get("growth_multiples"):
                    growth_mult = diffusion_res["growth_multiples"].get("3_year_multiplier")

                if growth_mult is not None and not np.isnan(growth_mult):
                    payoff_res = self.payoff_engine.simulate_payoff(
                        base_growth_multiplier=growth_mult,
                        current_valuation_multiple=25.0,
                        target_valuation_multiple_range=(20.0, 45.0)
                    )
                else:
                    payoff_res = {
                        "status": "NOT_PREDICTABLE",
                        "metrics": {
                            "asymmetry_ratio": None,
                            "prob_10x_super_return_pct": None,
                            "prob_positive_return_pct": None
                        }
                    }

                # 合法可投资产映射
                mapped_assets = self.asset_mapper.map_topic_to_assets(tid)

                # 综合决策融合 (研究评分与校准概率分离)
                synthesized = self.synthesizer.synthesize_opportunity(
                    topic_id=tid,
                    topic_name=tname,
                    macro_diag=latest_macro,
                    inflection_diag=inflection_res,
                    diffusion_res=diffusion_res,
                    profit_pool_res=profit_res,
                    survival_diag=survival_res,
                    payoff_res=payoff_res,
                    as_of_data_date=cutoff_date,
                    independent_calibration_prob=None  # 未执行独立历史校准前，强制为 None
                )

                # 附加详情
                synthesized["domain_id"] = domain_id
                synthesized["stage_description"] = diffusion_res.get("stage_description", "")
                synthesized["penetration_pct"] = diffusion_res.get("current_penetration_pct")
                synthesized["highest_margin_tier"] = profit_res.get("highest_margin_tier", "UPSTREAM")
                synthesized["legal_assets"] = mapped_assets
                synthesized["inflection_evidence"] = inflection_res.get("evidence", {})
                synthesized["survival_status"] = survival_res.get("status", "INSUFFICIENT_DATA")

                # 持久化记录到 model_forecasts
                fc = synthesized["forecast_record"]
                conn.execute("""
                    INSERT OR REPLACE INTO model_forecasts
                    (forecast_id, target_id, target_level, model_name, model_version, as_of_data_date, prediction_created_at, horizon_years, predicted_metric, point_estimate, ci_low, ci_high, confidence_score, prediction_status, is_verified_forecast, scenarios_json, drivers_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    fc["forecast_id"], fc["target_id"], fc["target_level"],
                    fc["model_name"], fc["model_version"], fc["as_of_data_date"],
                    fc["prediction_created_at"], fc["horizon_years"], fc["predicted_metric"],
                    fc["point_estimate"], fc["confidence_interval_low"], fc["confidence_interval_high"],
                    fc["confidence_score"], fc["prediction_status"], fc["is_verified_forecast"],
                    json.dumps(fc["scenarios"]), json.dumps(fc["driver_features"])
                ])

                pool_candidates.append(synthesized)

            # 默认按客观研究评分降序排序 (研究关注度榜单)
            pool_candidates.sort(key=lambda x: x["research_score"], reverse=True)

        finally:
            conn.close()

        return pool_candidates

    def get_research_candidates(self, cutoff_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """获取研究关注度排序榜单 (基于宏观、技术与产业链客观综合分)"""
        candidates = self.build_opportunity_pool(cutoff_date)
        return sorted(candidates, key=lambda x: x["research_score"], reverse=True)

    def get_verified_forecast_ranking(self, cutoff_date: Optional[date] = None) -> Dict[str, Any]:
        """获取经过独立校准与真实验证的正式预测排名 (如无达标模型则严格返回空)"""
        candidates = self.build_opportunity_pool(cutoff_date)
        verified = [c for c in candidates if c.get("is_calibrated") is True and c.get("prediction_status") == PredictionStatus.PREDICTED.value]

        if not verified:
            return {
                "status": "NO_VERIFIED_FORECASTS",
                "ranking": [],
                "verified_rankings": [],
                "message": "当前模型尚未基于真实验证样本完成独立经验概率校准，为保护决策真实性，禁止输出虚构增长倍数及虚假预测排名。"
            }

        verified.sort(key=lambda x: x.get("calibrated_prediction_probability") or 0.0, reverse=True)
        return {
            "status": "VERIFIED",
            "ranking": verified,
            "verified_rankings": verified,
            "message": f"共获得 {len(verified)} 项经过独立校准的正式预测标的。"
        }
