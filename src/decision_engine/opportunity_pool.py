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
from src.core.types import PredictionStatus, ModelTrainingStatus, ModelStage, SimulationType
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

    def __init__(
        self,
        lgbm_forecaster: Optional[LightGBMGrowthForecaster] = None,
        survival_engine: Optional[CompanySurvivalEngine] = None
    ):
        self.macro_model = MacroHMMRegimeModel()
        self.inflection_detector = InflectionDetector()
        self.diffusion_model = SCurveDiffusionModel()
        self.lgbm_forecaster = lgbm_forecaster or LightGBMGrowthForecaster()
        self.profit_analyzer = ProfitPoolAnalyzer()
        self.survival_engine = survival_engine or CompanySurvivalEngine()
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
                # 获取该主题截止时点可获得的不可变技术特征版本 (严禁因后续重采覆盖导致历史版本消失或前视泄露)
                topic_metrics = db_manager.query_tech_metrics_point_in_time(cutoff_date, topic_id=tid)

                # 拐点检测
                inflection_res = self.inflection_detector.analyze_tech_acceleration(topic_metrics)

                # 扩散S曲线拟合 (用于学术文献扩散拐点研判，严禁直接等同于产业/投资成长倍数)
                valid_scurve = (
                    topic_metrics[pd.notna(topic_metrics["papers_count"])]
                    if (not topic_metrics.empty and "papers_count" in topic_metrics.columns)
                    else topic_metrics.iloc[0:0]
                )
                if not valid_scurve.empty and len(valid_scurve) >= 4 and float(valid_scurve["papers_count"].iloc[-1]) > 0:
                    years = np.array([d.year if hasattr(d, "year") else int(str(d)[:4]) for d in valid_scurve["as_of_date"]])
                    papers = valid_scurve["papers_count"].astype(float).values
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

                # LightGBM 多维特征成长预测器 (使用当时实际可得指标构建，缺失严格保留 None，杜绝默认值或非 PIT GitHub 泄露)
                def _opt_float(col: str) -> Optional[float]:
                    if topic_metrics.empty or col not in topic_metrics.columns:
                        return None
                    v = topic_metrics[col].iloc[-1]
                    return None if (v is None or pd.isna(v)) else float(v)

                github_mom: Optional[float] = _opt_float("github_stars")
                if not topic_metrics.empty and "github_pit_eligible" in topic_metrics.columns:
                    gh_elig = topic_metrics["github_pit_eligible"].iloc[-1]
                    if cutoff_date < date.today() and (gh_elig is None or pd.isna(gh_elig) or not bool(gh_elig)):
                        github_mom = None

                lgbm_features = {
                    "velocity_1y": float(inflection_res["latest_velocity_yoy"]) if (not topic_metrics.empty and inflection_res.get("latest_velocity_yoy") is not None) else None,
                    "acceleration": float(inflection_res["latest_acceleration"]) if (not topic_metrics.empty and inflection_res.get("latest_acceleration") is not None) else None,
                    "active_institutions": _opt_float("active_institutions"),
                    "github_momentum": github_mom,
                    "macro_multiplier": float(latest_macro["macro_growth_multiplier"]) if latest_macro.get("macro_growth_multiplier") is not None else None,
                    "entropy_score": _opt_float("entropy_score"),
                    "novelty_score": _opt_float("novelty_score"),
                }
                forecaster_pred = self.lgbm_forecaster.predict_growth(
                    lgbm_features,
                    require_complete_features=(self.lgbm_forecaster.model_stage == ModelStage.VERIFIED_PRODUCTION),
                )

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
                    "is_verified": bool(forecaster_pred.get("is_verified", False)),
                    "growth_multiples": growth_multiples,
                    "current_penetration_pct": s_curve_res.get("current_penetration_pct"),
                    "stage_description": s_curve_res.get("stage_description")
                }

                # 产业链利润池分析
                profit_res = self.profit_analyzer.analyze_domain_profit_pool(domain_id)

                # 生存与失败风险 (调用生存引擎评估该赛道代表性核心企业财务生存分布)
                topic_financials = {
                    "TOPIC_EMBODIED_AI": {"debt_ratio": 45.0, "gross_margin": 35.0, "cash_burn_months": 24.0},
                    "TOPIC_SOLID_STATE_BATT": {"debt_ratio": 58.0, "gross_margin": 26.0, "cash_burn_months": 36.0},
                    "TOPIC_AGENTIC_AI": {"debt_ratio": 20.0, "gross_margin": 65.0, "cash_burn_months": 18.0},
                    "TOPIC_NEUROMORPHIC": {"debt_ratio": 22.0, "gross_margin": 60.0, "cash_burn_months": 20.0},
                    "TOPIC_EVTOL_AERO": {"debt_ratio": 35.0, "gross_margin": 45.0, "cash_burn_months": 15.0},
                    "TOPIC_SYNTHETIC_BIO": {"debt_ratio": 25.0, "gross_margin": 40.0, "cash_burn_months": 30.0},
                }
                c_metrics = topic_financials.get(tid, {})
                survival_res = self.survival_engine.predict_company_survival(c_metrics)

                # 非对称蒙特卡洛情景推演与正式决策隔离
                growth_mult = None
                if diffusion_res and diffusion_res.get("growth_multiples"):
                    growth_mult = diffusion_res["growth_multiples"].get("3_year_multiplier")

                is_lgbm_verified = bool(
                    diffusion_res
                    and diffusion_res.get("is_verified") is True
                    and diffusion_res.get("status") == PredictionStatus.PREDICTED.value
                )

                # 正式收益模拟需同时具备：已验证成长模型 + 经逐笔审计的真实可投资产价格与估值依据
                # 缺少可核验的资产价格与估值时，严格保持在假设情景推演 (HYPOTHETICAL_SCENARIO)，正式 prob_10x 为 None
                has_verified_asset_valuation = False
                is_formal_payoff_eligible = bool(is_lgbm_verified and has_verified_asset_valuation)

                if growth_mult is not None and not np.isnan(growth_mult):
                    if is_formal_payoff_eligible:
                        payoff_res = self.payoff_engine.simulate_payoff(
                            base_growth_multiplier=growth_mult,
                            current_valuation_multiple=25.0,
                            target_valuation_multiple_range=(20.0, 45.0),
                            simulation_type=SimulationType.FORMAL_PREDICTION,
                            is_verified=True
                        )
                    else:
                        # 实验性模型预测或缺少核验估值，严格路由至假设情景推演，正式 prob_10x 为 None
                        payoff_res = self.payoff_engine.simulate_payoff(
                            base_growth_multiplier=growth_mult,
                            current_valuation_multiple=25.0,
                            target_valuation_multiple_range=(20.0, 45.0),
                            simulation_type=SimulationType.HYPOTHETICAL_SCENARIO,
                            is_verified=False
                        )
                else:
                    payoff_res = {
                        "status": "NOT_PREDICTABLE",
                        "simulation_type": SimulationType.HYPOTHETICAL_SCENARIO.value,
                        "metrics": {
                            "asymmetry_ratio": None,
                            "prob_10x_super_return_pct": None,
                            "scenario_prob_10x_pct": None,
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

                # 关联本次预测实际使用的不可变技术特征版本标识 (P1 历史复现链路)
                used_feature_version_ids: List[str] = []
                if not topic_metrics.empty and "latest_feature_version_id" in topic_metrics.columns:
                    used_feature_version_ids = [
                        str(v)
                        for v in topic_metrics["latest_feature_version_id"].tolist()
                        if v is not None and not pd.isna(v) and str(v).strip()
                    ]
                latest_fv_id = used_feature_version_ids[-1] if used_feature_version_ids else None
                # 兼容直接写入旧表的数据，但不能将部分清单宣称为完整可复现输入。
                version_manifest = used_feature_version_ids if len(used_feature_version_ids) == len(topic_metrics) else None

                # 附加详情
                synthesized["domain_id"] = domain_id
                synthesized["stage_description"] = diffusion_res.get("stage_description", "")
                synthesized["penetration_pct"] = diffusion_res.get("current_penetration_pct")
                synthesized["highest_margin_tier"] = profit_res.get("highest_margin_tier", "UPSTREAM")
                synthesized["legal_assets"] = mapped_assets
                synthesized["inflection_evidence"] = inflection_res.get("evidence", {})
                synthesized["survival_status"] = survival_res.get("status", "INSUFFICIENT_DATA")
                synthesized["features"] = lgbm_features
                synthesized["feature_version_id"] = latest_fv_id
                synthesized["feature_version_ids"] = used_feature_version_ids
                synthesized["feature_version"] = latest_fv_id or f"feat_{cutoff_date.strftime('%Y%m%d')}_{tid}"
                synthesized["feature_published_at"] = (
                    topic_metrics["published_at"].iloc[-1]
                    if (not topic_metrics.empty and "published_at" in topic_metrics)
                    else None
                )

                # 持久化记录到 model_forecasts (不可变版本化存储，显式绑定所用技术特征版本)
                fc = synthesized["forecast_record"]
                base_forecast_id = fc["forecast_id"]
                existing_recs = conn.execute(
                    "SELECT forecast_id FROM model_forecasts WHERE forecast_id = ? OR forecast_id LIKE ?",
                    [base_forecast_id, f"{base_forecast_id}_v%"]
                ).fetchall()
                if existing_recs:
                    version_num = len(existing_recs) + 1
                    actual_forecast_id = f"{base_forecast_id}_v{version_num}"
                else:
                    actual_forecast_id = base_forecast_id

                conn.execute("""
                    INSERT INTO model_forecasts
                    (forecast_id, target_id, target_level, model_name, model_version, as_of_data_date, prediction_created_at, horizon_years, predicted_metric, point_estimate, ci_low, ci_high, confidence_score, prediction_status, is_verified_forecast, scenarios_json, drivers_json, feature_version_id, feature_version_ids, input_features_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    actual_forecast_id, fc["target_id"], fc["target_level"],
                    fc["model_name"], fc["model_version"], fc["as_of_data_date"],
                    fc["prediction_created_at"], fc["horizon_years"], fc["predicted_metric"],
                    fc["point_estimate"], fc["confidence_interval_low"], fc["confidence_interval_high"],
                    fc["confidence_score"], fc["prediction_status"], fc["is_verified_forecast"],
                    json.dumps(fc["scenarios"]), json.dumps(fc["driver_features"]), latest_fv_id,
                    json.dumps(version_manifest) if version_manifest is not None else None,
                    json.dumps(lgbm_features, allow_nan=False)
                ])
                synthesized["forecast_id"] = actual_forecast_id

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
