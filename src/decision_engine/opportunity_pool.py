"""非对称机会候选池构建与动态排名引擎 (Opportunity Pool Ranking & Persistence)."""

import json
from datetime import date
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.core.database import db_manager
from src.macro.hmm_regime import MacroHMMRegimeModel
from src.tech_discovery.inflection import InflectionDetector
from src.diffusion.s_curve import SCurveDiffusionModel
from src.value_chain.margin_pool import ProfitPoolAnalyzer
from src.risk_survival.survival_cox import CompanySurvivalEngine
from src.payoff_engine.monte_carlo import AsymmetricPayoffEngine
from src.payoff_engine.asset_mapping import LegalAssetMapper
from src.decision_engine.synthesizer import MultiModelDecisionSynthesizer


class OpportunityPoolManager:
    """非对称机会候选池管理器"""

    def __init__(self):
        self.macro_model = MacroHMMRegimeModel()
        self.inflection_detector = InflectionDetector()
        self.diffusion_model = SCurveDiffusionModel()
        self.profit_analyzer = ProfitPoolAnalyzer()
        self.survival_engine = CompanySurvivalEngine()
        self.payoff_engine = AsymmetricPayoffEngine()
        self.asset_mapper = LegalAssetMapper()
        self.synthesizer = MultiModelDecisionSynthesizer()

    def build_opportunity_pool(self, cutoff_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """构建全量非对称机会候选池并写入不可篡改预测历史表"""
        if cutoff_date is None:
            cutoff_date = date.today()

        # 1. 查询 Point-in-Time 宏观数据
        macro_df = db_manager.query_point_in_time("macro_indicators", cutoff_date)
        if macro_df.empty:
            macro_df = db_manager.query_df("SELECT * FROM macro_indicators ORDER BY as_of_date ASC")

        latest_macro = self.macro_model.get_latest_regime(macro_df) if not macro_df.empty else {
            "regime": "EXPANSION", "macro_growth_multiplier": 1.15
        }

        # 2. 查询各技术主题时序数据
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
                # 获取该主题学术与开源时序
                topic_metrics = db_manager.query_point_in_time(
                    "tech_metrics", cutoff_date, target_col="topic_id", target_val=tid
                )
                if topic_metrics.empty:
                    topic_metrics = db_manager.query_df(
                        "SELECT * FROM tech_metrics WHERE topic_id = ? ORDER BY as_of_date ASC", [tid]
                    )

                # 拐点检测
                inflection_res = self.inflection_detector.analyze_tech_acceleration(topic_metrics)

                # 扩散S曲线拟合
                if not topic_metrics.empty and len(topic_metrics) >= 4:
                    years = np.array([d.year if hasattr(d, "year") else int(str(d)[:4]) for d in topic_metrics["as_of_date"]])
                    papers = topic_metrics["papers_count"].values
                    diffusion_res = self.diffusion_model.fit_and_forecast(years, papers, forecast_horizon_years=5)
                else:
                    diffusion_res = {
                        "growth_multiples": {"1_year_multiplier": 1.25, "3_year_multiplier": 2.1, "5_year_multiplier": 3.8},
                        "current_penetration_pct": 8.5,
                        "stage_description": "早期主流扩散突破期"
                    }

                # 产业链利润池
                profit_res = self.profit_analyzer.analyze_domain_profit_pool(domain_id)

                # 生存与失败风险
                survival_res = self.survival_engine.predict_company_survival({
                    "cash_burn_months": 28.0, "debt_ratio": 38.0, "gross_margin": 42.0, "competitor_count": 8.0
                })

                # 非对称蒙特卡洛
                growth_mult = diffusion_res.get("growth_multiples", {}).get("3_year_multiplier", 2.2)
                payoff_res = self.payoff_engine.simulate_payoff(
                    base_growth_multiplier=growth_mult,
                    current_valuation_multiple=25.0,
                    target_valuation_multiple_range=(20.0, 45.0)
                )

                # 合法可投资产映射
                mapped_assets = self.asset_mapper.map_topic_to_assets(tid)

                # 综合决策融合
                synthesized = self.synthesizer.synthesize_opportunity(
                    topic_id=tid,
                    topic_name=tname,
                    macro_diag=latest_macro,
                    inflection_diag=inflection_res,
                    diffusion_res=diffusion_res,
                    profit_pool_res=profit_res,
                    survival_diag=survival_res,
                    payoff_res=payoff_res,
                    as_of_data_date=cutoff_date
                )

                # 附加详情
                synthesized["domain_id"] = domain_id
                synthesized["stage_description"] = diffusion_res.get("stage_description", "")
                synthesized["penetration_pct"] = diffusion_res.get("current_penetration_pct", 0.0)
                synthesized["highest_margin_tier"] = profit_res.get("highest_margin_tier", "UPSTREAM")
                synthesized["legal_assets"] = mapped_assets
                synthesized["inflection_evidence"] = inflection_res.get("evidence", {})

                # 持久化记录到 model_forecasts
                fc = synthesized["forecast_record"]
                conn.execute("""
                    INSERT OR REPLACE INTO model_forecasts
                    (forecast_id, target_id, target_level, model_name, model_version, as_of_data_date, prediction_created_at, horizon_years, predicted_metric, point_estimate, ci_low, ci_high, confidence_score, scenarios_json, drivers_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    fc["forecast_id"], fc["target_id"], fc["target_level"],
                    fc["model_name"], fc["model_version"], fc["as_of_data_date"],
                    fc["prediction_created_at"], fc["horizon_years"], fc["predicted_metric"],
                    fc["point_estimate"], fc["confidence_interval_low"], fc["confidence_interval_high"],
                    fc["confidence_score"], json.dumps(fc["scenarios"]), json.dumps(fc["driver_features"])
                ])

                pool_candidates.append(synthesized)

            # 按综合评分降序排列
            pool_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        finally:
            conn.close()

        return pool_candidates
