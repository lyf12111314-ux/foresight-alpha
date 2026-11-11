"""FastAPI REST 服务接口 (Foresight Alpha REST API Layer)."""

from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List
from datetime import date

from src.core.database import db_manager
from src.macro.hmm_regime import MacroHMMRegimeModel
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.monitoring.alerts import AlertManager
from src.backtest.engine import PointInTimeBacktestEngine
from src.reporting.dossier import ResearchDossierGenerator

app = FastAPI(
    title="Foresight Alpha - 新兴产业与非对称机会智能预测系统 API",
    version="1.0.0",
    description="面向未来3-10年硬科技与新兴产业非对称机会的智能预测与决策支持服务"
)

pool_manager = OpportunityPoolManager()
alert_manager = AlertManager()
backtest_engine = PointInTimeBacktestEngine()
dossier_gen = ResearchDossierGenerator()
macro_model = MacroHMMRegimeModel()


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "system": "Foresight Alpha", "version": "1.0.0"}


@app.get("/api/v1/macro/status")
def get_macro_status() -> Dict[str, Any]:
    macro_df = db_manager.query_df("SELECT * FROM macro_indicators ORDER BY as_of_date ASC")
    if macro_df.empty:
        raise HTTPException(status_code=404, detail="No macro indicators found in database.")
    return macro_model.get_latest_regime(macro_df)


@app.get("/api/v1/opportunities")
def get_opportunity_pool() -> List[Dict[str, Any]]:
    return pool_manager.build_opportunity_pool()


@app.get("/api/v1/alerts")
def get_active_alerts() -> List[Dict[str, Any]]:
    return alert_manager.get_active_alerts()


@app.get("/api/v1/backtest/latest")
def get_latest_backtest() -> Dict[str, Any]:
    df = db_manager.query_df("SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT 1")
    if df.empty:
        # 实时运行一次基准回测
        return backtest_engine.run_walk_forward_backtest()
    return df.iloc[0].to_dict()


@app.get("/api/v1/report/{topic_id}")
def generate_topic_report(topic_id: str) -> Dict[str, str]:
    pool = pool_manager.build_opportunity_pool()
    target = next((c for c in pool if c["topic_id"] == topic_id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found in opportunity pool.")
    report_content = dossier_gen.generate_dossier(target)
    return {"topic_id": topic_id, "markdown": report_content}
