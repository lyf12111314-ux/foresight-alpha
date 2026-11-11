"""端到端系统全流程闭环集成测试 (End-to-End System Integration Tests)."""

import pytest
from src.core.database import db_manager
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.monitoring.monitor import SystemMonitor
from src.reporting.dossier import ResearchDossierGenerator


def test_end_to_end_decision_and_reporting_loop():
    # 1. 验证数据库底座就绪
    raw_cnt = db_manager.query_df("SELECT COUNT(*) as c FROM raw_records").iloc[0]["c"]
    assert raw_cnt > 0

    # 2. 运行机会候选池构建
    mgr = OpportunityPoolManager()
    pool = mgr.build_opportunity_pool()
    assert len(pool) >= 4

    top_candidate = pool[0]
    assert top_candidate["composite_score"] > 0
    assert "topic_name" in top_candidate
    assert "legal_assets" in top_candidate

    # 3. 运行动态监控巡检
    monitor = SystemMonitor()
    alerts = monitor.run_daily_scan()
    assert isinstance(alerts, list)

    # 4. 验证 14 章节智能研报生成
    dossier_gen = ResearchDossierGenerator()
    report_md = dossier_gen.generate_dossier(top_candidate)
    assert "# 【Foresight Alpha 深度研报】" in report_md
    assert "## 1. 技术和产业定义" in report_md
    assert "## 14. 数据来源与溯源凭证" in report_md
