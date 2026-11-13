"""端到端系统全流程闭环集成测试 (End-to-End System Integration Tests)."""

from datetime import date, datetime, timedelta
import pytest
import src.core.database as db_module
from src.core.database import DatabaseManager
import src.decision_engine.opportunity_pool as opp_pool
import src.monitoring.alerts as mon_alerts
import src.monitoring.monitor as mon_mod
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.monitoring.monitor import SystemMonitor
from src.reporting.dossier import ResearchDossierGenerator


def test_end_to_end_decision_and_reporting_loop(tmp_path, monkeypatch):
    temp_db = DatabaseManager(db_path=str(tmp_path / "e2e_test.duckdb"))
    monkeypatch.setattr(db_module, "db_manager", temp_db)
    monkeypatch.setattr(opp_pool, "db_manager", temp_db)
    monkeypatch.setattr(mon_alerts, "db_manager", temp_db)
    monkeypatch.setattr(mon_mod, "db_manager", temp_db)

    conn = temp_db.get_connection()
    try:
        conn.execute(
            """
            INSERT INTO raw_records
            (record_id, source_name, source_url, as_of_date, published_at, ingested_at, quality_status, data_version, raw_hash, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "rec_e2e_1", "TEST_FIXTURE", "https://example.invalid/e2e",
                date(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 1),
                "SYNTHETIC", "v1", "0" * 64, "{}"
            ],
        )
        base = date(2020, 1, 1)
        for i in range(12):
            d = base + timedelta(days=90 * i)
            conn.execute(
                """
                INSERT INTO macro_indicators
                (indicator_code, as_of_date, published_at, indicator_name, value, unit, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ["FED_RATE", d, d, "Federal Funds Rate", 1.5 + 0.3 * (i % 4), "%", "TEST_FIXTURE"],
            )
        topics = [
            ("TOPIC_EMBODIED_AI", 10),
            ("TOPIC_SOLID_STATE_BATT", 25),
            ("TOPIC_AGENTIC_AI", 40),
            ("TOPIC_NEUROMORPHIC", 15),
        ]
        for tid, base_papers in topics:
            for yr in range(2020, 2025):
                d = date(yr, 6, 30)
                conn.execute(
                    """
                    INSERT INTO tech_metrics
                    (topic_id, as_of_date, published_at, papers_count, citations_count, active_institutions, github_stars, github_forks, commit_velocity, entropy_score, novelty_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [tid, d, d, base_papers * (yr - 2019), base_papers * 10, 5 + yr - 2020, 100 * (yr - 2019), 20, 1.5, 0.7, 0.8],
                )
    finally:
        conn.close()

    # 1. 验证数据库底座就绪
    raw_cnt = temp_db.query_df("SELECT COUNT(*) as c FROM raw_records").iloc[0]["c"]
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
