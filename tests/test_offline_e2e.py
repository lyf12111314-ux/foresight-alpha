"""无外部网络依赖的可重复离线端到端集成测试 (Offline E2E Integration Tests).

严格遵循 P0-02 规范：
1. 使用临时 DuckDB 文件，绝不访问或污染真实生产数据库 (`data/storage.duckdb`)。
2. 全程阻断外部网络请求 (requests / httpx / urllib / socket.create_connection)。
3. 使用明确标记为 `SYNTHETIC` / `TEST_ONLY` 的受控测试夹具数据。
4. 覆盖：空库初始化 -> 模拟入库 -> 宏观识别 -> 机会池生成 -> 监控巡检 -> 滚动回测 -> 研报生成。
5. 验证无真实模型与未成熟标签时正确返回 None / 未校准 / 不可用状态，绝不伪造正式预测。
6. 不修改仓库内历史示范导出文件，测试结束自动清理临时文件。
"""

import os
import json
import socket
import tempfile
import urllib.request
from datetime import date, datetime
import pytest
import requests
import httpx

import src.core.database as db_module
from src.core.database import DatabaseManager
from src.macro.hmm_regime import MacroHMMRegimeModel
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.monitoring.monitor import SystemMonitor
from src.backtest.engine import PointInTimeBacktestEngine
from src.reporting.dossier import ResearchDossierGenerator


@pytest.fixture
def block_external_network(monkeypatch):
    """全局阻断外部网络请求，确保离线测试绝对不依赖公网"""
    def _deny_network(*args, **kwargs):
        raise RuntimeError("OFFLINE_E2E_FORBIDDEN_NETWORK_CALL: 离线测试严禁发起外部网络请求！")

    monkeypatch.setattr(requests.Session, "request", _deny_network)
    monkeypatch.setattr(requests, "get", _deny_network)
    monkeypatch.setattr(requests, "post", _deny_network)
    monkeypatch.setattr(httpx.Client, "request", _deny_network)
    monkeypatch.setattr(httpx.AsyncClient, "request", _deny_network)
    monkeypatch.setattr(urllib.request, "urlopen", _deny_network)
    monkeypatch.setattr(socket, "create_connection", _deny_network)


@pytest.fixture
def isolated_temp_duckdb(monkeypatch):
    """创建隔离的临时 DuckDB 数据库实例并注入全调用链"""
    prod_db_path = os.path.abspath(db_module.db_manager.db_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db_path = os.path.abspath(os.path.join(tmpdir, "offline_e2e_test.duckdb"))
        # 防误用保护：严禁临时路径指向真实生产数据库
        assert tmp_db_path != prod_db_path, "保护拦截：离线测试不得指向生产数据库！"

        temp_mgr = DatabaseManager(db_path=tmp_db_path)

        # 使用 DuckDB 原生系统表查询验证空库表结构已成功初始化
        conn = temp_mgr.get_connection()
        try:
            tables_df = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main';"
            ).df()
            table_names = set(tables_df["table_name"].tolist())
            for expected in ["raw_records", "tech_metrics", "macro_indicators", "company_financials", "model_forecasts"]:
                assert expected in table_names, f"DuckDB 空库初始化缺失核心表: {expected}"
        finally:
            conn.close()

        # 替换全链路模块中的 db_manager
        import src.ingestion.pipeline as ing_pipe
        import src.backtest.engine as bt_eng
        import src.decision_engine.opportunity_pool as opp_pool
        import src.monitoring.alerts as mon_alerts
        import src.monitoring.monitor as mon_mod

        monkeypatch.setattr(db_module, "db_manager", temp_mgr)
        monkeypatch.setattr(ing_pipe, "db_manager", temp_mgr)
        monkeypatch.setattr(bt_eng, "db_manager", temp_mgr)
        monkeypatch.setattr(opp_pool, "db_manager", temp_mgr)
        monkeypatch.setattr(mon_alerts, "db_manager", temp_mgr)
        monkeypatch.setattr(mon_mod, "db_manager", temp_mgr)

        yield temp_mgr


def _seed_controlled_synthetic_data(temp_mgr: DatabaseManager) -> None:
    """向临时 DuckDB 写入明确标记为 SYNTHETIC / TEST_ONLY 的受控测试数据"""
    conn = temp_mgr.get_connection()
    try:
        # 1. 插入 raw_records (明确标记 SYNTHETIC 与 TEST_ONLY_SYNTHETIC_E2E)
        conn.execute("""
            INSERT INTO raw_records
            (record_id, source_name, source_url, as_of_date, published_at, ingested_at, quality_status, data_version, raw_hash, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, [
            "SYNTH_REC_001", "Offline_Test_Fixture", "synthetic://test_only",
            date(2024, 1, 1), datetime(2024, 1, 2, 0, 0, 0), datetime(2024, 1, 2, 0, 0, 0),
            "SYNTHETIC", "TEST_ONLY_SYNTHETIC_E2E", "0" * 64,
            json.dumps({"note": "TEST_ONLY controlled fixture"})
        ])

        # 2. 插入宏观时序指标 (2015-2024，确保历史截面 2021-12-31 有 >=4 期独立样本)
        for yr in range(2015, 2025):
            conn.execute("""
                INSERT INTO macro_indicators
                (indicator_code, as_of_date, published_at, indicator_name, value, unit, source)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, [
                "MACRO_GDP_GROWTH", date(yr, 12, 31), datetime(yr, 12, 31, 0, 0, 0),
                "GDP Growth (SYNTHETIC)", 4.5 + (yr - 2015) * 0.25, "%", "SYNTHETIC_TEST_ONLY"
            ])
            conn.execute("""
                INSERT INTO macro_indicators
                (indicator_code, as_of_date, published_at, indicator_name, value, unit, source)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, [
                "MACRO_PMI_MFG", date(yr, 12, 31), datetime(yr, 12, 31, 0, 0, 0),
                "Manufacturing PMI (SYNTHETIC)", 49.0 + ((yr - 2015) % 5) * 0.8, "index", "SYNTHETIC_TEST_ONLY"
            ])

        # 3. 插入技术主题时序指标 (2018-2024)
        topics = [
            ("TOPIC_EMBODIED_AI", 1.45),
            ("TOPIC_SOLID_STATE_BATT", 1.30),
            ("TOPIC_AGENTIC_AI", 1.55),
            ("TOPIC_NEUROMORPHIC", 1.15),
            ("TOPIC_EVTOL_AERO", 1.25),
            ("TOPIC_SYNTHETIC_BIO", 1.20),
        ]
        for tid, growth_rate in topics:
            base_papers = 100
            for idx, yr in enumerate(range(2018, 2025)):
                papers = int(base_papers * (growth_rate ** idx))
                conn.execute("""
                    INSERT INTO tech_metrics
                    (topic_id, as_of_date, published_at, papers_count, citations_count, active_institutions, github_stars, github_forks, commit_velocity, entropy_score, novelty_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    tid, date(yr, 12, 31), datetime(yr, 12, 31, 12, 0, 0),
                    papers, papers * 5, 20 + idx * 10, 500 + idx * 300, 100 + idx * 50,
                    10.0 + idx * 2.0, 0.55 + (idx % 3) * 0.05, 0.60 + (idx % 2) * 0.08
                ])

        # 4. 插入企业财务指标用于监控巡检
        conn.execute("""
            INSERT INTO company_financials
            (company_id, as_of_date, published_at, revenue, revenue_yoy, gross_margin, net_profit, ocf, rd_expense, rd_ratio, roic, ar_turnover_days, debt_to_asset)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, [
            "SYNTH_CO_01", date(2024, 6, 30), datetime(2024, 8, 15, 0, 0, 0),
            100.0, 0.25, 0.35, -10.0, -5.0, 20.0, 0.20, 0.05, 180.0, 0.45
        ])
    finally:
        conn.close()


def test_offline_e2e_network_block_is_active(block_external_network):
    """验证离线端到端测试环境确实阻断外部网络请求"""
    with pytest.raises(RuntimeError, match="OFFLINE_E2E_FORBIDDEN_NETWORK_CALL"):
        requests.get("https://api.openalex.org/works", timeout=1)


def test_offline_e2e_full_pipeline_with_synthetic_data(block_external_network, isolated_temp_duckdb):
    """离线端到端完整流水线验证：使用临时 DuckDB 与 TEST_ONLY 模拟数据"""
    # 1. 写入受控模拟数据
    _seed_controlled_synthetic_data(isolated_temp_duckdb)

    # 2. 宏观状态识别
    macro_df = isolated_temp_duckdb.query_df("SELECT * FROM macro_indicators ORDER BY as_of_date ASC")
    assert not macro_df.empty
    macro_model = MacroHMMRegimeModel()
    macro_status = macro_model.get_latest_regime(macro_df)
    assert "regime" in macro_status
    assert "macro_growth_multiplier" in macro_status

    # 3. 构建非对称机会候选池 (无真实生产模型)
    pool_mgr = OpportunityPoolManager()
    pool = pool_mgr.build_opportunity_pool(cutoff_date=date(2024, 12, 31))
    assert len(pool) >= 4

    for candidate in pool:
        assert candidate["composite_score"] >= 0
        # 严格校验：在未经过真实验证模型训练的情况下，正式预测相关字段必须为 None / 未校准
        assert candidate["growth_3y_multiple"] is None
        assert candidate["calibrated_prediction_probability"] is None
        assert candidate["prob_10x_super_return_pct"] is None
        assert candidate["is_calibrated"] is False

    # 4. 动态监控巡检
    monitor = SystemMonitor()
    alerts = monitor.run_daily_scan()
    assert isinstance(alerts, list)

    # 5. Point-in-Time 历史滚动回测
    bt_engine = PointInTimeBacktestEngine()
    bt_res = bt_engine.run_walk_forward_backtest(
        cutoff_date=date(2021, 12, 31),
        evaluation_date=date(2024, 12, 31)
    )
    assert "metrics" in bt_res

    # 6. 生成 14 章节智能研报 (仅在内存中验证，不改写仓库历史示范文件)
    dossier_gen = ResearchDossierGenerator()
    report_md = dossier_gen.generate_dossier(pool[0])
    assert "# 【Foresight Alpha 深度研报】" in report_md
    assert "## 1. 技术和产业定义" in report_md
    assert "## 14. 数据来源与溯源凭证" in report_md
