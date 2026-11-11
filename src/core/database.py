"""DuckDB 嵌入式数据库管理器 (DuckDB Persistence & Query Layer).

严格遵循 Point-in-Time 时间审计隔离机制。
"""

import os
import json
import duckdb
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.core.config import get_project_root, get_system_config


class DatabaseManager:
    """系统统一 DuckDB 持久层引擎"""

    def __init__(self, db_path: Optional[str] = None):
        root = get_project_root()
        if db_path is None:
            cfg = get_system_config()
            rel_path = cfg.get("storage", {}).get("database_path", "data/storage.duckdb")
            self.db_path = str(root / rel_path)
        else:
            self.db_path = str(root / db_path) if not os.path.isabs(db_path) else db_path

        # 确保数据目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_schema()

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """获取 DuckDB 连接 (支持嵌入式高性能并发读写)"""
        return duckdb.connect(self.db_path)

    def init_schema(self):
        """初始化系统核心数据表架构"""
        conn = self.get_connection()
        try:
            # 1. 原始多源数据存储 (包含溯源元数据与首次公开时间)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_records (
                    record_id VARCHAR PRIMARY KEY,
                    source_name VARCHAR NOT NULL,
                    source_url VARCHAR,
                    as_of_date DATE NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    ingested_at TIMESTAMP NOT NULL,
                    quality_status VARCHAR NOT NULL,
                    data_version VARCHAR NOT NULL,
                    raw_hash VARCHAR,
                    payload JSON
                );
            """)

            # 2. 技术主题时序指标 (学术论文、开源指标、新颖度)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tech_metrics (
                    topic_id VARCHAR NOT NULL,
                    as_of_date DATE NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    papers_count BIGINT DEFAULT 0,
                    citations_count BIGINT DEFAULT 0,
                    active_institutions INTEGER DEFAULT 0,
                    github_stars BIGINT DEFAULT 0,
                    github_forks BIGINT DEFAULT 0,
                    commit_velocity DOUBLE DEFAULT 0.0,
                    entropy_score DOUBLE DEFAULT 0.0,
                    novelty_score DOUBLE DEFAULT 0.0,
                    PRIMARY KEY (topic_id, as_of_date)
                );
            """)

            # 3. 宏观经济时序序列 (GDP, PMI, PPI, M2, 社融等)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_indicators (
                    indicator_code VARCHAR NOT NULL,
                    as_of_date DATE NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    indicator_name VARCHAR NOT NULL,
                    value DOUBLE NOT NULL,
                    unit VARCHAR,
                    source VARCHAR,
                    PRIMARY KEY (indicator_code, as_of_date)
                );
            """)

            # 4. 企业财务与经营时序
            conn.execute("""
                CREATE TABLE IF NOT EXISTS company_financials (
                    company_id VARCHAR NOT NULL,
                    as_of_date DATE NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    revenue DOUBLE,
                    revenue_yoy DOUBLE,
                    gross_margin DOUBLE,
                    net_profit DOUBLE,
                    ocf DOUBLE,
                    rd_expense DOUBLE,
                    rd_ratio DOUBLE,
                    roic DOUBLE,
                    ar_turnover_days DOUBLE,
                    debt_to_asset DOUBLE,
                    PRIMARY KEY (company_id, as_of_date)
                );
            """)

            # 5. 模型不可篡改预测历史表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_forecasts (
                    forecast_id VARCHAR PRIMARY KEY,
                    target_id VARCHAR NOT NULL,
                    target_level INTEGER NOT NULL,
                    model_name VARCHAR NOT NULL,
                    model_version VARCHAR NOT NULL,
                    as_of_data_date DATE NOT NULL,
                    prediction_created_at TIMESTAMP NOT NULL,
                    horizon_years INTEGER NOT NULL,
                    predicted_metric VARCHAR NOT NULL,
                    point_estimate DOUBLE NOT NULL,
                    ci_low DOUBLE,
                    ci_high DOUBLE,
                    confidence_score DOUBLE,
                    scenarios_json JSON,
                    drivers_json JSON
                );
            """)

            # 6. 预警事件日志
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_alerts (
                    alert_id VARCHAR PRIMARY KEY,
                    target_id VARCHAR NOT NULL,
                    target_level INTEGER NOT NULL,
                    severity VARCHAR NOT NULL,
                    rule_type VARCHAR NOT NULL,
                    title VARCHAR NOT NULL,
                    description VARCHAR NOT NULL,
                    triggered_at TIMESTAMP NOT NULL,
                    evidence_json JSON,
                    is_active BOOLEAN DEFAULT TRUE,
                    invalidation_notes VARCHAR
                );
            """)

            # 7. 历史回测结果评估日志
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    run_id VARCHAR PRIMARY KEY,
                    cutoff_date DATE NOT NULL,
                    evaluation_date DATE NOT NULL,
                    horizon_years INTEGER NOT NULL,
                    precision_3 DOUBLE,
                    precision_5 DOUBLE,
                    recall_5 DOUBLE,
                    pr_auc DOUBLE,
                    brier_score DOUBLE,
                    excess_vs_random DOUBLE,
                    excess_vs_momentum DOUBLE,
                    created_at TIMESTAMP NOT NULL
                );
            """)
        finally:
            conn.close()

    def query_df(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """执行 SQL 查询并返回 Pandas DataFrame"""
        conn = self.get_connection()
        try:
            if params:
                return conn.execute(sql, params).df()
            return conn.execute(sql).df()
        finally:
            conn.close()

    def execute(self, sql: str, params: Optional[List[Any]] = None):
        """执行更新/写入 SQL"""
        conn = self.get_connection()
        try:
            if params:
                conn.execute(sql, params)
            else:
                conn.execute(sql)
        finally:
            conn.close()

    def query_point_in_time(
        self,
        table_name: str,
        cutoff_date: date,
        target_col: Optional[str] = None,
        target_val: Optional[Any] = None
    ) -> pd.DataFrame:
        """严格的 Point-in-Time 查询接口。
        
        确保在 cutoff_date 时点，只读取当时已公开披露的数据 (published_at <= cutoff_date)，
        彻底防止未来信息泄露 (Lookahead Bias)。
        """
        sql = f"""
            SELECT * FROM {table_name}
            WHERE published_at <= ?
        """
        params = [cutoff_date]
        if target_col and target_val:
            sql += f" AND {target_col} = ?"
            params.append(target_val)
        sql += " ORDER BY as_of_date ASC"
        return self.query_df(sql, params)


# 全局单例管理器
db_manager = DatabaseManager()
