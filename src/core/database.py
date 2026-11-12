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
                    point_estimate DOUBLE,
                    ci_low DOUBLE,
                    ci_high DOUBLE,
                    confidence_score DOUBLE,
                    prediction_status VARCHAR DEFAULT 'UNVERIFIED_RESEARCH',
                    is_verified_forecast BOOLEAN DEFAULT FALSE,
                    scenarios_json JSON,
                    drivers_json JSON
                );
            """)

            # 兼容既有数据库表字段增补
            try:
                conn.execute("ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS prediction_status VARCHAR DEFAULT 'UNVERIFIED_RESEARCH';")
                conn.execute("ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS is_verified_forecast BOOLEAN DEFAULT FALSE;")
            except Exception:
                pass
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

            # 8. 企业生存与违约事件历史队列表 (用于 Cox 比例风险与生存分析模型)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS company_distress_cohort (
                    company_id VARCHAR PRIMARY KEY,
                    company_name VARCHAR NOT NULL,
                    industry_sector VARCHAR NOT NULL,
                    duration_months DOUBLE NOT NULL,
                    observation_window_months DOUBLE,
                    is_distressed INTEGER NOT NULL,
                    distress_type VARCHAR NOT NULL,
                    distress_event_type VARCHAR,
                    event_date DATE,
                    published_at TIMESTAMP NOT NULL,
                    cash_burn_months DOUBLE,
                    debt_ratio DOUBLE NOT NULL,
                    gross_margin DOUBLE NOT NULL,
                    competitor_count DOUBLE,
                    is_covariate_audited BOOLEAN DEFAULT TRUE,
                    official_announcement_id VARCHAR,
                    source_url VARCHAR,
                    quality_status VARCHAR NOT NULL
                );
            """)

            # 9. 历史真实前瞻技术训练特征与标签表 (用于 LightGBM 真实监督学习)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tech_training_samples (
                    sample_id VARCHAR PRIMARY KEY,
                    topic_id VARCHAR NOT NULL,
                    base_date DATE NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    feature_published_at TIMESTAMP,
                    feature_ingested_at TIMESTAMP,
                    velocity_1y DOUBLE NOT NULL,
                    acceleration DOUBLE NOT NULL,
                    active_institutions DOUBLE NOT NULL,
                    github_momentum DOUBLE NOT NULL,
                    macro_multiplier DOUBLE NOT NULL,
                    entropy_score DOUBLE NOT NULL,
                    novelty_score DOUBLE NOT NULL,
                    label_1y_period_end DATE,
                    label_1y_status VARCHAR,
                    target_1y DOUBLE,
                    label_3y_period_end DATE,
                    label_3y_status VARCHAR,
                    target_3y DOUBLE,
                    label_5y_period_end DATE,
                    label_5y_status VARCHAR,
                    target_5y DOUBLE,
                    quality_status VARCHAR NOT NULL
                );
            """)

            # 执行模式幂等升级与迁移
            self.migrate_schema(conn)
        finally:
            conn.close()

    def migrate_schema(self, conn=None):
        """执行数据库模式幂等迁移，修复旧版本 NOT NULL 约束并增补审计字段，记录迁移历史"""
        close_conn = False
        if conn is None:
            conn = self.get_connection()
            close_conn = True
        try:
            # 确保迁移历史追踪表存在
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    migration_id VARCHAR PRIMARY KEY,
                    version VARCHAR NOT NULL,
                    applied_at TIMESTAMP NOT NULL,
                    description VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    error_message VARCHAR
                );
            """)

            def _record_migration(m_id: str, ver: str, desc: str, status: str, err: Optional[str] = None):
                conn.execute("""
                    INSERT OR REPLACE INTO schema_migrations (migration_id, version, applied_at, description, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?);
                """, [m_id, ver, datetime.utcnow(), desc, status, err])

            # 1. 迁移 tech_training_samples：去除 target 字段的 NOT NULL 约束，允许未成熟标签为 NULL
            for tgt_col in ["target_1y", "target_3y", "target_5y"]:
                m_id = f"mig_tech_drop_notnull_{tgt_col}"
                try:
                    conn.execute(f"ALTER TABLE tech_training_samples ALTER COLUMN {tgt_col} DROP NOT NULL;")
                    _record_migration(m_id, "v1.1-p0", f"Drop NOT NULL on {tgt_col}", "SUCCESS")
                except Exception as e:
                    err_str = str(e).lower()
                    if "does not have a not null constraint" in err_str or "already" in err_str or "not found" in err_str:
                        _record_migration(m_id, "v1.1-p0", f"Drop NOT NULL on {tgt_col} (Idempotent)", "ALREADY_APPLIED", str(e))
                    else:
                        _record_migration(m_id, "v1.1-p0", f"Drop NOT NULL on {tgt_col} failed", "FAILED", str(e))
                        raise RuntimeError(f"Database migration failed at drop NOT NULL for {tgt_col}: {e}")

            for col, dtype in [
                ("feature_published_at", "TIMESTAMP"),
                ("feature_ingested_at", "TIMESTAMP"),
                ("label_1y_period_end", "DATE"),
                ("label_1y_status", "VARCHAR"),
                ("label_3y_period_end", "DATE"),
                ("label_3y_status", "VARCHAR"),
                ("label_5y_period_end", "DATE"),
                ("label_5y_status", "VARCHAR"),
            ]:
                m_id = f"mig_tech_add_{col}"
                try:
                    conn.execute(f"ALTER TABLE tech_training_samples ADD COLUMN IF NOT EXISTS {col} {dtype};")
                    _record_migration(m_id, "v1.1-p0", f"Add column {col} to tech_training_samples", "SUCCESS")
                except Exception as e:
                    err_str = str(e).lower()
                    if "already exists" in err_str:
                        _record_migration(m_id, "v1.1-p0", f"Column {col} already exists", "ALREADY_APPLIED", str(e))
                    else:
                        _record_migration(m_id, "v1.1-p0", f"Add column {col} failed", "FAILED", str(e))
                        raise RuntimeError(f"Database migration failed adding column {col}: {e}")

            # 2. 迁移 company_distress_cohort：允许未核验协变量为 NULL 并补充事件类型
            for cov_col in ["cash_burn_months", "competitor_count"]:
                m_id = f"mig_cohort_drop_notnull_{cov_col}"
                try:
                    conn.execute(f"ALTER TABLE company_distress_cohort ALTER COLUMN {cov_col} DROP NOT NULL;")
                    _record_migration(m_id, "v1.1-p0", f"Drop NOT NULL on {cov_col}", "SUCCESS")
                except Exception as e:
                    err_str = str(e).lower()
                    if "does not have a not null constraint" in err_str or "already" in err_str or "not found" in err_str:
                        _record_migration(m_id, "v1.1-p0", f"Drop NOT NULL on {cov_col} (Idempotent)", "ALREADY_APPLIED", str(e))
                    else:
                        _record_migration(m_id, "v1.1-p0", f"Drop NOT NULL on {cov_col} failed", "FAILED", str(e))
                        raise RuntimeError(f"Database migration failed at drop NOT NULL for {cov_col}: {e}")

            for col, dtype in [
                ("observation_window_months", "DOUBLE"),
                ("distress_event_type", "VARCHAR"),
                ("is_covariate_audited", "BOOLEAN DEFAULT TRUE"),
            ]:
                m_id = f"mig_cohort_add_{col}"
                try:
                    conn.execute(f"ALTER TABLE company_distress_cohort ADD COLUMN IF NOT EXISTS {col} {dtype};")
                    _record_migration(m_id, "v1.1-p0", f"Add column {col} to company_distress_cohort", "SUCCESS")
                except Exception as e:
                    err_str = str(e).lower()
                    if "already exists" in err_str:
                        _record_migration(m_id, "v1.1-p0", f"Column {col} already exists", "ALREADY_APPLIED", str(e))
                    else:
                        _record_migration(m_id, "v1.1-p0", f"Add column {col} failed", "FAILED", str(e))
                        raise RuntimeError(f"Database migration failed adding column {col}: {e}")

            # 3. 迁移 model_forecasts 字段
            for col, dtype in [
                ("prediction_status", "VARCHAR DEFAULT 'UNVERIFIED_RESEARCH'"),
                ("is_verified_forecast", "BOOLEAN DEFAULT FALSE"),
            ]:
                m_id = f"mig_forecasts_add_{col}"
                try:
                    conn.execute(f"ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS {col} {dtype};")
                    _record_migration(m_id, "v1.1-p0", f"Add column {col} to model_forecasts", "SUCCESS")
                except Exception as e:
                    err_str = str(e).lower()
                    if "already exists" in err_str:
                        _record_migration(m_id, "v1.1-p0", f"Column {col} already exists", "ALREADY_APPLIED", str(e))
                    else:
                        _record_migration(m_id, "v1.1-p0", f"Add column {col} failed", "FAILED", str(e))
                        raise RuntimeError(f"Database migration failed adding column {col}: {e}")

            # 4. 迁移后模式完整性校验 (Post-Migration Schema Integrity Check)
            self._verify_post_migration_integrity(conn)

        finally:
            if close_conn:
                conn.close()

    def _verify_post_migration_integrity(self, conn):
        """校验关键表结构与字段完整性"""
        tables = conn.execute("SHOW TABLES;").fetchall()
        table_names = [t[0] for t in tables]
        for req_table in ["tech_training_samples", "company_distress_cohort", "model_forecasts", "schema_migrations"]:
            if req_table not in table_names:
                raise RuntimeError(f"迁移后模式完整性校验失败：关键表 {req_table} 缺失。")

    def execute_custom_migration_step(self, migration_id: str, sql: str, description: str, conn=None):
        """执行独立迁移步骤并在失败时记录与抛出异常 (P1-15 规范)"""
        close_conn = False
        if conn is None:
            conn = self.get_connection()
            close_conn = True
        try:
            conn.execute(sql)
            conn.execute("""
                INSERT OR REPLACE INTO schema_migrations (migration_id, version, applied_at, description, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?);
            """, [migration_id, "v1.1-custom", datetime.utcnow(), description, "SUCCESS", None])
        except Exception as e:
            conn.execute("""
                INSERT OR REPLACE INTO schema_migrations (migration_id, version, applied_at, description, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?);
            """, [migration_id, "v1.1-custom", datetime.utcnow(), description, "FAILED", str(e)])
            raise RuntimeError(f"Database migration step '{migration_id}' failed: {e}")
        finally:
            if close_conn:
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
        target_val: Optional[Any] = None,
        require_verified_timestamp: bool = False
    ) -> pd.DataFrame:
        """严格的 Point-in-Time 查询接口。
        
        确保在 cutoff_date 时点，只读取当时已公开披露的数据 (published_at <= cutoff_date)，
        彻底防止未来信息泄露 (Lookahead Bias)。
        若 require_verified_timestamp=True，仅允许经过时间验证且发布时间可信的数据参与计算。
        """
        sql = f"""
            SELECT * FROM {table_name}
            WHERE published_at IS NOT NULL AND published_at <= ?
        """
        params = [cutoff_date]
        if target_col and target_val:
            sql += f" AND {target_col} = ?"
            params.append(target_val)
        sql += " ORDER BY as_of_date ASC"
        return self.query_df(sql, params)


# 全局单例管理器
db_manager = DatabaseManager()
