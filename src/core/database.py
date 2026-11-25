"""DuckDB 嵌入式数据库管理器 (DuckDB Persistence & Query Layer).

严格遵循 Point-in-Time 时间审计隔离机制。
"""

import os
import json
import hashlib
import uuid
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
                    payload JSON,
                    snapshot_id VARCHAR,
                    logical_record_id VARCHAR,
                    ingestion_version INTEGER DEFAULT 1
                );
            """)

            # 2. 技术主题时序指标 (学术论文、开源指标、新颖度与完整特征溯源)
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
                    snapshot_id VARCHAR,
                    historical_eligibility VARCHAR,
                    institution_count_scope VARCHAR,
                    citations_cutoff_year INTEGER,
                    github_pit_eligible BOOLEAN,
                    sampled_citations_count BIGINT,
                    citation_count_scope VARCHAR,
                    sampled_institutions_count INTEGER,
                    feature_lineage_json JSON,
                    latest_feature_version_id VARCHAR,
                    PRIMARY KEY (topic_id, as_of_date)
                );
            """)

            # 2.1 不可变技术特征历史版本表 (V1.2 P1: 禁止后续采集覆盖旧特征版本)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tech_metric_versions (
                    feature_version_id VARCHAR PRIMARY KEY,
                    topic_id VARCHAR NOT NULL,
                    as_of_date DATE NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    ingested_at TIMESTAMP NOT NULL,
                    feature_computed_at TIMESTAMP NOT NULL,
                    processor_code_version VARCHAR NOT NULL,
                    feature_lineage_json JSON,
                    source_snapshot_ids JSON,
                    papers_count BIGINT,
                    citations_count BIGINT,
                    sampled_citations_count BIGINT,
                    citation_count_scope VARCHAR,
                    active_institutions INTEGER,
                    sampled_institutions_count INTEGER,
                    institution_count_scope VARCHAR,
                    github_stars BIGINT,
                    github_forks BIGINT,
                    commit_velocity DOUBLE,
                    entropy_score DOUBLE,
                    novelty_score DOUBLE,
                    github_pit_eligible BOOLEAN,
                    snapshot_id VARCHAR,
                    citations_cutoff_year INTEGER,
                    historical_eligibility VARCHAR,
                    feature_payload_sha256 VARCHAR NOT NULL
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
                    drivers_json JSON,
                    feature_version_id VARCHAR
                );
            """)

            # 兼容既有数据库表字段增补
            try:
                conn.execute("ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS prediction_status VARCHAR DEFAULT 'UNVERIFIED_RESEARCH';")
                conn.execute("ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS is_verified_forecast BOOLEAN DEFAULT FALSE;")
                conn.execute("ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS feature_version_id VARCHAR;")
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
                    created_at TIMESTAMP NOT NULL,
                    backtest_type VARCHAR DEFAULT 'ML_PREDICTION_BACKTEST',
                    forecast_status VARCHAR,
                    historical_model_version VARCHAR,
                    dataset_version VARCHAR,
                    evaluable_sample_count INTEGER,
                    unevaluable_sample_count INTEGER,
                    validation_status VARCHAR
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

            # 10. 原始 API 响应快照与三层真实性存证表 (V1.2 P0)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_snapshots (
                    snapshot_id VARCHAR PRIMARY KEY,
                    source_name VARCHAR NOT NULL,
                    request_url VARCHAR NOT NULL,
                    query_params JSON,
                    http_status INTEGER,
                    ingested_at_utc TIMESTAMP NOT NULL,
                    as_of_date DATE NOT NULL,
                    verified_first_available_at TIMESTAMP,
                    quality_status VARCHAR NOT NULL,
                    raw_sha256 VARCHAR,
                    snapshot_path VARCHAR,
                    processor_code_version VARCHAR NOT NULL,
                    historical_eligibility VARCHAR NOT NULL,
                    error_message VARCHAR,
                    vintage_evidence_source VARCHAR,
                    content_integrity_verified BOOLEAN DEFAULT FALSE,
                    source_authenticity_status VARCHAR,
                    historical_availability_status VARCHAR
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

            # 4. 迁移 backtest_runs 补充回测类型、模型版本与样本审计字段
            for col, dtype in [
                ("backtest_type", "VARCHAR DEFAULT 'ML_PREDICTION_BACKTEST'"),
                ("forecast_status", "VARCHAR"),
                ("historical_model_version", "VARCHAR"),
                ("dataset_version", "VARCHAR"),
                ("evaluable_sample_count", "INTEGER"),
                ("unevaluable_sample_count", "INTEGER"),
                ("validation_status", "VARCHAR")
            ]:
                m_id = f"mig_backtest_add_{col}"
                try:
                    conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS {col} {dtype};")
                    _record_migration(m_id, "v1.1-p0", f"Add column {col} to backtest_runs", "SUCCESS")
                except Exception as e:
                    err_str = str(e).lower()
                    if "already exists" in err_str:
                        _record_migration(m_id, "v1.1-p0", f"Column {col} already exists", "ALREADY_APPLIED", str(e))
                    else:
                        _record_migration(m_id, "v1.1-p0", f"Add column {col} failed", "FAILED", str(e))

            # 4.5. V1.2 原始快照表与 tech_metrics/raw_records 审计列幂等增补
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_snapshots (
                    snapshot_id VARCHAR PRIMARY KEY,
                    source_name VARCHAR NOT NULL,
                    request_url VARCHAR NOT NULL,
                    query_params JSON,
                    http_status INTEGER,
                    ingested_at_utc TIMESTAMP NOT NULL,
                    as_of_date DATE NOT NULL,
                    verified_first_available_at TIMESTAMP,
                    quality_status VARCHAR NOT NULL,
                    raw_sha256 VARCHAR,
                    snapshot_path VARCHAR,
                    processor_code_version VARCHAR NOT NULL,
                    historical_eligibility VARCHAR NOT NULL,
                    error_message VARCHAR,
                    vintage_evidence_source VARCHAR,
                    content_integrity_verified BOOLEAN DEFAULT FALSE,
                    source_authenticity_status VARCHAR,
                    historical_availability_status VARCHAR
                );
            """)
            for col, dtype in [
                ("vintage_evidence_source", "VARCHAR"),
            ]:
                m_id = f"mig_raw_snapshots_add_{col}"
                try:
                    conn.execute(f"ALTER TABLE raw_snapshots ADD COLUMN IF NOT EXISTS {col} {dtype};")
                    _record_migration(m_id, "v1.2-p0", f"Add column {col} to raw_snapshots", "SUCCESS")
                except Exception as e:
                    err_str = str(e).lower()
                    if "already exists" in err_str:
                        _record_migration(m_id, "v1.2-p0", f"Column {col} already exists", "ALREADY_APPLIED", str(e))
                    else:
                        _record_migration(m_id, "v1.2-p0", f"Add column {col} failed", "FAILED", str(e))
                        raise RuntimeError(f"Database migration failed adding column {col} to raw_snapshots: {e}")

            for col, dtype in [
                ("snapshot_id", "VARCHAR"),
                ("historical_eligibility", "VARCHAR"),
                ("institution_count_scope", "VARCHAR"),
                ("citations_cutoff_year", "INTEGER"),
                ("github_pit_eligible", "BOOLEAN"),
                ("sampled_citations_count", "BIGINT"),
                ("citation_count_scope", "VARCHAR"),
                ("sampled_institutions_count", "INTEGER"),
                ("feature_lineage_json", "JSON"),
                ("latest_feature_version_id", "VARCHAR"),
            ]:
                m_id = f"mig_tech_metrics_add_{col}"
                try:
                    conn.execute(f"ALTER TABLE tech_metrics ADD COLUMN IF NOT EXISTS {col} {dtype};")
                    _record_migration(m_id, "v1.2-p0", f"Add column {col} to tech_metrics", "SUCCESS")
                except Exception as e:
                    err_str = str(e).lower()
                    if "already exists" in err_str:
                        _record_migration(m_id, "v1.2-p0", f"Column {col} already exists", "ALREADY_APPLIED", str(e))
                    else:
                        _record_migration(m_id, "v1.2-p0", f"Add column {col} failed", "FAILED", str(e))
                        raise RuntimeError(f"Database migration failed adding column {col} to tech_metrics: {e}")

            for col, dtype in [
                ("feature_version_id", "VARCHAR"),
            ]:
                m_id = f"mig_model_forecasts_add_{col}"
                try:
                    conn.execute(f"ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS {col} {dtype};")
                    _record_migration(m_id, "v1.2-p1", f"Add column {col} to model_forecasts", "SUCCESS")
                except Exception as e:
                    err_str = str(e).lower()
                    if "already exists" in err_str:
                        _record_migration(m_id, "v1.2-p1", f"Column {col} already exists", "ALREADY_APPLIED", str(e))
                    else:
                        _record_migration(m_id, "v1.2-p1", f"Add column {col} failed", "FAILED", str(e))
                        raise RuntimeError(f"Database migration failed adding column {col} to model_forecasts: {e}")

            for col, dtype in [
                ("snapshot_id", "VARCHAR"),
                ("logical_record_id", "VARCHAR"),
                ("ingestion_version", "INTEGER DEFAULT 1"),
            ]:
                m_id = f"mig_raw_records_add_{col}"
                try:
                    conn.execute(f"ALTER TABLE raw_records ADD COLUMN IF NOT EXISTS {col} {dtype};")
                    _record_migration(m_id, "v1.2-p0", f"Add column {col} to raw_records", "SUCCESS")
                except Exception as e:
                    err_str = str(e).lower()
                    if "already exists" in err_str:
                        _record_migration(m_id, "v1.2-p0", f"Column {col} already exists", "ALREADY_APPLIED", str(e))
                    else:
                        _record_migration(m_id, "v1.2-p0", f"Add column {col} failed", "FAILED", str(e))
                        raise RuntimeError(f"Database migration failed adding {col} to raw_records: {e}")

            # 4.6. V1.2 P0/P1 最终整改：受控历史归档认证表与不可变技术特征版本表幂等建表及旧库回填
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_archive_attestations (
                    attestation_id VARCHAR PRIMARY KEY,
                    snapshot_id VARCHAR NOT NULL,
                    archive_id VARCHAR NOT NULL,
                    archive_source_authority VARCHAR NOT NULL,
                    archive_file_path VARCHAR NOT NULL,
                    archive_file_sha256 VARCHAR NOT NULL,
                    snapshot_sha256 VARCHAR NOT NULL,
                    verified_published_at TIMESTAMP NOT NULL,
                    as_of_date DATE NOT NULL,
                    evidence_origin VARCHAR NOT NULL,
                    certification_mode VARCHAR NOT NULL,
                    is_test_only BOOLEAN DEFAULT FALSE,
                    verified_at_utc TIMESTAMP NOT NULL,
                    attestation_seal VARCHAR NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tech_metric_versions (
                    feature_version_id VARCHAR PRIMARY KEY,
                    topic_id VARCHAR NOT NULL,
                    as_of_date DATE NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    ingested_at TIMESTAMP NOT NULL,
                    feature_computed_at TIMESTAMP NOT NULL,
                    processor_code_version VARCHAR NOT NULL,
                    feature_lineage_json JSON,
                    source_snapshot_ids JSON,
                    papers_count BIGINT,
                    citations_count BIGINT,
                    sampled_citations_count BIGINT,
                    citation_count_scope VARCHAR,
                    active_institutions INTEGER,
                    sampled_institutions_count INTEGER,
                    institution_count_scope VARCHAR,
                    github_stars BIGINT,
                    github_forks BIGINT,
                    commit_velocity DOUBLE,
                    entropy_score DOUBLE,
                    novelty_score DOUBLE,
                    github_pit_eligible BOOLEAN,
                    snapshot_id VARCHAR,
                    citations_cutoff_year INTEGER,
                    historical_eligibility VARCHAR,
                    feature_payload_sha256 VARCHAR NOT NULL
                );
            """)

            # 将旧版 V1.1/V1.2 数据库中尚未绑定版本 ID 的 tech_metrics 幂等回填至 tech_metric_versions
            legacy_df = conn.execute(
                "SELECT * FROM tech_metrics WHERE latest_feature_version_id IS NULL;"
            ).df()
            if not legacy_df.empty:
                for _, lrow in legacy_df.iterrows():
                    ld = lrow.to_dict()
                    def _clean(v):
                        return None if (v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v)) else v

                    payload_hash = self._compute_feature_payload_sha256(ld)
                    as_of_str = str(_clean(ld.get("as_of_date")))[:10]
                    fv_id = f"tmv_migrated_{ld['topic_id']}_{as_of_str}_{payload_hash[:12]}"
                    pub_ts = _clean(ld.get("published_at")) or datetime.utcnow()
                    snap_list = [_clean(ld.get("snapshot_id"))] if _clean(ld.get("snapshot_id")) else []
                    conn.execute("""
                        INSERT INTO tech_metric_versions (
                            feature_version_id, topic_id, as_of_date, published_at, ingested_at,
                            feature_computed_at, processor_code_version, feature_lineage_json,
                            source_snapshot_ids, papers_count, citations_count, sampled_citations_count,
                            citation_count_scope, active_institutions, sampled_institutions_count,
                            institution_count_scope, github_stars, github_forks, commit_velocity,
                            entropy_score, novelty_score, github_pit_eligible, snapshot_id,
                            citations_cutoff_year, historical_eligibility, feature_payload_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(feature_version_id) DO NOTHING;
                    """, [
                        fv_id,
                        ld["topic_id"],
                        _clean(ld.get("as_of_date")),
                        pub_ts,
                        pub_ts,
                        pub_ts,
                        "legacy-migrated-v1.1",
                        _clean(ld.get("feature_lineage_json")),
                        json.dumps(snap_list, ensure_ascii=False),
                        _clean(ld.get("papers_count")),
                        _clean(ld.get("citations_count")),
                        _clean(ld.get("sampled_citations_count")),
                        _clean(ld.get("citation_count_scope")),
                        _clean(ld.get("active_institutions")),
                        _clean(ld.get("sampled_institutions_count")),
                        _clean(ld.get("institution_count_scope")),
                        _clean(ld.get("github_stars")),
                        _clean(ld.get("github_forks")),
                        _clean(ld.get("commit_velocity")),
                        _clean(ld.get("entropy_score")),
                        _clean(ld.get("novelty_score")),
                        _clean(ld.get("github_pit_eligible")),
                        _clean(ld.get("snapshot_id")),
                        _clean(ld.get("citations_cutoff_year")),
                        _clean(ld.get("historical_eligibility")) or "HISTORICAL_VINTAGE_UNAVAILABLE",
                        payload_hash,
                    ])
                    conn.execute("""
                        UPDATE tech_metrics
                        SET latest_feature_version_id = ?
                        WHERE topic_id = ? AND as_of_date = ? AND latest_feature_version_id IS NULL;
                    """, [fv_id, ld["topic_id"], _clean(ld.get("as_of_date"))])
                _record_migration(
                    "mig_backfill_legacy_tech_metric_versions",
                    "v1.2-p1",
                    f"Backfilled {len(legacy_df)} legacy tech_metrics rows to tech_metric_versions",
                    "SUCCESS",
                )

            # 5. 迁移后模式完整性校验 (Post-Migration Schema Integrity Check)
            self._verify_post_migration_integrity(conn)

        finally:
            if close_conn:
                conn.close()

    def _verify_post_migration_integrity(self, conn):
        """校验关键表结构与字段完整性"""
        tables = conn.execute("SHOW TABLES;").fetchall()
        table_names = [t[0] for t in tables]
        for req_table in [
            "tech_training_samples",
            "company_distress_cohort",
            "model_forecasts",
            "schema_migrations",
            "raw_snapshots",
            "tech_metric_versions",
            "historical_archive_attestations",
        ]:
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

    def save_raw_snapshot(self, snapshot_meta: Dict[str, Any], conn=None) -> None:
        """写入原始 API 响应快照元数据至 raw_snapshots (不可覆盖既有快照记录)。

        第一层最小权限收敛 (P0)：普通快照保存接口仅保存原始证据与采集声明，
        严禁直接授予 STRICT_PIT_VERIFIED、VERIFIED_HISTORICAL_ARCHIVE 或认证历史首次公开时间。
        """
        close_conn = False
        if conn is None:
            conn = self.get_connection()
            close_conn = True
        try:
            qparams = snapshot_meta.get("query_params", {})
            qparams_json = json.dumps(qparams, sort_keys=True, ensure_ascii=False) if not isinstance(qparams, str) else qparams

            allowed_ordinary_elig = {
                "CURRENT_SNAPSHOT_HISTORICAL_STATS",
                "CURRENT_STATE_ONLY",
                "HISTORICAL_VINTAGE_UNAVAILABLE",
            }
            req_elig = str(snapshot_meta.get("historical_eligibility") or "HISTORICAL_VINTAGE_UNAVAILABLE")
            safe_elig = req_elig if req_elig in allowed_ordinary_elig else "HISTORICAL_VINTAGE_UNAVAILABLE"

            req_avail = str(snapshot_meta.get("historical_availability_status") or safe_elig)
            safe_avail = req_avail if req_avail in allowed_ordinary_elig else "HISTORICAL_VINTAGE_UNAVAILABLE"

            req_auth = str(snapshot_meta.get("source_authenticity_status") or "UNVERIFIED_EXTERNAL_API")
            safe_auth = "UNVERIFIED_CALLER_CLAIM" if req_auth == "VERIFIED_HISTORICAL_ARCHIVE" else req_auth

            conn.execute("""
                INSERT INTO raw_snapshots (
                    snapshot_id, source_name, request_url, query_params, http_status,
                    ingested_at_utc, as_of_date, verified_first_available_at, quality_status,
                    raw_sha256, snapshot_path, processor_code_version, historical_eligibility,
                    error_message, vintage_evidence_source, content_integrity_verified,
                    source_authenticity_status, historical_availability_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO NOTHING;
            """, [
                snapshot_meta["snapshot_id"],
                snapshot_meta["source_name"],
                snapshot_meta["request_url"],
                qparams_json,
                snapshot_meta.get("http_status"),
                snapshot_meta["ingested_at_utc"],
                snapshot_meta["as_of_date"],
                None,  # 普通采集严禁写入已认证的历史首次公开时间
                snapshot_meta["quality_status"],
                snapshot_meta.get("raw_sha256"),
                snapshot_meta.get("snapshot_path"),
                snapshot_meta.get("processor_code_version", "v1.2-p0-data-provenance"),
                safe_elig,
                snapshot_meta.get("error_message"),
                snapshot_meta.get("vintage_evidence_source"),
                bool(snapshot_meta.get("content_integrity_verified", False)),
                safe_auth,
                safe_avail,
            ])
        finally:
            if close_conn:
                conn.close()

    def save_historical_archive_attestation(self, att: Dict[str, Any], conn=None) -> None:
        """持久化受控历史归档核验器签发的认证记录（不可覆盖既有认证记录）。"""
        close_conn = False
        if conn is None:
            conn = self.get_connection()
            close_conn = True
        try:
            conn.execute("""
                INSERT INTO historical_archive_attestations (
                    attestation_id, snapshot_id, archive_id, archive_source_authority,
                    archive_file_path, archive_file_sha256, snapshot_sha256,
                    verified_published_at, as_of_date, evidence_origin,
                    certification_mode, is_test_only, verified_at_utc, attestation_seal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attestation_id) DO NOTHING;
            """, [
                att["attestation_id"],
                att["snapshot_id"],
                att["archive_id"],
                att["archive_source_authority"],
                att["archive_file_path"],
                att["archive_file_sha256"],
                att["snapshot_sha256"],
                att["verified_published_at"],
                att["as_of_date"],
                att["evidence_origin"],
                att["certification_mode"],
                bool(att.get("is_test_only", False)),
                att["verified_at_utc"],
                att["attestation_seal"],
            ])
        finally:
            if close_conn:
                conn.close()

    def get_historical_archive_attestation(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """按 snapshot_id 查询最新受控历史归档认证记录。"""
        df = self.query_df(
            """
            SELECT * FROM historical_archive_attestations
            WHERE snapshot_id = ?
            ORDER BY verified_at_utc DESC LIMIT 1;
            """,
            [str(snapshot_id)],
        )
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    @staticmethod
    def _compute_feature_payload_sha256(metric: Dict[str, Any]) -> str:
        """计算技术特征版本的规范内容摘要 SHA-256。"""
        def _norm(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            if isinstance(v, (date, datetime)):
                return v.isoformat()
            return v

        canonical = {
            "topic_id": _norm(metric.get("topic_id")),
            "as_of_date": _norm(metric.get("as_of_date")),
            "papers_count": _norm(metric.get("papers_count")),
            "citations_count": _norm(metric.get("citations_count")),
            "sampled_citations_count": _norm(metric.get("sampled_citations_count")),
            "citation_count_scope": _norm(metric.get("citation_count_scope")),
            "active_institutions": _norm(metric.get("active_institutions")),
            "sampled_institutions_count": _norm(metric.get("sampled_institutions_count")),
            "institution_count_scope": _norm(metric.get("institution_count_scope")),
            "entropy_score": _norm(metric.get("entropy_score")),
            "novelty_score": _norm(metric.get("novelty_score")),
            "snapshot_id": _norm(metric.get("snapshot_id")),
            "feature_lineage_json": _norm(metric.get("feature_lineage_json")),
        }
        raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def append_tech_metric_version(
        self,
        metric: Dict[str, Any],
        conn=None,
        update_latest: bool = True,
    ) -> str:
        """不可变追加技术特征历史版本至 tech_metric_versions，并按需刷新 tech_metrics 当前最新视图。"""
        close_conn = False
        if conn is None:
            conn = self.get_connection()
            close_conn = True
        try:
            topic_id = str(metric["topic_id"])
            as_of_date_val = metric["as_of_date"]
            pub_ts = metric.get("published_at") or datetime.utcnow()
            ing_ts = metric.get("ingested_at") or pub_ts
            comp_ts = metric.get("feature_computed_at") or datetime.utcnow()
            proc_ver = str(metric.get("processor_code_version") or "v1.2-p1-feature-versioning")

            lineage_val = metric.get("feature_lineage_json")
            if isinstance(lineage_val, dict):
                lineage_json = json.dumps(lineage_val, sort_keys=True, ensure_ascii=False)
            else:
                lineage_json = lineage_val

            src_snaps = metric.get("source_snapshot_ids")
            if src_snaps is None:
                src_snaps = [metric["snapshot_id"]] if metric.get("snapshot_id") else []
            src_snaps_json = (
                json.dumps(src_snaps, ensure_ascii=False)
                if not isinstance(src_snaps, str)
                else src_snaps
            )

            metric_for_hash = dict(metric)
            metric_for_hash["feature_lineage_json"] = lineage_json
            payload_sha = self._compute_feature_payload_sha256(metric_for_hash)

            existing_cnt = conn.execute(
                "SELECT COUNT(*) FROM tech_metric_versions WHERE topic_id = ? AND as_of_date = ?;",
                [topic_id, as_of_date_val],
            ).fetchone()[0]
            seq_num = int(existing_cnt or 0) + 1
            as_of_str = as_of_date_val.isoformat() if isinstance(as_of_date_val, (date, datetime)) else str(as_of_date_val)[:10]
            ts_tag = comp_ts.strftime("%Y%m%dT%H%M%S") if isinstance(comp_ts, datetime) else datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            fv_id = str(metric.get("feature_version_id") or f"tmv_{topic_id}_{as_of_str}_v{seq_num}_{ts_tag}_{payload_sha[:8]}")
            if conn.execute("SELECT 1 FROM tech_metric_versions WHERE feature_version_id = ? LIMIT 1;", [fv_id]).fetchone():
                fv_id = f"{fv_id}_{hashlib.sha256(f'{fv_id}:{datetime.utcnow().isoformat()}'.encode()).hexdigest()[:6]}"

            # 严禁在特征版本表中直接标为 STRICT_PIT_VERIFIED
            raw_elig = str(metric.get("historical_eligibility") or "HISTORICAL_VINTAGE_UNAVAILABLE")
            safe_elig = (
                "HISTORICAL_VINTAGE_UNAVAILABLE"
                if raw_elig == "STRICT_PIT_VERIFIED"
                else raw_elig
            )

            conn.execute("""
                INSERT INTO tech_metric_versions (
                    feature_version_id, topic_id, as_of_date, published_at, ingested_at,
                    feature_computed_at, processor_code_version, feature_lineage_json,
                    source_snapshot_ids, papers_count, citations_count, sampled_citations_count,
                    citation_count_scope, active_institutions, sampled_institutions_count,
                    institution_count_scope, github_stars, github_forks, commit_velocity,
                    entropy_score, novelty_score, github_pit_eligible, snapshot_id,
                    citations_cutoff_year, historical_eligibility, feature_payload_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                fv_id,
                topic_id,
                as_of_date_val,
                pub_ts,
                ing_ts,
                comp_ts,
                proc_ver,
                lineage_json,
                src_snaps_json,
                metric.get("papers_count"),
                metric.get("citations_count"),
                metric.get("sampled_citations_count"),
                metric.get("citation_count_scope"),
                metric.get("active_institutions"),
                metric.get("sampled_institutions_count"),
                metric.get("institution_count_scope"),
                metric.get("github_stars"),
                metric.get("github_forks"),
                metric.get("commit_velocity"),
                metric.get("entropy_score"),
                metric.get("novelty_score"),
                bool(metric.get("github_pit_eligible", False)),
                metric.get("snapshot_id"),
                metric.get("citations_cutoff_year"),
                safe_elig,
                payload_sha,
            ])

            if update_latest:
                conn.execute("""
                    INSERT OR REPLACE INTO tech_metrics (
                        topic_id, as_of_date, published_at, papers_count, citations_count,
                        sampled_citations_count, citation_count_scope, active_institutions,
                        sampled_institutions_count, github_stars, github_forks, commit_velocity,
                        entropy_score, novelty_score, snapshot_id, historical_eligibility,
                        institution_count_scope, citations_cutoff_year, feature_lineage_json,
                        github_pit_eligible, latest_feature_version_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    topic_id,
                    as_of_date_val,
                    pub_ts,
                    metric.get("papers_count"),
                    metric.get("citations_count"),
                    metric.get("sampled_citations_count"),
                    metric.get("citation_count_scope"),
                    metric.get("active_institutions"),
                    metric.get("sampled_institutions_count"),
                    metric.get("github_stars"),
                    metric.get("github_forks"),
                    metric.get("commit_velocity"),
                    metric.get("entropy_score"),
                    metric.get("novelty_score"),
                    metric.get("snapshot_id"),
                    safe_elig,
                    metric.get("institution_count_scope"),
                    metric.get("citations_cutoff_year"),
                    lineage_json,
                    bool(metric.get("github_pit_eligible", False)),
                    fv_id,
                ])
            return fv_id
        finally:
            if close_conn:
                conn.close()

    def get_tech_metric_versions(
        self,
        topic_id: str,
        as_of_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """按时间升序查询某技术主题(及年度)的全部不可变特征历史版本。"""
        if as_of_date is None:
            return self.query_df(
                """
                SELECT * FROM tech_metric_versions
                WHERE topic_id = ?
                ORDER BY as_of_date ASC, ingested_at ASC, feature_computed_at ASC;
                """,
                [topic_id],
            )
        return self.query_df(
            """
            SELECT * FROM tech_metric_versions
            WHERE topic_id = ? AND as_of_date = ?
            ORDER BY ingested_at ASC, feature_computed_at ASC;
            """,
            [topic_id, as_of_date],
        )

    def get_tech_metric_by_version(self, feature_version_id: str) -> Optional[Dict[str, Any]]:
        """历史研究复现：按显式 feature_version_id 精确还原当时的技术特征版本，绝不自动切换至新版本。"""
        df = self.query_df(
            "SELECT * FROM tech_metric_versions WHERE feature_version_id = ? LIMIT 1;",
            [str(feature_version_id)],
        )
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_latest_tech_metric(
        self,
        topic_id: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[Dict[str, Any]]:
        """当前研究查询：读取当前最新的技术特征版本记录。"""
        if as_of_date is not None:
            df = self.query_df(
                "SELECT * FROM tech_metrics WHERE topic_id = ? AND as_of_date = ? LIMIT 1;",
                [topic_id, as_of_date],
            )
        else:
            df = self.query_df(
                "SELECT * FROM tech_metrics WHERE topic_id = ? ORDER BY as_of_date DESC LIMIT 1;",
                [topic_id],
            )
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def query_tech_metrics_point_in_time(
        self,
        cutoff_date: date,
        topic_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """按 cutoff_date 从不可变 tech_metric_versions 检索每个 (topic_id, as_of_date) 当时已公开的最新版本。

        同时向后兼容直接向 tech_metrics 写入但未走 append_tech_metric_version 的历史/测试数据，
        确保二次采集更新 tech_metrics.published_at 后不会遮蔽两次采集之间的历史版本。
        """
        cutoff_ts = self._resolve_cutoff_ts(cutoff_date)
        params: List[Any] = [cutoff_ts]
        topic_clause = ""
        if topic_id is not None:
            topic_clause = " AND topic_id = ?"
            params.append(topic_id)

        ver_df = self.query_df(
            f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT
                    topic_id, as_of_date, published_at, papers_count, citations_count,
                    active_institutions, github_stars, github_forks, commit_velocity,
                    entropy_score, novelty_score, snapshot_id, historical_eligibility,
                    institution_count_scope, citations_cutoff_year, github_pit_eligible,
                    sampled_citations_count, citation_count_scope, sampled_institutions_count,
                    feature_lineage_json, feature_version_id AS latest_feature_version_id,
                    feature_version_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY topic_id, as_of_date
                        ORDER BY ingested_at DESC, feature_computed_at DESC
                    ) AS rn
                FROM tech_metric_versions
                WHERE published_at IS NOT NULL AND published_at <= ? {topic_clause}
            ) t
            WHERE rn = 1
            ORDER BY as_of_date ASC;
            """,
            params,
        )

        tm_df = self.query_point_in_time(
            "tech_metrics",
            cutoff_date,
            target_col="topic_id" if topic_id is not None else None,
            target_val=topic_id,
        )
        if ver_df.empty:
            return tm_df
        if tm_df.empty:
            return ver_df

        seen_keys = {
            (str(r["topic_id"]), str(r["as_of_date"])[:10])
            for _, r in ver_df.iterrows()
        }
        extra_rows = tm_df[
            ~tm_df.apply(lambda r: (str(r["topic_id"]), str(r["as_of_date"])[:10]) in seen_keys, axis=1)
        ]
        if extra_rows.empty:
            return ver_df
        combined = pd.concat([ver_df, extra_rows], ignore_index=True)
        return combined.sort_values("as_of_date").reset_index(drop=True)

    def append_raw_record(self, record: Dict[str, Any], conn=None) -> str:
        """不可变追加原始采集记录至 raw_records (严禁使用 INSERT OR REPLACE 覆盖历史采集事件)。

        兼容规则：
        - 首次出现 logical_record_id 时以 record_id = logical_record_id, ingestion_version = 1 入库，完全兼容现有按 record_id 查询；
        - 同一逻辑记录再次采集时保留旧版本，新增版本号与唯一 record_id 独立落盘。
        """
        close_conn = False
        if conn is None:
            conn = self.get_connection()
            close_conn = True
        try:
            logical_id = str(record.get("logical_record_id") or record["record_id"])
            existing = conn.execute(
                """
                SELECT record_id, COALESCE(ingestion_version, 1)
                FROM raw_records
                WHERE logical_record_id = ? OR record_id = ?;
                """,
                [logical_id, logical_id],
            ).fetchall()

            if not existing:
                assigned_id = logical_id
                version_num = 1
            else:
                max_ver = max(int(r[1]) for r in existing)
                version_num = max_ver + 1
                snap_ref = record.get("snapshot_id") or record.get("raw_hash") or uuid.uuid4().hex
                tag = str(snap_ref)[-8:] if str(snap_ref).startswith(("OPENALEX_", "GITHUB_", "openalex_")) else str(snap_ref)[:8]
                assigned_id = f"{logical_id}__v{version_num}__{tag}"
                # 确保高并发或同内容同秒多次重采下 record_id 唯一
                collision = conn.execute("SELECT 1 FROM raw_records WHERE record_id = ? LIMIT 1;", [assigned_id]).fetchone()
                if collision:
                    assigned_id = f"{logical_id}__v{version_num}__{uuid.uuid4().hex[:8]}"

            payload_val = record.get("payload")
            if payload_val is not None and not isinstance(payload_val, str):
                payload_val = json.dumps(payload_val, ensure_ascii=False, sort_keys=True, default=str)

            conn.execute("""
                INSERT INTO raw_records (
                    record_id, source_name, source_url, as_of_date, published_at,
                    ingested_at, quality_status, data_version, raw_hash, payload,
                    snapshot_id, logical_record_id, ingestion_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                assigned_id,
                record["source_name"],
                record.get("source_url"),
                record["as_of_date"],
                record["published_at"],
                record["ingested_at"],
                record["quality_status"],
                record.get("data_version", "v1.2"),
                record.get("raw_hash"),
                payload_val,
                record.get("snapshot_id"),
                logical_id,
                version_num,
            ])
            return assigned_id
        finally:
            if close_conn:
                conn.close()

    def get_raw_record_versions(
        self,
        logical_record_id: str,
        cutoff_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """按时间升序检索某一逻辑记录的全部不可变采集版本 (支持截止日期 PIT 过滤)"""
        if cutoff_date is None:
            return self.query_df(
                """
                SELECT * FROM raw_records
                WHERE logical_record_id = ? OR record_id = ?
                ORDER BY COALESCE(ingestion_version, 1) ASC, ingested_at ASC;
                """,
                [logical_record_id, logical_record_id],
            )
        cutoff_ts = self._resolve_cutoff_ts(cutoff_date)
        return self.query_df(
            """
            SELECT * FROM raw_records
            WHERE (logical_record_id = ? OR record_id = ?)
              AND published_at IS NOT NULL AND published_at <= ?
            ORDER BY COALESCE(ingestion_version, 1) ASC, ingested_at ASC;
            """,
            [logical_record_id, logical_record_id, cutoff_ts],
        )

    def get_latest_raw_record(
        self,
        logical_record_id: str,
        cutoff_date: Optional[date] = None,
    ) -> Optional[Dict[str, Any]]:
        """获取指定截止日(或当前)最新的不可变采集记录"""
        df = self.get_raw_record_versions(logical_record_id, cutoff_date=cutoff_date)
        if df.empty:
            return None
        return df.iloc[-1].to_dict()

    def get_raw_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """按 snapshot_id 查询原始快照审计记录"""
        df = self.query_df("SELECT * FROM raw_snapshots WHERE snapshot_id = ? LIMIT 1;", [snapshot_id])
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def verify_snapshot_file_integrity(self, snapshot_id: str) -> Dict[str, Any]:
        """重新读取快照原始字节计算 SHA-256，校验内容完整性是否被篡改"""
        rec = self.get_raw_snapshot(snapshot_id)
        if not rec:
            return {"verified": False, "reason": f"未找到快照记录: {snapshot_id}"}
        snap_path = rec.get("snapshot_path")
        expected_hash = rec.get("raw_sha256")
        if not snap_path or not expected_hash:
            return {"verified": False, "reason": "快照记录缺少文件路径或原始 SHA-256 摘要"}
        resolved_path = snap_path if os.path.isabs(snap_path) else os.path.join(str(get_project_root()), snap_path)
        if not os.path.exists(resolved_path):
            return {"verified": False, "reason": f"快照文件不存在: {resolved_path}"}
        with open(resolved_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        if actual_hash.lower() != str(expected_hash).lower():
            return {
                "verified": False,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "reason": "快照文件内容已被修改，SHA-256 一致性校验失败"
            }
        return {
            "verified": True,
            "snapshot_id": snapshot_id,
            "snapshot_path": resolved_path,
            "sha256": actual_hash,
            "historical_eligibility": rec.get("historical_eligibility"),
            "verified_first_available_at": rec.get("verified_first_available_at")
        }

    @staticmethod
    def _resolve_cutoff_ts(cutoff_date: date) -> datetime:
        if isinstance(cutoff_date, datetime):
            return cutoff_date
        cutoff_ts = datetime(cutoff_date.year, cutoff_date.month, cutoff_date.day, 23, 59, 59, 999999)
        if cutoff_date == date.today():
            utc_now = datetime.utcnow()
            if utc_now > cutoff_ts:
                cutoff_ts = utc_now
        return cutoff_ts

    def query_point_in_time(
        self,
        table_name: str,
        cutoff_date: date,
        target_col: Optional[str] = None,
        target_val: Optional[Any] = None,
        require_verified_timestamp: bool = False
    ) -> pd.DataFrame:
        """严格的 Point-in-Time 查询接口。
        
        确保在 cutoff_date 时点，只读取当时已公开披露的数据 (published_at <= cutoff_date 当日结束时刻)，
        彻底防止未来信息泄露 (Lookahead Bias)。
        若 require_verified_timestamp=True，仅允许经过时间验证且发布时间可信的数据参与计算。
        """
        cutoff_ts = self._resolve_cutoff_ts(cutoff_date)

        sql = f"""
            SELECT * FROM {table_name}
            WHERE published_at IS NOT NULL AND published_at <= ?
        """
        params = [cutoff_ts]
        if target_col and target_val:
            sql += f" AND {target_col} = ?"
            params.append(target_val)

        if require_verified_timestamp:
            try:
                cols_info = self.query_df(f"PRAGMA table_info('{table_name}')")
                col_names = cols_info["name"].tolist() if not cols_info.empty else []
                if "quality_status" in col_names:
                    sql += " AND quality_status = 'VERIFIED'"
                elif "is_verified" in col_names:
                    sql += " AND (is_verified = TRUE OR is_verified = 1)"
                else:
                    # 表中无数据质量/核验标识，在严格校验下拒绝未核验时序数据
                    sql += " AND 1 = 0"
            except Exception as e:
                # 严格模式下严禁静默忽略元数据检查异常，防止放宽过滤导致未核验数据或未来信息泄漏
                raise RuntimeError(f"Point-in-Time 严格审计模式错误: 无法检查表 '{table_name}' 的数据质量元数据: {e}")

        try:
            cols_info = self.query_df(f"PRAGMA table_info('{table_name}')")
            col_names = cols_info["name"].tolist() if not cols_info.empty else []
            if "as_of_date" in col_names:
                sql += " ORDER BY as_of_date ASC"
            elif "base_date" in col_names:
                sql += " ORDER BY base_date ASC"
        except Exception:
            pass

        return self.query_df(sql, params)


# 全局单例管理器
db_manager = DatabaseManager()
