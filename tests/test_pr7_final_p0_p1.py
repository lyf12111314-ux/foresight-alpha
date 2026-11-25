"""PR #7 最终整改专项测试：P0 阻断历史归档自我认证 (P0-A..F) 与 P1 不可变技术特征版本 (P1-A..F).

说明：
- 本测试文件全部使用离线临时 DuckDB 与受控测试夹具，仅验证程序功能与安全门禁；
- 绝不将离线测试凭证或模拟数据计入真实生产级历史归档或模型样本外验证。
"""

import hashlib
import json
from datetime import date, datetime
import duckdb

import src.backtest.engine as bt_eng
import src.core.database as db_module
import src.decision_engine.opportunity_pool as opp_pool
import src.ingestion.pipeline as ing_pipe
from src.backtest.engine import PointInTimeBacktestEngine
from src.core.database import DatabaseManager
from src.core.historical_archive import HistoricalArchiveCertifier
from src.core.types import (
    CitationCountScope,
    DataQualityStatus,
    HistoricalEligibilityStatus,
    InstitutionCountScope,
)
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.ingestion.openalex import OpenAlexCollector
from src.ingestion.pipeline import IngestionPipeline


def _setup_isolated_db(tmp_path, monkeypatch, name: str = "pr7_final_test.duckdb") -> DatabaseManager:
    temp_db = DatabaseManager(db_path=str(tmp_path / name))
    monkeypatch.setattr(db_module, "db_manager", temp_db)
    monkeypatch.setattr(ing_pipe, "db_manager", temp_db)
    monkeypatch.setattr(bt_eng, "db_manager", temp_db)
    monkeypatch.setattr(opp_pool, "db_manager", temp_db)
    return temp_db


def _create_ordinary_snapshot(temp_db: DatabaseManager, snap_dir: str, payload: dict, as_of: date, prefix: str = "oa_2020"):
    collector = OpenAlexCollector(snapshot_dir=snap_dir)
    raw_bytes = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    meta = collector.save_raw_response_snapshot(
        snapshot_dir=snap_dir,
        request_url="https://api.openalex.org/works",
        query_params={"search": '"solid-state battery"', "filter": f"publication_year:{as_of.year}"},
        http_status=200,
        raw_bytes=raw_bytes,
        as_of_date=as_of,
        verified_first_available_at=datetime(as_of.year, 12, 31, 8, 0, 0),
        quality_status=DataQualityStatus.VERIFIED,
        historical_eligibility=HistoricalEligibilityStatus.STRICT_PIT_VERIFIED,
        file_prefix=prefix,
    )
    return collector, meta, raw_bytes


# =============================================================================
# P0-A ~ P0-F：彻底阻断历史归档自我认证
# =============================================================================

def test_p0_a_caller_forged_authority_and_flags_rejected(tmp_path, monkeypatch):
    """P0-A: 调用方直接传入 STRICT_PIT_VERIFIED、VERIFIED_HISTORICAL_ARCHIVE 与伪造权威来源名，不能获得正式资格"""
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    snap_dir = str(tmp_path / "p0a_snaps")
    collector, meta, raw_bytes = _create_ordinary_snapshot(
        temp_db, snap_dir, {"meta": {"count": 1841}, "group_by": [{"key": "2020", "count": 1841}]}, date(2020, 12, 31), "p0a"
    )
    temp_db.save_raw_snapshot(meta)

    forged_record = {
        "snapshot_id": meta["snapshot_id"],
        "as_of_date": date(2020, 12, 31),
        "verified_first_available_at": datetime(2020, 12, 31, 0, 0, 0),
        "historical_eligibility": HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value,
        "source_authenticity_status": "VERIFIED_HISTORICAL_ARCHIVE",
        "vintage_evidence_source": "OpenAlex_Official_2020_Archive_Release",
        "raw_sha256": meta["raw_sha256"],
    }
    res = collector.evaluate_historical_pit_eligibility(forged_record, cutoff_date=date(2020, 12, 31), db=temp_db)
    assert res["eligible"] is False
    assert res["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value

    # 即便直接调用生产认证入口传入伪造权威名，也因不在受信任外部权威注册表中而被拒绝
    arch_file = tmp_path / "p0a_archive.json"
    arch_file.write_bytes(raw_bytes)
    prod_attempt = HistoricalArchiveCertifier.certify_snapshot_production(
        db=temp_db,
        snapshot_id=meta["snapshot_id"],
        archive_id="FAKE-DOI-2020",
        archive_source_authority="OpenAlex_Official_2020_Archive_Release",
        archive_file_path=str(arch_file),
        verified_published_at=datetime(2020, 12, 31, 8, 0, 0),
    )
    assert prod_attempt["certified"] is False
    assert prod_attempt["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value


def test_p0_b_fabricated_2020_archive_via_save_raw_snapshot_cannot_certify(tmp_path, monkeypatch):
    """P0-B: 向普通数据保存接口提交一份虚构的 2020 年归档文件，即使 SHA-256 正确，也不能自动认定为 2020 年实际公开版本"""
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    snap_dir = str(tmp_path / "p0b_snaps")
    collector, meta, raw_bytes = _create_ordinary_snapshot(
        temp_db, snap_dir, {"meta": {"count": 1807}, "group_by": [{"key": "2020", "count": 1807}]}, date(2020, 12, 31), "p0b"
    )
    # 第一层采集器保存时已降权，绝不写出 STRICT_PIT_VERIFIED
    assert meta["historical_eligibility"] != HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value
    assert meta["verified_first_available_at"] is None
    assert meta["raw_sha256"] == hashlib.sha256(raw_bytes).hexdigest()

    # 调用方在入库前试图手动把字典改为 STRICT_PIT_VERIFIED + VERIFIED_HISTORICAL_ARCHIVE
    meta["historical_eligibility"] = HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value
    meta["source_authenticity_status"] = "VERIFIED_HISTORICAL_ARCHIVE"
    meta["verified_first_available_at"] = datetime(2020, 12, 31, 10, 0, 0)
    meta["vintage_evidence_source"] = "OpenAlex_Official_2020_Archive_Release"
    temp_db.save_raw_snapshot(meta)

    # save_raw_snapshot 必须强制降权，不得写入 STRICT_PIT_VERIFIED / VERIFIED_HISTORICAL_ARCHIVE / verified_first_available_at
    stored = temp_db.get_raw_snapshot(meta["snapshot_id"])
    assert stored["historical_eligibility"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value
    assert stored["source_authenticity_status"] == "UNVERIFIED_CALLER_CLAIM"
    assert stored["verified_first_available_at"] is None or str(stored["verified_first_available_at"]) == "NaT"

    res = collector.evaluate_historical_pit_eligibility({"snapshot_id": meta["snapshot_id"]}, cutoff_date=date(2020, 12, 31), db=temp_db)
    assert res["eligible"] is False
    assert res["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value


def test_p0_c_direct_db_tampering_cannot_bypass_controlled_certification(tmp_path, monkeypatch):
    """P0-C: 直接修改数据库 raw_snapshots 认证字段或伪插 historical_archive_attestations 记录，均不能绕过受控认证流程"""
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    snap_dir = str(tmp_path / "p0c_snaps")
    collector, meta, raw_bytes = _create_ordinary_snapshot(
        temp_db, snap_dir, {"meta": {"count": 1807}, "group_by": [{"key": "2020", "count": 1807}]}, date(2020, 12, 31), "p0c"
    )
    temp_db.save_raw_snapshot(meta)

    # 攻击 1：直接 SQL UPDATE 篡改 raw_snapshots 的来源认证与历史资格列
    temp_db.execute(
        """
        UPDATE raw_snapshots
        SET historical_eligibility = 'STRICT_PIT_VERIFIED',
            source_authenticity_status = 'VERIFIED_HISTORICAL_ARCHIVE',
            vintage_evidence_source = 'OpenAlex_Official_2020_Archive_Release',
            quality_status = 'VERIFIED',
            verified_first_available_at = ?
        WHERE snapshot_id = ?;
        """,
        [datetime(2020, 12, 31, 8, 0, 0), meta["snapshot_id"]],
    )
    res_sql_raw = collector.evaluate_historical_pit_eligibility({"snapshot_id": meta["snapshot_id"]}, cutoff_date=date(2020, 12, 31), db=temp_db)
    assert res_sql_raw["eligible"] is False
    assert res_sql_raw["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value

    # 攻击 2：绕过 HistoricalArchiveCertifier 直接向 historical_archive_attestations 写入伪造认证与伪造封签
    arch_file = tmp_path / "p0c_arch.json"
    arch_file.write_bytes(raw_bytes)
    temp_db.save_historical_archive_attestation(
        {
            "attestation_id": "att_forged_sql_001",
            "snapshot_id": meta["snapshot_id"],
            "archive_id": "FORGED-ARCHIVE-2020",
            "archive_source_authority": "OpenAlex_Official_2020_Archive_Release",
            "archive_file_path": str(arch_file),
            "archive_file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "snapshot_sha256": meta["raw_sha256"],
            "verified_published_at": datetime(2020, 12, 31, 8, 0, 0),
            "as_of_date": date(2020, 12, 31),
            "evidence_origin": "direct_sql_forged",
            "certification_mode": "PRODUCTION_VERIFIED_ARCHIVE",
            "is_test_only": False,
            "verified_at_utc": datetime.utcnow(),
            "attestation_seal": "0" * 64,
        }
    )
    res_sql_att = collector.evaluate_historical_pit_eligibility(
        {"snapshot_id": meta["snapshot_id"]}, cutoff_date=date(2020, 12, 31), db=temp_db, allow_test_attestation=True
    )
    assert res_sql_att["eligible"] is False
    assert res_sql_att["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value
    assert "封签" in res_sql_att["reason"]


def test_p0_d_modified_original_content_invalidates_attestation(tmp_path, monkeypatch):
    """P0-D: 原始内容（快照文件或归档文件）在认证后被修改，正式认证必须立即失效"""
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    snap_dir = str(tmp_path / "p0d_snaps")
    collector, meta, raw_bytes = _create_ordinary_snapshot(
        temp_db, snap_dir, {"meta": {"count": 1807}, "group_by": [{"key": "2020", "count": 1807}]}, date(2020, 12, 31), "p0d"
    )
    temp_db.save_raw_snapshot(meta)

    arch_file = tmp_path / "p0d_archive_manifest.json"
    arch_file.write_text(
        json.dumps({"archive_id": "TEST-ARCH-2020", "published_at": "2020-12-31T08:00:00", "content_sha256": meta["raw_sha256"]}),
        encoding="utf-8",
    )
    cert = HistoricalArchiveCertifier.certify_snapshot_with_test_credential(
        db=temp_db,
        snapshot_id=meta["snapshot_id"],
        archive_id="TEST-ARCH-2020",
        archive_file_path=str(arch_file),
        verified_published_at=datetime(2020, 12, 31, 8, 0, 0),
    )
    assert cert["certified"] is True
    assert collector.evaluate_historical_pit_eligibility(
        {"snapshot_id": meta["snapshot_id"]}, cutoff_date=date(2020, 12, 31), db=temp_db, allow_test_attestation=True
    )["eligible"] is True

    # 篡改原始快照文件内容 -> 认证立即失效
    with open(meta["snapshot_path"], "wb") as f:
        f.write(b'{"meta": {"count": 9999}, "group_by": [{"key": "2020", "count": 9999}]}')
    res_snap_tampered = collector.evaluate_historical_pit_eligibility(
        {"snapshot_id": meta["snapshot_id"]}, cutoff_date=date(2020, 12, 31), db=temp_db, allow_test_attestation=True
    )
    assert res_snap_tampered["eligible"] is False
    assert res_snap_tampered["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value

    # 还原快照文件，改为篡改归档清单文件 -> 认证同样立即失效
    with open(meta["snapshot_path"], "wb") as f:
        f.write(raw_bytes)
    arch_file.write_text(
        json.dumps({"archive_id": "TEST-ARCH-2020", "published_at": "2020-12-31T08:00:00", "content_sha256": "a" * 64}),
        encoding="utf-8",
    )
    res_arch_tampered = collector.evaluate_historical_pit_eligibility(
        {"snapshot_id": meta["snapshot_id"]}, cutoff_date=date(2020, 12, 31), db=temp_db, allow_test_attestation=True
    )
    assert res_arch_tampered["eligible"] is False
    assert res_arch_tampered["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value


def test_p0_e_archive_without_pre_cutoff_publication_proof_rejected(tmp_path, monkeypatch):
    """P0-E: 归档记录真实存在，但缺少发布时间或发布时间晚于预测截止日期，必须拒绝历史资格"""
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    snap_dir = str(tmp_path / "p0e_snaps")
    collector, meta, raw_bytes = _create_ordinary_snapshot(
        temp_db, snap_dir, {"meta": {"count": 1807}, "group_by": [{"key": "2020", "count": 1807}]}, date(2020, 12, 31), "p0e"
    )
    temp_db.save_raw_snapshot(meta)

    arch_file = tmp_path / "p0e_archive.json"
    arch_file.write_bytes(raw_bytes)

    # 无可核验发布时间 -> 认证阶段直接拒绝
    cert_no_time = HistoricalArchiveCertifier.certify_snapshot_with_test_credential(
        db=temp_db,
        snapshot_id=meta["snapshot_id"],
        archive_id="TEST-ARCH-NO-TIME",
        archive_file_path=str(arch_file),
        verified_published_at=None,
    )
    assert cert_no_time["certified"] is False

    # 发布时间晚于 2020-12-31 预测截止日 (如 2021-02-01 才归档公开) -> 2020-12-31 截断时必须拒绝
    cert_late = HistoricalArchiveCertifier.certify_snapshot_with_test_credential(
        db=temp_db,
        snapshot_id=meta["snapshot_id"],
        archive_id="TEST-ARCH-2021-02",
        archive_file_path=str(arch_file),
        verified_published_at=datetime(2021, 2, 1, 9, 0, 0),
    )
    assert cert_late["certified"] is True
    res_2020 = collector.evaluate_historical_pit_eligibility(
        {"snapshot_id": meta["snapshot_id"]}, cutoff_date=date(2020, 12, 31), db=temp_db, allow_test_attestation=True
    )
    assert res_2020["eligible"] is False
    assert res_2020["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value


def test_p0_f_controlled_offline_test_credential_isolated_from_production(tmp_path, monkeypatch):
    """P0-F: 受控离线测试凭证可验证正向逻辑，但绝不得进入正式生产 PIT 准入"""
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    snap_dir = str(tmp_path / "p0f_snaps")
    collector, meta, raw_bytes = _create_ordinary_snapshot(
        temp_db, snap_dir, {"meta": {"count": 1807}, "group_by": [{"key": "2020", "count": 1807}]}, date(2020, 12, 31), "p0f"
    )
    temp_db.save_raw_snapshot(meta)

    arch_manifest = tmp_path / "p0f_manifest.json"
    arch_manifest.write_text(
        json.dumps(
            {
                "archive_id": "OFFLINE-TEST-ARCHIVE-2020-12-31",
                "published_at": "2020-12-31T12:00:00",
                "files": [{"name": "2020_count.raw.json", "sha256": meta["raw_sha256"]}],
            }
        ),
        encoding="utf-8",
    )

    cert = HistoricalArchiveCertifier.certify_snapshot_with_test_credential(
        db=temp_db,
        snapshot_id=meta["snapshot_id"],
        archive_id="OFFLINE-TEST-ARCHIVE-2020-12-31",
        archive_file_path=str(arch_manifest),
        verified_published_at=datetime(2020, 12, 31, 12, 0, 0),
    )
    assert cert["certified"] is True
    assert cert["is_test_only"] is True
    assert cert["certification_mode"] == "TEST_ONLY_OFFLINE_CREDENTIAL"

    # 1) 显式启用离线测试模式 allow_test_attestation=True 时正向核验通过
    ok_test_mode = collector.evaluate_historical_pit_eligibility(
        {"snapshot_id": meta["snapshot_id"], "raw_sha256": meta["raw_sha256"]},
        cutoff_date=date(2020, 12, 31),
        db=temp_db,
        allow_test_attestation=True,
    )
    assert ok_test_mode["eligible"] is True
    assert ok_test_mode["status"] == HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value
    assert ok_test_mode["is_test_only"] is True

    # 2) 默认生产模式 (allow_test_attestation=False) 严禁将离线测试凭证视作生产级历史归档
    prod_mode = collector.evaluate_historical_pit_eligibility(
        {"snapshot_id": meta["snapshot_id"], "raw_sha256": meta["raw_sha256"]},
        cutoff_date=date(2020, 12, 31),
        db=temp_db,
    )
    assert prod_mode["eligible"] is False
    assert prod_mode["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value

    # 3) PointInTimeBacktestEngine 正式严格回测同样拒绝仅具离线测试凭证的样本
    engine = PointInTimeBacktestEngine()
    bt_model = engine.train_historical_forecaster(cutoff_date=date(2020, 12, 31), horizon_years=3, require_verified=True)
    assert bt_model.model_stage.value == "UNTRAINED"


# =============================================================================
# P1-A ~ P1-F：技术特征不可变历史版本保留、旧库迁移与预测版本关联
# =============================================================================

def _make_mock_openalex_rec(pipe: IngestionPipeline, papers_val: int, pub_dt: datetime, tag: str) -> dict:
    raw_bytes = json.dumps({"meta": {"count": papers_val}, "group_by": [{"key": "2020", "count": papers_val}]}).encode("utf-8")
    meta = pipe.openalex.save_raw_response_snapshot(
        snapshot_dir=pipe.openalex.snapshot_dir,
        request_url="https://api.openalex.org/works",
        query_params={"search": '"solid-state battery"', "year": 2020, "tag": tag},
        http_status=200,
        raw_bytes=raw_bytes,
        as_of_date=date(2020, 12, 31),
        verified_first_available_at=None,
        quality_status=DataQualityStatus.UNVERIFIED,
        historical_eligibility=HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS,
        ingested_at_utc=pub_dt,
        file_prefix=f"ssb_2020_{tag}",
    )
    return {
        "status": "OK",
        "as_of_date": date(2020, 12, 31),
        "published_at": pub_dt,
        "papers_count": papers_val,
        "citations_count": None,
        "sampled_citations_count": 419 if papers_val == 1841 else 425,
        "citation_count_scope": CitationCountScope.SAMPLED_TOP_CITED_BIASED.value,
        "active_institutions": None,
        "sampled_institutions_count": 32,
        "institution_count_scope": InstitutionCountScope.SAMPLED_WORKS_ONLY.value,
        "citations_cutoff_year": 2020,
        "historical_eligibility": HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value,
        "snapshot_id": meta["snapshot_id"],
        "snapshot_meta": meta,
        "snapshot_metas": [meta],
        "feature_lineage": {
            "papers_count": {"snapshot_id": meta["snapshot_id"], "raw_sha256": meta["raw_sha256"], "value": papers_val},
            "citations_count": {"scope": CitationCountScope.SAMPLED_TOP_CITED_BIASED.value, "formal_value": None},
        },
        "raw_payload": raw_bytes.decode("utf-8"),
        "provenance": {
            "record_id": "oa_ssb_2020_metrics",
            "source_name": "OpenAlex_API",
            "source_url": "https://api.openalex.org/works",
            "as_of_date": date(2020, 12, 31),
            "published_at": pub_dt,
            "ingested_at": pub_dt,
            "quality_status": DataQualityStatus.UNVERIFIED.value,
            "data_version": f"v_{tag}",
            "raw_hash": meta["raw_sha256"],
            "snapshot_id": meta["snapshot_id"],
        },
    }


def _run_two_ssb_captures(tmp_path, monkeypatch):
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    pipe = IngestionPipeline()
    pipe.openalex.snapshot_dir = str(tmp_path / "p1_snaps")
    monkeypatch.setattr(pipe.macro, "collect", lambda: {})
    monkeypatch.setattr(pipe.enterprise, "get_core_enterprise_financials", lambda: [])
    monkeypatch.setattr(pipe.enterprise, "get_historical_distress_cohort", lambda: [])
    monkeypatch.setattr(pipe.github, "search_repositories", lambda *a, **k: [])
    monkeypatch.setattr(pipe.tech_dataset, "get_historical_tech_training_samples", lambda: [])

    rec_v1 = _make_mock_openalex_rec(pipe, 1841, datetime(2026, 11, 23, 13, 37, 0), "20261123")
    rec_v2 = _make_mock_openalex_rec(pipe, 1847, datetime(2026, 11, 24, 9, 15, 0), "20261124")

    monkeypatch.setattr(pipe.openalex, "fetch_topic_research_trends", lambda q, *a, **k: [rec_v1] if "solid-state" in q else [])
    pipe.run_all(verbose=False)

    monkeypatch.setattr(pipe.openalex, "fetch_topic_research_trends", lambda q, *a, **k: [rec_v2] if "solid-state" in q else [])
    pipe.run_all(verbose=False)
    return temp_db, rec_v1, rec_v2


def test_p1_a_repeated_ingestion_preserves_both_tech_metric_versions(tmp_path, monkeypatch):
    """P1-A: 同一主题同一统计年度分两次写入 1841 与 1847，验证两个不可变版本均完整保留且不污染引用口径"""
    temp_db, rec_v1, rec_v2 = _run_two_ssb_captures(tmp_path, monkeypatch)

    versions_df = temp_db.get_tech_metric_versions("TOPIC_SOLID_STATE_BATT", date(2020, 12, 31))
    assert len(versions_df) == 2
    assert list(versions_df["papers_count"]) == [1841, 1847]
    assert len(set(versions_df["feature_version_id"])) == 2
    assert len(set(versions_df["feature_payload_sha256"])) == 2
    assert list(versions_df["snapshot_id"]) == [rec_v1["snapshot_id"], rec_v2["snapshot_id"]]
    # 引用口径保持隔离：正式 citations_count 均为 NULL，样本引用存 sampled_citations_count
    assert versions_df["citations_count"].isna().all()
    assert list(versions_df["sampled_citations_count"]) == [419, 425]
    assert all(s == CitationCountScope.SAMPLED_TOP_CITED_BIASED.value for s in versions_df["citation_count_scope"])


def test_p1_b_current_research_query_reads_latest_feature_version(tmp_path, monkeypatch):
    """P1-B: 当前研究查询能够读取最新版本 (1847) 并携带对应 latest_feature_version_id"""
    temp_db, _rec_v1, rec_v2 = _run_two_ssb_captures(tmp_path, monkeypatch)
    versions_df = temp_db.get_tech_metric_versions("TOPIC_SOLID_STATE_BATT", date(2020, 12, 31))
    v2_id = versions_df.iloc[1]["feature_version_id"]

    latest = temp_db.get_latest_tech_metric("TOPIC_SOLID_STATE_BATT", date(2020, 12, 31))
    assert latest is not None
    assert latest["papers_count"] == 1847
    assert latest["snapshot_id"] == rec_v2["snapshot_id"]
    assert latest["latest_feature_version_id"] == v2_id


def test_p1_c_historical_reproduction_reads_exact_old_feature_version(tmp_path, monkeypatch):
    """P1-C: 指定旧 feature_version_id 复现时，准确还原 1841 与原始快照关联，绝不自动切换至新版本 1847"""
    temp_db, rec_v1, _rec_v2 = _run_two_ssb_captures(tmp_path, monkeypatch)
    versions_df = temp_db.get_tech_metric_versions("TOPIC_SOLID_STATE_BATT", date(2020, 12, 31))
    v1_id = versions_df.iloc[0]["feature_version_id"]

    reproduced = temp_db.get_tech_metric_by_version(v1_id)
    assert reproduced is not None
    assert reproduced["feature_version_id"] == v1_id
    assert reproduced["papers_count"] == 1841
    assert reproduced["snapshot_id"] == rec_v1["snapshot_id"]
    src_snaps = json.loads(reproduced["source_snapshot_ids"])
    assert rec_v1["snapshot_id"] in src_snaps
    lineage = json.loads(reproduced["feature_lineage_json"])
    assert lineage["papers_count"]["value"] == 1841


def test_p1_d_version_preservation_does_not_grant_2020_pit_eligibility(tmp_path, monkeypatch):
    """P1-D: 两个版本均为 2026 年采集的 2020 年度数据，保存不可变版本绝不自动授予 2020 年严格 PIT 资格"""
    temp_db, _rec_v1, _rec_v2 = _run_two_ssb_captures(tmp_path, monkeypatch)
    versions_df = temp_db.get_tech_metric_versions("TOPIC_SOLID_STATE_BATT", date(2020, 12, 31))

    for _, row in versions_df.iterrows():
        assert row["historical_eligibility"] == HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value
        pit_res = OpenAlexCollector.evaluate_historical_pit_eligibility(
            {"snapshot_id": row["snapshot_id"]}, cutoff_date=date(2020, 12, 31), db=temp_db
        )
        assert pit_res["eligible"] is False
        assert pit_res["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value


def test_p1_e_legacy_v11_v12_database_migration_idempotent_and_preserves_data(tmp_path):
    """P1-E: 对旧版 V1.1/V1.2 结构数据库执行迁移，原有数据完整保留并回填至 tech_metric_versions，且重复迁移幂等"""
    legacy_db_path = str(tmp_path / "legacy_v11.duckdb")
    conn = duckdb.connect(legacy_db_path)
    try:
        # 构造不含 tech_metric_versions、不含 latest_feature_version_id 的旧版表结构
        conn.execute("""
            CREATE TABLE tech_metrics (
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
        conn.execute(
            """
            INSERT INTO tech_metrics (topic_id, as_of_date, published_at, papers_count, citations_count, active_institutions)
            VALUES ('TOPIC_SOLID_STATE_BATT', DATE '2020-12-31', TIMESTAMP '2020-12-31 00:00:00', 1807, 419, 32);
            """
        )
    finally:
        conn.close()

    # 第一次初始化触发迁移与回填
    migrated_db = DatabaseManager(db_path=legacy_db_path)
    latest_row = migrated_db.get_latest_tech_metric("TOPIC_SOLID_STATE_BATT", date(2020, 12, 31))
    assert latest_row is not None
    assert latest_row["papers_count"] == 1807
    assert latest_row["latest_feature_version_id"] is not None

    v_df_1 = migrated_db.get_tech_metric_versions("TOPIC_SOLID_STATE_BATT", date(2020, 12, 31))
    assert len(v_df_1) == 1
    assert v_df_1.iloc[0]["papers_count"] == 1807
    assert v_df_1.iloc[0]["feature_version_id"] == latest_row["latest_feature_version_id"]

    # 第二次重复执行迁移，必须保持幂等（不报错、不重复插入版本）
    migrated_db.migrate_schema()
    v_df_2 = migrated_db.get_tech_metric_versions("TOPIC_SOLID_STATE_BATT", date(2020, 12, 31))
    assert len(v_df_2) == 1
    assert v_df_2.iloc[0]["feature_version_id"] == v_df_1.iloc[0]["feature_version_id"]


def test_p1_f_model_prediction_explicitly_linked_to_tech_feature_version(tmp_path, monkeypatch):
    """P1-F: 一次模型预测在返回结果与 model_forecasts 落库记录中均明确关联所用 feature_version_id，后续新采集不破坏复现"""
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)

    # 先写入 v1 (papers=1841)
    fv_v1 = temp_db.append_tech_metric_version(
        {
            "topic_id": "TOPIC_SOLID_STATE_BATT",
            "as_of_date": date(2020, 12, 31),
            "published_at": datetime(2026, 11, 23, 12, 0, 0),
            "ingested_at": datetime(2026, 11, 23, 12, 0, 0),
            "papers_count": 1841,
            "citations_count": None,
            "sampled_citations_count": 419,
            "citation_count_scope": CitationCountScope.SAMPLED_TOP_CITED_BIASED.value,
            "active_institutions": None,
            "institution_count_scope": InstitutionCountScope.SAMPLED_WORKS_ONLY.value,
            "snapshot_id": "snap_ssb_v1",
            "historical_eligibility": HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value,
        }
    )

    pool_mgr = OpportunityPoolManager()
    candidates = pool_mgr.build_opportunity_pool(cutoff_date=date(2026, 11, 25))
    ssb = next(c for c in candidates if c["topic_id"] == "TOPIC_SOLID_STATE_BATT")
    assert ssb["feature_version_id"] == fv_v1
    assert fv_v1 in ssb["feature_version_ids"]

    fc_df = temp_db.query_df(
        "SELECT forecast_id, feature_version_id FROM model_forecasts WHERE forecast_id = ? LIMIT 1;",
        [ssb["forecast_id"]],
    )
    assert not fc_df.empty
    linked_fv_id = fc_df.iloc[0]["feature_version_id"]
    assert linked_fv_id == fv_v1

    # 后续再采集写入 v2 (papers=1847)，不会破坏原预测所绑定的 v1 特征复现
    fv_v2 = temp_db.append_tech_metric_version(
        {
            "topic_id": "TOPIC_SOLID_STATE_BATT",
            "as_of_date": date(2020, 12, 31),
            "published_at": datetime(2026, 11, 24, 12, 0, 0),
            "ingested_at": datetime(2026, 11, 24, 12, 0, 0),
            "papers_count": 1847,
            "citations_count": None,
            "sampled_citations_count": 425,
            "citation_count_scope": CitationCountScope.SAMPLED_TOP_CITED_BIASED.value,
            "active_institutions": None,
            "institution_count_scope": InstitutionCountScope.SAMPLED_WORKS_ONLY.value,
            "snapshot_id": "snap_ssb_v2",
            "historical_eligibility": HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value,
        }
    )
    assert fv_v2 != fv_v1
    reproduced_for_forecast = temp_db.get_tech_metric_by_version(linked_fv_id)
    assert reproduced_for_forecast["papers_count"] == 1841
    assert reproduced_for_forecast["snapshot_id"] == "snap_ssb_v1"
