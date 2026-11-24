"""V1.2 第一阶段数据真实性、原始快照存证与 PIT 准入专项测试 (Tests A–G).

严格区分：
- 所有 CI 用例使用离线 Mock 与隔离临时目录/DuckDB，绝不发起公网请求，绝不混淆工程测试与真实线上数据验证。
"""

import hashlib
import json
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import requests

import src.backtest.engine as bt_eng
import src.core.database as db_module
import src.decision_engine.opportunity_pool as opp_pool
import src.ingestion.pipeline as ing_pipe
from src.core.database import DatabaseManager
from src.core.types import (
    CitationCountScope,
    DataQualityStatus,
    HistoricalEligibilityStatus,
    InstitutionCountScope,
    ModelStage,
    ModelTrainingStatus,
    OOSValidationSample,
    PredictionStatus,
    ProvenanceMetadata,
    TrustedDatasetRegistration,
)
from src.decision_engine.model_validator import TrustedValidationRegistry
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.diffusion.lgbm_forecaster import LightGBMGrowthForecaster
from src.ingestion.base import BaseCollector
from src.ingestion.github import GithubCollector
from src.ingestion.openalex import OpenAlexCollector
from src.ingestion.pipeline import IngestionPipeline
from src.backtest.engine import PointInTimeBacktestEngine
from src.tech_discovery.inflection import InflectionDetector


def _mock_json_response(payload: dict, status_code: int = 200) -> MagicMock:
    raw_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.content = raw_bytes
    resp.text = raw_bytes.decode("utf-8")
    resp.json.return_value = payload
    return resp


def _setup_isolated_db(tmp_path, monkeypatch) -> DatabaseManager:
    temp_db = DatabaseManager(db_path=str(tmp_path / "v12_p0_test.duckdb"))
    monkeypatch.setattr(db_module, "db_manager", temp_db)
    monkeypatch.setattr(ing_pipe, "db_manager", temp_db)
    monkeypatch.setattr(bt_eng, "db_manager", temp_db)
    monkeypatch.setattr(opp_pool, "db_manager", temp_db)
    return temp_db


# =============================================================================
# 测试 A：未来引用不能进入历史特征 & 机构数禁用估算公式
# =============================================================================

def test_a_future_citations_filtered_and_missing_counts_never_zeroed(tmp_path, monkeypatch):
    """2020 年论文总引用 2528，但 2020 年当年仅 107 次：必须过滤 2021+ 引用且标注历史版本限制；缺失 counts_by_year 不可填 0"""
    snap_dir = str(tmp_path / "openalex_snaps")
    collector = OpenAlexCollector(snapshot_dir=snap_dir)

    count_payload = {"meta": {"count": 1807}, "group_by": [{"key": "2020", "count": 1807}]}
    works_payload = {
        "results": [
            {
                "id": "https://openalex.org/W3005210566",
                "publication_date": "2020-02-05",
                "cited_by_count": 2528,
                "counts_by_year": [
                    {"year": 2020, "cited_by_count": 107},
                    {"year": 2021, "cited_by_count": 320},
                    {"year": 2022, "cited_by_count": 480},
                    {"year": 2026, "cited_by_count": 621},
                ],
                "authorships": [
                    {"institutions": [{"id": "https://openalex.org/I1", "display_name": "Inst A"}]},
                    {"institutions": [{"id": "https://openalex.org/I2", "display_name": "Inst B"}]},
                ],
            },
            {
                "id": "https://openalex.org/W_MISSING_COUNTS",
                "publication_date": "2020-06-01",
                "cited_by_count": 500,
                "counts_by_year": None,  # 缺失按年拆分引用，不得回退到 500 也不得填 0
                "authorships": [
                    {"institutions": [{"id": "https://openalex.org/I2", "display_name": "Inst B"}]},
                    {"institutions": [{"ror": "https://ror.org/03", "display_name": "Inst C"}]},
                ],
            },
        ]
    }
    inst_agg_truncated = {
        "meta": {"count": 1807, "groups_count": 200},
        "group_by": [{"key": f"https://openalex.org/I{i}", "count": 5} for i in range(200)],
    }

    def fake_fetch(url, params=None, **kwargs):
        gb = (params or {}).get("group_by")
        if gb == "publication_year":
            return _mock_json_response(count_payload, 200), 200, None
        if gb == "authorships.institutions.id":
            return _mock_json_response(inst_agg_truncated, 200), 200, None
        return _mock_json_response(works_payload, 200), 200, None

    monkeypatch.setattr(collector, "fetch_raw_with_audit", fake_fetch)
    rows = collector.fetch_topic_research_trends('"solid-state battery"', start_year=2020, end_year=2020, save_snapshots=True)
    assert len(rows) == 1
    rec = rows[0]

    assert rec["status"] == "OK"
    assert rec["papers_count"] == 1807
    # 严禁将仅来自 2 篇高被引样本的 107 当作全领域 1807 篇总引用；仅写入 sampled_citations_count 并标注采样口径
    assert rec["citations_count"] is None
    assert rec["sampled_citations_count"] == 107
    assert rec["citation_count_scope"] == CitationCountScope.SAMPLED_TOP_CITED_BIASED.value
    assert rec["citations_cutoff_year"] == 2020
    assert rec["works_with_pit_counts"] == 1
    assert rec["works_missing_pit_counts"] == 1
    assert set(rec["feature_lineage"].keys()) >= {"papers_count", "citations_count", "active_institutions"}

    # 机构数去重严格为 3 (I1, I2, ror/03)，因 group_by 触及 200 上限且仅部分采样，绝不启用 *12/*0.15 估算，完整口径字段为 None
    assert rec["sampled_institutions_count"] == 3
    assert rec["institution_count_scope"] == InstitutionCountScope.SAMPLED_WORKS_ONLY.value
    assert rec["active_institutions"] is None

    # 即使过滤了未来年份引用，仍显式标记为当前快照而非 2020 历史快照
    assert rec["historical_eligibility"] == HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value
    assert rec["verified_first_available_at"] is None
    assert "后续索引补录" in rec["limitation_note"]

    # 单独验证缺失/空异常 counts_by_year 绝不填零
    c_miss, st_miss = OpenAlexCollector.extract_pit_citations_from_work({"cited_by_count": 200, "counts_by_year": []}, 2020)
    assert c_miss is None
    assert st_miss == "CITATIONS_BY_YEAR_INCONSISTENT_EMPTY"


def test_a_complete_institution_aggregation_accepted_below_cap(tmp_path, monkeypatch):
    """当 OpenAlex group_by=authorships.institutions.id 未触及 200 上限时，准确统计完整去重机构数"""
    collector = OpenAlexCollector(snapshot_dir=str(tmp_path / "snaps"))

    def fake_fetch(url, params=None, **kwargs):
        gb = (params or {}).get("group_by")
        if gb == "publication_year":
            return _mock_json_response({"meta": {"count": 50}, "group_by": [{"key": "2020", "count": 50}]}, 200), 200, None
        if gb == "authorships.institutions.id":
            groups = [{"key": f"https://openalex.org/I{i}", "count": 2} for i in range(18)]
            groups.append({"key": "unknown", "count": 3})
            return _mock_json_response({"meta": {"count": 50, "groups_count": 19}, "group_by": groups}, 200), 200, None
        return _mock_json_response({"results": [{"id": "W1", "cited_by_count": 10, "counts_by_year": [{"year": 2020, "cited_by_count": 4}]}]}, 200), 200, None

    monkeypatch.setattr(collector, "fetch_raw_with_audit", fake_fetch)
    rows = collector.fetch_topic_research_trends("niche-topic", start_year=2020, end_year=2020, save_snapshots=True)
    assert rows[0]["institution_count_scope"] == InstitutionCountScope.COMPLETE_AGGREGATION.value
    assert rows[0]["active_institutions"] == 18


# =============================================================================
# 测试 B：原始快照内容被修改时 SHA-256 校验必须失败
# =============================================================================

def test_b_tampered_snapshot_fails_integrity_verification(tmp_path, monkeypatch):
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    snap_dir = str(tmp_path / "snaps")
    collector = OpenAlexCollector(snapshot_dir=snap_dir)

    raw_bytes = json.dumps({"group_by": [{"key": "2020", "count": 1807}]}).encode("utf-8")
    meta1 = collector.save_raw_response_snapshot(
        snapshot_dir=snap_dir,
        request_url="https://api.openalex.org/works",
        query_params={"search": '"solid-state battery"', "filter": "publication_year:2020"},
        http_status=200,
        raw_bytes=raw_bytes,
        as_of_date=date(2020, 12, 31),
        verified_first_available_at=None,
        quality_status=DataQualityStatus.UNVERIFIED,
        historical_eligibility=HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS,
        file_prefix="pilot_test",
    )
    # 相同内容再次采集：不覆盖历史采集记录，产生独立 snapshot_id 且保留两份元数据
    meta2 = collector.save_raw_response_snapshot(
        snapshot_dir=snap_dir,
        request_url="https://api.openalex.org/works",
        query_params={"search": '"solid-state battery"', "filter": "publication_year:2020"},
        http_status=200,
        raw_bytes=raw_bytes,
        as_of_date=date(2020, 12, 31),
        verified_first_available_at=None,
        quality_status=DataQualityStatus.UNVERIFIED,
        historical_eligibility=HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS,
        file_prefix="pilot_test",
    )
    assert meta1["snapshot_id"] != meta2["snapshot_id"]
    assert meta1["raw_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert meta1["content_integrity_verified"] is True
    assert meta1["source_authenticity_status"] == "REAL_HTTP_200_RESPONSE"
    assert meta1["historical_availability_status"] == HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value

    temp_db.save_raw_snapshot(meta1)
    assert temp_db.verify_snapshot_file_integrity(meta1["snapshot_id"])["verified"] is True

    # 故意篡改磁盘快照内容
    with open(meta1["snapshot_path"], "wb") as f:
        f.write(b'{"group_by": [{"key": "2020", "count": 999999}]}')

    direct_check = BaseCollector.verify_snapshot_integrity(meta1["snapshot_path"], meta1["raw_sha256"])
    db_check = temp_db.verify_snapshot_file_integrity(meta1["snapshot_id"])
    assert direct_check["content_integrity_verified"] is False
    assert db_check["verified"] is False
    assert "篡改" in db_check["reason"] or "一致" in db_check["reason"]


# =============================================================================
# 测试 C：API 超时、限流 429、非 200 与畸形格式绝不变成 papers_count=0 的正常记录
# =============================================================================

@pytest.mark.parametrize(
    "mock_behavior",
    [
        "timeout",
        "rate_limit_429",
        "server_500",
        "malformed_json",
    ],
)
def test_c_api_failures_never_become_zero_paper_observation(tmp_path, monkeypatch, mock_behavior):
    snap_dir = str(tmp_path / f"snaps_{mock_behavior}")
    collector = OpenAlexCollector(snapshot_dir=snap_dir)

    def fake_get(url, params=None, timeout=None):
        if mock_behavior == "timeout":
            raise requests.Timeout("Simulated read timeout")
        if mock_behavior == "rate_limit_429":
            r = MagicMock(spec=requests.Response)
            r.status_code = 429
            return r
        if mock_behavior == "server_500":
            r = MagicMock(spec=requests.Response)
            r.status_code = 500
            return r
        # malformed JSON (HTTP 200 但无 group_by 键)
        r = MagicMock(spec=requests.Response)
        r.status_code = 200
        r.content = b'{"unexpected": "payload"}'
        r.text = '{"unexpected": "payload"}'
        r.json.return_value = {"unexpected": "payload"}
        return r

    monkeypatch.setattr(collector.session, "get", fake_get)
    monkeypatch.setattr("src.ingestion.base.time.sleep", lambda *_args, **_kwargs: None)

    rows = collector.fetch_topic_research_trends("solid-state battery", start_year=2020, end_year=2020, save_snapshots=True)
    assert len(rows) == 1
    rec = rows[0]
    assert rec["status"] == "FETCH_FAILED"
    assert rec["papers_count"] is None
    assert rec["citations_count"] is None
    assert rec["active_institutions"] is None
    assert rec["raw_sha256"] is None
    assert rec["quality_status"] == DataQualityStatus.UNAVAILABLE.value
    assert rec["error_message"]


# =============================================================================
# 测试 D：2025 年创建的 GitHub 仓库与实时 Star 禁止进入 2020 年历史回测
# =============================================================================

def test_d_github_future_repos_and_live_stars_isolated_from_2020_backtest(tmp_path, monkeypatch):
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    gh = GithubCollector(snapshot_dir=str(tmp_path / "gh_snaps"))

    future_repo = {
        "repo_id": "913533710",
        "full_name": "stanford-developers/steer-opencell-design",
        "stars": 1200,
        "forks": 85,
        "open_issues": 10,
        "created_at_str": "2025-01-07T21:44:38Z",
        "updated_at": datetime(2026, 10, 30, 17, 27, 27),
    }
    older_repo = {
        "repo_id": "123456",
        "full_name": "legacy/old-battery-repo",
        "stars": 9500,
        "forks": 1200,
        "open_issues": 40,
        "created_at_str": "2019-05-01T10:00:00Z",
        "updated_at": datetime(2026, 11, 20, 8, 0, 0),
    }

    pit_check = gh.filter_repositories_for_historical_cutoff([future_repo, older_repo], cutoff_date=date(2020, 12, 31))
    assert pit_check["excluded_future_repos_count"] == 1
    assert pit_check["excluded_future_repos"][0]["full_name"] == "stanford-developers/steer-opencell-design"
    assert pit_check["existed_on_cutoff_count"] == 1
    # 即使 2019 年已建库，也不能使用当下 9500 Star 或线性倒推充当 2020 特征
    assert pit_check["github_momentum"] is None
    assert pit_check["existed_repos"][0]["stargazers_count"] is None
    assert pit_check["pit_eligible"] is False
    assert pit_check["historical_eligibility"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value

    # 验证 OpportunityPoolManager 对历史时点 github_pit_eligible=FALSE 不会把当下 Star 喂入模型特征
    conn = temp_db.get_connection()
    try:
        conn.execute(
            """
            INSERT INTO tech_metrics
            (topic_id, as_of_date, published_at, papers_count, citations_count, active_institutions, github_stars, github_forks, commit_velocity, entropy_score, novelty_score, github_pit_eligible)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ["TOPIC_SOLID_STATE_BATT", date(2020, 12, 31), datetime(2020, 12, 31), 1807, 419, 32, 9500, 1200, 0.0, 0.7, 0.8, False],
        )
    finally:
        conn.close()

    pool_mgr = OpportunityPoolManager()
    candidates = pool_mgr.build_opportunity_pool(cutoff_date=date(2020, 12, 31))
    ssb = next(c for c in candidates if c["topic_id"] == "TOPIC_SOLID_STATE_BATT")
    assert ssb["features"]["github_momentum"] is None


# =============================================================================
# 测试 E：2026 年采集的 2020 年论文统计无历史快照证明时，严格 PIT 验证必须拒绝
# =============================================================================

def test_e_2026_ingested_2020_stats_rejected_by_strict_2020_pit(tmp_path, monkeypatch):
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    collector = OpenAlexCollector(snapshot_dir=str(tmp_path / "snaps"))

    record_2026_for_2020 = {
        "year": 2020,
        "as_of_date": date(2020, 12, 31),
        "ingested_at_utc": datetime(2026, 11, 23, 13, 37, 14),
        "verified_first_available_at": None,
        "historical_eligibility": HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value,
        "raw_sha256": "411a2f7adeea3ef76b628eee722461dcf755f410b8da1d8324fe92119be1ae8f",
    }
    elig = collector.evaluate_historical_pit_eligibility(record_2026_for_2020, cutoff_date=date(2020, 12, 31))
    assert elig["eligible"] is False
    assert elig["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value

    # 验证严格历史回测 train_historical_forecaster(require_verified=True) 同样拒绝仅含当下统计而无 STRICT_PIT_VERIFIED 快照的数据
    engine = PointInTimeBacktestEngine()
    hist_model = engine.train_historical_forecaster(cutoff_date=date(2020, 12, 31), horizon_years=3, require_verified=True)
    assert hist_model.model_stage == ModelStage.UNTRAINED
    assert hist_model.training_status == ModelTrainingStatus.MODEL_UNAVAILABLE_AS_OF_DATE


# =============================================================================
# 测试 F：虚假血缘、临时自生成 prov_hist_ 哈希与自报审计资质一律被阻断
# =============================================================================

def test_f_fake_provenance_and_self_declared_audit_blocked(tmp_path, monkeypatch):
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    forecaster = LightGBMGrowthForecaster()

    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.random.randn(n) + 2.0, 3: np.random.randn(n) + 3.0, 5: np.random.randn(n) + 4.0}

    # F1: 非 64 位十六进制 SHA-256 的虚假血缘 (如 prov_hist_0) 必须被 fit_verified 拒绝
    fake_short_provs = [{"quality_status": "VERIFIED", "raw_hash": f"prov_hist_{i}"} for i in range(n)]
    res_f1 = forecaster.fit_verified(
        features=features,
        labels_by_horizon=labels,
        quality_status=DataQualityStatus.VERIFIED,
        provenance_records=fake_short_provs,
    )
    assert res_f1["is_verified"] is False
    assert forecaster.model_stage == ModelStage.UNTRAINED

    # F2: 即便哈希是 64 位，若历史资格仅是 CURRENT_SNAPSHOT_HISTORICAL_STATS，fit_verified 也拒绝
    ineligible_hist_provs = [
        {
            "quality_status": "VERIFIED",
            "raw_hash": "a" * 64,
            "historical_eligibility": HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value,
        }
        for _ in range(n)
    ]
    res_f2 = forecaster.fit_verified(
        features=features,
        labels_by_horizon=labels,
        quality_status=DataQualityStatus.VERIFIED,
        provenance_records=ineligible_hist_provs,
    )
    assert res_f2["is_verified"] is False
    assert forecaster.model_stage == ModelStage.UNTRAINED

    # F3: 即便库里强行写入了 quality_status='VERIFIED' 的样本，缺少 raw_records/raw_snapshots 严格历史快照证明，backtest engine 拒绝训练
    conn = temp_db.get_connection()
    try:
        for i in range(25):
            conn.execute(
                """
                INSERT INTO tech_training_samples
                (sample_id, topic_id, base_date, published_at, feature_published_at, feature_ingested_at, velocity_1y, acceleration, active_institutions, github_momentum, macro_multiplier, entropy_score, novelty_score, label_1y_period_end, label_1y_status, target_1y, label_3y_period_end, label_3y_status, target_3y, label_5y_period_end, label_5y_status, target_5y, quality_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    f"FAKE_V_{i}", f"T_{i % 4}", date(2016, 12, 31), datetime(2016, 12, 31), datetime(2016, 12, 31), datetime(2026, 11, 23),
                    10.0 + i, 2.0, 15.0, 50.0, 1.0, 0.7, 0.8,
                    date(2017, 12, 31), "MATURED", 1.5,
                    date(2019, 12, 31), "MATURED", 2.5,
                    date(2021, 12, 31), "NOT_MATURED", None,
                    "VERIFIED",
                ],
            )
    finally:
        conn.close()

    engine = PointInTimeBacktestEngine()
    bt_forecaster = engine.train_historical_forecaster(cutoff_date=date(2020, 12, 31), horizon_years=3, require_verified=True)
    assert bt_forecaster.model_stage == ModelStage.UNTRAINED
    assert "HISTORICAL_VINTAGE_UNAVAILABLE" in bt_forecaster.rejection_reason

    # F4: TrustedValidationRegistry 不得仅凭 self 声明或伪造权威名称就授权生产准入
    sample = OOSValidationSample(
        sample_id="SELF_CERT_0",
        dataset_version="v1.2-self",
        feature_snapshot_id="snap_self",
        decision_date=date(2021, 1, 1),
        feature_available_at=datetime(2020, 12, 31),
        label_period_end=date(2022, 1, 1),
        label_observed_at=datetime(2022, 1, 2),
        label_source="Self",
        label_quality_status=DataQualityStatus.VERIFIED,
        source_record_id="rec_self",
        data_quality_status=DataQualityStatus.VERIFIED,
        features={k: 1.0 for k in forecaster.feature_names},
        labels={1: 1.5, 3: 2.5, 5: 3.5},
    )
    reg_self = TrustedDatasetRegistration(
        dataset_id="REG_SELF_DECLARED_AUTH",
        dataset_version="v1.2-self",
        audit_status=DataQualityStatus.VERIFIED,
        audit_authority="self",
        is_audited_production_ready=True,
        has_independent_audit_certificate=True,
        samples=[sample],
    )
    TrustedValidationRegistry.register(reg_self)
    stored = TrustedValidationRegistry.get("REG_SELF_DECLARED_AUTH")
    assert stored.is_audited_production_ready is False
    assert stored.audit_status == DataQualityStatus.UNVERIFIED


# =============================================================================
# 测试 G：特征缺失时禁止用固定默认值制造正式预测，且 is_verified 契约正确传递
# =============================================================================

def test_g_missing_features_refused_in_formal_mode_and_is_verified_contract(tmp_path, monkeypatch):
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)

    # 构造带完整性缺失 (active_institutions IS NULL, github_stars IS NULL) 的技术指标
    conn = temp_db.get_connection()
    try:
        for yr in [2018, 2019, 2020]:
            conn.execute(
                """
                INSERT INTO tech_metrics
                (topic_id, as_of_date, published_at, papers_count, citations_count, active_institutions, github_stars, github_forks, commit_velocity, entropy_score, novelty_score, github_pit_eligible)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ["TOPIC_SOLID_STATE_BATT", date(yr, 12, 31), datetime(yr, 12, 31), 900 + yr, 200, None, None, None, None, None, None, False],
            )
    finally:
        conn.close()

    # 构造一个标记为 VERIFIED_PRODUCTION 的测试器验证正式预测阶段遇特征缺失必须拒绝
    forecaster = LightGBMGrowthForecaster()
    n = 25
    features = np.random.randn(n, 7)
    labels = {1: np.ones(n) * 1.5, 3: np.ones(n) * 2.5, 5: np.ones(n) * 3.5}
    provs = [
        ProvenanceMetadata(
            record_id=f"VER_{i}",
            source_name="Audit_Vault",
            as_of_date=date(2019, 1, 1),
            published_at=datetime(2019, 1, 2),
            quality_status=DataQualityStatus.VERIFIED,
            raw_hash="b" * 64,
        )
        for i in range(n)
    ]
    forecaster.fit_verified(features, labels, quality_status=DataQualityStatus.VERIFIED, provenance_records=provs)
    # 模拟已通过独立验证进入 VERIFIED_PRODUCTION（仅测试下游特征缺失拦截逻辑）
    forecaster.model_stage = ModelStage.VERIFIED_PRODUCTION
    forecaster.is_out_of_sample_validated = True

    pool_mgr = OpportunityPoolManager(lgbm_forecaster=forecaster)
    pool = pool_mgr.build_opportunity_pool(cutoff_date=date(2020, 12, 31))
    ssb = next(c for c in pool if c["topic_id"] == "TOPIC_SOLID_STATE_BATT")

    assert ssb["features"]["active_institutions"] is None
    assert ssb["features"]["github_momentum"] is None
    assert ssb["inflection_evidence"]["active_institutions"] is None
    # 因为缺失 active_institutions 与 github_momentum 等关键特征，正式预测严格返回 INSUFFICIENT_FEATURES 且点估计为 None
    direct_formal_pred = forecaster.predict_growth(ssb["features"], require_complete_features=True)
    assert direct_formal_pred["status"] == "INSUFFICIENT_FEATURES"
    assert direct_formal_pred["is_verified"] is False
    assert direct_formal_pred["forecasts"]["3y"]["point_estimate"] is None

    assert ssb["growth_3y_multiple"] is None
    assert ssb["prediction_status"] == PredictionStatus.NOT_PREDICTABLE.value
    assert ssb["prob_10x_super_return_pct"] is None

    # 若特征全部真实完整，is_verified 应正确传播为 True
    complete_features = {k: 1.0 for k in forecaster.feature_names}
    ok_pred = forecaster.predict_growth(complete_features, require_complete_features=True)
    assert ok_pred["status"] == "PREDICTED"
    assert ok_pred["is_verified"] is True


def test_p1_same_day_pit_visibility_and_pipeline_snapshot_consistency(tmp_path, monkeypatch):
    """回归验证 P1 修复：当日非零点 ingested_utc 可被 date.today() 检索，且 GitHub 快照落库无悬空引用"""
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    # 若恰好是 UTC 零点附近也保证用非零点时刻测试
    non_midnight_today = datetime(now_utc.year, now_utc.month, now_utc.day, 14, 25, 30)
    conn = temp_db.get_connection()
    try:
        conn.execute(
            """
            INSERT INTO tech_metrics
            (topic_id, as_of_date, published_at, papers_count, citations_count, active_institutions, github_pit_eligible)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ["TOPIC_SOLID_STATE_BATT", date(2020, 12, 31), non_midnight_today, 1847, 419, None, False],
        )
    finally:
        conn.close()

    # 当日实盘研究视角 (date.today()) 应能查到今日刚采入库、non-midnight TIMESTAMP 的记录
    df_today = temp_db.query_point_in_time("tech_metrics", date.today(), target_col="topic_id", target_val="TOPIC_SOLID_STATE_BATT")
    assert len(df_today) == 1
    assert df_today["papers_count"].iloc[0] == 1847

    # 历史回测视角 (2020-12-31) 依然严格隔离该 2026 年才采集的记录
    df_2020 = temp_db.query_point_in_time("tech_metrics", date(2020, 12, 31), target_col="topic_id", target_val="TOPIC_SOLID_STATE_BATT")
    assert df_2020.empty

    # 模拟 pipeline 写入 GitHub 快照，确认 raw_records.snapshot_id 在 raw_snapshots 中完整存在且文件可哈希校验
    pipe = IngestionPipeline()
    pipe.openalex.snapshot_dir = str(tmp_path / "oa_snaps")
    pipe.github.snapshot_dir = str(tmp_path / "gh_snaps")

    def fake_gh_get(url, params=None, timeout=None):
        payload = {
            "items": [
                {
                    "id": 913533710,
                    "full_name": "stanford-developers/steer-opencell-design",
                    "stargazers_count": 12,
                    "forks_count": 3,
                    "open_issues_count": 1,
                    "created_at": "2025-01-07T21:44:38Z",
                    "updated_at": "2026-10-30T17:27:27Z",
                    "html_url": "https://github.com/stanford-developers/steer-opencell-design",
                    "description": "test",
                }
            ]
        }
        return _mock_json_response(payload, 200)

    monkeypatch.setattr(pipe.github.session, "get", fake_gh_get)
    repos = pipe.github.search_repositories("solid-state-battery", max_items=1, save_snapshots=True)
    assert len(repos) == 1 and repos[0]["snapshot_meta"] is not None

    conn = temp_db.get_connection()
    try:
        temp_db.save_raw_snapshot(repos[0]["snapshot_meta"], conn=conn)
    finally:
        conn.close()
    assert temp_db.verify_snapshot_file_integrity(repos[0]["snapshot_id"])["verified"] is True



# =============================================================================
# 测试 H (PR #7 P0 整改)：伪造历史状态/时间/64位哈希不得通过 STRICT PIT 核验，真实库内归档快照可通过
# =============================================================================

def test_h_forged_pit_eligibility_rejected_and_db_backed_verified_accepted(tmp_path, monkeypatch):
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    snap_dir = str(tmp_path / "h_snaps")
    collector = OpenAlexCollector(snapshot_dir=snap_dir)

    # H1: 调用方直接自填 STRICT_PIT_VERIFIED + 历史发布时间 + 伪造64位SHA256 + 未登记 snapshot_id，必须拒绝
    forged_unregistered = {
        "snapshot_id": "OA_FORGED_UNREGISTERED_2020",
        "as_of_date": date(2020, 12, 31),
        "verified_first_available_at": datetime(2020, 12, 31, 12, 0, 0),
        "historical_eligibility": HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value,
        "source_authenticity_status": "VERIFIED_HISTORICAL_ARCHIVE",
        "vintage_evidence_source": "forged_caller_claim",
        "raw_sha256": "f" * 64,
    }
    res_h1 = collector.evaluate_historical_pit_eligibility(forged_unregistered, cutoff_date=date(2020, 12, 31), db=temp_db)
    assert res_h1["eligible"] is False
    assert res_h1["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value

    # H2: 仅普通当下 HTTP 快照在 raw_snapshots 登记，即使调用方伪传 STRICT_PIT_VERIFIED 与 2020 时间，仍必须拒绝
    raw_bytes = json.dumps({"meta": {"count": 1807}, "group_by": [{"key": "2020", "count": 1807}]}).encode("utf-8")
    current_meta = collector.save_raw_response_snapshot(
        snapshot_dir=snap_dir,
        request_url="https://api.openalex.org/works",
        query_params={"search": '"solid-state battery"', "filter": "publication_year:2020", "group_by": "publication_year"},
        http_status=200,
        raw_bytes=raw_bytes,
        as_of_date=date(2020, 12, 31),
        verified_first_available_at=None,
        quality_status=DataQualityStatus.UNVERIFIED,
        historical_eligibility=HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS,
        file_prefix="oa_2020",
    )
    temp_db.save_raw_snapshot(current_meta)

    forged_override = {
        "snapshot_id": current_meta["snapshot_id"],
        "as_of_date": date(2020, 12, 31),
        "verified_first_available_at": datetime(2020, 12, 31, 0, 0, 0),
        "historical_eligibility": HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value,
        "raw_sha256": current_meta["raw_sha256"],
    }
    res_h2 = collector.evaluate_historical_pit_eligibility(forged_override, cutoff_date=date(2020, 12, 31), db=temp_db)
    assert res_h2["eligible"] is False
    assert res_h2["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value

    # H3: 库中即便已登记历史归档记录，但若磁盘文件遭篡改或证据来源无效 ("self")，仍必须拒绝
    tampered_archive = collector.save_raw_response_snapshot(
        snapshot_dir=snap_dir,
        request_url="https://archive.example.org/openalex/2020",
        query_params={"year": 2020},
        http_status=200,
        raw_bytes=raw_bytes,
        as_of_date=date(2020, 12, 31),
        verified_first_available_at=datetime(2020, 12, 31, 8, 0, 0),
        quality_status=DataQualityStatus.VERIFIED,
        historical_eligibility=HistoricalEligibilityStatus.STRICT_PIT_VERIFIED,
        file_prefix="oa_arch_bad",
    )
    tampered_archive["source_authenticity_status"] = "VERIFIED_HISTORICAL_ARCHIVE"
    tampered_archive["vintage_evidence_source"] = "OpenAlex_Official_2020_Monthly_Dump"
    temp_db.save_raw_snapshot(tampered_archive)
    with open(tampered_archive["snapshot_path"], "wb") as f:
        f.write(b'{"tampered": true}')
    res_h3 = collector.evaluate_historical_pit_eligibility(
        {"snapshot_id": tampered_archive["snapshot_id"], "raw_sha256": tampered_archive["raw_sha256"]},
        cutoff_date=date(2020, 12, 31),
        db=temp_db,
    )
    assert res_h3["eligible"] is False
    assert res_h3["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value

    # H3b: 磁盘文件完整但 vintage_evidence_source 为自宣 ("self")，仍必须拒绝
    self_vouch_archive = collector.save_raw_response_snapshot(
        snapshot_dir=snap_dir,
        request_url="https://archive.example.org/openalex/2020-self",
        query_params={"year": 2020, "tag": "self_vouch"},
        http_status=200,
        raw_bytes=raw_bytes,
        as_of_date=date(2020, 12, 31),
        verified_first_available_at=datetime(2020, 12, 31, 8, 0, 0),
        quality_status=DataQualityStatus.VERIFIED,
        historical_eligibility=HistoricalEligibilityStatus.STRICT_PIT_VERIFIED,
        file_prefix="oa_arch_self",
    )
    self_vouch_archive["source_authenticity_status"] = "VERIFIED_HISTORICAL_ARCHIVE"
    self_vouch_archive["vintage_evidence_source"] = "self"
    temp_db.save_raw_snapshot(self_vouch_archive)
    res_h3b = collector.evaluate_historical_pit_eligibility(
        {"snapshot_id": self_vouch_archive["snapshot_id"], "raw_sha256": self_vouch_archive["raw_sha256"]},
        cutoff_date=date(2020, 12, 31),
        db=temp_db,
    )
    assert res_h3b["eligible"] is False
    assert res_h3b["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value

    # H4: 真实完整库内历史归档快照（文件完整 + 哈希匹配 + VERIFIED_HISTORICAL_ARCHIVE + 第三方归档证据）在截断日后通过、截断日前拒绝
    valid_archive = collector.save_raw_response_snapshot(
        snapshot_dir=snap_dir,
        request_url="https://archive.openalex.org/releases/2020-12-31",
        query_params={"year": 2020, "topic": "solid-state battery"},
        http_status=200,
        raw_bytes=raw_bytes,
        as_of_date=date(2020, 12, 31),
        verified_first_available_at=datetime(2020, 12, 31, 8, 0, 0),
        quality_status=DataQualityStatus.VERIFIED,
        historical_eligibility=HistoricalEligibilityStatus.STRICT_PIT_VERIFIED,
        file_prefix="oa_arch_ok",
    )
    valid_archive["source_authenticity_status"] = "VERIFIED_HISTORICAL_ARCHIVE"
    valid_archive["vintage_evidence_source"] = "OpenAlex_Official_2020_Archive_Release"
    temp_db.save_raw_snapshot(valid_archive)

    res_h4_ok = collector.evaluate_historical_pit_eligibility(
        {"snapshot_id": valid_archive["snapshot_id"], "raw_sha256": valid_archive["raw_sha256"]},
        cutoff_date=date(2020, 12, 31),
        db=temp_db,
    )
    assert res_h4_ok["eligible"] is True
    assert res_h4_ok["status"] == HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value

    res_h4_early = collector.evaluate_historical_pit_eligibility(
        {"snapshot_id": valid_archive["snapshot_id"], "raw_sha256": valid_archive["raw_sha256"]},
        cutoff_date=date(2020, 6, 30),
        db=temp_db,
    )
    assert res_h4_early["eligible"] is False
    assert res_h4_early["status"] == HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value


# =============================================================================
# 测试 I (PR #7 P1 整改)：raw_records 不可变历史留痕（禁止覆盖同一主题同一年度多次采集）
# =============================================================================

def test_i_immutable_raw_records_preserve_versions_without_overwrite(tmp_path, monkeypatch):
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    pipe = IngestionPipeline()
    pipe.openalex.snapshot_dir = str(tmp_path / "oa_hist_snaps")

    # 关闭无关数据源，聚焦验证同一来源同一主题同一年度多次采集的不可变版本留痕
    monkeypatch.setattr(pipe.macro, "collect", lambda: {})
    monkeypatch.setattr(pipe.enterprise, "get_core_enterprise_financials", lambda: [])
    monkeypatch.setattr(pipe.enterprise, "get_historical_distress_cohort", lambda: [])
    monkeypatch.setattr(pipe.github, "search_repositories", lambda *a, **k: [])
    monkeypatch.setattr(pipe.tech_dataset, "get_historical_tech_training_samples", lambda: [])

    logical_id = "oa_ssb_2020_metrics"

    def _make_run_record(papers_val: int, pub_dt: datetime, ing_dt: datetime, snap_tag: str):
        raw_bytes = json.dumps({"meta": {"count": papers_val}, "group_by": [{"key": "2020", "count": papers_val}]}).encode("utf-8")
        meta = pipe.openalex.save_raw_response_snapshot(
            snapshot_dir=pipe.openalex.snapshot_dir,
            request_url="https://api.openalex.org/works",
            query_params={"topic": "ssb", "year": 2020, "tag": snap_tag},
            http_status=200,
            raw_bytes=raw_bytes,
            as_of_date=date(2020, 12, 31),
            verified_first_available_at=None,
            quality_status=DataQualityStatus.UNVERIFIED,
            historical_eligibility=HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS,
            file_prefix=f"count_{snap_tag}",
        )
        return {
            "status": "OK",
            "as_of_date": date(2020, 12, 31),
            "published_at": pub_dt,
            "papers_count": papers_val,
            "citations_count": None,
            "sampled_citations_count": 50,
            "citation_count_scope": CitationCountScope.SAMPLED_TOP_CITED_BIASED.value,
            "active_institutions": 12,
            "sampled_institutions_count": 12,
            "institution_count_scope": InstitutionCountScope.COMPLETE_AGGREGATION.value,
            "citations_cutoff_year": 2020,
            "historical_eligibility": HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value,
            "snapshot_id": meta["snapshot_id"],
            "snapshot_meta": meta,
            "snapshot_metas": [meta],
            "feature_lineage": {
                "papers_count": {"snapshot_id": meta["snapshot_id"], "raw_sha256": meta["raw_sha256"]}
            },
            "raw_payload": raw_bytes.decode("utf-8"),
            "provenance": {
                "record_id": logical_id,
                "source_name": "OpenAlex_API",
                "source_url": "https://api.openalex.org/works",
                "as_of_date": date(2020, 12, 31),
                "published_at": pub_dt,
                "ingested_at": ing_dt,
                "quality_status": DataQualityStatus.UNVERIFIED.value,
                "data_version": f"v_{snap_tag}",
                "raw_hash": meta["raw_sha256"],
                "snapshot_id": meta["snapshot_id"],
            },
        }

    run1_rec = _make_run_record(1800, datetime(2020, 12, 31, 10, 0, 0), datetime(2020, 12, 31, 10, 0, 0), "2020_v1")
    run2_rec = _make_run_record(1855, datetime(2022, 6, 15, 12, 0, 0), datetime(2022, 6, 15, 12, 0, 0), "2022_v2")

    monkeypatch.setattr(pipe.openalex, "fetch_topic_research_trends", lambda q, *a, **k: [run1_rec] if "solid-state" in q else [])
    pipe.run_all(verbose=False)

    monkeypatch.setattr(pipe.openalex, "fetch_topic_research_trends", lambda q, *a, **k: [run2_rec] if "solid-state" in q else [])
    pipe.run_all(verbose=False)

    all_versions = temp_db.get_raw_record_versions(logical_id)
    assert len(all_versions) == 2
    assert list(all_versions["ingestion_version"]) == [1, 2]
    assert len(set(all_versions["record_id"])) == 2
    assert len(set(all_versions["snapshot_id"])) == 2
    assert len(set(all_versions["raw_hash"])) == 2

    # PIT 版本检索：2021 视角仅看见 v1 (1800)，2023 视角可见最新 v2 (1855)
    v_2021 = temp_db.get_latest_raw_record(logical_id, cutoff_date=date(2021, 12, 31))
    assert v_2021 is not None
    assert v_2021["ingestion_version"] == 1
    assert json.loads(v_2021["payload"])["meta"]["count"] == 1800

    v_2023 = temp_db.get_latest_raw_record(logical_id, cutoff_date=date(2023, 12, 31))
    assert v_2023 is not None
    assert v_2023["ingestion_version"] == 2
    assert json.loads(v_2023["payload"])["meta"]["count"] == 1855


# =============================================================================
# 测试 J (PR #7 P1 整改)：引用统计口径隔离（样本引用禁止冒充全领域总量进入主指标）
# =============================================================================

def test_j_citation_scope_isolation_and_inflection_guard(tmp_path, monkeypatch):
    collector = OpenAlexCollector(snapshot_dir=str(tmp_path / "j_snaps"))

    def fake_fetch(url, params=None, **kwargs):
        gb = (params or {}).get("group_by")
        if gb == "publication_year":
            return _mock_json_response({"meta": {"count": 1000}, "group_by": [{"key": "2020", "count": 1000}]}, 200), 200, None
        if gb == "authorships.institutions.id":
            return _mock_json_response({"meta": {"count": 1000, "groups_count": 2}, "group_by": [{"key": "https://openalex.org/I1", "count": 500}]}, 200), 200, None
        works_payload = {
            "results": [
                {"id": "W1", "counts_by_year": [{"year": 2020, "cited_by_count": 150}]},
                {"id": "W2", "counts_by_year": [{"year": 2020, "cited_by_count": 100}]},
            ]
        }
        return _mock_json_response(works_payload, 200), 200, None

    monkeypatch.setattr(collector, "fetch_raw_with_audit", fake_fetch)
    rows = collector.fetch_topic_research_trends("solid-state battery", start_year=2020, end_year=2020, save_snapshots=True)
    rec = rows[0]

    # papers_count=1000 但仅采样 2 篇作品：正式 citations_count 必须为 None，样本求和仅存 sampled_citations_count=250
    assert rec["papers_count"] == 1000
    assert rec["citations_count"] is None
    assert rec["sampled_citations_count"] == 250
    assert rec["citation_count_scope"] == CitationCountScope.SAMPLED_TOP_CITED_BIASED.value

    # 下游 InflectionDetector 对 SAMPLED_TOP_CITED_BIASED 指标不得输出为正式全领域 citations_latest
    detector = InflectionDetector()
    df_biased = pd.DataFrame([
        {
            "as_of_date": date(2018, 12, 31),
            "papers_count": 400,
            "citations_count": None,
            "sampled_citations_count": 120,
            "citation_count_scope": CitationCountScope.SAMPLED_TOP_CITED_BIASED.value,
            "active_institutions": 10,
        },
        {
            "as_of_date": date(2019, 12, 31),
            "papers_count": 650,
            "citations_count": None,
            "sampled_citations_count": 180,
            "citation_count_scope": CitationCountScope.SAMPLED_TOP_CITED_BIASED.value,
            "active_institutions": 14,
        },
        {
            "as_of_date": date(2020, 12, 31),
            "papers_count": 1000,
            "citations_count": 250,  # 即便脏数据误填，也因 scope!=COMPLETE_DOMAIN_TOTAL 被拦截
            "sampled_citations_count": 250,
            "citation_count_scope": CitationCountScope.SAMPLED_TOP_CITED_BIASED.value,
            "active_institutions": 20,
        },
    ])
    res = detector.analyze_tech_acceleration(df_biased)
    assert res["status"] == "VALIDATED"
    assert res["evidence"]["citations_latest"] is None
    assert res["evidence"]["sampled_citations_latest"] == 250
    assert res["evidence"]["citation_count_scope"] == CitationCountScope.SAMPLED_TOP_CITED_BIASED.value


# =============================================================================
# 测试 K (PR #7 P1 整改)：特征级快照血缘独立绑证（仅篡改 works_sample 快照必须被 citations_count 校验捕获）
# =============================================================================

def test_k_feature_lineage_tampering_detects_works_sample_corruption(tmp_path, monkeypatch):
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    collector = OpenAlexCollector(snapshot_dir=str(tmp_path / "k_snaps"))

    def fake_fetch(url, params=None, **kwargs):
        gb = (params or {}).get("group_by")
        if gb == "publication_year":
            return _mock_json_response({"meta": {"count": 1807}, "group_by": [{"key": "2020", "count": 1807}]}, 200), 200, None
        if gb == "authorships.institutions.id":
            return _mock_json_response({"meta": {"count": 1807, "groups_count": 15}, "group_by": [{"key": f"https://openalex.org/I{i}", "count": 10} for i in range(15)]}, 200), 200, None
        works_payload = {
            "results": [
                {"id": "W1", "counts_by_year": [{"year": 2020, "cited_by_count": 107}]},
            ]
        }
        return _mock_json_response(works_payload, 200), 200, None

    monkeypatch.setattr(collector, "fetch_raw_with_audit", fake_fetch)
    rows = collector.fetch_topic_research_trends('"solid-state battery"', start_year=2020, end_year=2020, save_snapshots=True)
    rec = rows[0]
    for smeta in rec["snapshot_metas"]:
        temp_db.save_raw_snapshot(smeta)

    lineage = rec["feature_lineage"]
    assert set(lineage.keys()) >= {"papers_count", "citations_count", "active_institutions"}
    assert lineage["citations_count"]["scope"] == CitationCountScope.SAMPLED_TOP_CITED_BIASED.value
    assert lineage["papers_count"]["snapshot_id"] != lineage["citations_count"]["snapshot_id"]
    assert lineage["citations_count"]["snapshot_id"] != lineage["active_institutions"]["snapshot_id"]

    # 未篡改前：papers_count 与 active_institutions 完整口径校验通过；citations_count 允许非完整口径下复算通过，要求完整口径则拒绝
    assert OpenAlexCollector.verify_feature_lineage(rec, "papers_count", db=temp_db)["verified"] is True
    assert OpenAlexCollector.verify_feature_lineage(rec, "active_institutions", db=temp_db)["verified"] is True
    assert OpenAlexCollector.verify_feature_lineage(rec, "citations_count", db=temp_db, require_complete_scope=True)["verified"] is False
    cit_ok = OpenAlexCollector.verify_feature_lineage(rec, "citations_count", db=temp_db, require_complete_scope=False)
    assert cit_ok["verified"] is True
    assert cit_ok["recomputed_value"] == 107

    # 仅篡改 works_sample 原始快照文件（保持 count 与 institutions 原始快照完全不动）
    works_snap_path = lineage["citations_count"]["snapshot_path"]
    with open(works_snap_path, "wb") as f:
        f.write(json.dumps({"results": [{"id": "W1", "counts_by_year": [{"year": 2020, "cited_by_count": 9999}]}]}).encode("utf-8"))

    # papers_count 仍应独立验证通过，但 citations_count / sampled_citations_count 必须立即检出篡改并失败
    assert OpenAlexCollector.verify_feature_lineage(rec, "papers_count", db=temp_db)["verified"] is True
    cit_tampered = OpenAlexCollector.verify_feature_lineage(rec, "citations_count", db=temp_db, require_complete_scope=False)
    assert cit_tampered["verified"] is False
    assert "SHA-256" in cit_tampered["reason"] or "一致" in cit_tampered["reason"]


def test_k2_active_institutions_sampled_and_failed_lineage_verification(tmp_path, monkeypatch):
    """验证 PR #7 [P1-01]：active_institutions 在 200 上限采样或机构接口失败时正确绑定 works_sample 并严格检出篡改"""
    temp_db = _setup_isolated_db(tmp_path, monkeypatch)
    collector = OpenAlexCollector(snapshot_dir=str(tmp_path / "k2_snaps"))

    # 场景 1：institutions 达到 200 截断上限 -> SAMPLED_WORKS_ONLY 主血缘必须指向 works_sample，并附带 institutions_snapshot 截断证据
    def fake_fetch_capped(url, params=None, **kwargs):
        gb = (params or {}).get("group_by")
        if gb == "publication_year":
            return _mock_json_response({"meta": {"count": 1807}, "group_by": [{"key": "2020", "count": 1807}]}, 200), 200, None
        if gb == "authorships.institutions.id":
            groups = [{"key": f"https://openalex.org/I{i}", "count": 5} for i in range(200)]
            return _mock_json_response({"meta": {"count": 1807, "groups_count": 200}, "group_by": groups}, 200), 200, None
        works_payload = {
            "results": [
                {
                    "id": "W1",
                    "publication_date": "2020-03-01",
                    "counts_by_year": [{"year": 2020, "cited_by_count": 80}],
                    "authorships": [
                        {"institutions": [{"id": "https://openalex.org/I1"}, {"id": "https://openalex.org/I2"}]},
                    ],
                },
                {
                    "id": "W2",
                    "publication_date": "2020-06-01",
                    "counts_by_year": [{"year": 2020, "cited_by_count": 27}],
                    "authorships": [
                        {"institutions": [{"id": "https://openalex.org/I2"}, {"ror": "https://ror.org/03"}]},
                    ],
                },
            ]
        }
        return _mock_json_response(works_payload, 200), 200, None

    monkeypatch.setattr(collector, "fetch_raw_with_audit", fake_fetch_capped)
    rows = collector.fetch_topic_research_trends('"solid-state battery"', start_year=2020, end_year=2020, save_snapshots=True)
    rec = rows[0]
    for smeta in rec["snapshot_metas"]:
        temp_db.save_raw_snapshot(smeta)

    assert rec["active_institutions"] is None
    assert rec["sampled_institutions_count"] == 3
    assert rec["institution_count_scope"] == InstitutionCountScope.SAMPLED_WORKS_ONLY.value

    inst_lin = rec["feature_lineage"]["active_institutions"]
    assert inst_lin["snapshot_id"] == rec["feature_lineage"]["citations_count"]["snapshot_id"]
    assert inst_lin["raw_sha256"] == rec["feature_lineage"]["citations_count"]["raw_sha256"]
    assert inst_lin["institutions_snapshot"]["raw_sha256"]

    assert OpenAlexCollector.verify_feature_lineage(rec, "active_institutions", db=temp_db, require_complete_scope=True)["verified"] is False
    inst_ok = OpenAlexCollector.verify_feature_lineage(rec, "active_institutions", db=temp_db, require_complete_scope=False)
    assert inst_ok["verified"] is True
    assert inst_ok["recomputed_value"] == 3

    # 支持直接传入带 feature_lineage_json 字符串的 tech_metrics 行字典
    tm_row_like = {"feature_lineage_json": json.dumps(rec["feature_lineage"], ensure_ascii=False)}
    assert OpenAlexCollector.verify_feature_lineage(tm_row_like, "active_institutions", db=temp_db, require_complete_scope=False)["recomputed_value"] == 3

    # 仅篡改 works_sample 快照：active_institutions 与 citations_count 均应失败，papers_count 仍通过
    with open(inst_lin["snapshot_path"], "wb") as f:
        f.write(json.dumps({"results": [{"id": "W1", "authorships": []}]}).encode("utf-8"))

    assert OpenAlexCollector.verify_feature_lineage(rec, "papers_count", db=temp_db)["verified"] is True
    assert OpenAlexCollector.verify_feature_lineage(rec, "active_institutions", db=temp_db, require_complete_scope=False)["verified"] is False

    # 场景 2：institutions 聚合接口 HTTP 500 失败而 works_sample 成功 -> 仍必须绑定 works_sample 有效快照并通过采样校验
    collector2 = OpenAlexCollector(snapshot_dir=str(tmp_path / "k2_fail_snaps"))

    def fake_fetch_inst_fail(url, params=None, **kwargs):
        gb = (params or {}).get("group_by")
        if gb == "publication_year":
            return _mock_json_response({"meta": {"count": 500}, "group_by": [{"key": "2020", "count": 500}]}, 200), 200, None
        if gb == "authorships.institutions.id":
            r500 = MagicMock(spec=requests.Response)
            r500.status_code = 500
            return r500, 500, "INST_HTTP_500_FAILED"
        works_payload = {
            "results": [
                {
                    "id": "W10",
                    "publication_date": "2020-02-01",
                    "counts_by_year": [{"year": 2020, "cited_by_count": 40}],
                    "authorships": [
                        {"institutions": [{"id": "https://openalex.org/I10"}, {"id": "https://openalex.org/I11"}]},
                    ],
                }
            ]
        }
        return _mock_json_response(works_payload, 200), 200, None

    monkeypatch.setattr(collector2, "fetch_raw_with_audit", fake_fetch_inst_fail)
    rows2 = collector2.fetch_topic_research_trends("solid-state-electrolyte", start_year=2020, end_year=2020, save_snapshots=True)
    rec2 = rows2[0]
    for smeta in rec2["snapshot_metas"]:
        temp_db.save_raw_snapshot(smeta)

    inst_lin2 = rec2["feature_lineage"]["active_institutions"]
    assert inst_lin2["snapshot_id"] == rec2["feature_lineage"]["citations_count"]["snapshot_id"]
    assert inst_lin2["raw_sha256"] is not None
    assert inst_lin2["institutions_snapshot"]["http_status"] == 500
    assert inst_lin2["institutions_snapshot"]["raw_sha256"] is None

    ok2 = OpenAlexCollector.verify_feature_lineage(rec2, "active_institutions", db=temp_db, require_complete_scope=False)
    assert ok2["verified"] is True
    assert ok2["recomputed_value"] == 2
