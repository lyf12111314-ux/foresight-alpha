"""V1.2 第一阶段数据真实性、原始快照存证与 PIT 准入专项测试 (Tests A–G).

严格区分：
- 所有 CI 用例使用离线 Mock 与隔离临时目录/DuckDB，绝不发起公网请求，绝不混淆工程测试与真实线上数据验证。
"""

import hashlib
import json
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

import src.backtest.engine as bt_eng
import src.core.database as db_module
import src.decision_engine.opportunity_pool as opp_pool
import src.ingestion.pipeline as ing_pipe
from src.core.database import DatabaseManager
from src.core.types import (
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
    # 严禁使用 2528 + 500；仅取 2020 年真实未未来泄露部分 107
    assert rec["citations_count"] == 107
    assert rec["citations_cutoff_year"] == 2020
    assert rec["works_with_pit_counts"] == 1
    assert rec["works_missing_pit_counts"] == 1

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
