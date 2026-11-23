"""OpenAlex 学术文献与全球科研成果采集器 (OpenAlex API Client — V1.2 PIT & Provenance Overhaul)."""

import json
import os
from datetime import datetime, date, timezone
from typing import Dict, Any, List, Optional, Tuple, Set
from src.ingestion.base import BaseCollector
from src.core.types import (
    DataQualityStatus,
    HistoricalEligibilityStatus,
    InstitutionCountScope,
)


class OpenAlexCollector(BaseCollector):
    """OpenAlex 学术数据采集器 (具备原始响应存证、引用时间过滤、机构口径区分与历史资格审计)"""

    OPENALEX_GROUP_CAP = 200

    def __init__(
        self,
        base_url: str = "https://api.openalex.org",
        snapshot_dir: Optional[str] = None,
    ):
        super().__init__(source_name="OpenAlex", base_url=base_url)
        self.snapshot_dir = snapshot_dir or os.path.abspath("data/snapshots/openalex")

    @staticmethod
    def extract_pit_citations_from_work(
        work: Dict[str, Any],
        cutoff_year: int,
    ) -> Tuple[Optional[int], str]:
        """从单篇论文的 counts_by_year 中严格剔除 cutoff_year 之后的未来引用。

        严禁回退使用总 cited_by_count；缺少或格式异常时显式返回 None 与缺失状态，不自动填零。
        """
        if not isinstance(work, dict):
            return None, "CITATIONS_BY_YEAR_MALFORMED"

        counts = work.get("counts_by_year")
        if counts is None or not isinstance(counts, list):
            return None, "CITATIONS_BY_YEAR_UNAVAILABLE"

        if len(counts) == 0:
            total_cited = work.get("cited_by_count")
            if total_cited is None:
                return None, "CITATIONS_BY_YEAR_UNAVAILABLE"
            try:
                if int(total_cited) == 0:
                    return 0, "OK"
            except (TypeError, ValueError):
                return None, "CITATIONS_BY_YEAR_MALFORMED"
            # 总引用 > 0 但 counts_by_year 为空，禁止以 0 充当真实观测
            return None, "CITATIONS_BY_YEAR_INCONSISTENT_EMPTY"

        pit_total = 0
        valid_entries = 0
        for item in counts:
            if not isinstance(item, dict) or "year" not in item or "cited_by_count" not in item:
                return None, "CITATIONS_BY_YEAR_MALFORMED"
            try:
                yr = int(item["year"])
                cnt = int(item["cited_by_count"])
            except (TypeError, ValueError):
                return None, "CITATIONS_BY_YEAR_MALFORMED"
            if cnt < 0:
                return None, "CITATIONS_BY_YEAR_MALFORMED"
            valid_entries += 1
            if yr <= cutoff_year:
                pit_total += cnt

        if valid_entries == 0:
            return None, "CITATIONS_BY_YEAR_UNAVAILABLE"
        return pit_total, "OK"

    @staticmethod
    def extract_institution_ids_from_works(works: List[Dict[str, Any]]) -> Tuple[Set[str], int]:
        """严格按作者机构稳定 ID (优先 OpenAlex id，其次 ror) 去重，不使用任何经验倍数估算。"""
        inst_ids: Set[str] = set()
        works_with_inst = 0
        if not isinstance(works, list):
            return inst_ids, 0
        for w in works:
            if not isinstance(w, dict):
                continue
            work_has_inst = False
            for auth in (w.get("authorships") or []):
                if not isinstance(auth, dict):
                    continue
                for inst in (auth.get("institutions") or []):
                    if not isinstance(inst, dict):
                        continue
                    stable_id = inst.get("id") or inst.get("ror")
                    if stable_id and str(stable_id).strip():
                        inst_ids.add(str(stable_id).strip())
                        work_has_inst = True
            if work_has_inst:
                works_with_inst += 1
        return inst_ids, works_with_inst

    @staticmethod
    def evaluate_historical_pit_eligibility(
        record: Dict[str, Any],
        cutoff_date: date,
    ) -> Dict[str, Any]:
        """严格检查记录是否具备历史预测时点 cutoff_date 的 Point-in-Time 回测资格。

        禁止将当前查询获得的历史年份统计 (CURRENT_SNAPSHOT_HISTORICAL_STATS) 自动升级为严格 PIT 证据。
        """
        eligibility = record.get("historical_eligibility")
        verified_first_avail = record.get("verified_first_available_at")
        raw_sha256 = record.get("raw_sha256") or record.get("raw_hash")

        if eligibility == HistoricalEligibilityStatus.CURRENT_STATE_ONLY.value:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "仅反映查询当日实时状态 (CURRENT_STATE_ONLY)，禁止用于历史回测。",
            }

        if not raw_sha256:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "缺少原始内容 SHA-256 存证。",
            }

        if not verified_first_avail:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": (
                    f"记录仅具备 {eligibility or 'CURRENT_SNAPSHOT'} 资格，"
                    "缺少可核验的历史首次公开时间 (verified_first_available_at)，"
                    "无法证明与预测时点当时可获取的版本一致。"
                ),
            }

        if isinstance(verified_first_avail, str):
            try:
                avail_dt = datetime.fromisoformat(verified_first_avail.replace("Z", "+00:00"))
            except ValueError:
                return {
                    "eligible": False,
                    "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                    "reason": f"无法解析 verified_first_available_at: {verified_first_avail}",
                }
        elif isinstance(verified_first_avail, datetime):
            avail_dt = verified_first_avail
        elif isinstance(verified_first_avail, date):
            avail_dt = datetime(verified_first_avail.year, verified_first_avail.month, verified_first_avail.day)
        else:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "verified_first_available_at 类型无效。",
            }

        if avail_dt.date() > cutoff_date:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"数据核验首次可获取时间 {avail_dt.date()} 晚于预测截断日 {cutoff_date}。",
            }

        if eligibility != HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"当前资格状态为 {eligibility}，未达 STRICT_PIT_VERIFIED 严格历史版本证明。",
            }

        return {
            "eligible": True,
            "status": HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value,
            "reason": None,
        }

    def fetch_topic_research_trends(
        self,
        query: str,
        start_year: int = 2018,
        end_year: int = 2024,
        save_snapshots: bool = True,
        citations_cutoff_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """按发表年份拉取 OpenAlex 文献动态，保存原始快照并严格分离数据使用资格。

        重要限制说明：
        1. 当前 API 返回的历史年份论文计数与按 counts_by_year 过滤后的引用数，仍属于
           CURRENT_SNAPSHOT_HISTORICAL_STATS（受后续索引补录影响，不可直接冒充当年实时快照）。
        2. 若缺少 counts_by_year 或无法完成全量机构去重，绝不使用估算公式或自动填零替代。
        3. 若 API 超时、限流 (429)、非 200 或响应畸形，显式返回 FETCH_FAILED，绝不产出 papers_count=0 的正常记录。
        """
        url = f"{self.base_url}/works"
        results: List[Dict[str, Any]] = []

        for year in range(start_year, end_year + 1):
            as_of_date = date(year, 12, 31)
            eff_cutoff_year = citations_cutoff_year if citations_cutoff_year is not None else year
            ingested_utc = datetime.now(timezone.utc).replace(tzinfo=None)

            # 1) 查询该年度文献总数 (group_by=publication_year)
            count_params = {
                "search": query,
                "filter": f"publication_year:{year}",
                "group_by": "publication_year",
            }
            resp, http_status, fetch_err = self.fetch_raw_with_audit(url, params=count_params)

            papers_count: Optional[int] = None
            parse_error: Optional[str] = None
            raw_bytes: Optional[bytes] = None
            count_data: Optional[Dict[str, Any]] = None

            if resp is not None and http_status == 200:
                raw_bytes = resp.content if isinstance(resp.content, (bytes, bytearray)) else resp.text.encode("utf-8")
                try:
                    count_data = resp.json()
                    if not isinstance(count_data, dict):
                        parse_error = "MALFORMED_JSON_NOT_OBJECT"
                    else:
                        group_data = count_data.get("group_by")
                        if not isinstance(group_data, list):
                            parse_error = "MALFORMED_RESPONSE_MISSING_GROUP_BY"
                        elif len(group_data) == 0:
                            # 合法空分组：需在 meta.count == 0 时才确认是真实 0 篇
                            meta_cnt = (count_data.get("meta") or {}).get("count")
                            if meta_cnt == 0:
                                papers_count = 0
                            else:
                                parse_error = "EMPTY_GROUP_BY_WITH_UNKNOWN_META"
                        else:
                            raw_cnt = group_data[0].get("count")
                            if raw_cnt is None:
                                parse_error = "MALFORMED_GROUP_COUNT_MISSING"
                            else:
                                papers_count = int(raw_cnt)
                except (ValueError, TypeError, KeyError) as exc:
                    parse_error = f"JSON_PARSE_ERROR: {exc}"
            else:
                parse_error = fetch_err or f"HTTP_{http_status}_FAILED"

            count_snap_meta = None
            if save_snapshots:
                count_snap_meta = self.save_raw_response_snapshot(
                    snapshot_dir=self.snapshot_dir,
                    request_url=url,
                    query_params=count_params,
                    http_status=http_status,
                    raw_bytes=raw_bytes if parse_error is None else None,
                    as_of_date=as_of_date,
                    verified_first_available_at=None,
                    quality_status=DataQualityStatus.UNVERIFIED if parse_error is None else DataQualityStatus.UNAVAILABLE,
                    historical_eligibility=(
                        HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS
                        if parse_error is None
                        else HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE
                    ),
                    error_message=parse_error,
                    ingested_at_utc=ingested_utc,
                    file_prefix=f"openalex_{query}_{year}_count",
                )

            if parse_error is not None or papers_count is None:
                results.append({
                    "year": year,
                    "status": "FETCH_FAILED",
                    "error_message": parse_error,
                    "http_status": http_status,
                    "as_of_date": as_of_date,
                    "published_at": None,
                    "verified_first_available_at": None,
                    "ingested_at_utc": ingested_utc,
                    "papers_count": None,
                    "citations_count": None,
                    "citations_status": "FETCH_FAILED",
                    "citations_cutoff_year": eff_cutoff_year,
                    "active_institutions": None,
                    "sampled_institutions_count": None,
                    "institution_count_scope": InstitutionCountScope.UNAVAILABLE.value,
                    "historical_eligibility": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                    "snapshot_id": count_snap_meta["snapshot_id"] if count_snap_meta else None,
                    "snapshot_path": None,
                    "raw_sha256": None,
                    "snapshot_meta": count_snap_meta,
                    "quality_status": DataQualityStatus.UNAVAILABLE.value,
                })
                continue

            # 2) 拉取代表性论文样本，严格按 counts_by_year <= eff_cutoff_year 过滤引用并按稳定 ID 去重机构
            sample_params = {
                "search": query,
                "filter": f"publication_year:{year}",
                "per-page": 10,
                "sort": "cited_by_count:desc",
            }
            sample_resp, sample_status, sample_err = self.fetch_raw_with_audit(url, params=sample_params)
            sample_works: List[Dict[str, Any]] = []
            sample_parse_err: Optional[str] = None
            sample_raw_bytes: Optional[bytes] = None

            if sample_resp is not None and sample_status == 200:
                sample_raw_bytes = (
                    sample_resp.content
                    if isinstance(sample_resp.content, (bytes, bytearray))
                    else sample_resp.text.encode("utf-8")
                )
                try:
                    sample_json = sample_resp.json()
                    if isinstance(sample_json, dict) and isinstance(sample_json.get("results"), list):
                        sample_works = sample_json["results"]
                    else:
                        sample_parse_err = "MALFORMED_WORKS_RESULTS"
                except (ValueError, TypeError) as exc:
                    sample_parse_err = f"SAMPLE_JSON_PARSE_ERROR: {exc}"
            else:
                sample_parse_err = sample_err or f"SAMPLE_HTTP_{sample_status}_FAILED"

            if save_snapshots:
                self.save_raw_response_snapshot(
                    snapshot_dir=self.snapshot_dir,
                    request_url=url,
                    query_params=sample_params,
                    http_status=sample_status,
                    raw_bytes=sample_raw_bytes if sample_parse_err is None else None,
                    as_of_date=as_of_date,
                    verified_first_available_at=None,
                    quality_status=DataQualityStatus.UNVERIFIED if sample_parse_err is None else DataQualityStatus.UNAVAILABLE,
                    historical_eligibility=HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS
                    if sample_parse_err is None
                    else HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE,
                    error_message=sample_parse_err,
                    ingested_at_utc=ingested_utc,
                    file_prefix=f"openalex_{query}_{year}_works_sample",
                )

            # 统计 cutoff_year 之前真实发生的历史引用（绝不回退到 cited_by_count）
            pit_citations_sum: Optional[int] = None
            citations_status = "CITATIONS_BY_YEAR_UNAVAILABLE"
            works_with_pit_counts = 0
            works_missing_pit_counts = 0
            excluded_future_pub_works = 0
            max_cutoff_date = date(eff_cutoff_year, 12, 31)

            eligible_works: List[Dict[str, Any]] = []
            for w in sample_works:
                if not isinstance(w, dict):
                    continue
                pub_d_str = w.get("publication_date")
                if pub_d_str:
                    try:
                        pub_d = date.fromisoformat(str(pub_d_str)[:10])
                        if pub_d > max_cutoff_date:
                            excluded_future_pub_works += 1
                            continue
                    except ValueError:
                        pass
                eligible_works.append(w)

            if papers_count == 0 and sample_parse_err is None:
                pit_citations_sum = 0
                citations_status = "OK_ZERO_WORKS"
            elif sample_parse_err is None and eligible_works:
                running_sum = 0
                for w in eligible_works:
                    c_val, c_stat = self.extract_pit_citations_from_work(w, cutoff_year=eff_cutoff_year)
                    if c_stat == "OK" and c_val is not None:
                        running_sum += c_val
                        works_with_pit_counts += 1
                    else:
                        works_missing_pit_counts += 1
                if works_with_pit_counts > 0:
                    pit_citations_sum = running_sum
                    citations_status = (
                        "OK_PARTIAL_SAMPLED_WORKS"
                        if works_missing_pit_counts > 0 or len(eligible_works) < papers_count
                        else "OK_COMPLETE_PIT_COUNTS"
                    )
                else:
                    pit_citations_sum = None
                    citations_status = "CITATIONS_BY_YEAR_UNAVAILABLE"
            elif sample_parse_err is not None:
                pit_citations_sum = None
                citations_status = f"WORKS_SAMPLE_FAILED:{sample_parse_err}"

            # 3) 机构统计：先看按采样去重，只有完整聚合接口且未达上限才允许填充 active_institutions
            sampled_inst_ids, _ = self.extract_institution_ids_from_works(eligible_works)
            sampled_inst_count: Optional[int] = len(sampled_inst_ids) if sample_parse_err is None else None
            active_institutions: Optional[int] = None
            inst_scope = InstitutionCountScope.UNAVAILABLE.value

            if papers_count == 0 and parse_error is None:
                active_institutions = 0
                sampled_inst_count = 0
                inst_scope = InstitutionCountScope.COMPLETE_AGGREGATION.value
            elif sample_parse_err is None:
                # 尝试 OpenAlex group_by=authorships.institutions.id 获取全量机构去重（仅当结果未达 200 上限时才可认定完整）
                inst_params = {
                    "search": query,
                    "filter": f"publication_year:{year}",
                    "group_by": "authorships.institutions.id",
                }
                inst_resp, inst_status, _ = self.fetch_raw_with_audit(url, params=inst_params)
                used_complete_agg = False
                if inst_resp is not None and inst_status == 200:
                    try:
                        inst_json = inst_resp.json()
                        inst_groups = inst_json.get("group_by") if isinstance(inst_json, dict) else None
                        meta_info = (inst_json.get("meta") or {}) if isinstance(inst_json, dict) else {}
                        groups_count = meta_info.get("groups_count")
                        if isinstance(inst_groups, list):
                            valid_group_ids = {
                                str(g["key"]).strip()
                                for g in inst_groups
                                if isinstance(g, dict) and g.get("key") and str(g["key"]).lower() != "unknown"
                            }
                            if groups_count is not None and int(groups_count) < self.OPENALEX_GROUP_CAP and len(inst_groups) < self.OPENALEX_GROUP_CAP:
                                active_institutions = len(valid_group_ids)
                                inst_scope = InstitutionCountScope.COMPLETE_AGGREGATION.value
                                used_complete_agg = True
                    except (ValueError, TypeError):
                        pass

                if not used_complete_agg:
                    # 无法取得全量唯一机构数时，严禁用 *12 或 *0.15 放大采样数，正式特征保持 None 并明确标为采样统计
                    active_institutions = None
                    inst_scope = InstitutionCountScope.SAMPLED_WORKS_ONLY.value

            snap_sha = count_snap_meta["raw_sha256"] if count_snap_meta else self.calculate_bytes_hash(raw_bytes or b"")
            snap_id = count_snap_meta["snapshot_id"] if count_snap_meta else None
            snap_path = count_snap_meta["snapshot_path"] if count_snap_meta else None

            # 明确记录数据时间语义、版本与当前限制（不通过伪造 published_at 冒充历史版本）
            record_published_at = ingested_utc
            canonical_payload = json.dumps({
                "query": query,
                "year": year,
                "as_of_date": as_of_date.isoformat(),
                "ingested_at_utc": ingested_utc.isoformat(),
                "papers_count": papers_count,
                "citations_count": pit_citations_sum,
                "citations_status": citations_status,
                "citations_cutoff_year": eff_cutoff_year,
                "active_institutions": active_institutions,
                "sampled_institutions_count": sampled_inst_count,
                "institution_count_scope": inst_scope,
                "historical_eligibility": HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value,
                "raw_snapshot_id": snap_id,
                "raw_snapshot_sha256": snap_sha,
            }, ensure_ascii=False, sort_keys=True)

            prov = self.build_provenance(
                record_id=f"OPENALEX_{query}_{year}",
                as_of_date=as_of_date,
                published_at=record_published_at,
                source_url=count_snap_meta["request_url"] if count_snap_meta else f"{url}?search={query}&filter=publication_year:{year}",
                quality_status=DataQualityStatus.UNVERIFIED,
                raw_hash=snap_sha,
                snapshot_id=snap_id,
                snapshot_path=snap_path,
                historical_eligibility=HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value,
            )

            results.append({
                "year": year,
                "status": "OK",
                "as_of_date": as_of_date,
                "published_at": record_published_at,
                "verified_first_available_at": None,
                "ingested_at_utc": ingested_utc,
                "papers_count": papers_count,
                "citations_count": pit_citations_sum,
                "citations_status": citations_status,
                "citations_cutoff_year": eff_cutoff_year,
                "works_with_pit_counts": works_with_pit_counts,
                "works_missing_pit_counts": works_missing_pit_counts,
                "excluded_future_pub_works": excluded_future_pub_works,
                "active_institutions": active_institutions,
                "sampled_institutions_count": sampled_inst_count,
                "institution_count_scope": inst_scope,
                "historical_eligibility": HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value,
                "limitation_note": (
                    "已按 counts_by_year 过滤未来年份引用且机构未估算；"
                    "但当前 API 返回的历史年份统计包含后续索引补录，"
                    "不代表与目标历史年末当年实际可获快照完全一致。"
                ),
                "snapshot_id": snap_id,
                "snapshot_path": snap_path,
                "raw_sha256": snap_sha,
                "snapshot_meta": count_snap_meta,
                "quality_status": DataQualityStatus.UNVERIFIED.value,
                "provenance": prov.model_dump(),
                "raw_payload": canonical_payload,
            })

        return results

    def collect(self, **kwargs) -> Dict[str, Any]:
        query = kwargs.get("query", "solid-state battery")
        start_year = kwargs.get("start_year", 2018)
        end_year = kwargs.get("end_year", 2025)
        return {
            "data": self.fetch_topic_research_trends(
                query=query,
                start_year=start_year,
                end_year=end_year,
                save_snapshots=kwargs.get("save_snapshots", True),
                citations_cutoff_year=kwargs.get("citations_cutoff_year"),
            )
        }
