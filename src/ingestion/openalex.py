"""OpenAlex 学术文献与全球科研成果采集器 (OpenAlex API Client — V1.2 PIT & Provenance Overhaul)."""

import json
import os
from datetime import datetime, date, timezone
from typing import Dict, Any, List, Optional, Tuple, Set
import pandas as pd
import src.core.database as db_module
from src.ingestion.base import BaseCollector
from src.core.types import (
    CitationCountScope,
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
    def _filter_eligible_works_by_cutoff(
        works: List[Dict[str, Any]], cutoff_year: Optional[int]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """过滤掉发表日期晚于 cutoff_year-12-31 的未来论文，并返回 (合格论文列表, 被排除篇数)。"""
        if not isinstance(works, list):
            return [], 0
        max_cutoff = date(int(cutoff_year), 12, 31) if cutoff_year is not None else None
        eligible: List[Dict[str, Any]] = []
        excluded = 0
        for w in works:
            if not isinstance(w, dict):
                continue
            pub_s = w.get("publication_date")
            if max_cutoff is not None and pub_s:
                try:
                    if date.fromisoformat(str(pub_s)[:10]) > max_cutoff:
                        excluded += 1
                        continue
                except ValueError:
                    pass
            eligible.append(w)
        return eligible, excluded

    @staticmethod
    def evaluate_historical_pit_eligibility(
        record: Dict[str, Any],
        cutoff_date: date,
        db: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """严格检查记录是否具备历史预测时点 cutoff_date 的 Point-in-Time 回测资格。

        安全铁律 (P0)：
        1. 严禁信任调用方传入的 historical_eligibility / verified_first_available_at / raw_sha256 字段；
        2. 必须以数据库 raw_snapshots 中已登记且有真实磁盘文件支撑的记录为准；
        3. 必须重新计算磁盘快照字节 SHA-256 并核验第三方历史归档证据、首次可获取时间与截止日期；
        4. 无可验证历史版本证据时一律返回 HISTORICAL_VINTAGE_UNAVAILABLE。
        """
        cutoff_d = (
            cutoff_date.date()
            if isinstance(cutoff_date, datetime)
            else (cutoff_date if isinstance(cutoff_date, date) else pd.to_datetime(cutoff_date).date())
        )
        db_mgr = db if db is not None else db_module.db_manager
        snapshot_id = record.get("snapshot_id")
        if not snapshot_id:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "缺少 snapshot_id，未在系统数据库 raw_snapshots 登记快照记录。",
            }

        snap_rec = db_mgr.get_raw_snapshot(str(snapshot_id))
        if not snap_rec:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"快照 {snapshot_id} 未在系统数据库 raw_snapshots 注册，禁止信任自报历史资格字段。",
            }

        # 核验磁盘快照文件真实存在且重新计算实际 SHA-256 一致
        integrity = db_mgr.verify_snapshot_file_integrity(str(snapshot_id))
        if not integrity.get("verified"):
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"快照文件完整性校验失败：{integrity.get('reason')}",
            }

        # 若调用方同时附带了 raw_sha256 / raw_hash，必须与已核验的磁盘实际哈希完全一致
        caller_hash = record.get("raw_sha256") or record.get("raw_hash")
        actual_sha = str(integrity.get("sha256") or snap_rec.get("raw_sha256") or "").lower()
        if caller_hash and str(caller_hash).strip().lower() != actual_sha:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "调用方传入的 SHA-256 与数据库及磁盘实际快照不一致。",
            }

        db_eligibility = snap_rec.get("historical_eligibility")
        if db_eligibility == HistoricalEligibilityStatus.CURRENT_STATE_ONLY.value:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "数据库登记该快照仅反映当日实时状态 (CURRENT_STATE_ONLY)，禁止用于历史回测。",
            }

        if db_eligibility != HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": (
                    f"数据库登记历史资格为 {db_eligibility or 'UNKNOWN'}，"
                    "未达到 STRICT_PIT_VERIFIED 严格历史快照证明。"
                ),
            }

        if str(snap_rec.get("quality_status") or "").strip() != DataQualityStatus.VERIFIED.value:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"数据库快照 quality_status 为 {snap_rec.get('quality_status')}，未达到 VERIFIED。",
            }

        if snap_rec.get("error_message") or (
            snap_rec.get("http_status") is not None and not pd.isna(snap_rec.get("http_status")) and int(snap_rec.get("http_status")) != 200
        ):
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"数据库快照包含错误或非 200 响应 (http_status={snap_rec.get('http_status')})。",
            }

        source_auth = str(snap_rec.get("source_authenticity_status") or "").strip()
        if source_auth != "VERIFIED_HISTORICAL_ARCHIVE":
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": (
                    f"快照来源证据为 {source_auth or 'NONE'}，缺少权威历史归档来源凭证 (VERIFIED_HISTORICAL_ARCHIVE)。"
                ),
            }

        vintage_src = str(snap_rec.get("vintage_evidence_source") or "").strip()
        if vintage_src.lower() in {"", "self", "manual", "auto", "internal", "internal_research_prototype"}:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "缺少有效的独立历史首次公开时间来源证据 (vintage_evidence_source)。",
            }

        v_first = snap_rec.get("verified_first_available_at")
        if v_first is None or pd.isna(v_first):
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "数据库快照缺少已核验的首次公开时间 (verified_first_available_at)。",
            }

        try:
            avail_dt = pd.to_datetime(v_first)
            if pd.isna(avail_dt):
                raise ValueError("NaT")
        except Exception:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"无法解析数据库 verified_first_available_at: {v_first}",
            }

        if avail_dt.date() > cutoff_d:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"历史快照核验首次公开时间 {avail_dt.date()} 晚于预测截断日 {cutoff_d}。",
            }

        db_as_of = snap_rec.get("as_of_date")
        if db_as_of is not None and not pd.isna(db_as_of):
            as_of_d = db_as_of.date() if isinstance(db_as_of, datetime) else (
                db_as_of if isinstance(db_as_of, date) else pd.to_datetime(db_as_of).date()
            )
            if as_of_d > cutoff_d:
                return {
                    "eligible": False,
                    "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                    "reason": f"数据所属日期 {as_of_d} 晚于预测截断日 {cutoff_d}。",
                }

        return {
            "eligible": True,
            "status": HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value,
            "snapshot_id": str(snapshot_id),
            "sha256": actual_sha,
            "verified_first_available_at": avail_dt.isoformat(),
            "vintage_evidence_source": vintage_src,
            "reason": None,
        }

    @classmethod
    def verify_feature_lineage(
        cls,
        record_or_lineage: Dict[str, Any],
        feature_name: str,
        db: Optional[Any] = None,
        require_complete_scope: bool = True,
    ) -> Dict[str, Any]:
        """独立核验单个输出特征与其对应原始响应快照的完整性、范围口径和可重算一致性。"""
        if isinstance(record_or_lineage, dict):
            if isinstance(record_or_lineage.get("feature_lineage"), dict):
                lineage_map = record_or_lineage["feature_lineage"]
            elif isinstance(record_or_lineage.get("feature_lineage_json"), str):
                try:
                    parsed = json.loads(record_or_lineage["feature_lineage_json"])
                    lineage_map = parsed if isinstance(parsed, dict) else None
                except Exception:
                    lineage_map = None
            else:
                lineage_map = record_or_lineage
        elif isinstance(record_or_lineage, str):
            try:
                parsed = json.loads(record_or_lineage)
                lineage_map = parsed if isinstance(parsed, dict) else None
            except Exception:
                lineage_map = None
        else:
            lineage_map = None

        if not isinstance(lineage_map, dict) or feature_name not in lineage_map:
            return {"verified": False, "feature": feature_name, "reason": f"缺少特征 {feature_name} 的快照血缘映射"}

        f_lin = lineage_map[feature_name]
        if not isinstance(f_lin, dict):
            return {"verified": False, "feature": feature_name, "reason": "特征血缘结构无效"}

        snap_id = f_lin.get("snapshot_id")
        snap_path = f_lin.get("snapshot_path")
        expected_sha = f_lin.get("raw_sha256")
        scope = f_lin.get("scope")

        if not snap_id or not snap_path or not expected_sha:
            return {"verified": False, "feature": feature_name, "reason": f"特征 {feature_name} 缺失关联快照 ID、文件路径或 SHA-256"}

        if db is not None:
            db_snap = db.get_raw_snapshot(str(snap_id))
            if not db_snap:
                return {"verified": False, "feature": feature_name, "reason": f"特征依赖的快照 {snap_id} 未登记在 raw_snapshots"}
            if str(db_snap.get("raw_sha256") or "").lower() != str(expected_sha).lower():
                return {"verified": False, "feature": feature_name, "reason": f"快照 {snap_id} 数据库哈希与特征记录不一致"}

        # 1) 重新读取对应原始文件校验 SHA-256 未被篡改
        integrity = BaseCollector.verify_snapshot_integrity(snap_path, expected_sha)
        if not integrity.get("content_integrity_verified"):
            return {
                "verified": False,
                "feature": feature_name,
                "snapshot_id": snap_id,
                "reason": f"特征 {feature_name} 对应快照完整性校验失败: {integrity.get('reason')}",
            }

        # 2) 重新读取原始 JSON 重算对应指标
        resolved = BaseCollector._resolve_snapshot_path(snap_path)
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
        except Exception as exc:
            return {"verified": False, "feature": feature_name, "reason": f"重读快照 JSON 失败: {exc}"}

        recomputed_val: Any = None
        if feature_name == "papers_count":
            group_data = raw_json.get("group_by") if isinstance(raw_json, dict) else None
            if not isinstance(group_data, list):
                return {"verified": False, "feature": feature_name, "reason": "文献数快照缺少 group_by 列表"}
            if len(group_data) == 0:
                meta_cnt = (raw_json.get("meta") or {}).get("count")
                if meta_cnt != 0:
                    return {"verified": False, "feature": feature_name, "reason": "文献数快照为空且 meta.count != 0"}
                recomputed_val = 0
            else:
                try:
                    raw_cnt = group_data[0].get("count") if isinstance(group_data[0], dict) else None
                    if raw_cnt is None:
                        raise ValueError("missing count")
                    recomputed_val = int(raw_cnt)
                except (TypeError, ValueError) as exc:
                    return {"verified": False, "feature": feature_name, "reason": f"文献数快照 count 格式无效: {exc}"}
            recorded_val = f_lin.get("value")
            if recomputed_val != recorded_val:
                return {"verified": False, "feature": feature_name, "reason": f"文献数重算不一致: {recomputed_val} != {recorded_val}"}

        elif feature_name == "citations_count":
            results_list = raw_json.get("results") if isinstance(raw_json, dict) else None
            if not isinstance(results_list, list):
                return {"verified": False, "feature": feature_name, "reason": "引用量快照缺少 results 论文列表"}
            cutoff_yr = f_lin.get("cutoff_year")
            if cutoff_yr is None:
                return {"verified": False, "feature": feature_name, "reason": "引用量血缘缺少 cutoff_year"}
            eligible_list, _ = cls._filter_eligible_works_by_cutoff(results_list, int(cutoff_yr))
            calc_sum = 0
            valid_works = 0
            for w in eligible_list:
                c_val, c_stat = cls.extract_pit_citations_from_work(w, int(cutoff_yr))
                if c_stat == "OK" and c_val is not None:
                    calc_sum += c_val
                    valid_works += 1
            recomputed_val = calc_sum if valid_works > 0 else (0 if f_lin.get("sampled_value") == 0 and len(results_list) == 0 else None)
            expected_val = f_lin.get("formal_value") if scope == CitationCountScope.COMPLETE_DOMAIN_TOTAL.value else f_lin.get("sampled_value")
            if recomputed_val != expected_val:
                return {"verified": False, "feature": feature_name, "reason": f"引用量重算不一致: {recomputed_val} != {expected_val}"}

        elif feature_name == "active_institutions":
            method = f_lin.get("method")
            if method == "empty_domain_zero_papers":
                meta_cnt = (raw_json.get("meta") or {}).get("count")
                if meta_cnt != 0 or f_lin.get("formal_value") != 0:
                    return {"verified": False, "feature": feature_name, "reason": "零论文零机构重算核验失败"}
                recomputed_val = 0
            elif scope == InstitutionCountScope.COMPLETE_AGGREGATION.value:
                inst_groups = raw_json.get("group_by") if isinstance(raw_json, dict) else None
                if not isinstance(inst_groups, list):
                    return {"verified": False, "feature": feature_name, "reason": "机构聚合快照缺少 group_by"}
                meta_cnt = (raw_json.get("meta") or {}).get("groups_count")
                if meta_cnt is None or int(meta_cnt) >= cls.OPENALEX_GROUP_CAP or len(inst_groups) >= cls.OPENALEX_GROUP_CAP:
                    return {"verified": False, "feature": feature_name, "reason": "机构聚合达到上限或缺少 groups_count，不得标记为 COMPLETE_AGGREGATION"}
                valid_ids = {
                    str(g["key"]).strip()
                    for g in inst_groups
                    if isinstance(g, dict) and g.get("key") and str(g["key"]).lower() != "unknown"
                }
                recomputed_val = len(valid_ids)
                if recomputed_val != f_lin.get("formal_value"):
                    return {"verified": False, "feature": feature_name, "reason": f"机构数重算不一致: {recomputed_val} != {f_lin.get('formal_value')}"}
            elif scope == InstitutionCountScope.SAMPLED_WORKS_ONLY.value:
                if f_lin.get("formal_value") is not None:
                    return {"verified": False, "feature": feature_name, "reason": "采样机构口径不得填充 formal_value"}
                if isinstance(raw_json, dict) and isinstance(raw_json.get("results"), list):
                    eligible_list, _ = cls._filter_eligible_works_by_cutoff(raw_json["results"], f_lin.get("cutoff_year"))
                    recomputed_val = len(cls.extract_institution_ids_from_works(eligible_list)[0])
                    if f_lin.get("sampled_value") is None or recomputed_val != f_lin.get("sampled_value"):
                        return {
                            "verified": False,
                            "feature": feature_name,
                            "reason": f"采样机构数重算不一致: {recomputed_val} != {f_lin.get('sampled_value')}",
                        }
                    inst_proof = f_lin.get("institutions_snapshot")
                    if isinstance(inst_proof, dict) and inst_proof.get("snapshot_id"):
                        proof_sha = inst_proof.get("raw_sha256")
                        proof_path = inst_proof.get("snapshot_path")
                        if db is not None:
                            db_inst = db.get_raw_snapshot(str(inst_proof["snapshot_id"]))
                            if not db_inst:
                                return {"verified": False, "feature": feature_name, "reason": "机构辅助快照未登记在 raw_snapshots"}
                            db_proof_sha = db_inst.get("raw_sha256")
                            if proof_sha:
                                if str(db_proof_sha or "").lower() != str(proof_sha).lower():
                                    return {"verified": False, "feature": feature_name, "reason": "机构辅助快照数据库哈希不一致"}
                            elif db_proof_sha is not None:
                                return {"verified": False, "feature": feature_name, "reason": "机构辅助快照哈希缺失但库中存在哈希"}
                        if proof_sha and proof_path:
                            inst_int = BaseCollector.verify_snapshot_integrity(proof_path, proof_sha)
                            if not inst_int.get("content_integrity_verified"):
                                return {"verified": False, "feature": feature_name, "reason": f"机构辅助快照完整性校验失败: {inst_int.get('reason')}"}
                            try:
                                with open(BaseCollector._resolve_snapshot_path(proof_path), "r", encoding="utf-8") as inf:
                                    inst_raw = json.load(inf)
                            except Exception as exc:
                                return {"verified": False, "feature": feature_name, "reason": f"重读机构辅助快照失败: {exc}"}
                            inst_groups = inst_raw.get("group_by") if isinstance(inst_raw, dict) else None
                            inst_meta = (inst_raw.get("meta") or {}) if isinstance(inst_raw, dict) else {}
                            g_cnt = inst_meta.get("groups_count")
                            if not isinstance(inst_groups, list):
                                return {"verified": False, "feature": feature_name, "reason": "机构辅助快照缺少 group_by"}
                            if g_cnt is not None and int(g_cnt) < cls.OPENALEX_GROUP_CAP and len(inst_groups) < cls.OPENALEX_GROUP_CAP:
                                return {"verified": False, "feature": feature_name, "reason": "机构辅助快照未达 200 上限却降级为采样"}
                        elif not inst_proof.get("error_message"):
                            return {"verified": False, "feature": feature_name, "reason": "机构辅助快照缺少哈希且无失败说明"}
                elif isinstance(raw_json, dict) and isinstance(raw_json.get("group_by"), list):
                    if f_lin.get("sampled_value") is not None:
                        return {"verified": False, "feature": feature_name, "reason": "仅机构聚合快照存在 sampled_value 无来源"}
                else:
                    return {"verified": False, "feature": feature_name, "reason": "采样机构快照缺少 results 或 group_by"}
            elif scope != InstitutionCountScope.UNAVAILABLE.value:
                return {"verified": False, "feature": feature_name, "reason": f"未支持的机构口径或方法: scope={scope}, method={method}"}

        # 3) 统计口径是否满足正式使用要求
        ineligible_scopes = {
            InstitutionCountScope.UNAVAILABLE.value,
            InstitutionCountScope.SAMPLED_WORKS_ONLY.value,
            CitationCountScope.SAMPLED_TOP_CITED_BIASED.value,
            CitationCountScope.UNAVAILABLE.value,
        }
        if require_complete_scope and scope in ineligible_scopes:
            return {
                "verified": False,
                "feature": feature_name,
                "snapshot_id": snap_id,
                "scope": scope,
                "recomputed_value": recomputed_val,
                "content_integrity_verified": True,
                "reason": f"特征 {feature_name} 统计口径为 {scope}，不得冒充完整总体指标进入正式特征。",
            }

        return {
            "verified": True,
            "feature": feature_name,
            "snapshot_id": snap_id,
            "scope": scope,
            "recomputed_value": recomputed_val,
            "sha256": integrity.get("actual_sha256"),
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
                    "sampled_citations_count": None,
                    "citation_count_scope": CitationCountScope.UNAVAILABLE.value,
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
                    "snapshot_metas": [count_snap_meta] if count_snap_meta else [],
                    "feature_lineage": {},
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

            sample_snap_meta = None
            if save_snapshots:
                sample_snap_meta = self.save_raw_response_snapshot(
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

            # 统计 cutoff_year 之前真实发生的历史引用（严格区分采样引用与完整领域引用）
            sampled_citations_sum: Optional[int] = None
            formal_citations_count: Optional[int] = None
            citation_count_scope = CitationCountScope.UNAVAILABLE.value
            citations_status = "CITATIONS_BY_YEAR_UNAVAILABLE"
            works_with_pit_counts = 0
            works_missing_pit_counts = 0
            eligible_works, excluded_future_pub_works = self._filter_eligible_works_by_cutoff(
                sample_works, eff_cutoff_year
            )

            if papers_count == 0 and sample_parse_err is None:
                sampled_citations_sum = 0
                formal_citations_count = 0
                citation_count_scope = CitationCountScope.COMPLETE_DOMAIN_TOTAL.value
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
                    sampled_citations_sum = running_sum
                    # 仅当覆盖了领域全部论文且均具备有效按年拆分引用、不存在排序采样偏差时才为完整总体统计
                    if len(eligible_works) >= papers_count and works_missing_pit_counts == 0 and excluded_future_pub_works == 0:
                        formal_citations_count = running_sum
                        citation_count_scope = CitationCountScope.COMPLETE_DOMAIN_TOTAL.value
                        citations_status = "OK_COMPLETE_PIT_COUNTS"
                    else:
                        # 按当前 cited_by_count 倒序采样的子集只能放入 sampled_citations_count，正式字段严格置 None
                        formal_citations_count = None
                        citation_count_scope = CitationCountScope.SAMPLED_TOP_CITED_BIASED.value
                        citations_status = "OK_PARTIAL_SAMPLED_WORKS"
                else:
                    sampled_citations_sum = None
                    formal_citations_count = None
                    citation_count_scope = CitationCountScope.UNAVAILABLE.value
                    citations_status = "CITATIONS_BY_YEAR_UNAVAILABLE"
            elif sample_parse_err is not None:
                sampled_citations_sum = None
                formal_citations_count = None
                citation_count_scope = CitationCountScope.UNAVAILABLE.value
                citations_status = f"WORKS_SAMPLE_FAILED:{sample_parse_err}"

            # 3) 机构统计：先看按采样去重，只有完整聚合接口且未达上限才允许填充 active_institutions
            sampled_inst_ids, _ = self.extract_institution_ids_from_works(eligible_works)
            sampled_inst_count: Optional[int] = len(sampled_inst_ids) if sample_parse_err is None else None
            active_institutions: Optional[int] = None
            inst_scope = InstitutionCountScope.UNAVAILABLE.value
            inst_snap_meta = None

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
                inst_resp, inst_status, inst_err = self.fetch_raw_with_audit(url, params=inst_params)
                used_complete_agg = False
                inst_raw_bytes: Optional[bytes] = None
                inst_parse_err: Optional[str] = None
                if inst_resp is not None and inst_status == 200:
                    inst_raw_bytes = (
                        inst_resp.content
                        if isinstance(inst_resp.content, (bytes, bytearray))
                        else inst_resp.text.encode("utf-8")
                    )
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
                        else:
                            inst_parse_err = "INST_MALFORMED_GROUP_BY"
                    except (ValueError, TypeError) as exc:
                        inst_parse_err = f"INST_JSON_PARSE_ERROR: {exc}"
                else:
                    inst_parse_err = inst_err or f"INST_HTTP_{inst_status}_FAILED"

                if save_snapshots:
                    inst_snap_meta = self.save_raw_response_snapshot(
                        snapshot_dir=self.snapshot_dir,
                        request_url=url,
                        query_params=inst_params,
                        http_status=inst_status,
                        raw_bytes=inst_raw_bytes if inst_parse_err is None else None,
                        as_of_date=as_of_date,
                        verified_first_available_at=None,
                        quality_status=DataQualityStatus.UNVERIFIED if inst_parse_err is None else DataQualityStatus.UNAVAILABLE,
                        historical_eligibility=HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS
                        if inst_parse_err is None
                        else HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE,
                        error_message=inst_parse_err,
                        ingested_at_utc=ingested_utc,
                        file_prefix=f"openalex_{query}_{year}_institutions",
                    )

                if not used_complete_agg:
                    # 无法取得全量唯一机构数时，严禁用 *12 或 *0.15 放大采样数，正式特征保持 None 并明确标为采样统计
                    active_institutions = None
                    inst_scope = InstitutionCountScope.SAMPLED_WORKS_ONLY.value

            snap_sha = count_snap_meta["raw_sha256"] if count_snap_meta else self.calculate_bytes_hash(raw_bytes or b"")
            snap_id = count_snap_meta["snapshot_id"] if count_snap_meta else None
            snap_path = count_snap_meta["snapshot_path"] if count_snap_meta else None

            inst_has_file = bool(
                inst_snap_meta and inst_snap_meta.get("raw_sha256") and inst_snap_meta.get("snapshot_path")
            )
            sample_has_file = bool(
                sample_snap_meta and sample_snap_meta.get("raw_sha256") and sample_snap_meta.get("snapshot_path")
            )
            if papers_count == 0:
                inst_primary_meta = count_snap_meta
                inst_method = "empty_domain_zero_papers"
            elif inst_scope == InstitutionCountScope.COMPLETE_AGGREGATION.value and inst_has_file:
                inst_primary_meta = inst_snap_meta
                inst_method = "openalex_group_by_institution_id_lt_200_cap"
            else:
                inst_primary_meta = sample_snap_meta if sample_has_file else (inst_snap_meta if inst_has_file else None)
                inst_method = "sampled_works_institution_id_dedup"

            inst_lineage_entry: Dict[str, Any] = {
                "snapshot_id": inst_primary_meta["snapshot_id"] if inst_primary_meta else None,
                "raw_sha256": inst_primary_meta["raw_sha256"] if inst_primary_meta else None,
                "snapshot_path": inst_primary_meta["snapshot_path"] if inst_primary_meta else None,
                "scope": inst_scope,
                "method": inst_method,
                "cutoff_year": eff_cutoff_year,
                "formal_value": active_institutions,
                "sampled_value": sampled_inst_count,
            }
            if inst_snap_meta is not None and inst_primary_meta is not inst_snap_meta:
                inst_lineage_entry["institutions_snapshot"] = {
                    "snapshot_id": inst_snap_meta.get("snapshot_id"),
                    "raw_sha256": inst_snap_meta.get("raw_sha256"),
                    "snapshot_path": inst_snap_meta.get("snapshot_path"),
                    "http_status": inst_snap_meta.get("http_status"),
                    "error_message": inst_snap_meta.get("error_message"),
                }

            all_snap_metas = [m for m in [count_snap_meta, sample_snap_meta, inst_snap_meta] if m is not None]
            feature_lineage = {
                "papers_count": {
                    "snapshot_id": snap_id,
                    "raw_sha256": snap_sha,
                    "snapshot_path": snap_path,
                    "scope": CitationCountScope.COMPLETE_DOMAIN_TOTAL.value,
                    "method": "openalex_group_by_publication_year",
                    "value": papers_count,
                },
                "citations_count": {
                    "snapshot_id": sample_snap_meta["snapshot_id"] if sample_snap_meta else None,
                    "raw_sha256": sample_snap_meta["raw_sha256"] if sample_snap_meta else None,
                    "snapshot_path": sample_snap_meta["snapshot_path"] if sample_snap_meta else None,
                    "scope": citation_count_scope,
                    "method": "sum_counts_by_year_le_cutoff",
                    "cutoff_year": eff_cutoff_year,
                    "formal_value": formal_citations_count,
                    "sampled_value": sampled_citations_sum,
                    "sampled_works_count": len(sample_works),
                    "sampling_order": "cited_by_count:desc",
                },
                "active_institutions": inst_lineage_entry,
            }

            # 明确记录数据时间语义、版本与当前限制（不通过伪造 published_at 冒充历史版本）
            record_published_at = ingested_utc
            canonical_payload = json.dumps({
                "query": query,
                "year": year,
                "as_of_date": as_of_date.isoformat(),
                "ingested_at_utc": ingested_utc.isoformat(),
                "papers_count": papers_count,
                "citations_count": formal_citations_count,
                "sampled_citations_count": sampled_citations_sum,
                "citation_count_scope": citation_count_scope,
                "citations_status": citations_status,
                "citations_cutoff_year": eff_cutoff_year,
                "active_institutions": active_institutions,
                "sampled_institutions_count": sampled_inst_count,
                "institution_count_scope": inst_scope,
                "historical_eligibility": HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value,
                "feature_lineage": feature_lineage,
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
                "citations_count": formal_citations_count,
                "sampled_citations_count": sampled_citations_sum,
                "citation_count_scope": citation_count_scope,
                "citations_status": citations_status,
                "citations_cutoff_year": eff_cutoff_year,
                "works_with_pit_counts": works_with_pit_counts,
                "works_missing_pit_counts": works_missing_pit_counts,
                "excluded_future_pub_works": excluded_future_pub_works,
                "active_institutions": active_institutions,
                "sampled_institutions_count": sampled_inst_count,
                "institution_count_scope": inst_scope,
                "historical_eligibility": HistoricalEligibilityStatus.CURRENT_SNAPSHOT_HISTORICAL_STATS.value,
                "sample_selection_bias_note": (
                    "仅当覆盖完整领域论文时填充 citations_count；按当前 cited_by_count 倒序采样前 10 篇存在选择偏差，"
                    "采样引用放入 sampled_citations_count，正式 citations_count 严格置 None。"
                ),
                "limitation_note": (
                    "已按 counts_by_year 过滤未来年份引用且机构未估算；"
                    "但当前 API 返回的历史年份统计包含后续索引补录，"
                    "不代表与目标历史年末当年实际可获快照完全一致。"
                ),
                "snapshot_id": snap_id,
                "snapshot_path": snap_path,
                "raw_sha256": snap_sha,
                "snapshot_meta": count_snap_meta,
                "snapshot_metas": all_snap_metas,
                "feature_lineage": feature_lineage,
                "quality_status": DataQualityStatus.UNVERIFIED.value,
                "provenance": prov.model_dump(),
                "raw_payload": canonical_payload,
            })

        return results

    def collect(self, **kwargs) -> Dict[str, Any]:
        query = kwargs.get("query", '"solid-state battery"')
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
