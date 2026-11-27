"""受控历史归档认证与证据核验中心 (Controlled Historical Archive Certifier).

严格实现三层职责分离 (PR #7 P0 最终整改)：
1. 第一层（普通采集 `save_raw_snapshot`）：仅记录原始响应、哈希、采集时间与未认证声明，
   严禁直接授予 `STRICT_PIT_VERIFIED` 或 `VERIFIED_HISTORICAL_ARCHIVE`。
2. 第二层（受控历史归档核验 `HistoricalArchiveCertifier`）：验证真实存在的历史归档文件、
   第三方稳定归档标识、归档发布时间、内容与快照的实际字节/清单对应关系，并生成进程内不可伪造的
   HMAC-SHA256 认证封签 (`attestation_seal`) 持久化至 `historical_archive_attestations`。
   同时将离线测试凭证 (`is_test_only=True`) 与生产级历史归档物理隔离。
3. 第三层（历史预测准入 `verify_attestation`）：评估时忽略调用方自报状态与 `raw_snapshots`
   中任何人工篡改列，重新校验封签、磁盘快照 SHA-256、归档文件 SHA-256、内容对应关系与截断日期。
"""

import hashlib
import hmac
import json
import os
import secrets
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from src.core.config import get_project_root
from src.core.types import HistoricalEligibilityStatus


# 进程内受控签发密钥：不落盘、不写库、不暴露给业务调用方，阻断直接篡改数据库/JSON 伪造认证
_RUNTIME_CERTIFIER_KEY: bytes = secrets.token_bytes(32)

# 当前受信任的外部第三方生产级历史归档权威注册表。
# 截至 V1.2 第一阶段，系统尚未接入真实可独立验真的外部第三方历史归档源，因此保持为空；
# 任何内部模块自填字符串（如 "OpenAlex_Official_2020_Archive_Release"）一律不得获得生产级认证。
TRUSTED_PRODUCTION_ARCHIVE_AUTHORITIES: Dict[str, Dict[str, Any]] = {}


def _resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(str(get_project_root()), path_str)


def _normalize_date(d: Any) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    parsed = pd.to_datetime(d)
    if pd.isna(parsed):
        raise ValueError(f"无法解析日期: {d}")
    return parsed.date()


def _normalize_timestamp(ts: Any) -> Optional[datetime]:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    try:
        parsed = pd.to_datetime(ts)
        if pd.isna(parsed):
            return None
        dt = parsed.to_pydatetime()
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def _compute_attestation_seal(fields: Dict[str, Any]) -> str:
    canonical_payload = json.dumps(
        {
            "attestation_id": str(fields["attestation_id"]),
            "snapshot_id": str(fields["snapshot_id"]),
            "archive_id": str(fields["archive_id"]),
            "archive_source_authority": str(fields["archive_source_authority"]),
            "archive_file_sha256": str(fields["archive_file_sha256"]).lower(),
            "snapshot_sha256": str(fields["snapshot_sha256"]).lower(),
            "verified_published_at": str(fields["verified_published_at"]),
            "as_of_date": str(fields["as_of_date"]),
            "certification_mode": str(fields["certification_mode"]),
            "is_test_only": bool(fields["is_test_only"]),
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hmac.new(_RUNTIME_CERTIFIER_KEY, canonical_payload, hashlib.sha256).hexdigest()


def _verify_archive_content_correspondence(
    archive_file_path: str,
    snapshot_sha256: str,
    expected_archive_sha256: Optional[str] = None,
    verified_published_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """核验真实存在的历史归档文件与目标快照的哈希一致性及内容对应关系。"""
    if not archive_file_path:
        return {"verified": False, "reason": "缺少历史归档文件路径 (archive_file_path)。"}

    resolved = _resolve_path(archive_file_path)
    if not os.path.exists(resolved) or not os.path.isfile(resolved):
        return {"verified": False, "reason": f"历史归档文件不存在: {resolved}"}

    with open(resolved, "rb") as f:
        archive_bytes = f.read()
    actual_archive_sha = hashlib.sha256(archive_bytes).hexdigest()

    if expected_archive_sha256 and actual_archive_sha.lower() != str(expected_archive_sha256).lower():
        return {
            "verified": False,
            "actual_archive_sha256": actual_archive_sha,
            "reason": "历史归档文件内容已被修改，SHA-256 校验失败。",
        }

    snap_sha_low = str(snapshot_sha256).lower()

    # 情况 1：归档文件本身与目标快照字节完全一致
    if actual_archive_sha.lower() == snap_sha_low:
        return {
            "verified": True,
            "actual_archive_sha256": actual_archive_sha,
            "correspondence_type": "EXACT_ARCHIVE_BYTES_MATCH",
        }

    # 情况 2：归档文件为结构化归档清单/封装 JSON，需显式包含目标快照的 SHA-256 与匹配发布时间
    try:
        archive_obj = json.loads(archive_bytes.decode("utf-8"))
    except Exception:
        return {
            "verified": False,
            "actual_archive_sha256": actual_archive_sha,
            "reason": "历史归档文件既不与快照字节一致，也非有效 JSON 归档清单，无法证明内容对应关系。",
        }

    if not isinstance(archive_obj, dict):
        return {
            "verified": False,
            "actual_archive_sha256": actual_archive_sha,
            "reason": "历史归档清单格式无效（非 JSON 对象）。",
        }

    matched_entry: Optional[Dict[str, Any]] = None
    top_sha = archive_obj.get("content_sha256") or archive_obj.get("raw_sha256")
    if top_sha and str(top_sha).strip().lower() == snap_sha_low:
        matched_entry = archive_obj
    else:
        for list_key in ("files", "records", "snapshots"):
            items = archive_obj.get(list_key)
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    cand_sha = item.get("sha256") or item.get("raw_sha256") or item.get("content_sha256")
                    if cand_sha and str(cand_sha).strip().lower() == snap_sha_low:
                        matched_entry = item
                        break
            if matched_entry is not None:
                break

    if matched_entry is None:
        return {
            "verified": False,
            "actual_archive_sha256": actual_archive_sha,
            "reason": "历史归档清单中未找到与目标快照 SHA-256 对应的内容条目。",
        }

    # 若归档清单内含发布时间，需与申报的归档发布时间日期一致
    embedded_pub = matched_entry.get("published_at") or archive_obj.get("published_at")
    if embedded_pub is not None and verified_published_at is not None:
        emb_dt = _normalize_timestamp(embedded_pub)
        if emb_dt is None or emb_dt.date() != verified_published_at.date():
            return {
                "verified": False,
                "actual_archive_sha256": actual_archive_sha,
                "reason": f"归档清单内嵌发布时间 ({embedded_pub}) 与申报发布时间 ({verified_published_at.isoformat()}) 不一致。",
            }

    return {
        "verified": True,
        "actual_archive_sha256": actual_archive_sha,
        "correspondence_type": "ARCHIVE_MANIFEST_SHA256_MATCH",
    }


class HistoricalArchiveCertifier:
    """受控历史归档认证器：独立于普通数据采集，禁止调用方自报通过。"""

    @classmethod
    def certify_snapshot_production(
        cls,
        db: Any,
        snapshot_id: str,
        archive_id: str,
        archive_source_authority: str,
        archive_file_path: str,
        verified_published_at: Any,
        evidence_origin: str = "external_archive",
    ) -> Dict[str, Any]:
        """正式生产级历史归档认证入口。

        仅允许已在 `TRUSTED_PRODUCTION_ARCHIVE_AUTHORITIES` 注册的独立外部权威源，
        当前注册表为空，因此任何内部模块或调用方均无法自宣获得生产级历史 PIT 认证。
        """
        auth = str(archive_source_authority or "").strip()
        if auth not in TRUSTED_PRODUCTION_ARCHIVE_AUTHORITIES:
            return {
                "certified": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": (
                    f"来源 '{auth or 'EMPTY'}' 不在受信任的外部独立历史归档注册表内，"
                    "禁止内部模块仅凭字符串声明自认证生产级历史 PIT 资格。"
                ),
            }

        return cls._issue_attestation(
            db=db,
            snapshot_id=snapshot_id,
            archive_id=archive_id,
            archive_source_authority=auth,
            archive_file_path=archive_file_path,
            verified_published_at=verified_published_at,
            evidence_origin=evidence_origin,
            is_test_only=False,
        )

    @classmethod
    def certify_snapshot_with_test_credential(
        cls,
        db: Any,
        snapshot_id: str,
        archive_id: str,
        archive_file_path: str,
        verified_published_at: Any,
        test_authority: str = "OFFLINE_CONTROLLED_TEST_ARCHIVE",
        evidence_origin: str = "controlled_offline_test_fixture",
    ) -> Dict[str, Any]:
        """受控离线测试凭证认证入口：仅用于验证正向核验逻辑，物理隔离于生产级准入。"""
        return cls._issue_attestation(
            db=db,
            snapshot_id=snapshot_id,
            archive_id=archive_id,
            archive_source_authority=test_authority,
            archive_file_path=archive_file_path,
            verified_published_at=verified_published_at,
            evidence_origin=evidence_origin,
            is_test_only=True,
        )

    @classmethod
    def _issue_attestation(
        cls,
        db: Any,
        snapshot_id: str,
        archive_id: str,
        archive_source_authority: str,
        archive_file_path: str,
        verified_published_at: Any,
        evidence_origin: str,
        is_test_only: bool,
    ) -> Dict[str, Any]:
        if not snapshot_id or not str(archive_id or "").strip():
            return {
                "certified": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "缺少 snapshot_id 或第三方归档稳定标识 archive_id。",
            }

        pub_dt = _normalize_timestamp(verified_published_at)
        if pub_dt is None:
            return {
                "certified": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "缺少可核验的历史归档发布时间 (verified_published_at)。",
            }

        snap_rec = db.get_raw_snapshot(str(snapshot_id))
        if not snap_rec:
            return {
                "certified": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"快照 {snapshot_id} 未在 raw_snapshots 登记。",
            }

        integrity = db.verify_snapshot_file_integrity(str(snapshot_id))
        if not integrity.get("verified"):
            return {
                "certified": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"快照文件完整性校验失败: {integrity.get('reason')}",
            }

        actual_snap_sha = str(integrity["sha256"]).lower()
        corr = _verify_archive_content_correspondence(
            archive_file_path=archive_file_path,
            snapshot_sha256=actual_snap_sha,
            expected_archive_sha256=None,
            verified_published_at=pub_dt,
        )
        if not corr.get("verified"):
            return {
                "certified": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"历史归档证据核验失败: {corr.get('reason')}",
            }

        as_of_d = _normalize_date(snap_rec["as_of_date"])
        cert_mode = "TEST_ONLY_OFFLINE_CREDENTIAL" if is_test_only else "PRODUCTION_VERIFIED_ARCHIVE"
        now_utc = datetime.utcnow()
        attestation_id = f"att_{'test' if is_test_only else 'prod'}_{snapshot_id}_{secrets.token_hex(6)}"

        seal_fields = {
            "attestation_id": attestation_id,
            "snapshot_id": str(snapshot_id),
            "archive_id": str(archive_id).strip(),
            "archive_source_authority": str(archive_source_authority).strip(),
            "archive_file_sha256": corr["actual_archive_sha256"],
            "snapshot_sha256": actual_snap_sha,
            "verified_published_at": pub_dt.isoformat(),
            "as_of_date": as_of_d.isoformat(),
            "certification_mode": cert_mode,
            "is_test_only": bool(is_test_only),
        }
        seal = _compute_attestation_seal(seal_fields)

        att_record = {
            "attestation_id": attestation_id,
            "snapshot_id": str(snapshot_id),
            "archive_id": str(archive_id).strip(),
            "archive_source_authority": str(archive_source_authority).strip(),
            "archive_file_path": archive_file_path,
            "archive_file_sha256": corr["actual_archive_sha256"],
            "snapshot_sha256": actual_snap_sha,
            "verified_published_at": pub_dt,
            "as_of_date": as_of_d,
            "evidence_origin": evidence_origin,
            "certification_mode": cert_mode,
            "is_test_only": bool(is_test_only),
            "verified_at_utc": now_utc,
            "attestation_seal": seal,
        }
        db.save_historical_archive_attestation(att_record)
        return {
            "certified": True,
            "status": HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value,
            "attestation_id": attestation_id,
            "certification_mode": cert_mode,
            "is_test_only": bool(is_test_only),
            "correspondence_type": corr.get("correspondence_type"),
            "snapshot_sha256": actual_snap_sha,
            "archive_file_sha256": corr["actual_archive_sha256"],
            "verified_published_at": pub_dt.isoformat(),
        }

    @classmethod
    def verify_attestation(
        cls,
        db: Any,
        snapshot_id: str,
        cutoff_date: Any,
        allow_test_attestation: bool = False,
        caller_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """第三层受控准入核验：不信任调用方或 raw_snapshots 自填列，重验封签与磁盘原始证据。"""
        cutoff_d = _normalize_date(cutoff_date)
        if not snapshot_id:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "缺少 snapshot_id，未在系统数据库 raw_snapshots 登记快照记录。",
            }

        snap_rec = db.get_raw_snapshot(str(snapshot_id))
        if not snap_rec:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"快照 {snapshot_id} 未在系统数据库 raw_snapshots 注册，禁止信任自报历史资格字段。",
            }

        integrity = db.verify_snapshot_file_integrity(str(snapshot_id))
        if not integrity.get("verified"):
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"快照文件完整性校验失败：{integrity.get('reason')}",
            }

        actual_sha = str(integrity.get("sha256") or "").lower()
        if caller_hash and str(caller_hash).strip().lower() != actual_sha:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "调用方传入的 SHA-256 与数据库及磁盘实际快照不一致。",
            }

        if snap_rec.get("error_message") is not None and not pd.isna(snap_rec.get("error_message")) and str(snap_rec.get("error_message")).strip():
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"快照包含采集错误记录: {snap_rec.get('error_message')}",
            }

        att = db.get_historical_archive_attestation(str(snapshot_id))
        if not att:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "缺少经受控历史归档核验流程签发的有效认证记录，禁止凭 raw_snapshots 字段或调用方自报放行。",
            }

        pub_dt = _normalize_timestamp(att.get("verified_published_at"))
        if pub_dt is None:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "认证记录缺少有效的历史归档发布时间。",
            }

        try:
            att_as_of_d = _normalize_date(att.get("as_of_date"))
        except Exception:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "认证记录缺少有效的数据所属日期 (as_of_date)。",
            }

        is_test_only = bool(att.get("is_test_only"))
        cert_mode = str(att.get("certification_mode") or "")

        # 核验受控封签，阻断直接通过 SQL INSERT/UPDATE 伪造认证记录
        expected_seal = _compute_attestation_seal(
            {
                "attestation_id": att.get("attestation_id"),
                "snapshot_id": att.get("snapshot_id"),
                "archive_id": att.get("archive_id"),
                "archive_source_authority": att.get("archive_source_authority"),
                "archive_file_sha256": att.get("archive_file_sha256"),
                "snapshot_sha256": att.get("snapshot_sha256"),
                "verified_published_at": pub_dt.isoformat(),
                "as_of_date": att_as_of_d.isoformat(),
                "certification_mode": cert_mode,
                "is_test_only": is_test_only,
            }
        )
        stored_seal = str(att.get("attestation_seal") or "")
        if not stored_seal or not hmac.compare_digest(expected_seal, stored_seal):
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "历史归档认证封签校验失败：检测到未经受控流程写入或被篡改的认证记录。",
            }

        if is_test_only or cert_mode != "PRODUCTION_VERIFIED_ARCHIVE":
            if not allow_test_attestation:
                return {
                    "eligible": False,
                    "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                    "reason": "该记录仅使用离线测试凭证 (TEST_ONLY_OFFLINE_CREDENTIAL)，严禁作为生产级严格 PIT 历史归档准入。",
                }
        else:
            if str(att.get("archive_source_authority") or "").strip() not in TRUSTED_PRODUCTION_ARCHIVE_AUTHORITIES:
                return {
                    "eligible": False,
                    "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                    "reason": "生产认证来源不在当前受信任外部历史归档注册表中。",
                }

        # 重新比对当前磁盘快照 SHA-256 与认证时记录是否一致
        if actual_sha != str(att.get("snapshot_sha256") or "").lower():
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": "原始快照磁盘内容在认证后发生变更，历史认证已失效。",
            }

        # 重新读取归档文件核验 SHA-256 与内容对应关系
        corr = _verify_archive_content_correspondence(
            archive_file_path=str(att.get("archive_file_path") or ""),
            snapshot_sha256=actual_sha,
            expected_archive_sha256=str(att.get("archive_file_sha256") or ""),
            verified_published_at=pub_dt,
        )
        if not corr.get("verified"):
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"历史归档证据复验失败: {corr.get('reason')}",
            }

        if pub_dt.date() > cutoff_d:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"历史归档核验发布时间 {pub_dt.date()} 晚于预测截断日 {cutoff_d}。",
            }

        if att_as_of_d > cutoff_d:
            return {
                "eligible": False,
                "status": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
                "reason": f"数据所属日期 {att_as_of_d} 晚于预测截断日 {cutoff_d}。",
            }

        return {
            "eligible": True,
            "status": HistoricalEligibilityStatus.STRICT_PIT_VERIFIED.value,
            "snapshot_id": str(snapshot_id),
            "attestation_id": str(att.get("attestation_id")),
            "archive_id": str(att.get("archive_id")),
            "archive_source_authority": str(att.get("archive_source_authority")),
            "certification_mode": cert_mode,
            "is_test_only": is_test_only,
            "sha256": actual_sha,
            "verified_first_available_at": pub_dt.isoformat(),
            "reason": None,
        }
