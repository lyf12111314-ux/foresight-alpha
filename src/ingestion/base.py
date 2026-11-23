"""数据采集抽象基类 (Base Collector with Provenance & Point-in-Time Support)."""

import hashlib
import json
import os
import re
import time
import uuid
import requests
from abc import ABC, abstractmethod
from datetime import datetime, date, timezone
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlencode

from src.core.config import get_project_root
from src.core.types import (
    ProvenanceMetadata,
    DataQualityStatus,
    HistoricalEligibilityStatus,
    RawSnapshotRecord,
)

PROCESSOR_CODE_VERSION = "v1.2-p0-data-provenance"


class BaseCollector(ABC):
    """数据采集器基类"""

    def __init__(self, source_name: str, base_url: str, timeout_sec: int = 15):
        self.source_name = source_name
        self.base_url = base_url
        self.timeout_sec = timeout_sec
        self.processor_code_version = PROCESSOR_CODE_VERSION
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ForesightAlpha-ResearchAgent/1.2 (+https://github.com/lyf12111314-ux/foresight-alpha)"
        })

    def calculate_hash(self, raw_content: str) -> str:
        """计算原始载荷字符串的 SHA-256 哈希值"""
        return hashlib.sha256(raw_content.encode("utf-8")).hexdigest()

    def calculate_bytes_hash(self, raw_bytes: bytes) -> str:
        """计算原始字节流的真实 SHA-256 哈希值"""
        return hashlib.sha256(raw_bytes).hexdigest()

    @staticmethod
    def _portable_path(path: str) -> str:
        """若路径位于仓库根目录内，转为跨机器可复用的相对路径（POSIX 风格）"""
        abs_p = os.path.abspath(path)
        root_p = os.path.abspath(str(get_project_root()))
        try:
            if os.path.commonpath([root_p, abs_p]) == root_p:
                return os.path.relpath(abs_p, root_p).replace(os.sep, "/")
        except ValueError:
            pass
        return abs_p

    @staticmethod
    def _resolve_snapshot_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(str(get_project_root()), path)

    @staticmethod
    def verify_snapshot_integrity(snapshot_path: str, expected_sha256: str) -> Dict[str, Any]:
        """重新读取快照原始字节计算 SHA-256，校验内容完整性（不等于历史时点可信）"""
        if not snapshot_path or not expected_sha256:
            return {"content_integrity_verified": False, "reason": "缺少快照路径或预期 SHA-256"}
        resolved = BaseCollector._resolve_snapshot_path(snapshot_path)
        if not os.path.exists(resolved):
            return {"content_integrity_verified": False, "reason": f"快照文件不存在: {resolved}"}
        with open(resolved, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        matched = actual_hash.lower() == str(expected_sha256).lower()
        return {
            "content_integrity_verified": matched,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_hash,
            "reason": None if matched else "原始快照内容已发生篡改或损坏，SHA-256 不一致"
        }

    def build_provenance(
        self,
        record_id: str,
        as_of_date: date,
        published_at: datetime,
        source_url: Optional[str] = None,
        quality_status: DataQualityStatus = DataQualityStatus.VALIDATED,
        raw_hash: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        snapshot_path: Optional[str] = None,
        historical_eligibility: Optional[str] = None
    ) -> ProvenanceMetadata:
        """构建标准化溯源元数据"""
        return ProvenanceMetadata(
            record_id=record_id,
            source_name=self.source_name,
            source_url=source_url,
            as_of_date=as_of_date,
            published_at=published_at,
            ingested_at=datetime.utcnow(),
            quality_status=quality_status,
            data_version=self.processor_code_version,
            raw_hash=raw_hash,
            snapshot_id=snapshot_id,
            snapshot_path=snapshot_path,
            historical_eligibility=historical_eligibility,
        )

    def fetch_raw_with_audit(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        backoff_sec: float = 1.0
    ) -> Tuple[Optional[requests.Response], Optional[int], Optional[str]]:
        """带详细 HTTP 状态码与异常信息捕获的健壮抓取"""
        last_status: Optional[int] = None
        last_error: Optional[str] = None
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout_sec)
                last_status = getattr(resp, "status_code", None)
                if last_status == 200:
                    return resp, 200, None
                elif last_status in [429, 500, 502, 503, 504]:
                    last_error = f"HTTP {last_status} (attempt {attempt + 1}/{max_retries})"
                    if attempt < max_retries - 1:
                        time.sleep(backoff_sec * (2 ** attempt))
                else:
                    last_error = f"HTTP {last_status} non-retryable response"
                    return None, last_status, last_error
            except requests.RequestException as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < max_retries - 1:
                    time.sleep(backoff_sec * (2 ** attempt))
        return None, last_status, last_error or "Request failed after retries"

    def fetch_with_retry(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        backoff_sec: float = 1.0
    ) -> Optional[requests.Response]:
        """具备指数退避重试的健壮 HTTP 抓取 (向后兼容接口)"""
        resp, _, _ = self.fetch_raw_with_audit(
            url=url, params=params, max_retries=max_retries, backoff_sec=backoff_sec
        )
        return resp

    def save_raw_response_snapshot(
        self,
        snapshot_dir: str,
        request_url: str,
        query_params: Optional[Dict[str, Any]],
        http_status: Optional[int],
        raw_bytes: Optional[bytes],
        as_of_date: date,
        verified_first_available_at: Optional[datetime] = None,
        quality_status: DataQualityStatus = DataQualityStatus.UNVERIFIED,
        historical_eligibility: HistoricalEligibilityStatus = HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE,
        error_message: Optional[str] = None,
        ingested_at_utc: Optional[datetime] = None,
        file_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """保存原始 HTTP 响应字节与审计元数据，明确分离内容完整性、来源真实性与历史可获得性。

        - 使用不可覆盖的唯一 snapshot_id；
        - 若响应字节存在，写入原始 .raw.json（内容按 sha256 去重引用或独立存储）+ 独立采集事件 .meta.json，不丢失每次采集日志；
        - 采集失败时 (raw_bytes is None 或 http_status != 200) 绝不生成伪造正常哈希。
        """
        os.makedirs(snapshot_dir, exist_ok=True)
        now_utc = ingested_at_utc or datetime.now(timezone.utc).replace(tzinfo=None)
        ts_str = now_utc.strftime("%Y%m%dT%H%M%SZ")
        safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", file_prefix or self.source_name.lower())
        rand_hex = uuid.uuid4().hex[:8]
        qparams = query_params or {}
        full_url = f"{request_url}?{urlencode(qparams)}" if qparams else request_url

        if raw_bytes is not None and http_status == 200 and not error_message:
            raw_sha256 = self.calculate_bytes_hash(raw_bytes)
            snapshot_id = f"{safe_prefix}_{ts_str}_{raw_sha256[:12]}_{rand_hex}"
            # 允许按内容哈希去重保存物理载荷文件，同时每次采集均写入不可覆盖的元数据记录
            raw_filename = f"{safe_prefix}_{raw_sha256[:16]}.raw.json"
            raw_abs_path = os.path.abspath(os.path.join(snapshot_dir, raw_filename))
            if not os.path.exists(raw_abs_path):
                with open(raw_abs_path, "wb") as f:
                    f.write(raw_bytes)
            raw_path = self._portable_path(raw_abs_path)
            integrity_check = self.verify_snapshot_integrity(raw_path, raw_sha256)
            content_integrity_verified = bool(integrity_check.get("content_integrity_verified", False))
            source_authenticity = "REAL_HTTP_200_RESPONSE"
        else:
            raw_sha256 = None
            raw_path = None
            snapshot_id = f"{safe_prefix}_{ts_str}_failed_{rand_hex}"
            content_integrity_verified = False
            source_authenticity = "FETCH_FAILED_NO_RESPONSE"

        record = RawSnapshotRecord(
            snapshot_id=snapshot_id,
            source_name=self.source_name,
            request_url=full_url,
            query_params=qparams,
            http_status=http_status,
            ingested_at_utc=now_utc,
            as_of_date=as_of_date,
            verified_first_available_at=verified_first_available_at,
            quality_status=quality_status,
            raw_sha256=raw_sha256,
            snapshot_path=raw_path,
            processor_code_version=self.processor_code_version,
            historical_eligibility=historical_eligibility,
            error_message=error_message,
            content_integrity_verified=content_integrity_verified,
            source_authenticity_status=source_authenticity,
            historical_availability_status=historical_eligibility.value,
        )

        meta_abs_path = os.path.abspath(os.path.join(snapshot_dir, f"{snapshot_id}.meta.json"))
        meta_dict = record.model_dump(mode="json")
        meta_dict["meta_path"] = self._portable_path(meta_abs_path)
        with open(meta_abs_path, "w", encoding="utf-8") as mf:
            json.dump(meta_dict, mf, ensure_ascii=False, indent=2, default=str)

        return meta_dict

    @abstractmethod
    def collect(self, **kwargs) -> Dict[str, Any]:
        """子类具体采集逻辑"""
        pass
