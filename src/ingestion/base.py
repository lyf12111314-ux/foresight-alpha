"""数据采集抽象基类 (Base Collector with Provenance & Point-in-Time Support)."""

import hashlib
import json
import time
import requests
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Dict, Any, Optional

from src.core.types import ProvenanceMetadata, DataQualityStatus


class BaseCollector(ABC):
    """数据采集器基类"""

    def __init__(self, source_name: str, base_url: str, timeout_sec: int = 15):
        self.source_name = source_name
        self.base_url = base_url
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ForesightAlpha-ResearchAgent/1.0 (+https://github.com/foresight-alpha)"
        })

    def calculate_hash(self, raw_content: str) -> str:
        """计算原始载荷的 SHA-256 哈希值"""
        return hashlib.sha256(raw_content.encode("utf-8")).hexdigest()

    def build_provenance(
        self,
        record_id: str,
        as_of_date: date,
        published_at: datetime,
        source_url: Optional[str] = None,
        quality_status: DataQualityStatus = DataQualityStatus.VALIDATED,
        raw_hash: Optional[str] = None
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
            raw_hash=raw_hash
        )

    def fetch_with_retry(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        backoff_sec: float = 1.0
    ) -> Optional[requests.Response]:
        """具备指数退避重试的健壮 HTTP 抓取"""
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout_sec)
                if resp.status_code == 200:
                    return resp
                elif resp.status_code in [429, 500, 502, 503, 504]:
                    time.sleep(backoff_sec * (2 ** attempt))
                else:
                    return None
            except requests.RequestException:
                time.sleep(backoff_sec * (2 ** attempt))
        return None

    @abstractmethod
    def collect(self, **kwargs) -> Dict[str, Any]:
        """子类具体采集逻辑"""
        pass
