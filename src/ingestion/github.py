"""GitHub 开源生态与技术热度采集器 (GitHub REST API Client — V1.2 PIT 隔离版)."""

import json
import os
from datetime import datetime, date, timezone
from typing import Dict, Any, List, Optional
from src.ingestion.base import BaseCollector
from src.core.types import DataQualityStatus, HistoricalEligibilityStatus


class GithubCollector(BaseCollector):
    """GitHub 开源生态采集器 (严格隔离实时观察与历史 PIT 回测)"""

    def __init__(
        self,
        base_url: str = "https://api.github.com",
        snapshot_dir: Optional[str] = None,
    ):
        super().__init__(source_name="GitHub", base_url=base_url)
        self.snapshot_dir = snapshot_dir or os.path.abspath("data/snapshots/github")

    @staticmethod
    def _parse_github_dt(dt_str: Optional[str]) -> Optional[datetime]:
        if not dt_str:
            return None
        s = str(dt_str).strip()
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            parsed = datetime.fromisoformat(s)
            return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed
        except ValueError:
            return None

    def search_repositories(
        self,
        topic_keyword: str,
        max_items: int = 15,
        save_snapshots: bool = False,
    ) -> List[Dict[str, Any]]:
        """检索与特定技术领域相关的当前开源项目实时状态 (严禁直接当作历史年份特征)"""
        url = f"{self.base_url}/search/repositories"
        params = {
            "q": f"{topic_keyword} stars:>50",
            "sort": "stars",
            "order": "desc",
            "per_page": min(max_items, 30),
        }
        resp, http_status, fetch_err = self.fetch_raw_with_audit(url, params=params)
        ingested_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        repos: List[Dict[str, Any]] = []

        self.last_snapshot_meta: Optional[Dict[str, Any]] = None
        if resp is None or http_status != 200:
            if save_snapshots:
                self.last_snapshot_meta = self.save_raw_response_snapshot(
                    snapshot_dir=self.snapshot_dir,
                    request_url=url,
                    query_params=params,
                    http_status=http_status,
                    raw_bytes=None,
                    as_of_date=ingested_utc.date(),
                    verified_first_available_at=None,
                    quality_status=DataQualityStatus.UNAVAILABLE,
                    historical_eligibility=HistoricalEligibilityStatus.CURRENT_STATE_ONLY,
                    error_message=fetch_err or f"GitHub HTTP {http_status} failed",
                    ingested_at_utc=ingested_utc,
                    file_prefix=f"github_{topic_keyword}",
                )
            return repos

        raw_bytes = resp.content if isinstance(resp.content, (bytes, bytearray)) else resp.text.encode("utf-8")
        parse_err: Optional[str] = None
        data: Any = None
        try:
            data = resp.json()
            if not isinstance(data, dict) or not isinstance(data.get("items"), list):
                parse_err = "GITHUB_MALFORMED_RESPONSE_MISSING_ITEMS"
        except (ValueError, TypeError) as exc:
            parse_err = f"GITHUB_JSON_PARSE_ERROR: {exc}"

        if parse_err is not None:
            if save_snapshots:
                self.last_snapshot_meta = self.save_raw_response_snapshot(
                    snapshot_dir=self.snapshot_dir,
                    request_url=url,
                    query_params=params,
                    http_status=http_status,
                    raw_bytes=None,
                    as_of_date=ingested_utc.date(),
                    verified_first_available_at=None,
                    quality_status=DataQualityStatus.UNAVAILABLE,
                    historical_eligibility=HistoricalEligibilityStatus.CURRENT_STATE_ONLY,
                    error_message=parse_err,
                    ingested_at_utc=ingested_utc,
                    file_prefix=f"github_{topic_keyword}",
                )
            return repos

        snap_meta = None
        if save_snapshots:
            snap_meta = self.save_raw_response_snapshot(
                snapshot_dir=self.snapshot_dir,
                request_url=url,
                query_params=params,
                http_status=http_status,
                raw_bytes=raw_bytes,
                as_of_date=ingested_utc.date(),
                verified_first_available_at=None,
                quality_status=DataQualityStatus.UNVERIFIED,
                historical_eligibility=HistoricalEligibilityStatus.CURRENT_STATE_ONLY,
                error_message=None,
                ingested_at_utc=ingested_utc,
                file_prefix=f"github_{topic_keyword}",
            )
            self.last_snapshot_meta = snap_meta

        for item in data.get("items", []):
            if not isinstance(item, dict):
                continue
            repo_id = str(item.get("id"))
            name = item.get("full_name")
            stars = item.get("stargazers_count")
            forks = item.get("forks_count")
            open_issues = item.get("open_issues_count")
            updated_at_str = item.get("updated_at")
            created_at_str = item.get("created_at")
            description = item.get("description") or ""

            created_at = self._parse_github_dt(created_at_str)
            updated_at = self._parse_github_dt(updated_at_str) or ingested_utc

            raw_json = json.dumps({
                "id": repo_id,
                "name": name,
                "stars": stars,
                "forks": forks,
                "open_issues": open_issues,
                "created_at": created_at_str,
                "updated_at": updated_at_str,
                "ingested_at_utc": ingested_utc.isoformat(),
                "historical_eligibility": HistoricalEligibilityStatus.CURRENT_STATE_ONLY.value,
            }, sort_keys=True)

            raw_hash = snap_meta["raw_sha256"] if snap_meta else self.calculate_hash(raw_json)
            prov = self.build_provenance(
                record_id=f"GITHUB_{repo_id}",
                as_of_date=ingested_utc.date(),
                published_at=ingested_utc,
                source_url=item.get("html_url"),
                quality_status=DataQualityStatus.UNVERIFIED,
                raw_hash=raw_hash,
                snapshot_id=snap_meta["snapshot_id"] if snap_meta else None,
                snapshot_path=snap_meta["snapshot_path"] if snap_meta else None,
                historical_eligibility=HistoricalEligibilityStatus.CURRENT_STATE_ONLY.value,
            )

            repos.append({
                "repo_id": repo_id,
                "full_name": name,
                "stars": stars,
                "forks": forks,
                "open_issues": open_issues,
                "description": description,
                "created_at": created_at,
                "created_at_str": created_at_str,
                "updated_at": updated_at,
                "ingested_at_utc": ingested_utc,
                "historical_eligibility": HistoricalEligibilityStatus.CURRENT_STATE_ONLY.value,
                "pit_eligible": False,
                "snapshot_id": snap_meta["snapshot_id"] if snap_meta else None,
                "snapshot_meta": snap_meta,
                "provenance": prov.model_dump(),
                "raw_payload": raw_json,
            })

        return repos

    @classmethod
    def filter_repositories_for_historical_cutoff(
        cls,
        repos: List[Dict[str, Any]],
        cutoff_date: date,
    ) -> Dict[str, Any]:
        """严格检查 GitHub 仓库集合在历史预测时点 cutoff_date 的可用性。

        规则：
        1. created_at 缺失或 created_at > cutoff_date 的未来仓库必须被排除。
        2. 即使仓库在 cutoff_date 之前已创建，实时 stargazers_count / forks_count 也只代表查询日状态，
           不得倒推、线性插值或填常数作为历史特征，统一置为 None 并标为 HISTORICAL_VINTAGE_UNAVAILABLE。
        """
        eligible_existed_repos: List[Dict[str, Any]] = []
        excluded_future_repos: List[Dict[str, Any]] = []

        for r in repos or []:
            if not isinstance(r, dict):
                continue
            cr_val = r.get("created_at") or r.get("created_at_str")
            cr_dt = cr_val if isinstance(cr_val, datetime) else cls._parse_github_dt(cr_val)
            if cr_dt is None or cr_dt.date() > cutoff_date:
                excluded_future_repos.append({
                    "repo_id": r.get("repo_id"),
                    "full_name": r.get("full_name"),
                    "created_at": str(cr_val),
                    "exclusion_reason": "CREATED_AFTER_CUTOFF_OR_UNKNOWN",
                })
                continue

            eligible_existed_repos.append({
                "repo_id": r.get("repo_id"),
                "full_name": r.get("full_name"),
                "created_at": cr_dt,
                # 严禁使用实时 Star/Fork 充当历史截断日观测值
                "stargazers_count": None,
                "forks_count": None,
                "open_issues_count": None,
                "pit_eligible": False,
                "historical_eligibility": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
            })

        return {
            "cutoff_date": cutoff_date.isoformat(),
            "pit_eligible": False,
            "historical_eligibility": HistoricalEligibilityStatus.HISTORICAL_VINTAGE_UNAVAILABLE.value,
            "github_momentum": None,
            "stargazers_count": None,
            "forks_count": None,
            "open_issues_count": None,
            "existed_on_cutoff_count": len(eligible_existed_repos),
            "excluded_future_repos_count": len(excluded_future_repos),
            "existed_repos": eligible_existed_repos,
            "excluded_future_repos": excluded_future_repos,
            "reason": (
                "GitHub Search API 仅提供查询时刻实时 Star/Fork；"
                "已剔除创建于截止日期之后的未来仓库，剩余已存在仓库因缺少历史 Star 快照，指标严格置为 None。"
            ),
        }

    def collect(self, **kwargs) -> Dict[str, Any]:
        keyword = kwargs.get("keyword", "solid-state-battery")
        return {"data": self.search_repositories(keyword, save_snapshots=kwargs.get("save_snapshots", False))}
