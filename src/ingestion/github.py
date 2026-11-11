"""GitHub 开源生态与技术热度采集器 (GitHub REST API Client)."""

import json
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from src.ingestion.base import BaseCollector
from src.core.types import DataQualityStatus


class GithubCollector(BaseCollector):
    """GitHub 开源生态采集器"""

    def __init__(self, base_url: str = "https://api.github.com"):
        super().__init__(source_name="GitHub", base_url=base_url)

    def search_repositories(
        self,
        topic_keyword: str,
        max_items: int = 15
    ) -> List[Dict[str, Any]]:
        """检索与特定技术领域相关的顶级开源项目指标"""
        url = f"{self.base_url}/search/repositories"
        params = {
            "q": f"{topic_keyword} stars:>50",
            "sort": "stars",
            "order": "desc",
            "per_page": min(max_items, 30)
        }
        resp = self.fetch_with_retry(url, params=params)
        repos = []
        if not resp or resp.status_code != 200:
            return repos

        try:
            data = resp.json()
            for item in data.get("items", []):
                repo_id = str(item.get("id"))
                name = item.get("full_name")
                stars = item.get("stargazers_count", 0)
                forks = item.get("forks_count", 0)
                open_issues = item.get("open_issues_count", 0)
                updated_at_str = item.get("updated_at", "")
                created_at_str = item.get("created_at", "")
                description = item.get("description") or ""

                try:
                    updated_at = datetime.strptime(updated_at_str, "%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    updated_at = datetime.utcnow()

                raw_json = json.dumps({
                    "id": repo_id,
                    "name": name,
                    "stars": stars,
                    "forks": forks,
                    "open_issues": open_issues,
                    "updated_at": updated_at_str,
                    "created_at": created_at_str,
                    "description": description
                })

                prov = self.build_provenance(
                    record_id=f"GITHUB_{repo_id}",
                    as_of_date=updated_at.date(),
                    published_at=updated_at,
                    source_url=item.get("html_url"),
                    quality_status=DataQualityStatus.UNVERIFIED,
                    raw_hash=self.calculate_hash(raw_json)
                )

                repos.append({
                    "repo_id": repo_id,
                    "full_name": name,
                    "stars": stars,
                    "forks": forks,
                    "open_issues": open_issues,
                    "description": description,
                    "updated_at": updated_at,
                    "provenance": prov.model_dump(),
                    "raw_payload": raw_json
                })
        except Exception as e:
            print(f"Error parsing GitHub response: {e}")

        return repos

    def collect(self, **kwargs) -> Dict[str, Any]:
        keyword = kwargs.get("keyword", "embodied-ai")
        return {"data": self.search_repositories(keyword)}
