"""arXiv 前沿科技预印本文献采集器 (arXiv API Client)."""

import xml.etree.ElementTree as ET
import json
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from src.ingestion.base import BaseCollector
from src.core.types import DataQualityStatus


class ArxivCollector(BaseCollector):
    """arXiv 预印本文献采集器"""

    def __init__(self, base_url: str = "https://export.arxiv.org/api/query"):
        super().__init__(source_name="arXiv", base_url=base_url)

    def fetch_preprints(
        self,
        search_query: str,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """检索指定技术关键词的前沿预印本论文"""
        params = {
            "search_query": f"all:{search_query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        resp = self.fetch_with_retry(self.base_url, params=params)
        preprints = []
        if not resp or resp.status_code != 200:
            return preprints

        try:
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                published_elem = entry.find("atom:published", ns)
                id_elem = entry.find("atom:id", ns)

                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""
                summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
                pub_str = published_elem.text.strip() if published_elem is not None else ""
                paper_id = id_elem.text.strip() if id_elem is not None else ""

                try:
                    pub_dt = datetime.strptime(pub_str, "%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pub_dt = datetime.utcnow()

                authors = []
                for author_elem in entry.findall("atom:author", ns):
                    name_elem = author_elem.find("atom:name", ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                raw_json = json.dumps({
                    "title": title,
                    "summary": summary,
                    "published": pub_str,
                    "authors": authors,
                    "id": paper_id
                }, ensure_ascii=False)

                prov = self.build_provenance(
                    record_id=f"ARXIV_{paper_id.split('/')[-1]}",
                    as_of_date=pub_dt.date(),
                    published_at=pub_dt,
                    source_url=paper_id,
                    quality_status=DataQualityStatus.VALIDATED,
                    raw_hash=self.calculate_hash(raw_json)
                )

                preprints.append({
                    "paper_id": paper_id,
                    "title": title,
                    "summary": summary,
                    "published_at": pub_dt,
                    "authors": authors,
                    "provenance": prov.model_dump(),
                    "raw_payload": raw_json
                })
        except Exception as e:
            print(f"Error parsing arXiv XML: {e}")

        return preprints

    def collect(self, **kwargs) -> Dict[str, Any]:
        query = kwargs.get("query", "robotics")
        limit = kwargs.get("limit", 30)
        return {"data": self.fetch_preprints(query, max_results=limit)}
