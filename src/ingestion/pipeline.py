"""多源数据采集与溯源流水线总线 (Ingestion Pipeline Coordinator)."""

import json
from datetime import datetime, date
from typing import Dict, Any, List

from src.core.database import db_manager
from src.ingestion.openalex import OpenAlexCollector
from src.ingestion.arxiv import ArxivCollector
from src.ingestion.github import GithubCollector
from src.ingestion.macro import MacroCollector
from src.ingestion.enterprise import EnterpriseCollector


class IngestionPipeline:
    """系统数据采集与清洗入库流水线"""

    def __init__(self):
        self.openalex = OpenAlexCollector()
        self.arxiv = ArxivCollector()
        self.github = GithubCollector()
        self.macro = MacroCollector()
        self.enterprise = EnterpriseCollector()

    def run_all(self, verbose: bool = True) -> Dict[str, int]:
        """执行全量数据采集与入库流水线"""
        stats = {
            "macro_indicators": 0,
            "enterprise_financials": 0,
            "tech_metrics": 0,
            "raw_records": 0
        }

        if verbose:
            print("[1/5] Ingesting Macro Economic Series...")
        macro_res = self.macro.collect()
        official_series = macro_res.get("official_series", [])
        wb_series = macro_res.get("world_bank", [])

        all_macro = official_series + wb_series
        conn = db_manager.get_connection()
        try:
            for item in all_macro:
                prov = item["provenance"]
                # 写入 raw_records 归档表
                conn.execute("""
                    INSERT OR REPLACE INTO raw_records 
                    (record_id, source_name, source_url, as_of_date, published_at, ingested_at, quality_status, data_version, raw_hash, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    prov["record_id"], prov["source_name"], prov["source_url"],
                    prov["as_of_date"], prov["published_at"], prov["ingested_at"],
                    prov["quality_status"], prov["data_version"], prov["raw_hash"],
                    item["raw_payload"]
                ])
                stats["raw_records"] += 1

                # 写入 clean macro 表
                conn.execute("""
                    INSERT OR REPLACE INTO macro_indicators 
                    (indicator_code, as_of_date, published_at, indicator_name, value, unit, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                """, [
                    item["indicator_code"], item["as_of_date"], item["published_at"],
                    item["indicator_name"], item["value"], item["unit"], item["source"]
                ])
                stats["macro_indicators"] += 1

            if verbose:
                print(f" -> Ingested {len(all_macro)} macro indicators.")

            if verbose:
                print("[2/5] Ingesting Core Enterprise Financials...")
            fin_records = self.enterprise.get_core_enterprise_financials()
            for fin in fin_records:
                prov = fin["provenance"]
                conn.execute("""
                    INSERT OR REPLACE INTO raw_records 
                    (record_id, source_name, source_url, as_of_date, published_at, ingested_at, quality_status, data_version, raw_hash, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    prov["record_id"], prov["source_name"], prov["source_url"],
                    prov["as_of_date"], prov["published_at"], prov["ingested_at"],
                    prov["quality_status"], prov["data_version"], prov["raw_hash"],
                    fin["raw_payload"]
                ])
                stats["raw_records"] += 1

                conn.execute("""
                    INSERT OR REPLACE INTO company_financials
                    (company_id, as_of_date, published_at, revenue, revenue_yoy, gross_margin, net_profit, ocf, rd_expense, rd_ratio, roic, ar_turnover_days, debt_to_asset)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    fin["company_id"], fin["as_of_date"], fin["published_at"],
                    fin["revenue"], fin["revenue_yoy"], fin["gross_margin"],
                    fin["net_profit"], fin["ocf"], fin["rd_expense"],
                    fin["rd_ratio"], fin["roic"], fin["ar_turnover_days"], fin["debt_to_asset"]
                ])
                stats["enterprise_financials"] += 1

            if verbose:
                print(f" -> Ingested {len(fin_records)} enterprise financial records.")

            if verbose:
                print("[3/5] Ingesting Scholarly Research Trends (OpenAlex)...")
            topics_to_track = [
                ("TOPIC_EMBODIED_AI", "embodied AI"),
                ("TOPIC_SOLID_STATE_BATT", "solid state battery"),
                ("TOPIC_NEUROMORPHIC", "neuromorphic computing"),
                ("TOPIC_AGENTIC_AI", "autonomous AI agents"),
                ("TOPIC_EVTOL_AERO", "eVTOL aircraft"),
                ("TOPIC_SYNTHETIC_BIO", "synthetic biology cell factory")
            ]

            for topic_id, query_str in topics_to_track:
                alex_records = self.openalex.fetch_topic_research_trends(query_str, 2018, 2024)
                for rec in alex_records:
                    prov = rec["provenance"]
                    conn.execute("""
                        INSERT OR REPLACE INTO raw_records 
                        (record_id, source_name, source_url, as_of_date, published_at, ingested_at, quality_status, data_version, raw_hash, payload)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, [
                        prov["record_id"], prov["source_name"], prov["source_url"],
                        prov["as_of_date"], prov["published_at"], prov["ingested_at"],
                        prov["quality_status"], prov["data_version"], prov["raw_hash"],
                        rec["raw_payload"]
                    ])
                    stats["raw_records"] += 1

                    # 基础新颖度与熵计算 (基于机构数与论文增长加速度)
                    entropy = min(1.0, round(rec["active_institutions"] / max(rec["papers_count"] * 0.2 + 1, 1), 3))
                    novelty = min(1.0, round(1.0 / (1.0 + (rec["year"] - 2018) * 0.15) * 0.95, 3))

                    conn.execute("""
                        INSERT OR REPLACE INTO tech_metrics
                        (topic_id, as_of_date, published_at, papers_count, citations_count, active_institutions, github_stars, github_forks, commit_velocity, entropy_score, novelty_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, [
                        topic_id, rec["as_of_date"], rec["published_at"],
                        rec["papers_count"], rec["citations_count"], rec["active_institutions"],
                        0, 0, 0.0, entropy, novelty
                    ])
                    stats["tech_metrics"] += 1

            if verbose:
                print("[4/5] Ingesting GitHub Ecosystem Dynamics...")
            github_keywords = [
                ("TOPIC_EMBODIED_AI", "embodied-ai"),
                ("TOPIC_AGENTIC_AI", "llm-agent"),
                ("TOPIC_NEUROMORPHIC", "spiking-neural-network")
            ]
            for topic_id, kw in github_keywords:
                repos = self.github.search_repositories(kw, max_items=5)
                tot_stars = sum(r["stars"] for r in repos)
                tot_forks = sum(r["forks"] for r in repos)
                for r in repos:
                    prov = r["provenance"]
                    conn.execute("""
                        INSERT OR REPLACE INTO raw_records 
                        (record_id, source_name, source_url, as_of_date, published_at, ingested_at, quality_status, data_version, raw_hash, payload)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, [
                        prov["record_id"], prov["source_name"], prov["source_url"],
                        prov["as_of_date"], prov["published_at"], prov["ingested_at"],
                        prov["quality_status"], prov["data_version"], prov["raw_hash"],
                        r["raw_payload"]
                    ])
                    stats["raw_records"] += 1

                # 更新最新日期的 GitHub 累计星标
                conn.execute("""
                    UPDATE tech_metrics
                    SET github_stars = ?, github_forks = ?
                    WHERE topic_id = ? AND as_of_date = (SELECT MAX(as_of_date) FROM tech_metrics WHERE topic_id = ?);
                """, [tot_stars, tot_forks, topic_id, topic_id])

            if verbose:
                print(f"[5/5] Ingestion Complete! Stats: {stats}")

        finally:
            conn.close()

        return stats
