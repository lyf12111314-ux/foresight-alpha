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
from src.ingestion.tech_dataset import HistoricalTechDatasetCollector


class IngestionPipeline:
    """系统数据采集与清洗入库流水线"""

    def __init__(self):
        self.openalex = OpenAlexCollector()
        self.arxiv = ArxivCollector()
        self.github = GithubCollector()
        self.macro = MacroCollector()
        self.enterprise = EnterpriseCollector()
        self.tech_dataset = HistoricalTechDatasetCollector()

    @staticmethod
    def _append_raw_record(conn, prov: Dict[str, Any], raw_payload: Any, snapshot_id: Any = None) -> str:
        """以不可变版本模式追加 raw_records，保留同一来源同一主题同一年度多次采集的历史版本。"""
        snap = snapshot_id if snapshot_id is not None else prov.get("snapshot_id")
        return db_manager.append_raw_record(
            {
                "record_id": prov["record_id"],
                "logical_record_id": prov.get("logical_record_id", prov["record_id"]),
                "source_name": prov["source_name"],
                "source_url": prov["source_url"],
                "as_of_date": prov["as_of_date"],
                "published_at": prov["published_at"],
                "ingested_at": prov["ingested_at"],
                "quality_status": prov["quality_status"],
                "data_version": prov["data_version"],
                "raw_hash": prov["raw_hash"],
                "payload": raw_payload,
                "snapshot_id": snap,
            },
            conn=conn,
        )

    def ingest_distress_cohort(self) -> int:
        """入库企业违约与存续历史真实队列 (P1-01 训练底座)"""
        records = self.enterprise.get_historical_distress_cohort()
        conn = db_manager.get_connection()
        count = 0
        try:
            for row in records:
                prov = row["provenance"]
                self._append_raw_record(conn, prov, row["raw_payload"])
                conn.execute("""
                    INSERT OR REPLACE INTO company_distress_cohort
                    (company_id, company_name, industry_sector, duration_months, observation_window_months, is_distressed, distress_type, distress_event_type, event_date, published_at, cash_burn_months, debt_ratio, gross_margin, competitor_count, is_covariate_audited, official_announcement_id, source_url, quality_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    row["company_id"], row["company_name"], row["industry_sector"],
                    row["duration_months"], row.get("observation_window_months", row["duration_months"]),
                    row["is_distressed"], row["distress_type"], row.get("distress_event_type", "CENSORED"),
                    row["event_date"], row["published_at"], row.get("cash_burn_months"),
                    row["debt_ratio"], row["gross_margin"], row.get("competitor_count"),
                    row.get("is_covariate_audited", True),
                    row["official_announcement_id"], row["source_url"], row["quality_status"]
                ])
                count += 1
        finally:
            conn.close()
        return count

    def ingest_tech_training_samples(self) -> int:
        """入库前沿技术多期真实特征与多期标签训练集 (P1-01 训练底座)"""
        samples = self.tech_dataset.get_historical_tech_training_samples()
        conn = db_manager.get_connection()
        count = 0
        try:
            for s in samples:
                prov = s["provenance"]
                self._append_raw_record(conn, prov, s["raw_payload"])
                conn.execute("""
                    INSERT OR REPLACE INTO tech_training_samples
                    (sample_id, topic_id, base_date, published_at, feature_published_at, feature_ingested_at, velocity_1y, acceleration, active_institutions, github_momentum, macro_multiplier, entropy_score, novelty_score, label_1y_period_end, label_1y_status, target_1y, label_3y_period_end, label_3y_status, target_3y, label_5y_period_end, label_5y_status, target_5y, quality_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    s["sample_id"], s["topic_id"], s["base_date"], s["published_at"],
                    s.get("feature_published_at", s["published_at"]),
                    s.get("feature_ingested_at", s["published_at"]),
                    s["velocity_1y"], s["acceleration"], s["active_institutions"],
                    s["github_momentum"], s["macro_multiplier"], s["entropy_score"],
                    s["novelty_score"],
                    s.get("label_1y_period_end"), s.get("label_1y_status"), s.get("target_1y"),
                    s.get("label_3y_period_end"), s.get("label_3y_status"), s.get("target_3y"),
                    s.get("label_5y_period_end"), s.get("label_5y_status"), s.get("target_5y"),
                    s["quality_status"]
                ])
                count += 1
        finally:
            conn.close()
        return count

    def run_all(self, verbose: bool = True) -> Dict[str, int]:
        """执行全量数据采集与入库流水线"""
        stats = {
            "macro_indicators": 0,
            "enterprise_financials": 0,
            "distress_cohort": 0,
            "tech_training_samples": 0,
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
                # 写入不可变 raw_records 归档表
                self._append_raw_record(conn, prov, item["raw_payload"])
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
                self._append_raw_record(conn, prov, fin["raw_payload"])
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

            # 2.1 入库企业违约与存续真实事件队列 (用于 Cox 模型)
            if verbose:
                print("[2.1/5] Ingesting Historical Distress & Survival Cohort...")
            distress_cohort = self.enterprise.get_historical_distress_cohort()
            for row in distress_cohort:
                prov = row["provenance"]
                self._append_raw_record(conn, prov, row["raw_payload"])
                stats["raw_records"] += 1

                conn.execute("""
                    INSERT OR REPLACE INTO company_distress_cohort
                    (company_id, company_name, industry_sector, duration_months, observation_window_months, is_distressed, distress_type, distress_event_type, event_date, published_at, cash_burn_months, debt_ratio, gross_margin, competitor_count, is_covariate_audited, official_announcement_id, source_url, quality_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    row["company_id"], row["company_name"], row["industry_sector"],
                    row["duration_months"], row.get("observation_window_months", row["duration_months"]),
                    row["is_distressed"], row["distress_type"], row.get("distress_event_type", "CENSORED"),
                    row["event_date"], row["published_at"], row.get("cash_burn_months"),
                    row["debt_ratio"], row["gross_margin"], row.get("competitor_count"),
                    row.get("is_covariate_audited", True),
                    row["official_announcement_id"], row["source_url"], row["quality_status"]
                ])
                stats["distress_cohort"] += 1

            if verbose:
                print(f" -> Ingested {len(distress_cohort)} historical enterprise distress cohort records.")

            if verbose:
                print("[3/5] Ingesting Scholarly Research Trends (OpenAlex)...")
            topics_to_track = [
                ("TOPIC_EMBODIED_AI", '"embodied AI"'),
                ("TOPIC_SOLID_STATE_BATT", '"solid-state battery"'),
                ("TOPIC_NEUROMORPHIC", '"neuromorphic computing"'),
                ("TOPIC_AGENTIC_AI", '"autonomous AI agents"'),
                ("TOPIC_EVTOL_AERO", '"eVTOL"'),
                ("TOPIC_SYNTHETIC_BIO", '"synthetic biology"')
            ]

            for topic_id, query_str in topics_to_track:
                alex_records = self.openalex.fetch_topic_research_trends(query_str, 2018, 2024)
                for rec in alex_records:
                    snap_metas = rec.get("snapshot_metas") or (
                        [rec["snapshot_meta"]] if rec.get("snapshot_meta") else []
                    )
                    seen_snap_ids = set()
                    for smeta in snap_metas:
                        sid = smeta.get("snapshot_id")
                        if sid and sid not in seen_snap_ids:
                            db_manager.save_raw_snapshot(smeta, conn=conn)
                            seen_snap_ids.add(sid)

                    # 采集失败绝不生成 papers_count=0 的正常观测记录
                    if rec.get("status") != "OK" or rec.get("papers_count") is None:
                        continue

                    prov = rec["provenance"]
                    self._append_raw_record(conn, prov, rec["raw_payload"], snapshot_id=rec.get("snapshot_id"))
                    stats["raw_records"] += 1

                    # 仅在完整真实机构数和论文数均可用时计算指标；缺失则保留 NULL，绝不用经验公式或年份衰减估算
                    inst_cnt = rec.get("active_institutions")
                    papers_cnt = rec["papers_count"]
                    entropy = (
                        min(1.0, round(inst_cnt / max(papers_cnt * 0.2 + 1, 1), 3))
                        if (inst_cnt is not None and papers_cnt > 0)
                        else None
                    )
                    lineage_dict = rec.get("feature_lineage", {})
                    lineage_json = json.dumps(
                        lineage_dict, ensure_ascii=False, sort_keys=True
                    )
                    source_snap_ids = sorted({
                        sid for sid in (
                            [rec.get("snapshot_id")]
                            + [m.get("snapshot_id") for m in snap_metas if isinstance(m, dict)]
                            + [
                                entry.get("snapshot_id")
                                for entry in lineage_dict.values()
                                if isinstance(entry, dict)
                            ]
                        )
                        if sid
                    })

                    db_manager.append_tech_metric_version(
                        {
                            "topic_id": topic_id,
                            "as_of_date": rec["as_of_date"],
                            "published_at": rec["published_at"],
                            "ingested_at": prov.get("ingested_at") or rec["published_at"],
                            "processor_code_version": getattr(self.openalex, "processor_code_version", "v1.2-p0-data-provenance"),
                            "feature_lineage_json": lineage_json,
                            "source_snapshot_ids": source_snap_ids,
                            "papers_count": papers_cnt,
                            "citations_count": rec.get("citations_count"),
                            "sampled_citations_count": rec.get("sampled_citations_count"),
                            "citation_count_scope": rec.get("citation_count_scope"),
                            "active_institutions": inst_cnt,
                            "sampled_institutions_count": rec.get("sampled_institutions_count"),
                            "institution_count_scope": rec.get("institution_count_scope"),
                            "github_stars": None,
                            "github_forks": None,
                            "commit_velocity": None,
                            "entropy_score": entropy,
                            "novelty_score": None,
                            "github_pit_eligible": False,
                            "snapshot_id": rec.get("snapshot_id"),
                            "citations_cutoff_year": rec.get("citations_cutoff_year"),
                            "historical_eligibility": rec.get("historical_eligibility"),
                        },
                        conn=conn,
                        update_latest=True,
                    )
                    stats["tech_metrics"] += 1

            if verbose:
                print("[4/5] Ingesting GitHub Ecosystem Dynamics (Current Observation Only — Non-PIT)...")
            github_keywords = [
                ("TOPIC_EMBODIED_AI", "embodied-ai"),
                ("TOPIC_AGENTIC_AI", "llm-agent"),
                ("TOPIC_NEUROMORPHIC", "spiking-neural-network")
            ]
            for topic_id, kw in github_keywords:
                repos = self.github.search_repositories(kw, max_items=5, save_snapshots=True)
                saved_gh_snaps = set()
                if getattr(self.github, "last_snapshot_meta", None):
                    lmeta = self.github.last_snapshot_meta
                    db_manager.save_raw_snapshot(lmeta, conn=conn)
                    saved_gh_snaps.add(lmeta["snapshot_id"])
                for r in repos:
                    if r.get("snapshot_meta") and r["snapshot_meta"]["snapshot_id"] not in saved_gh_snaps:
                        db_manager.save_raw_snapshot(r["snapshot_meta"], conn=conn)
                        saved_gh_snaps.add(r["snapshot_meta"]["snapshot_id"])
                    prov = r["provenance"]
                    self._append_raw_record(conn, prov, r["raw_payload"], snapshot_id=prov.get("snapshot_id"))
                    stats["raw_records"] += 1

                # 仅作为当前查询时点生态观察以不可变新版本追加写入（明确标为 github_pit_eligible=FALSE），绝不覆盖旧版本且无仓库时不写 0
                if repos:
                    star_vals = [r["stars"] for r in repos if r.get("stars") is not None]
                    fork_vals = [r["forks"] for r in repos if r.get("forks") is not None]
                    tot_stars = sum(star_vals) if star_vals else None
                    tot_forks = sum(fork_vals) if fork_vals else None
                    latest_tm_df = conn.execute(
                        "SELECT * FROM tech_metrics WHERE topic_id = ? ORDER BY as_of_date DESC LIMIT 1;",
                        [topic_id],
                    ).df()
                    if not latest_tm_df.empty:
                        base_row = latest_tm_df.iloc[0].to_dict()
                        def _v(val):
                            if val is None:
                                return None
                            try:
                                import pandas as _pd
                                if _pd.isna(val):
                                    return None
                            except Exception:
                                pass
                            return val

                        existing_snaps: List[str] = []
                        prev_fv_id = _v(base_row.get("latest_feature_version_id"))
                        if prev_fv_id:
                            prev_df = conn.execute(
                                "SELECT source_snapshot_ids FROM tech_metric_versions WHERE feature_version_id = ? LIMIT 1;",
                                [prev_fv_id],
                            ).df()
                            if not prev_df.empty and _v(prev_df.iloc[0]["source_snapshot_ids"]):
                                try:
                                    parsed_snaps = json.loads(prev_df.iloc[0]["source_snapshot_ids"])
                                    if isinstance(parsed_snaps, list):
                                        existing_snaps = [str(x) for x in parsed_snaps if x]
                                except Exception:
                                    pass
                        if _v(base_row.get("snapshot_id")):
                            existing_snaps.append(str(base_row["snapshot_id"]))
                        merged_snaps = sorted(set(existing_snaps) | {str(s) for s in saved_gh_snaps if s})

                        now_ts = datetime.utcnow()
                        db_manager.append_tech_metric_version(
                            {
                                "topic_id": topic_id,
                                "as_of_date": _v(base_row.get("as_of_date")),
                                # 复合版本包含本次才观测的实时 GitHub 指标，不能继承旧公开时间。
                                "published_at": max(_v(base_row.get("published_at")) or now_ts, now_ts),
                                "ingested_at": now_ts,
                                "feature_computed_at": now_ts,
                                "processor_code_version": getattr(self.github, "processor_code_version", "v1.2-p1-github-versioned"),
                                "feature_lineage_json": _v(base_row.get("feature_lineage_json")),
                                "source_snapshot_ids": merged_snaps,
                                "papers_count": _v(base_row.get("papers_count")),
                                "citations_count": _v(base_row.get("citations_count")),
                                "sampled_citations_count": _v(base_row.get("sampled_citations_count")),
                                "citation_count_scope": _v(base_row.get("citation_count_scope")),
                                "active_institutions": _v(base_row.get("active_institutions")),
                                "sampled_institutions_count": _v(base_row.get("sampled_institutions_count")),
                                "institution_count_scope": _v(base_row.get("institution_count_scope")),
                                "github_stars": tot_stars,
                                "github_forks": tot_forks,
                                "commit_velocity": _v(base_row.get("commit_velocity")),
                                "entropy_score": _v(base_row.get("entropy_score")),
                                "novelty_score": _v(base_row.get("novelty_score")),
                                "github_pit_eligible": False,
                                "snapshot_id": _v(base_row.get("snapshot_id")),
                                "citations_cutoff_year": _v(base_row.get("citations_cutoff_year")),
                                "historical_eligibility": _v(base_row.get("historical_eligibility")),
                            },
                            conn=conn,
                            update_latest=True,
                        )

            if verbose:
                print("[5/5] Ingesting Multi-Horizon Historical Tech Training Samples...")
            tech_samples = self.tech_dataset.get_historical_tech_training_samples()
            for s in tech_samples:
                prov = s["provenance"]
                self._append_raw_record(conn, prov, s["raw_payload"])
                stats["raw_records"] += 1

                conn.execute("""
                    INSERT OR REPLACE INTO tech_training_samples
                    (sample_id, topic_id, base_date, published_at, feature_published_at, feature_ingested_at, velocity_1y, acceleration, active_institutions, github_momentum, macro_multiplier, entropy_score, novelty_score, label_1y_period_end, label_1y_status, target_1y, label_3y_period_end, label_3y_status, target_3y, label_5y_period_end, label_5y_status, target_5y, quality_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, [
                    s["sample_id"], s["topic_id"], s["base_date"], s["published_at"],
                    s.get("feature_published_at", s["published_at"]),
                    s.get("feature_ingested_at", s["published_at"]),
                    s["velocity_1y"], s["acceleration"], s["active_institutions"],
                    s["github_momentum"], s["macro_multiplier"], s["entropy_score"],
                    s["novelty_score"],
                    s.get("label_1y_period_end"), s.get("label_1y_status"), s.get("target_1y"),
                    s.get("label_3y_period_end"), s.get("label_3y_status"), s.get("target_3y"),
                    s.get("label_5y_period_end"), s.get("label_5y_status"), s.get("target_5y"),
                    s["quality_status"]
                ])
                stats["tech_training_samples"] += 1

            if verbose:
                print(f" -> Ingested {len(tech_samples)} tech training samples.")
                print(f"[Finished] Full Ingestion Complete! Stats: {stats}")

        finally:
            conn.close()

        return stats
