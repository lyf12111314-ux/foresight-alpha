"""V1.2 第一阶段真实数据验证试点脚本：固态电池 (TOPIC_SOLID_STATE_BATT)。

用途：
1. 调用改造后的 OpenAlexCollector 实际请求 https://api.openalex.org/works (2018-2025)；
2. 将原始 HTTP 响应字节与采集元数据落盘到 data/snapshots/openalex/；
3. 从磁盘重新读取快照文件计算 SHA-256，校验内容完整性；
4. 提取文献发表时间与按年度截断的引用指标，对 2020-12-31 执行严格 PIT 资格判定；
5. 输出可人工复核的真实样本明细 JSON，便于同历史文档核对版本漂移。
"""

import json
import os
import sys
from datetime import date

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.base import BaseCollector
from src.ingestion.openalex import OpenAlexCollector


def run_pilot(snapshot_dir: str = "data/snapshots/openalex") -> dict:
    abs_snap_dir = os.path.abspath(os.path.join(PROJECT_ROOT, snapshot_dir)) if not os.path.isabs(snapshot_dir) else snapshot_dir
    collector = OpenAlexCollector(snapshot_dir=abs_snap_dir)

    records = collector.fetch_topic_research_trends(
        query='"solid-state battery"',
        start_year=2018,
        end_year=2025,
        save_snapshots=True,
    )

    summary_rows = []
    for rec in records:
        snap_path = rec.get("snapshot_path")
        raw_sha = rec.get("raw_sha256")
        integrity = (
            BaseCollector.verify_snapshot_integrity(snap_path, raw_sha)
            if (snap_path and raw_sha)
            else {"content_integrity_verified": False}
        )
        pit_2020 = collector.evaluate_historical_pit_eligibility(rec, cutoff_date=date(2020, 12, 31))
        rel_snap = os.path.relpath(snap_path, PROJECT_ROOT) if snap_path else None
        rel_meta = os.path.relpath(rec["snapshot_meta"]["meta_path"], PROJECT_ROOT) if (rec.get("snapshot_meta") and rec["snapshot_meta"].get("meta_path")) else None

        summary_rows.append({
            "year": rec["year"],
            "status": rec["status"],
            "http_status": (rec.get("snapshot_meta") or {}).get("http_status", rec.get("http_status")),
            "ingested_at_utc": rec["ingested_at_utc"].isoformat() if rec.get("ingested_at_utc") else None,
            "verified_first_available_at": rec.get("verified_first_available_at"),
            "papers_count": rec.get("papers_count"),
            "citations_count_pit_cutoff": rec.get("citations_count"),
            "sampled_citations_count": rec.get("sampled_citations_count"),
            "citation_count_scope": rec.get("citation_count_scope"),
            "citations_status": rec.get("citations_status"),
            "citations_cutoff_year": rec.get("citations_cutoff_year"),
            "active_institutions": rec.get("active_institutions"),
            "sampled_institutions_count": rec.get("sampled_institutions_count"),
            "institution_count_scope": rec.get("institution_count_scope"),
            "feature_lineage": rec.get("feature_lineage"),
            "historical_eligibility": rec.get("historical_eligibility"),
            "strict_pit_2020_check": pit_2020,
            "content_integrity_verified": bool(integrity.get("content_integrity_verified")),
            "raw_sha256": raw_sha,
            "snapshot_id": rec.get("snapshot_id"),
            "snapshot_path": rel_snap,
            "meta_path": rel_meta,
            "error_message": rec.get("error_message"),
        })

    manifest = {
        "topic_id": "TOPIC_SOLID_STATE_BATT",
        "query": '"solid-state battery"',
        "snapshot_dir": os.path.relpath(abs_snap_dir, PROJECT_ROOT),
        "records": summary_rows,
    }
    manifest_path = os.path.join(abs_snap_dir, "pilot_solid_state_battery_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)
    return manifest


if __name__ == "__main__":
    res = run_pilot()
    print(json.dumps(res, ensure_ascii=False, indent=2))
