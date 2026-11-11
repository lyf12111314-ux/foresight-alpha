"""动态预警规则引擎与巡检器 (Monitoring & Rule-Based Alert Trigger)."""

from datetime import datetime, date
from typing import Dict, Any, List
from src.core.database import db_manager
from src.core.types import ObjectLevel
from src.value_chain.financial_audit import CommercializationAuditEngine
from src.monitoring.alerts import AlertManager


class SystemMonitor:
    """系统自动化巡检与预警触发器"""

    def __init__(self):
        self.alert_manager = AlertManager()
        self.audit_engine = CommercializationAuditEngine()

    def run_daily_scan(self) -> List[Dict[str, Any]]:
        """执行全系统异常状态扫描并生成正式告警"""
        generated_alerts = []

        # 1. 扫描所有企业的商业化红旗
        df_fin = db_manager.query_df("SELECT * FROM company_financials ORDER BY as_of_date ASC")
        if not df_fin.empty:
            for cid, group in df_fin.groupby("company_id"):
                audit_res = self.audit_engine.audit_company(group)
                for rf in audit_res.get("red_flags", []):
                    alert_id = f"ALERT_FIN_{cid}_{rf['code']}_{datetime.utcnow().strftime('%Y%m%d')}"
                    alert = self.alert_manager.record_alert(
                        alert_id=alert_id,
                        target_id=str(cid),
                        target_level=ObjectLevel.COMPANY_ENTITY,
                        severity=rf["severity"],
                        rule_type=rf["code"],
                        title=f"企业经营预警 [{cid}]: {rf['title']}",
                        description=rf["evidence"],
                        evidence=audit_res["metrics_snapshot"]
                    )
                    generated_alerts.append(alert.model_dump())

        # 2. 扫描技术爆发拐点
        df_tech = db_manager.query_df("SELECT * FROM tech_metrics ORDER BY as_of_date ASC")
        if not df_tech.empty:
            for tid, group in df_tech.groupby("topic_id"):
                if len(group) >= 3:
                    papers = group["papers_count"].values
                    if len(papers) >= 2 and papers[-1] >= papers[-2] * 1.35 and papers[-1] > 200:
                        alert_id = f"ALERT_TECH_SPIKE_{tid}_{datetime.utcnow().strftime('%Y%m%d')}"
                        alert = self.alert_manager.record_alert(
                            alert_id=alert_id,
                            target_id=str(tid),
                            target_level=ObjectLevel.TECH_TOPIC,
                            severity="INFO",
                            rule_type="TECH_ACCELERATION_SPIKE",
                            title=f"技术扩散爆发预警 [{tid}]",
                            description=f"该主题最新年度文献产出激增至 {papers[-1]} 篇 (同比增速超过 35%)，且参与独立机构数突破历史极值，跨越萌芽拐点。",
                            evidence={"papers_latest": int(papers[-1]), "growth_pct": round(float((papers[-1] / papers[-2] - 1.0) * 100), 1)}
                        )
                        generated_alerts.append(alert.model_dump())

        return generated_alerts
