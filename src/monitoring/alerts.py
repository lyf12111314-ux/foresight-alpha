"""预警记录与失效审计存储 (Alert Persistence & Invalidation Audit)."""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.core.database import db_manager
from src.core.types import AlertRecord, ObjectLevel


class AlertManager:
    """系统预警生命周期管理器"""

    def record_alert(
        self,
        alert_id: str,
        target_id: str,
        target_level: ObjectLevel,
        severity: str,
        rule_type: str,
        title: str,
        description: str,
        evidence: Dict[str, Any]
    ) -> AlertRecord:
        """记录一条具备不可篡改证据链的正式预警"""
        now = datetime.utcnow()
        alert = AlertRecord(
            alert_id=alert_id,
            target_id=target_id,
            target_level=target_level,
            severity=severity,
            rule_type=rule_type,
            title=title,
            description=description,
            triggered_at=now,
            data_evidence=evidence,
            is_active=True
        )

        conn = db_manager.get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO model_alerts
                (alert_id, target_id, target_level, severity, rule_type, title, description, triggered_at, evidence_json, is_active, invalidation_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                alert.alert_id, alert.target_id, alert.target_level.value,
                alert.severity, alert.rule_type, alert.title,
                alert.description, alert.triggered_at,
                json.dumps(alert.data_evidence, ensure_ascii=False),
                alert.is_active, alert.invalidation_notes
            ])
        finally:
            conn.close()

        return alert

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取所有当前有效未解除的预警事件"""
        sql = "SELECT * FROM model_alerts WHERE is_active = TRUE ORDER BY triggered_at DESC"
        df = db_manager.query_df(sql)
        res = []
        for _, row in df.iterrows():
            item = row.to_dict()
            try:
                item["evidence"] = json.loads(item.get("evidence_json", "{}"))
            except Exception:
                item["evidence"] = {}
            res.append(item)
        return res

    def invalidate_alert(self, alert_id: str, reason: str):
        """对已失效或被事后纠正的预警进行显式失效标记 (审计追踪)"""
        sql = "UPDATE model_alerts SET is_active = FALSE, invalidation_notes = ? WHERE alert_id = ?"
        db_manager.execute(sql, [reason, alert_id])
