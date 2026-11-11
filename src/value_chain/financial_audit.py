"""企业商业化健康度与四大红旗诊断引擎 (Commercialization Red Flag Diagnosis)."""

import pandas as pd
from typing import Dict, Any, List


class CommercializationAuditEngine:
    """企业商业化质量与红旗缺陷审计器"""

    def audit_company(self, company_financials_df: pd.DataFrame) -> Dict[str, Any]:
        """对单一企业的财务时序进行商业化真实质量审计与红旗风险筛查"""
        if company_financials_df.empty:
            return {"status": "NO_DATA", "red_flags": [], "health_score": 50.0}

        df = company_financials_df.sort_values("as_of_date").copy()
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        red_flags = []

        # 红旗 1: 收入高速增长但毛利率持续失血 (恶性价格战与内卷)
        rev_growth = float(latest.get("revenue_yoy", 0.0))
        gm_latest = float(latest.get("gross_margin", 0.0))
        gm_prev = float(prev.get("gross_margin", 0.0))
        gm_drop = gm_prev - gm_latest

        if rev_growth > 20.0 and gm_drop >= 3.5:
            red_flags.append({
                "code": "MARGIN_SQUEEZE_PRICE_WAR",
                "severity": "CRITICAL",
                "title": "增收不增利与毛利失血红旗",
                "evidence": f"营收同比高增 {rev_growth}%，但毛利率由前期 {gm_prev}% 骤降至 {gm_latest}% (下滑 {round(gm_drop, 1)} pct)，表明行业竞争格局恶化，陷入降价冲量内卷陷阱。"
            })

        # 红旗 2: 账面净利润与经营现金流背离 (伪造繁荣/存货与应收虚增)
        net_profit = float(latest.get("net_profit", 0.0))
        ocf = float(latest.get("ocf", 0.0))
        if net_profit > 0 and ocf < 0:
            red_flags.append({
                "code": "CASHFLOW_EARNINGS_DISTORTION",
                "severity": "CRITICAL",
                "title": "净利润与经营现金流严重背离红旗",
                "evidence": f"账面实现净利润 {net_profit} 亿元，但实际经营现金流净流出 {ocf} 亿元，利润含金量低，面临大量应收账款积压或存货跌价减值隐患。"
            })

        # 红旗 3: 应收账款周转天数异常拉长 (下游回款恶化与产业链被动垫资)
        ar_days = float(latest.get("ar_turnover_days", 0.0))
        if ar_days > 180.0:
            red_flags.append({
                "code": "ACCOUNTS_RECEIVABLE_OVERHANG",
                "severity": "WARNING",
                "title": "应收账款周转周期过长",
                "evidence": f"应收账款周转天数高达 {ar_days} 天 (> 180天半年度安全阈值)，企业对下游议价权脆弱，存在大额坏账计提风险。"
            })

        # 红旗 4: 研发费用率与商业化造血能力失衡 (持续流血无商业闭环)
        rd_ratio = float(latest.get("rd_ratio", 0.0))
        roic = float(latest.get("roic", 0.0))
        if rd_ratio > 40.0 and roic < -10.0:
            red_flags.append({
                "code": "HIGH_RD_LOW_CONVERSION",
                "severity": "WARNING",
                "title": "研发向商业利润转化受阻",
                "evidence": f"研发费用率高达 {rd_ratio}%，但投入资本回报率 (ROIC) 仅为 {roic}%，尚未找到规模化可持续盈利商业模型。"
            })

        # 商业化综合健康度评分 (0-100)
        base_score = 75.0
        if rev_growth > 20.0:
            base_score += 10.0
        if gm_latest > 35.0:
            base_score += 10.0
        if ocf > 0:
            base_score += 10.0
        if roic > 15.0:
            base_score += 10.0

        for rf in red_flags:
            if rf["severity"] == "CRITICAL":
                base_score -= 25.0
            else:
                base_score -= 12.0

        health_score = max(5.0, min(100.0, base_score))

        return {
            "company_id": latest.get("company_id", ""),
            "health_score": round(health_score, 1),
            "red_flags_count": len(red_flags),
            "red_flags": red_flags,
            "metrics_snapshot": {
                "revenue_yoy": rev_growth,
                "gross_margin": gm_latest,
                "net_profit": net_profit,
                "operating_cashflow": ocf,
                "ar_days": ar_days,
                "roic": roic
            }
        }
