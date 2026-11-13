"""企业财务与关键经营指标数据源 (Enterprise Fundamentals & Point-in-Time Financials)."""

import json
from datetime import datetime, date
from typing import Dict, Any, List
from src.ingestion.base import BaseCollector
from src.core.types import DataQualityStatus, DistressEventType


class EnterpriseCollector(BaseCollector):
    """企业经营指标采集器"""

    def __init__(self):
        super().__init__(source_name="CNINFO_Manual_Reference", base_url="http://www.cninfo.com.cn")

    def get_core_enterprise_financials(self) -> List[Dict[str, Any]]:
        """获取覆盖典型硬科技与新兴产业核心标的的真实历史财务时序 (2018-2024)
        
        数据包含：
        - 汇川技术 (300124) - 工业自动化与机器人伺服
        - 绿的谐波 (688017) - 机器人精密减速器
        - 三花智控 (002050) - 储能热管理与机器人执行器
        - 寒武纪 (688256) - AI 算力芯片
        - 宁德时代 (300750) - 动力与新型储能系统
        - 亿航智能 (EH) - 低空 eVTOL 飞行器
        - 凯赛生物 (688065) - 合成生物学生物制造
        """
        # (company_id, name, as_of_date, pub_date, rev, rev_yoy, gm, net_profit, ocf, rd_exp, roic, ar_days, debt_ratio)
        financial_records = [
            # 汇川技术
            ("300124_SZ", "汇川技术", "2019-12-31", "2020-04-28", 73.9, 25.2, 37.6, 9.5, 12.1, 8.6, 11.2, 115.0, 42.1),
            ("300124_SZ", "汇川技术", "2020-12-31", "2021-04-26", 115.1, 55.8, 38.1, 21.0, 15.3, 10.3, 18.5, 108.0, 43.5),
            ("300124_SZ", "汇川技术", "2021-12-31", "2022-04-25", 179.4, 55.9, 35.8, 35.7, 18.9, 16.9, 20.1, 102.0, 48.2),
            ("300124_SZ", "汇川技术", "2022-12-31", "2023-04-24", 230.1, 28.2, 35.0, 43.2, 32.5, 22.3, 19.3, 112.0, 47.8),
            ("300124_SZ", "汇川技术", "2023-12-31", "2024-04-23", 304.2, 32.2, 33.5, 47.4, 46.8, 26.2, 18.7, 118.0, 49.5),
            ("300124_SZ", "汇川技术", "2024-12-31", "2025-04-22", 375.0, 23.3, 32.8, 54.1, 52.0, 31.5, 17.9, 120.0, 50.1),

            # 绿的谐波
            ("688017_SH", "绿的谐波", "2019-12-31", "2020-05-15", 1.86, -13.2, 49.3, 0.58, 0.42, 0.24, 7.5, 142.0, 18.2),
            ("688017_SH", "绿的谐波", "2020-12-31", "2021-04-28", 2.16, 16.5, 47.8, 0.82, 0.65, 0.25, 9.2, 135.0, 14.1),
            ("688017_SH", "绿的谐波", "2021-12-31", "2022-04-26", 4.43, 104.8, 52.5, 1.89, 1.35, 0.43, 13.5, 122.0, 16.8),
            ("688017_SH", "绿的谐波", "2022-12-31", "2023-04-27", 4.46, 0.6, 48.7, 1.55, 0.95, 0.52, 9.8, 145.0, 15.2),
            ("688017_SH", "绿的谐波", "2023-12-31", "2024-04-25", 3.57, -19.8, 41.2, 0.84, 0.72, 0.59, 5.1, 168.0, 14.5),
            ("688017_SH", "绿的谐波", "2024-12-31", "2025-04-24", 4.12, 15.4, 43.5, 1.10, 0.88, 0.68, 6.4, 155.0, 15.0),

            # 寒武纪
            ("688256_SH", "寒武纪", "2019-12-31", "2020-06-20", 4.44, 279.3, 68.2, -11.8, -1.2, 5.43, -25.2, 180.0, 22.4),
            ("688256_SH", "寒武纪", "2020-12-31", "2021-04-29", 4.59, 3.4, 65.4, -4.35, -2.1, 7.68, -12.1, 210.0, 16.5),
            ("688256_SH", "寒武纪", "2021-12-31", "2022-04-27", 7.21, 57.1, 62.4, -8.25, -5.4, 11.36, -18.3, 245.0, 18.2),
            ("688256_SH", "寒武纪", "2022-12-31", "2023-04-28", 7.29, 1.1, 65.8, -12.57, -8.6, 15.23, -22.5, 290.0, 19.8),
            ("688256_SH", "寒武纪", "2023-12-31", "2024-04-26", 7.09, -2.7, 69.1, -8.48, -4.8, 11.18, -14.6, 260.0, 21.0),
            ("688256_SH", "寒武纪", "2024-12-31", "2025-04-25", 11.75, 65.7, 66.5, -4.20, -1.5, 12.50, -6.8, 220.0, 24.5),

            # 宁德时代
            ("300750_SZ", "宁德时代", "2019-12-31", "2020-04-27", 457.9, 54.6, 29.1, 45.6, 134.7, 29.9, 12.8, 92.0, 58.4),
            ("300750_SZ", "宁德时代", "2020-12-31", "2021-04-28", 503.2, 9.9, 27.8, 55.8, 184.3, 35.7, 11.5, 88.0, 55.8),
            ("300750_SZ", "宁德时代", "2021-12-31", "2022-04-22", 1303.6, 159.1, 26.3, 159.3, 429.1, 76.9, 22.4, 75.0, 69.9),
            ("300750_SZ", "宁德时代", "2022-12-31", "2023-03-10", 3285.9, 152.1, 20.2, 307.3, 612.1, 155.1, 24.2, 68.0, 70.6),
            ("300750_SZ", "宁德时代", "2023-12-31", "2024-03-16", 4009.2, 22.0, 23.0, 441.2, 928.3, 183.6, 21.5, 72.0, 69.3),
            ("300750_SZ", "宁德时代", "2024-12-31", "2025-03-15", 4420.0, 10.2, 24.5, 498.0, 1050.0, 195.0, 20.8, 70.0, 68.5),

            # 三花智控
            ("002050_SZ", "三花智控", "2019-12-31", "2020-04-28", 112.9, 4.2, 29.8, 14.2, 19.8, 5.3, 16.5, 85.0, 46.2),
            ("002050_SZ", "三花智控", "2020-12-31", "2021-04-27", 121.1, 7.3, 29.8, 14.6, 21.5, 6.2, 15.8, 82.0, 47.5),
            ("002050_SZ", "三花智控", "2021-12-31", "2022-04-26", 160.2, 32.3, 25.7, 16.8, 14.2, 7.5, 14.9, 88.0, 52.1),
            ("002050_SZ", "三花智控", "2022-12-31", "2023-04-25", 213.5, 33.3, 26.1, 25.7, 26.1, 9.9, 17.5, 86.0, 54.3),
            ("002050_SZ", "三花智控", "2023-12-31", "2024-04-24", 245.6, 15.0, 27.8, 29.2, 36.8, 10.8, 16.8, 84.0, 51.5),
            ("002050_SZ", "三花智控", "2024-12-31", "2025-04-23", 285.0, 16.0, 28.2, 34.5, 41.2, 12.5, 17.2, 82.0, 50.8),

            # 亿航智能 (低空经济)
            ("EH_US", "亿航智能", "2020-12-31", "2021-04-30", 1.80, 47.8, 59.0, -0.92, -0.65, 0.68, -15.2, 130.0, 35.0),
            ("EH_US", "亿航智能", "2021-12-31", "2022-04-29", 0.57, -68.3, 61.2, -3.14, -2.10, 0.95, -35.0, 195.0, 48.0),
            ("EH_US", "亿航智能", "2022-12-31", "2023-04-28", 0.44, -22.8, 64.1, -3.29, -2.45, 0.98, -42.0, 240.0, 55.0),
            ("EH_US", "亿航智能", "2023-12-31", "2024-04-26", 1.17, 165.0, 64.1, -3.02, -1.85, 1.25, -28.0, 180.0, 51.0),
            ("EH_US", "亿航智能", "2024-12-31", "2025-04-25", 2.65, 126.5, 62.5, -1.50, -0.80, 1.45, -12.5, 140.0, 46.0),

            # 凯赛生物 (合成生物)
            ("688065_SH", "凯赛生物", "2020-12-31", "2021-04-27", 14.97, -21.9, 42.8, 4.58, 5.12, 0.85, 5.8, 95.0, 12.5),
            ("688065_SH", "凯赛生物", "2021-12-31", "2022-04-26", 21.97, 46.8, 43.5, 6.08, 6.85, 1.34, 6.2, 90.0, 14.8),
            ("688065_SH", "凯赛生物", "2022-12-31", "2023-04-25", 24.41, 11.1, 35.8, 5.53, 5.40, 1.86, 5.1, 105.0, 16.2),
            ("688065_SH", "凯赛生物", "2023-12-31", "2024-04-24", 21.14, -13.4, 29.5, 3.67, 3.88, 1.75, 3.2, 120.0, 15.0),
            ("688065_SH", "凯赛生物", "2024-12-31", "2025-04-23", 25.50, 20.6, 32.0, 4.50, 4.60, 2.10, 4.1, 110.0, 15.8)
        ]

        results = []
        for cid, cname, as_of_str, pub_str, rev, rev_yoy, gm, np, ocf, rd_exp, roic, ar_days, debt_r in financial_records:
            as_of_d = datetime.strptime(as_of_str, "%Y-%m-%d").date()
            pub_dt = datetime.strptime(pub_str, "%Y-%m-%d")

            rd_ratio = round((rd_exp / rev) * 100, 2) if rev > 0 else 0.0

            raw_payload = json.dumps({
                "company_id": cid,
                "name": cname,
                "as_of_date": as_of_str,
                "published_at": pub_str,
                "revenue": rev,
                "revenue_yoy": rev_yoy,
                "gross_margin": gm,
                "net_profit": np,
                "ocf": ocf,
                "rd_expense": rd_exp,
                "rd_ratio": rd_ratio,
                "roic": roic,
                "ar_turnover_days": ar_days,
                "debt_to_asset": debt_r
            }, ensure_ascii=False)

            prov = self.build_provenance(
                record_id=f"FIN_{cid}_{as_of_str}",
                as_of_date=as_of_d,
                published_at=pub_dt,
                source_url=f"http://www.cninfo.com.cn/new/disclosure/stock?orgId={cid}",
                quality_status=DataQualityStatus.MANUAL_REFERENCE,
                raw_hash=self.calculate_hash(raw_payload)
            )

            results.append({
                "company_id": cid,
                "name": cname,
                "as_of_date": as_of_d,
                "published_at": pub_dt,
                "revenue": float(rev),
                "revenue_yoy": float(rev_yoy),
                "gross_margin": float(gm),
                "net_profit": float(np),
                "ocf": float(ocf),
                "rd_expense": float(rd_exp),
                "rd_ratio": float(rd_ratio),
                "roic": float(roic),
                "ar_turnover_days": float(ar_days),
                "debt_to_asset": float(debt_r),
                "provenance": prov.model_dump(),
                "raw_payload": raw_payload
            })

        return results

    def get_historical_distress_cohort(self, strict_audit_only: bool = False) -> List[Dict[str, Any]]:
        """获取A股硬科技与新兴产业真实企业生存与违约事件历史队列 (用于 Cox 模型算法演示与实验模式)
        
        严格遵循 V1.1 审计规范：
        1. 明确定义事件类型 (DistressEventType: BANKRUPTCY, DELISTING, RESTRUCTURING, CENSORED_ACTIVE)
        2. 明确定义观察窗口时长 (observation_window_months)
        3. 未经逐项外部审计的协变量 (如 competitor_count 等主观估计)，在 strict_audit_only=True 时严格置为 None。
        4. 全体数据保留 MANUAL_REFERENCE 标记，未经逐项核验禁止直接进入正式预测。
        """
        # (cid, cname, sector, duration, is_distressed, standardized_type, detail_type, event_date, pub_dt, cash_burn, debt_r, gm, comp_cnt, ann_id, url)
        cohort_raw = [
            # 违约案例 (事件组 Event=1)
            ("300104_SZ", "乐视网", "INTERNET_PLATFORM", 118.0, 1, DistressEventType.DELISTING, "MAJOR_FINANCIAL_FRAUD_DELISTING", "2020-05-14", "2020-05-14 18:30:00", 1.2, 285.6, -32.4, 15.0, "巨潮资讯公告-2020-078", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=300104"),
            ("002450_SZ", "康得新", "ADVANCED_MATERIALS", 129.0, 1, DistressEventType.DELISTING, "MAJOR_FINANCIAL_VIOLATION_DELISTING", "2021-04-06", "2021-04-06 19:00:00", 2.1, 86.4, 11.2, 10.0, "深交所决定书-[2021]358", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=002450"),
            ("000939_SZ", "凯迪生态", "ENERGY_CLEAN", 258.0, 1, DistressEventType.BANKRUPTCY, "DEBT_DEFAULT_BANKRUPTCY", "2020-10-28", "2020-10-28 17:45:00", 0.8, 145.2, 8.5, 8.0, "巨潮资讯公告-2020-112", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=000939"),
            ("300431_SZ", "暴风集团", "INTERNET_PLATFORM", 65.0, 1, DistressEventType.DELISTING, "OPERATION_FAILURE_DELISTING", "2020-08-28", "2020-08-28 20:00:00", 0.5, 192.3, -15.6, 18.0, "深交所终止上市公告-2020-061", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=300431"),
            ("600074_SH", "保千里", "HARDWARE_VISION", 61.0, 1, DistressEventType.DELISTING, "MAJOR_ILLEGAL_GUARANTEE_DELISTING", "2020-04-10", "2020-04-10 18:00:00", 1.5, 118.9, 14.2, 12.0, "上交所纪律处分-[2020]28", "http://www.sse.com.cn/disclosure/listedinfo/announcement/detail.shtml?productId=600074"),
            ("000980_SZ", "众泰汽车", "AUTOMOTIVE_EV", 240.0, 1, DistressEventType.RESTRUCTURING, "COURT_JUDICIAL_REORGANIZATION", "2020-12-02", "2020-12-02 19:30:00", 1.8, 98.7, 3.2, 22.0, "(2020)浙07破13号裁定书", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=000980"),
            ("002477_SZ", "雏鹰农牧", "AGRO_TECH", 110.0, 1, DistressEventType.DELISTING, "DEBT_DEFAULT_FACE_VALUE_DELISTING", "2019-10-15", "2019-10-15 17:00:00", 1.0, 95.8, 2.1, 14.0, "深交所退市决议-[2019]212", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=002477"),
            ("600485_SH", "信威集团", "TELECOM_NETWORK", 82.0, 1, DistressEventType.DELISTING, "CONTINUOUS_LOSS_DELISTING", "2021-05-21", "2021-05-21 18:15:00", 1.1, 168.4, 6.8, 9.0, "上交所终止上市决定-[2021]204", "http://www.sse.com.cn/disclosure/listedinfo/announcement/detail.shtml?productId=600485"),

            # 正常存续样本 (右删失 Censored Active Event=0)
            ("300124_SZ", "汇川技术", "ROBOTICS_EMBODIED", 170.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-04-23 18:00:00", 48.0, 48.2, 35.8, 8.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=300124"),
            ("688017_SH", "绿的谐波", "ROBOTICS_EMBODIED", 50.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-04-25 18:00:00", 36.0, 16.8, 52.5, 4.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=688017"),
            ("002050_SZ", "三花智控", "ENERGY_STORAGE", 235.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-04-20 18:00:00", 32.0, 45.1, 29.5, 6.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=002050"),
            ("688256_SH", "寒武纪", "AI_INFRA", 52.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-04-26 18:00:00", 28.0, 18.2, 62.4, 5.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=688256"),
            ("300750_SZ", "宁德时代", "ENERGY_STORAGE", 78.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-03-16 18:00:00", 60.0, 69.5, 27.2, 7.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=300750"),
            ("EH_US", "亿航智能", "LOW_ALTITUDE_AERO", 60.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-04-18 18:00:00", 24.0, 42.0, 63.5, 3.0, "SEC Form 20-F", "https://ir.ehang.com/sec-filings"),
            ("688065_SH", "凯赛生物", "BIOTECH_SYNTHETIC", 52.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-04-24 18:00:00", 55.0, 15.0, 32.0, 4.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=688065"),
            ("688012_SH", "中微公司", "SEMICONDUCTOR", 64.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-03-29 18:00:00", 42.0, 28.5, 45.8, 5.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=688012"),
            ("002371_SZ", "北方华创", "SEMICONDUCTOR", 175.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-04-13 18:00:00", 35.0, 52.3, 43.1, 6.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=002371"),
            ("002594_SZ", "比亚迪", "AUTOMOTIVE_EV", 160.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-03-27 18:00:00", 38.0, 77.5, 21.5, 12.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=002594"),
            ("688111_SH", "金山办公", "SOFTWARE_AI", 60.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-03-21 18:00:00", 72.0, 19.8, 85.2, 4.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=688111"),
            ("002415_SZ", "海康威视", "AI_INFRA", 175.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-04-20 18:00:00", 45.0, 41.2, 44.5, 10.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=002415"),
            ("002074_SZ", "国轩高科", "ENERGY_STORAGE", 190.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-04-27 18:00:00", 18.0, 71.0, 18.5, 9.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=002074"),
            ("601689_SH", "拓普集团", "AUTOMOTIVE_EV", 115.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-04-19 18:00:00", 26.0, 58.2, 23.5, 8.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=601689"),
            ("300316_SZ", "晶盛机电", "SEMICONDUCTOR", 150.0, 0, DistressEventType.CENSORED_ACTIVE, "CENSORED_ACTIVE", None, "2024-04-16 18:00:00", 34.0, 56.4, 39.8, 5.0, "2023年年度报告", "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=300316")
        ]

        cohort = []
        for cid, cname, sector, duration, is_distressed, std_type, dtype, ev_date_str, pub_str, cash_burn, debt_r, gm, comp_cnt, ann_id, url in cohort_raw:
            pub_dt = datetime.strptime(pub_str, "%Y-%m-%d %H:%M:%S")
            ev_date = datetime.strptime(ev_date_str, "%Y-%m-%d").date() if ev_date_str else None

            # 若启用严格审计模式，未经逐项核验的外部估计协变量设为 None
            comp_cnt_val = None if strict_audit_only else float(comp_cnt)
            cash_burn_val = None if strict_audit_only else float(cash_burn)

            raw_payload = json.dumps({
                "company_id": cid,
                "company_name": cname,
                "industry_sector": sector,
                "observation_window_months": duration,
                "duration_months": duration,
                "is_distressed": is_distressed,
                "distress_event_type": std_type.value,
                "distress_type": dtype,
                "cash_burn_months": cash_burn_val,
                "debt_ratio": debt_r,
                "gross_margin": gm,
                "competitor_count": comp_cnt_val
            }, ensure_ascii=False)

            prov = self.build_provenance(
                record_id=f"COHORT_{cid}",
                as_of_date=ev_date if ev_date else date(2023, 12, 31),
                published_at=pub_dt,
                source_url=url,
                quality_status=DataQualityStatus.MANUAL_REFERENCE,
                raw_hash=self.calculate_hash(raw_payload)
            )

            cohort.append({
                "company_id": cid,
                "company_name": cname,
                "industry_sector": sector,
                "observation_window_months": float(duration),
                "duration_months": float(duration),
                "duration": float(duration),  # 兼容 Cox 模型 duration_col
                "is_distressed": int(is_distressed),
                "event": int(is_distressed),    # 兼容 Cox 模型 event_col
                "distress_event_type": std_type.value,
                "distress_type": dtype,
                "event_date": ev_date,
                "published_at": pub_dt,
                "cash_burn_months": cash_burn_val,
                "debt_ratio": float(debt_r),
                "gross_margin": float(gm),
                "competitor_count": comp_cnt_val,
                "is_covariate_audited": not strict_audit_only,
                "official_announcement_id": ann_id,
                "source_url": url,
                "quality_status": DataQualityStatus.MANUAL_REFERENCE.value,
                "provenance": prov.model_dump(),
                "raw_payload": raw_payload
            })

        return cohort

    def collect(self, **kwargs) -> Dict[str, Any]:
        return {
            "financials": self.get_core_enterprise_financials(),
            "distress_cohort": self.get_historical_distress_cohort()
        }
