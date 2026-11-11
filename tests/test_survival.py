"""企业生存与商业化红旗审计测试 (Survival & Commercialization Audit Tests)."""

import pytest
import pandas as pd
from datetime import date
from src.risk_survival.survival_cox import CompanySurvivalEngine
from src.value_chain.financial_audit import CommercializationAuditEngine


def test_survival_engine_cox():
    engine = CompanySurvivalEngine()
    engine.fit()
    metrics = {
        "cash_burn_months": 24.0,
        "debt_ratio": 45.0,
        "gross_margin": 35.0,
        "competitor_count": 8.0
    }
    res = engine.predict_company_survival(metrics)
    assert "partial_hazard_ratio" in res
    assert "survival_probabilities" in res
    assert res["survival_probabilities"]["surv_3y"] > 0


def test_commercialization_red_flag_audit():
    audit_engine = CommercializationAuditEngine()
    # 模拟增收不增利恶性价格战样本
    df = pd.DataFrame([
        {
            "as_of_date": date(2023, 12, 31),
            "company_id": "TEST_CO",
            "revenue": 100.0,
            "revenue_yoy": 15.0,
            "gross_margin": 45.0,
            "net_profit": 20.0,
            "ocf": 25.0,
            "ar_turnover_days": 80.0,
            "rd_ratio": 10.0,
            "roic": 15.0
        },
        {
            "as_of_date": date(2024, 12, 31),
            "company_id": "TEST_CO",
            "revenue": 160.0,
            "revenue_yoy": 60.0,
            "gross_margin": 32.0,  # 下滑 13%
            "net_profit": 5.0,
            "ocf": -15.0,         # 现金流转负
            "ar_turnover_days": 210.0,
            "rd_ratio": 8.0,
            "roic": 4.0
        }
    ])
    res = audit_engine.audit_company(df)
    assert res["red_flags_count"] >= 1
    flag_codes = [rf["code"] for rf in res["red_flags"]]
    assert "MARGIN_SQUEEZE_PRICE_WAR" in flag_codes
    assert "CASHFLOW_EARNINGS_DISTORTION" in flag_codes
