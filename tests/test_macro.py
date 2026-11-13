"""宏观周期与状态转换测试 (Macro Cycle & HMM Tests)."""

import pytest
import pandas as pd
from datetime import date, timedelta
from src.macro.hmm_regime import MacroHMMRegimeModel
from src.macro.cycle_filter import MacroCycleFilter
from src.core.database import DatabaseManager


def test_macro_hmm_regime_model(tmp_path):
    temp_db = DatabaseManager(db_path=str(tmp_path / "macro_test.duckdb"))
    conn = temp_db.get_connection()
    try:
        base = date(2020, 1, 1)
        for i in range(16):
            d = base + timedelta(days=90 * i)
            conn.execute(
                """
                INSERT INTO macro_indicators
                (indicator_code, as_of_date, published_at, indicator_name, value, unit, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ["FED_RATE", d, d, "Federal Funds Rate", 1.5 + 0.25 * (i % 5), "%", "TEST_FIXTURE"],
            )
            conn.execute(
                """
                INSERT INTO macro_indicators
                (indicator_code, as_of_date, published_at, indicator_name, value, unit, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ["GDP_YOY", d, d, "Real GDP YoY", 2.0 + 0.4 * ((i + 2) % 4), "%", "TEST_FIXTURE"],
            )
    finally:
        conn.close()

    macro_df = temp_db.query_df("SELECT * FROM macro_indicators ORDER BY as_of_date ASC")
    assert not macro_df.empty

    model = MacroHMMRegimeModel()
    model.fit(macro_df)
    regimes = model.predict_regimes(macro_df)
    assert not regimes.empty
    assert "regime" in regimes.columns
    assert "prob_expansion" in regimes.columns

    latest = model.get_latest_regime(macro_df)
    assert "regime" in latest
    assert "macro_growth_multiplier" in latest
    assert latest["macro_growth_multiplier"] > 0


def test_macro_cycle_filter_kondratiev():
    filter_engine = MacroCycleFilter()
    k_wave = filter_engine.analyze_kondratiev_wave()
    assert "第六次康波长波" in k_wave["current_wave"]
    assert "扩散" in k_wave["current_phase"]
