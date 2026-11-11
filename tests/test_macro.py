"""宏观周期与状态转换测试 (Macro Cycle & HMM Tests)."""

import pytest
import pandas as pd
from datetime import date
from src.macro.hmm_regime import MacroHMMRegimeModel
from src.macro.cycle_filter import MacroCycleFilter
from src.core.database import db_manager


def test_macro_hmm_regime_model():
    macro_df = db_manager.query_df("SELECT * FROM macro_indicators ORDER BY as_of_date ASC")
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
