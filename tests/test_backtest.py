"""Point-in-Time 回测与评估指标测试 (Backtest & Calibration Metrics Tests)."""

import pytest
from datetime import date
from src.backtest.metrics import BacktestMetricsCalculator
from src.backtest.engine import PointInTimeBacktestEngine


def test_backtest_metrics_calculation():
    calc = BacktestMetricsCalculator()
    preds = [("T1", 95.0), ("T2", 90.0), ("T3", 85.0), ("T4", 60.0), ("T5", 40.0)]
    labels = {"T1": 1, "T2": 1, "T3": 0, "T4": 1, "T5": 0}

    p3 = calc.precision_at_k(preds, labels, k=3)
    assert p3 == 0.667

    p5 = calc.precision_at_k(preds, labels, k=5)
    assert p5 == 0.60

    brier = calc.brier_score([0.9, 0.8, 0.2], [1, 1, 0])
    assert brier < 0.10


def test_walk_forward_engine():
    engine = PointInTimeBacktestEngine()
    res = engine.run_walk_forward_backtest(
        cutoff_date=date(2021, 12, 31),
        evaluation_date=date(2024, 12, 31)
    )
    assert "metrics" in res
    assert "precision_at_3" in res["metrics"]
    assert "brier_score" in res["metrics"]
    assert "excess_vs_random_baseline" in res["metrics"]
