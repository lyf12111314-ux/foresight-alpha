"""对比基准模型 (Benchmark Baselines: Random Walk & Simple Momentum Extrapolation)."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


class BaselineForecaster:
    """对比基准预测器"""

    @staticmethod
    def random_baseline(n_samples: int, seed: int = 42) -> np.ndarray:
        """随机基准 (在均匀分布区间内随机抽取)"""
        np.random.seed(seed)
        return np.random.uniform(0.8, 3.0, size=n_samples)

    @staticmethod
    def cagr_momentum_baseline(historical_series: List[float], horizon_years: int = 3) -> float:
        """简单复合历史增速 (CAGR) 外推基准"""
        if len(historical_series) < 2 or historical_series[0] <= 0:
            return 1.15 ** horizon_years

        n = len(historical_series) - 1
        historical_cagr = (historical_series[-1] / historical_series[0]) ** (1.0 / n) - 1.0
        # 截断极值防爆表
        clipped_cagr = max(-0.20, min(historical_cagr, 0.80))
        return round(float((1.0 + clipped_cagr) ** horizon_years), 2)

    @staticmethod
    def financial_filter_baseline(revenue_growth: float, gross_margin: float, roic: float) -> bool:
        """传统财务多因子静态筛选基准 (高成长 + 高毛利 + 高ROIC)"""
        return bool(revenue_growth >= 20.0 and gross_margin >= 30.0 and roic >= 10.0)
