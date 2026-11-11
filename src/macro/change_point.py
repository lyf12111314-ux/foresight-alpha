"""贝叶斯宏观变点与结构性突变检测 (Bayesian Macro Change Point Detection)."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class BayesianChangePointDetector:
    """贝叶斯在线变点检测器 (BOCPD)"""

    def __init__(self, hazard_rate: float = 1.0 / 30.0):
        # 先验：假设变点发生的先验期望间隔为 30 个时间步
        self.hazard_rate = hazard_rate

    def detect_change_points(
        self,
        series: pd.Series,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """检测一维时序中的突变点及其后验概率"""
        values = series.dropna().values
        dates = series.dropna().index
        n = len(values)
        if n < 5:
            return []

        # 简化版高斯分布均值突变后验递推
        mean = np.mean(values[:3])
        var = np.var(values[:3]) + 1e-4

        change_points = []
        rolling_mean = values[0]
        rolling_var = 1.0

        for t in range(1, n):
            val = values[t]
            # 计算当前点在历史分布下的偏离度 (Z-Score)
            z_score = abs(val - rolling_mean) / (np.sqrt(rolling_var) + 1e-6)

            # 简化的后验变点概率
            prob_cp = 1.0 / (1.0 + np.exp(-1.5 * (z_score - 2.0)))

            if prob_cp >= threshold:
                change_points.append({
                    "index": t,
                    "date": dates[t] if hasattr(dates[t], "strftime") else str(dates[t]),
                    "value": float(val),
                    "previous_mean": float(rolling_mean),
                    "posterior_probability": round(float(prob_cp), 4),
                    "z_score": round(float(z_score), 2)
                })
                # 变点发生后重置均值估计
                rolling_mean = val
                rolling_var = max(rolling_var * 0.5, 0.1)
            else:
                # 正常指数平滑更新
                rolling_mean = 0.8 * rolling_mean + 0.2 * val
                rolling_var = 0.8 * rolling_var + 0.2 * ((val - rolling_mean) ** 2)

        return change_points
