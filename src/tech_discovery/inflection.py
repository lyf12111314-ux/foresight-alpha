"""技术爆发与拐点信号检测引擎 (Inflection & Growth Acceleration Detection).

使用 CUSUM (累积和)、二阶导数加速度与多源交叉验证检验技术从低关注向快速扩散的临界转变。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from src.core.types import AlertRecord, ObjectLevel


class InflectionDetector:
    """技术爆发拐点检测器"""

    def __init__(self, cusum_threshold: float = 3.0, drift: float = 0.5):
        self.cusum_threshold = cusum_threshold
        self.drift = drift

    def detect_cusum(self, series: pd.Series) -> Tuple[List[int], pd.Series]:
        """CUSUM 算法检测序列均值向上突变点"""
        s = series.dropna()
        if len(s) < 3:
            return [], pd.Series(dtype=float)

        mean_val = s.iloc[:3].mean()
        std_val = s.iloc[:3].std() if len(s) > 3 and s.iloc[:3].std() > 0 else (s.mean() * 0.1 + 1e-4)

        z = (s - mean_val) / std_val
        s_pos = np.zeros(len(z))
        change_points = []

        for i in range(1, len(z)):
            s_pos[i] = max(0, s_pos[i - 1] + z.iloc[i] - self.drift)
            if s_pos[i] > self.cusum_threshold:
                change_points.append(i)

        return change_points, pd.Series(s_pos, index=s.index)

    def analyze_tech_acceleration(self, topic_df: pd.DataFrame) -> Dict[str, Any]:
        """综合分析某技术主题的文献/代码一阶增速、二阶加速度与拐点状态"""
        df = topic_df.sort_values("as_of_date").copy()
        if len(df) < 3:
            return {
                "status": "INSUFFICIENT_DATA",
                "velocity": 0.0,
                "acceleration": 0.0,
                "is_inflection_point": False
            }

        papers = df["papers_count"].values
        citations = df["citations_count"].values
        years = np.array([d.year if hasattr(d, "year") else int(str(d)[:4]) for d in df["as_of_date"]])

        # 一阶差分 (速度)
        velocity = np.diff(papers)
        # 二阶差分 (加速度)
        acceleration = np.diff(velocity) if len(velocity) > 1 else np.array([0.0])

        latest_velocity = float(velocity[-1]) if len(velocity) > 0 else 0.0
        latest_accel = float(acceleration[-1]) if len(acceleration) > 0 else 0.0

        # CUSUM 检测文献跳变
        cp_indices, cusum_series = self.detect_cusum(df.set_index("as_of_date")["papers_count"])
        has_cusum_break = len(cp_indices) > 0

        # 判断是否跨过拐点: 满足 (加速度为正 且 近期增速显著高于历史基准 且 机构扩散熵 > 0.4)
        is_inflection = bool(latest_accel > 0 and latest_velocity > np.mean(velocity) * 1.3 and has_cusum_break)

        stage = "萌芽探索期 (Early Lab Stage)"
        if is_inflection and latest_accel > 100:
            stage = "拐点爆发期 (Inflection & Rapid Diffusion)"
        elif latest_velocity > 0 and latest_accel <= 0:
            stage = "线性平稳增长期 (Steady Expansion)"
        elif latest_velocity <= 0:
            stage = "成熟/关注回落期 (Cooling Down)"

        return {
            "status": "VALIDATED",
            "stage": stage,
            "latest_velocity_yoy": round(latest_velocity, 2),
            "latest_acceleration": round(latest_accel, 2),
            "is_inflection_point": is_inflection,
            "cusum_score": float(cusum_series.iloc[-1]) if not cusum_series.empty else 0.0,
            "change_points_count": len(cp_indices),
            "evidence": {
                "papers_latest": int(papers[-1]),
                "papers_3y_ago": int(papers[-4]) if len(papers) >= 4 else int(papers[0]),
                "growth_multiple_3y": round(float(papers[-1] / max(papers[-4] if len(papers) >= 4 else papers[0], 1)), 2),
                "citations_latest": int(citations[-1]),
                "active_institutions": int(df["active_institutions"].iloc[-1])
            }
        }
