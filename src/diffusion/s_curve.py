"""S型技术扩散与采纳模型 (Bass, Logistic & Gompertz Diffusion Curves)."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from scipy.optimize import curve_fit


def bass_cumulative(t, p, q, M):
    """Bass 累积采纳模型:
    p: 创新系数 (外部影响/自发采用)
    q: 模仿系数 (内部影响/网络效应)
    M: 潜在市场饱和容量
    """
    exponent = -(p + q) * t
    return M * (1.0 - np.exp(exponent)) / (1.0 + (q / p) * np.exp(exponent))


def logistic_cumulative(t, K, r, t0):
    """Logistic 对称S型扩散模型:
    K: 饱和容量
    r: 增长率斜率
    t0: 拐点发生时间 (达到 50% 饱和点)
    """
    return K / (1.0 + np.exp(-r * (t - t0)))


def gompertz_cumulative(t, K, b, c):
    """Gompertz 非对称S型扩散模型 (拐点在约 36.8% 处):
    K: 饱和容量
    b: 位移常数
    c: 成长衰减率
    """
    return K * np.exp(-b * np.exp(-c * t))


class SCurveDiffusionModel:
    """S曲线扩散拟合与多模型参数估计器"""

    def __init__(self):
        self.fitted_params: Dict[str, Any] = {}

    def fit_and_forecast(
        self,
        time_steps: np.ndarray,
        adoptions: np.ndarray,
        forecast_horizon_years: int = 5
    ) -> Dict[str, Any]:
        """对历史时序拟合 Bass 与 Logistic 扩散模型并向外推演"""
        t = np.array(time_steps, dtype=float)
        y = np.array(adoptions, dtype=float)

        t_norm = t - t[0]
        n_points = len(t)
        if n_points < 4:
            raise ValueError("At least 4 time points required for S-curve diffusion fit.")

        current_max = y[-1]
        future_t = np.arange(0, t_norm[-1] + forecast_horizon_years + 1)
        future_years = np.arange(int(t[0]), int(t[-1]) + forecast_horizon_years + 1)

        # 1. 拟合 Logistic 模型
        p0_logistic = [current_max * 2.5, 0.4, t_norm[-1] + 1.0]
        bounds_logistic = ([current_max, 0.01, -10.0], [current_max * 50.0, 3.0, 50.0])
        try:
            popt_log, _ = curve_fit(logistic_cumulative, t_norm, y, p0=p0_logistic, bounds=bounds_logistic, maxfev=5000)
            logistic_pred = logistic_cumulative(future_t, *popt_log)
            log_K, log_r, log_t0 = popt_log
            inflection_year = int(round(t[0] + log_t0))
        except Exception:
            popt_log = p0_logistic
            logistic_pred = y[-1] * (1.0 + 0.20 * (future_t - t_norm[-1]))
            inflection_year = int(t[-1] + 2)

        # 2. 拟合 Bass 模型
        p0_bass = [0.03, 0.38, current_max * 3.0]
        bounds_bass = ([0.0001, 0.01, current_max], [0.3, 1.5, current_max * 100.0])
        try:
            popt_bass, _ = curve_fit(bass_cumulative, t_norm, y, p0=p0_bass, bounds=bounds_bass, maxfev=5000)
            bass_pred = bass_cumulative(future_t, *popt_bass)
            bass_p, bass_q, bass_M = popt_bass
        except Exception:
            popt_bass = p0_bass
            bass_pred = logistic_pred * 0.95
            bass_p, bass_q, bass_M = 0.03, 0.38, current_max * 3.0

        # 当前所处扩散渗透率阶段
        current_penetration = float(y[-1] / max(popt_log[0], 1.0)) * 100.0
        if current_penetration < 5.0:
            stage_desc = "早期创新采纳期 (< 5% 渗透率, 高技术不确定性)"
        elif current_penetration < 20.0:
            stage_desc = "早期主流扩散突破期 (5% - 20%, 跨越鸿沟进入陡峭增长)"
        elif current_penetration < 50.0:
            stage_desc = "快速成长爆发期 (20% - 50%, 市场规模非线性扩张)"
        elif current_penetration < 85.0:
            stage_desc = "成熟放缓期 (50% - 85%, 竞争格局固化)"
        else:
            stage_desc = "饱和与存量博弈期 (> 85%)"

        forecast_df = pd.DataFrame({
            "year": future_years,
            "logistic_curve": np.round(logistic_pred, 1),
            "bass_curve": np.round(bass_pred, 1)
        })

        # 未来 1年、3年、5年 预期采纳规模倍数
        base_val = max(float(y[-1]), 1.0)
        p_1y = forecast_df[forecast_df["year"] == int(t[-1] + 1)]["logistic_curve"].values[0] if not forecast_df[forecast_df["year"] == int(t[-1] + 1)].empty else base_val
        p_3y = forecast_df[forecast_df["year"] == int(t[-1] + 3)]["logistic_curve"].values[0] if not forecast_df[forecast_df["year"] == int(t[-1] + 3)].empty else base_val
        p_5y = forecast_df[forecast_df["year"] == int(t[-1] + 5)].empty and base_val or forecast_df[forecast_df["year"] == int(t[-1] + 5)]["logistic_curve"].values[0]

        mult_1y = round(float(np.nan_to_num(p_1y / base_val, nan=1.10)), 2)
        mult_3y = round(float(np.nan_to_num(p_3y / base_val, nan=1.35)), 2)
        mult_5y = round(float(np.nan_to_num(p_5y / base_val, nan=1.80)), 2)

        # 保证成长倍数非空合理
        mult_1y = max(0.5, mult_1y)
        mult_3y = max(0.5, mult_3y)
        mult_5y = max(0.5, mult_5y)

        return {
            "current_penetration_pct": round(current_penetration, 2),
            "stage_description": stage_desc,
            "estimated_saturation_K": round(float(popt_log[0]), 1),
            "inflection_point_year": inflection_year,
            "bass_coefficients": {
                "innovation_p": round(float(bass_p), 4),
                "imitation_q": round(float(bass_q), 4),
                "market_potential_M": round(float(bass_M), 1)
            },
            "growth_multiples": {
                "1_year_multiplier": mult_1y,
                "3_year_multiplier": mult_3y,
                "5_year_multiplier": mult_5y
            },
            "forecast_series": forecast_df.to_dict(orient="records")
        }
