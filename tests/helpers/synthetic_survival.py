"""测试专用合成企业生存数据生成器 (Synthetic Survival Dataset for Testing Only).

严格隔离于测试模块，禁止在正式生产训练与正式预测中自动调用。
"""

import numpy as np
import pandas as pd


def generate_synthetic_survival_cohort(n_samples: int = 250, random_state: int = 42) -> pd.DataFrame:
    """生成测试专用的企业生存队列样本 (用于 Cox 比例风险算法测试)
    
    事件定义 (测试模拟):
    duration: 存活或观察时间 (年)
    event: 1=破产/退市/违约重整, 0=正常存续(右删失)
    """
    np.random.seed(random_state)
    n = n_samples

    cash_burn_months = np.random.exponential(scale=18, size=n) + 4
    debt_ratio = np.random.beta(a=2, b=4, size=n) * 80.0
    gross_margin = np.random.normal(loc=32, scale=12, size=n)
    competitor_count = np.random.poisson(lam=12, size=n)

    risk_score = (
        (36.0 - cash_burn_months) * 0.08 +
        debt_ratio * 0.04 -
        gross_margin * 0.05 +
        competitor_count * 0.06
    )
    base_hazard = np.exp(risk_score)
    survival_years = np.random.exponential(scale=15.0 / (base_hazard + 0.1))
    observed_T = np.clip(survival_years, 0.5, 12.0)

    censoring_time = np.random.uniform(3.0, 10.0, size=n)
    event_E = (observed_T <= censoring_time).astype(int)
    duration_T = np.minimum(observed_T, censoring_time)

    return pd.DataFrame({
        "duration": duration_T,
        "event": event_E,
        "cash_burn_months": cash_burn_months,
        "debt_ratio": debt_ratio,
        "gross_margin": gross_margin,
        "competitor_count": competitor_count
    })
