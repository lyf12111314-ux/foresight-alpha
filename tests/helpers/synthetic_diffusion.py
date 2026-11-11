"""测试专用合成扩散数据生成器 (Synthetic Diffusion Dataset for Testing Only).

严格隔离于测试模块，禁止在正式生产训练与正式预测中自动调用。
"""

import numpy as np
from typing import Dict, Tuple


def generate_synthetic_diffusion_samples(
    n_samples: int = 300,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """生成测试专用的合成技术扩散样本 (用于算法拟合能力单元测试)"""
    np.random.seed(random_state)
    N = n_samples

    vel = np.random.exponential(scale=200, size=N) + 20
    acc = np.random.normal(loc=15, scale=50, size=N)
    inst = np.random.lognormal(mean=5.5, sigma=1.0, size=N)
    gh = np.random.exponential(scale=1000, size=N)
    macro = np.random.choice([0.7, 0.85, 1.15, 1.25], size=N, p=[0.2, 0.3, 0.3, 0.2])
    entropy = np.random.uniform(0.2, 0.95, size=N)
    novelty = np.random.uniform(0.1, 0.99, size=N)

    X = np.column_stack([vel, acc, inst, gh, macro, entropy, novelty])

    y_1y = np.clip(1.0 + (vel / 400.0) * 0.4 + (acc > 0) * 0.15 + (macro - 1.0) * 0.3 + np.random.normal(0, 0.08, N), 0.7, 3.5)
    y_3y = np.clip((y_1y ** 2.2) * (1.0 + entropy * 0.5) * (1.0 + (inst > 500) * 0.4) + np.random.normal(0, 0.25, N), 0.8, 12.0)
    y_5y = np.clip((y_3y ** 1.3) * (1.0 + (novelty > 0.6) * 0.6) + np.random.normal(0, 0.6, N), 0.9, 35.0)

    return X, {1: y_1y, 3: y_3y, 5: y_5y}
