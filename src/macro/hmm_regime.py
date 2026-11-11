"""宏观周期隐马尔可夫模型 (Gaussian HMM Macro Regime Switching Engine).

严禁机械式按固定年份预言周期，采用连续概率状态转移推断：
1. 繁荣扩张 (EXPANSION)
2. 过热放缓 (SLOWDOWN)
3. 紧缩衰退 (CONTRACTION)
4. 筑底复苏 (RECOVERY)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.mixture import GaussianMixture
from src.core.types import MacroRegime


class MacroHMMRegimeModel:
    """宏观经济状态隐马尔可夫与马尔可夫区制转换模型"""

    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.regime_labels = [
            MacroRegime.CONTRACTION,
            MacroRegime.RECOVERY,
            MacroRegime.EXPANSION,
            MacroRegime.SLOWDOWN
        ]
        self.gmm: Optional[GaussianMixture] = None
        self.transition_matrix: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.state_means_: Optional[np.ndarray] = None

    def _prepare_features(self, macro_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """从宏观面板中提取特征向量 (PMI, PPI, CPI, M2, 剪刀差, 利润同比)"""
        pivoted = macro_df.pivot(index="as_of_date", columns="indicator_code", values="value").sort_index()
        pivoted = pivoted.ffill().bfill()

        # 构建衍生因子
        if "CN_PMI" in pivoted.columns:
            pivoted["PMI_DIFF"] = pivoted["CN_PMI"] - 50.0
        else:
            pivoted["PMI_DIFF"] = 0.0

        if "CN_PPI_YOY" in pivoted.columns and "CN_CPI_YOY" in pivoted.columns:
            pivoted["PPI_CPI_SPREAD"] = pivoted["CN_PPI_YOY"] - pivoted["CN_CPI_YOY"]
        else:
            pivoted["PPI_CPI_SPREAD"] = 0.0

        feature_cols = [c for c in ["PMI_DIFF", "PPI_CPI_SPREAD", "CN_M2_YOY", "CN_PROFIT_YOY"] if c in pivoted.columns]
        self.feature_names = feature_cols
        return pivoted[feature_cols].values, feature_cols

    def fit(self, macro_df: pd.DataFrame):
        """拟合宏观状态转换模型"""
        X, cols = self._prepare_features(macro_df)
        if len(X) < 4:
            raise ValueError(f"Insufficient macro samples for HMM: {len(X)}")

        # 拟合高斯混合先验
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=200
        )
        states = self.gmm.fit_predict(X)

        # 根据 PMI_DIFF 与 利润同比 对各状态按经济活力升序重排索引
        # 确保 0=衰退, 1=复苏, 2=繁荣, 3=放缓
        means = self.gmm.means_
        vitality_score = means[:, 0] * 2.0 + (means[:, -1] if means.shape[1] > 1 else 0.0)
        sorted_indices = np.argsort(vitality_score)

        # 重新映射状态
        state_mapping = {old: new for new, old in enumerate(sorted_indices)}
        mapped_states = np.array([state_mapping[s] for s in states])
        self.state_means_ = means[sorted_indices]

        # 统计转移概率矩阵 P(S_t | S_{t-1})
        trans_counts = np.zeros((self.n_regimes, self.n_regimes))
        for t in range(1, len(mapped_states)):
            prev_s = mapped_states[t - 1]
            curr_s = mapped_states[t]
            trans_counts[prev_s, curr_s] += 1

        # Laplace 平滑得到转移矩阵
        trans_counts += 0.1
        self.transition_matrix = trans_counts / trans_counts.sum(axis=1, keepdims=True)

        return self

    def predict_regimes(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """推断全时序的历史宏观状态与最新状态置信度"""
        if self.gmm is None:
            self.fit(macro_df)

        pivoted = macro_df.pivot(index="as_of_date", columns="indicator_code", values="value").sort_index()
        pivoted = pivoted.ffill().bfill()
        pivoted["PMI_DIFF"] = pivoted.get("CN_PMI", 50.0) - 50.0
        pivoted["PPI_CPI_SPREAD"] = pivoted.get("CN_PPI_YOY", 0.0) - pivoted.get("CN_CPI_YOY", 0.0)

        X = pivoted[self.feature_names].values
        probs = self.gmm.predict_proba(X)

        results = []
        for i, date_val in enumerate(pivoted.index):
            curr_prob = probs[i]
            # 找到概率最大状态
            best_state_idx = int(np.argmax(curr_prob))
            regime_enum = self.regime_labels[best_state_idx]

            row = {
                "as_of_date": date_val,
                "regime": regime_enum.value,
                "confidence": float(curr_prob[best_state_idx]),
                "prob_contraction": float(curr_prob[0]),
                "prob_recovery": float(curr_prob[1]),
                "prob_expansion": float(curr_prob[2]),
                "prob_slowdown": float(curr_prob[3])
            }
            results.append(row)

        return pd.DataFrame(results)

    def get_latest_regime(self, macro_df: pd.DataFrame) -> Dict[str, Any]:
        """获取当前最新的宏观状态诊断"""
        df_res = self.predict_regimes(macro_df)
        latest = df_res.iloc[-1].to_dict()

        # 结合状态给出产业宏观阻力/助推系数
        regime = latest["regime"]
        if regime == MacroRegime.EXPANSION.value:
            macro_factor = 1.25
            desc = "经济处于繁荣扩张期，风险偏好高，成长型技术与重资产资本开支具备极佳外部扩张环境。"
        elif regime == MacroRegime.RECOVERY.value:
            macro_factor = 1.15
            desc = "经济筑底复苏，流动性先行释放，早期突破性硬科技和卡脖子环节享受政策与资金红利。"
        elif regime == MacroRegime.SLOWDOWN.value:
            macro_factor = 0.85
            desc = "增长面临放缓与通胀分化，重点防范恶性价格战与下游需求萎缩风险。"
        else:
            macro_factor = 0.70
            desc = "紧缩出清期，现金流为王，缺乏商业自我造血能力的纯技术概念将面临极高死亡率。"

        latest["macro_growth_multiplier"] = macro_factor
        latest["macro_implication"] = desc
        latest["transition_matrix"] = self.transition_matrix.tolist() if self.transition_matrix is not None else []
        return latest
