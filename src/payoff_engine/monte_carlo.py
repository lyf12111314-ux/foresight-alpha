"""非对称收益与蒙特卡洛情景模拟引擎 (Asymmetric Payoff & Monte Carlo Engine).

严格区分四大层次，绝不混淆：
1. 未来产业规模可能增长10倍 (Industry Market Growth)
2. 企业收入可能增长10倍 (Company Revenue Conversion)
3. 企业估值可能增长10倍 (Enterprise Valuation Expansion)
4. 投资者实际取得10倍净收益 (Investor Net Realized Return, 扣除稀释与流动性摩擦)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class AsymmetricPayoffEngine:
    """非对称机会极值分布与蒙特卡洛模拟器"""

    def __init__(self, n_simulations: int = 5000, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.random_seed = random_seed

    def simulate_payoff(
        self,
        base_growth_multiplier: float,
        current_valuation_multiple: float,
        target_valuation_multiple_range: Tuple[float, float],
        dilution_rate_est: float = 0.15,
        exit_friction_discount: float = 0.10
    ) -> Dict[str, Any]:
        """执行 5000 次蒙特卡洛路径推演，计算四层转化率与分位数收益分布"""
        np.random.seed(self.random_seed)

        # 1. 产业整体规模扩张模拟 (对数正态肥尾分布)
        mu_ind = np.log(max(base_growth_multiplier, 1.1))
        sigma_ind = 0.45
        sim_ind_growth = np.random.lognormal(mean=mu_ind, sigma=sigma_ind, size=self.n_simulations)

        # 2. 企业营收转化率 (企业是否能够跑赢或落后行业：β 与 特质冲击)
        # 很多企业技术虽好但份额丢失或内卷，转化系数中位数约 0.70-0.90
        market_share_factor = np.random.beta(a=3, b=3, size=self.n_simulations) * 1.6
        sim_rev_growth = sim_ind_growth * market_share_factor

        # 3. 估值乘数扩张/收缩 (P/S 或 P/E Multiple)
        low_mult, high_mult = target_valuation_multiple_range
        sim_exit_multiple = np.random.uniform(low_mult, high_mult, size=self.n_simulations)
        multiple_expansion_ratio = sim_exit_multiple / max(current_valuation_multiple, 1.0)
        sim_enterprise_valuation_growth = sim_rev_growth * multiple_expansion_ratio

        # 4. 投资者实际净收益 (考虑增发稀释、交易滑点、流动性折价)
        retention_rate = (1.0 - dilution_rate_est) * (1.0 - exit_friction_discount)
        sim_investor_net_multiple = sim_enterprise_valuation_growth * retention_rate

        # 分位数统计 (Q10, Q25, Q50, Q75, Q90, Q95)
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        q_results = {
            "industry_growth": {f"Q{int(q*100)}": round(float(np.quantile(sim_ind_growth, q)), 2) for q in quantiles},
            "company_revenue": {f"Q{int(q*100)}": round(float(np.quantile(sim_rev_growth, q)), 2) for q in quantiles},
            "enterprise_valuation": {f"Q{int(q*100)}": round(float(np.quantile(sim_enterprise_valuation_growth, q)), 2) for q in quantiles},
            "investor_net_return": {f"Q{int(q*100)}": round(float(np.quantile(sim_investor_net_multiple, q)), 2) for q in quantiles}
        }

        # 核心胜率与非对称赔率指标
        prob_positive = float(np.mean(sim_investor_net_multiple > 1.0)) * 100.0  # 获得正收益概率
        prob_3x = float(np.mean(sim_investor_net_multiple >= 3.0)) * 100.0       # 获得3倍以上收益概率
        prob_10x = float(np.mean(sim_investor_net_multiple >= 10.0)) * 100.0     # 获得10倍非对称大胜概率
        prob_loss_50 = float(np.mean(sim_investor_net_multiple <= 0.50)) * 100.0 # 亏损超过50%极端风险概率

        # 预期赔率 (上行中位数 / 下行中位数)
        upside = max(float(np.quantile(sim_investor_net_multiple, 0.90)) - 1.0, 0.1)
        downside = max(1.0 - float(np.quantile(sim_investor_net_multiple, 0.10)), 0.1)
        asymmetry_ratio = round(upside / downside, 2)

        return {
            "n_simulations": self.n_simulations,
            "quantiles_breakdown": q_results,
            "metrics": {
                "prob_positive_return_pct": round(prob_positive, 1),
                "prob_3x_return_pct": round(prob_3x, 1),
                "prob_10x_super_return_pct": round(prob_10x, 1),
                "prob_severe_loss_pct": round(prob_loss_50, 1),
                "asymmetry_ratio": asymmetry_ratio
            },
            "four_dimensions_comparison": {
                "industry_growth_median": q_results["industry_growth"]["Q50"],
                "company_rev_median": q_results["company_revenue"]["Q50"],
                "valuation_growth_median": q_results["enterprise_valuation"]["Q50"],
                "investor_net_median": q_results["investor_net_return"]["Q50"]
            }
        }
