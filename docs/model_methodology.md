# 《新兴产业与非对称财富机会智能预测系统》模型算法方法论与回测体系

## 一、宏观周期与区制转换 (Macro Regime Switching)

### 1.1 高斯隐马尔可夫模型 (Gaussian HMM)
宏观经济状态难以直接观测，定义离散潜在状态 $S_t \in \{0, 1, 2, 3\}$（分别对应：紧缩衰退、筑底复苏、繁荣扩张、过热放缓）。
观测向量 $Y_t = [\text{PMI}-50, \text{PPI}-\text{CPI}, \Delta \text{M2}, \text{Profit YoY}]^T$ 遵循多元高斯分布：
$$P(Y_t \mid S_t = k) \sim \mathcal{N}(\mu_k, \Sigma_k)$$
状态转移矩阵 $A_{ij} = P(S_t = j \mid S_{t-1} = i)$ 刻画经济周期在各象限之间的平滑转换。

---

## 二、S型扩散理论与非线性成长拟合

### 2.1 Bass 创新扩散模型
技术或产品在潜在市场 $M$ 中的采纳过程受“创新效应”（外部媒体、政策）与“模仿效应”（口碑效应、网络效应）共同驱动：
$$\frac{dN(t)}{dt} = \left[ p + q \frac{N(t)}{M} \right] [M - N(t)]$$
累积采纳量显式解：
$$N(t) = M \frac{1 - e^{-(p+q)t}}{1 + \frac{q}{p} e^{-(p+q)t}}$$
- $p$：创新系数 (Innovation Coefficient, 通常在 0.01 ~ 0.03 区间)
- $q$：模仿系数 (Imitation Coefficient, 通常在 0.30 ~ 0.50 区间)
- $M$：潜在市场容量上限 (Market Potential)

### 2.2 Gompertz 与 Logistic 模型
- 对称 S 曲线拐点位于 $50\% \cdot K$（Logistic 模型）。
- 非对称 S 曲线拐点位于 $\frac{1}{e} \approx 36.8\% \cdot K$（Gompertz 模型），更能真实刻画高技术门槛工业品早期爬坡艰难、后期快速放量的特征。

---

## 三、生存分析与经营失败风险模型 (Survival Analysis)

### 3.1 删失数据处理 (Right Censoring)
很多初创硬科技公司与未暴雷企业处于正常存续中，其“真实死亡时间”大于观测期（右删失 $E_i = 0$）。
采用 **Cox 比例风险模型 (Cox Proportional Hazards)**：
$$\lambda(t \mid X) = \lambda_0(t) \exp(X^T \beta)$$
其中协变量 $X$ 涵盖：
1. 现金消耗月数（Cash Burn Runway）
2. 资产负债率（Debt-to-Asset Ratio）
3. 销售毛利率（Gross Margin）
4. 同行业竞争密度（Competitor Density）
计算危险比率 $\text{HR} = \exp(\beta_k)$，度量各因子对企业违约/倒闭风险的边际影响。

---

## 四、非对称收益与蒙特卡洛情景模拟

### 4.1 四层转化链条
$$\text{产业规模增长} \xrightarrow{\times \beta_{\text{share}}} \text{企业收入增长} \xrightarrow{\times \frac{P/S_T}{P/S_0}} \text{企业估值增长} \xrightarrow{\times (1-\text{稀释})(1-\text{流动性摩擦})} \text{投资者实际净收益}$$

### 4.2 蒙特卡洛推演
采用 5,000 次路径模拟：
- 对产业规模增长施加肥尾对数正态冲击；
- 结合估值分位数收敛推导非对称赔率比（Asymmetry Ratio = 上行90%分位数收益 / 下行10%分位数回撤）。

---

## 五、Point-in-Time 滚动回测与概率校准体系

### 5.1 严防未来信息泄漏 (Lookahead Bias)
回测步进器在历史时点 $T$（如 2021-12-31）运行时：
$$\text{可使用数据集} = \{ D_i \mid \text{published\_at}(D_i) \le T \}$$
任何在 $T$ 时刻之后公开披露的年报修正、事后统计数据一律在物理内存中不可见。

### 5.2 评估指标
1. **Precision@K**：Top-K 机会命中率（实际成长倍数 $\ge 1.9\times$ 的比例）。
2. **PR-AUC**：精确率-召回率曲线下面积。
3. **Brier Score**：概率校准优度：
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
4. **超额胜率**：与均匀随机基准（Random Benchmark）和简单历史动量外推（CAGR Extrapolation）的直接净胜率差值。
