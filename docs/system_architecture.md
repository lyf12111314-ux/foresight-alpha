# 《新兴产业与非对称财富机会智能预测系统》总体技术架构文档 (V1.0)

## 一、系统定位与设计哲学

本系统（代号：**Foresight Alpha**）旨在面向未来 3—10 年宏观经济与科技浪潮演进，通过多层级真实数据建模与量化决策，尽可能提前识别具备“非线性增长特征”的新兴产业、细分赛道、核心企业及合法可参与投资或经营机会。

### 核心设计原则
1. **真实数据与严格溯源**：杜绝伪造市场数据或随意填补虚假特征。底层数据包含来源（Source）、公开披露时间戳（`published_at`）、数据所属时间戳（`as_of_date`）与采集时间戳（`ingested_at`）。
2. **严防未来信息泄露 (Point-in-Time)**：在历史切片点 $T$ 开展回测推理时，模型仅读取 `published_at <= T` 的公开信息。
3. **区分四层转化逻辑**：
   $$\text{产业规模扩张} \neq \text{企业营收转化} \neq \text{企业估值扩张} \neq \text{投资者实际净收益}$$
4. **多模型交叉验证**：结合连续时间高斯隐马尔可夫模型 (Gaussian HMM)、S型扩散理论 (Bass / Logistic)、LightGBM 多期回归预测器、Kaplan-Meier 与 Cox 比例风险生存分析、蒙特卡洛极值情景模拟。

---

## 二、五级研究对象架构 (Five-Tier Research Taxonomy)

```
[Level 1] 技术主题 (Tech Topic, 如: 具身智能、全固态电池、AI Agent)
   │
   ▼
[Level 2] 新兴产业及细分赛道 (Emerging Industry, 如: 人形机器人关节模组、新型大储系统)
   │
   ▼
[Level 3] 产业链环节 (Value Chain Segment, 如: 上游高精密减速器/芯片、中游本体集成)
   │
   ▼
[Level 4] 具体企业与项目 (Company Entity, 如: 汇川技术、绿的谐波、三花智控)
   │
   ▼
[Level 5] 可合法持有资产/经营权益 (Investable Asset, 如: A股/港股标的、公开科技ETF)
```

---

## 三、系统模块设计

### 1. 多源数据采集中心 (`src/ingestion/`)
- 学术前沿：OpenAlex API（全球高引文献与学者机构）+ arXiv API（最新预印本）。
- 开发者生态：GitHub REST API（星标增速、活跃贡献者、代码提交频次）。
- 宏观经济序列：世界银行官方 API (World Bank API) + 国家统计局/央行官方月度公报（制造业PMI、PPI、CPI、M2、社融存量、工业企业利润）。
- 企业经营与财务时序：涵盖典型硬科技标的经审计历史财务报表（营收增速、毛利率、净利润、自由现金流、研发费用率、ROIC、应收账款周转天数）。

### 2. 宏观周期研判 (`src/macro/`)
- **Gaussian HMM**：划分四阶段经济状态（繁荣扩张、过热放缓、紧缩衰退、筑底复苏），推算状态后验概率与马尔可夫转移矩阵。
- **HP 滤波与多周期分解**：剥离基钦库存周期（~40个月）与朱格拉中周期（7-10年），定位第六次康波技术革命（扩散期）。

### 3. 技术发现与爆发拐点检测 (`src/tech_discovery/`)
- 语义嵌入与同义技术对齐（TF-IDF + Cosine Similarity）。
- 基于香农熵（Shannon Entropy）的跨机构扩散多样性指数。
- CUSUM 累积和突变检测与二阶导数加速度分析。

### 4. 技术扩散与成长预测 (`src/diffusion/`)
- 非线性最小二乘拟合经典扩散模型：
  * Bass 创新扩散模型：$N(t) = M \cdot \frac{1 - e^{-(p+q)t}}{1 + \frac{q}{p} e^{-(p+q)t}}$
  * Logistic 模型：$N(t) = \frac{K}{1 + e^{-r(t - t_0)}}$
- LightGBM 多期回归预测器（1年、3年、5年复合增长区间），附带 80% 置信区间与特征重要性。

### 5. 产业链利润池与商业化红旗诊断 (`src/value_chain/`)
- 拓扑有向图（NetworkX DiGraph）映射产业链上下游依赖与利润分配。
- 四大商业化红旗缺陷自动审计：
  1. 增收不增利与恶性价格战（营收高增但毛利率剧烈下滑）
  2. 利润含金量低（账面盈利但经营现金流为负）
  3. 应收账款周转天数畸高（> 180天回款风险）
  4. 高研发低转化（高研发费用率伴随长期负 ROIC）

### 6. 企业生存分析 (`src/risk_survival/`)
- 引入生存分析（Survival Analysis）处理未破产的右删失样本。
- Kaplan-Meier 估计生存函数，Cox Proportional Hazards 模型评估各协变量的危险比率（Hazard Ratio）。

### 7. 非对称收益与蒙特卡洛推演 (`src/payoff_engine/`)
- 5,000 次路径蒙特卡洛模拟，考虑退出流动性折价与增发稀释。
- 严格映射境内外合法二级市场标的，对早期不可投资概念仅输出产业研究，绝不捏造虚拟收益。

### 8. 滚动回测与决策融合 (`src/decision_engine/`, `src/backtest/`)
- 贝叶斯概率校准与综合打分融合。
- 步进时间序列滚动回测（Walk-Forward），全面输出 Precision@K, Recall@K, Brier Score，并与随机基准和简单动量外推对比。
