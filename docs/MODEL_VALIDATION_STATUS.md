# Foresight Alpha V1.1 — 全系统模型验证状态透明化公示清单 (Model Validation Status)

> **审计核准时点**：`2026-11-12`  
> **审计标准**：Foresight Alpha V1.1 真实性、独立样本外准入门禁与可信度规范  
> **系统设计准则**：算法拟合不等于模型有效；测试通过不等于具备生产资格；未核验数据拟合的模型一律处于 `EXPERIMENTAL` 状态，严禁输出正式预测 (`is_verified=False`)；情景模拟必须与正式预测物理隔离。

---

## 一、系统核心模型验证状态总览

| 模型模块名称 | 算法/统计引擎 | 当前阶段 (ModelStage) | 算法拟合收敛 (`is_algorithm_fitted`) | 训练数据已核验 (`is_data_verified`) | 独立样本外审计 (`is_oos_validated`) | 预测概率已校准 (`is_calibrated`) | 正式生产预测准入 (`is_verified`) | 数据血缘与审计说明 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **产业多期成长预测器** | LightGBM Regressor (多期限复合增长) | `EXPERIMENTAL` (实验研究阶段) | ✅ 是 | ❌ 否 (`MANUAL_REFERENCE`) | ❌ 否 | ❌ 否 | ❌ **禁止** | 基于 45 个前沿技术跨周期历史样本拟合，未逐项完成生产级审计，仅供实验模式研究。独立样本外准入已部署 6 步审计流水线 (Max MAE <= 2.0 且超越基准模型)。 |
| **企业生存分析器** | Cox 比例风险回归 + Kaplan-Meier 估计 | `EXPERIMENTAL` (实验研究阶段) | ✅ 是 | ❌ 否 (`MANUAL_REFERENCE`) | ❌ 否 | ❌ 否 | ❌ **禁止** | 基于 23 家 A 股违约/存续历史队列拟合，协变量未经逐项审计，严禁断言实际企业破产概率。时间单位严格统一为 `MONTHS` (12m/36m/60m)，缺失财务数据直接拒绝。 |
| **技术扩散曲线模型** | Bass / Logistic S-Curve 非线性回归 | `BENCHMARK` (基准研判阶段) | ✅ 是 | ✅ 是 (公开文献量) | ❌ 否 | ❌ 否 | ❌ **禁止** | 用于拟合学术文献的渗透拐点与采纳阶段，严禁直接等同于产业财务成长倍数。 |
| **宏观周期状态机** | GMM-Regime (4 状态高斯混合聚类识别) | `BENCHMARK` (基准研判阶段) | ✅ 是 | ✅ 是 (世界银行/国家统计局) | ⚠️ 部分 | ❌ 否 | ❌ **禁止** | 输出宏观景气周期与成长阻尼系数，作为环境先验输入，不单独承担标的预测。采用高斯混合聚类与经验转移频率，非 Baum-Welch 隐马尔可夫模型。 |
| **技术动能拐点检测器** | 物理二阶差分滤波与加速度检测 | `VERIFIED_PRODUCTION` (确定性分析工具) | ✅ 是 | ✅ 是 (无未来时序) | ✅ 是 | N/A | ✅ **分析工具** | 基于一阶增速与二阶加速度计算拐点信号，属于无参/规则型确定性研判，无过拟合风险。 |
| **产业链利润池分析器** | 产业链图谱拓扑与环节毛利中枢聚合 | `VERIFIED_PRODUCTION` (确定性分析工具) | ✅ 是 | ✅ 是 (CNINFO 审计年报) | N/A | N/A | ✅ **分析工具** | 统计上下游产业链毛利率分布，沉淀最高壁垒环节，属于客观历史描述性统计。 |
| **非对称赔率引擎** | 蒙特卡洛随机折现与极值模拟 | `HYPOTHETICAL_SCENARIO` (情景模拟阶段) | ✅ 是 | N/A (参数化推演) | ❌ 否 | ❌ 否 | ❌ **禁止** | 严格区分 `FORMAL_PREDICTION` 与 `HYPOTHETICAL_SCENARIO`；实验模型接入时正式 `prob_10x_super_return_pct` 严格为 `None`，携带明确免责声明。 |
| **多模型综合决策器** | 决策融合与客观研究评分加权 | `BENCHMARK` (研究排序阶段) | ✅ 是 | ⚠️ 部分 | ❌ 否 | ❌ 否 | ❌ **禁止** | 输出客观非概率研究综合评分 (0~100) 用于研究排序，未验证时校准概率字段严格设为 `None`，预测记录不可变版本化存储。 |
| **滚动回测评估引擎** | Point-in-Time Walk-Forward 评估器 | `VERIFIED_PRODUCTION` (评估工具) | ✅ 是 | ✅ 是 (物理时序隔离) | ✅ 是 | ✅ 是 | ✅ **评估工具** | 严格按历史截面独立训练历史模型；历史样本不足时返回 `MODEL_UNAVAILABLE_AS_OF_DATE`；超期评估返回 `NOT_MATURED`；不可篡改增量入库。 |

---

## 二、状态定义与准入原则

### 1. 验证执行类型分类 (ValidationExecutionType)
- **`FORMAL_PRODUCTION_AUDIT` (正式生产准入验证)**：
  对独立样本外核验数据集执行严格 6 步审计，仅在所有样本具备真实血缘、时序无未来泄漏、误差低于生产硬性上限且战胜基准模型时，方允许晋升为 `VERIFIED_PRODUCTION`。
- **`TEST_ONLY` (工程与代码逻辑测试专用)**：
  专用于 CI/CD 与单元测试验证算法收敛性。即使测试指标优秀，模型阶段强制保持 `BENCHMARK`，绝不晋升生产状态。
- **`EXPERIMENTAL` (实验性研究验证)**：
  针对探索性模型进行评估，保留实验记录，不参与生产预测。

### 2. 情景模拟隔离类型 (SimulationType)
- **`FORMAL_PREDICTION` (正式预测推演)**：
  基于生产级验证模型输出与真实估值中枢推演，具备独立校准胜率。
- **`HYPOTHETICAL_SCENARIO` (假设情景推演)**：
  人工或系统设定的敏感性情景分析。正式决策指标 `prob_10x_super_return_pct` 严格为 `None`，估算值置于 `scenario_prob_10x_pct`，并附带明确免责声明。
- **`EXPERIMENTAL_RESEARCH` (实验性研究估算)**：
  科研探索性推演，不作商业与投资决策参考。

### 3. 生产准入硬性门槛与可信注册中心
- **服务端可信验证数据集注册中心 (`TrustedValidationRegistry`)**：
  正式生产验证准入只接受经独立审计登记的数据集引用 (`registered_dataset_id`)。
  - **规范化 SHA-256 摘要校验**：服务端根据样本集特征、标签、时序与来源严格重新计算规范哈希 (`compute_samples_checksum`)，任何手工伪造的 64 位哈希均会被立即识别拦截 (`REJECTED_CHECKSUM_TAMPERED`)。
  - **样本替换防御**：调用方试图复用已注册 ID 传入篡改后的样本时，直接触发 `REJECTED_SAMPLE_SUBSTITUTION` 拦截。
  - **权威证书门禁**：注册数据集必须具备权威第三方审计凭证 (`has_independent_audit_certificate=True`)。系统预置示例数据集 (`REG_OOS_TECH_V1_2023`, `REG_OOS_COHORT_V1_2023`) 严格标记为未获生产审计证书，正式生产验证入口保持阻断 (`REJECTED_DATASET_NOT_AUDITED_FOR_PRODUCTION`)。
  - **防作弊反向审计**：严格校验验证标签与模型自身预测值的贴合度 (`atol <= 0.25` 或相关系数 $r > 0.999$)，检测到模型自生标签作弊时直接拒绝晋升 (`REJECTED_SELF_GENERATED_CHEAT`)。
- **三轨独立历史回测体系 (`P0-02`)**：
  历史回测分为三类，物理隔离存储于 `backtest_experiments` 表，彻底消除混淆：
  1. `ML_VALIDATED_BACKTEST`：仅允许调用基于 VERIFIED 数据训练且通过独立 OOS 生产审计的模型。若历史模型未达标，机器学习指标严格为 `None`，不伪造准确率。
  2. `ML_EXPERIMENTAL_BACKTEST`：调用历史实验模型，回测状态显式标记为 `EXPERIMENTAL`，预测状态为 `EXPERIMENTAL_PREDICTION`，绝不标记为 `VALIDATED`。
  3. `RESEARCH_RANKING_BACKTEST`：纯规则型客观研究排序回测，与机器学习模型预测彻底物理隔离。
- **端到端动态特征生成流水线 (`P0-03`)**：
  从数据库历史时序查询真实动态特征（动态计算 `entropy_score`, `novelty_score`，严禁硬编码默认值 0.60/0.70）。
  每个候选标的均携带完整特征向量字典存储于 `c["features"]`。正式历史预测遇到特征缺失时严格返回 `status="INSUFFICIENT_FEATURES"`，特征公开时间晚于历史时点时严格拒绝前视偏差。
- **Point-in-Time 真实时钟严格模式 (`P0-04 / P1-02`)**：
  历史模型训练严格排除 `feature_published_at IS NULL` 记录；`query_point_in_time` 在 `require_verified_timestamp=True` 时执行严格时序审计，若底层表元数据校验失败，直接抛出 `RuntimeError`，绝不静默放宽过滤条件。
- **LightGBM 成长预测器**：
  生产准入硬性上限 $\text{MAE} \le 2.0$ 且必须显著超越全预测训练集均值/中位数基准 (Naive Baseline)。调用方传入宽松阈值会被系统自动截断回硬性上限。
- **Cox 企业生存风险模型**：
  生产准入硬性下限 $C\text{-index} \ge 0.60$ 且必须超越随机基准 (0.50)。正式预测 (`is_formal=True`) 必须全部具备 4 项核心协变量 (`cash_burn_months`, `debt_ratio`, `gross_margin`, `competitor_count`)，严禁默认值填充。
- **双轨排序治理机制**：
  在当前阶段，全系统正式预测排名严格返回空 (`NO_VERIFIED_FORECASTS`)，真实反映系统能力边界。所有实验预测与研究报告均显著标注 `UNVERIFIED_RESEARCH (实验研究模式 - 未获生产验证)`。
