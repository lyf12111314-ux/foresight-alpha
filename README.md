# 新兴产业与非对称财富机会智能预测系统 (Foresight Alpha V1.0)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Database: DuckDB](https://img.shields.io/badge/database-DuckDB-yellow.svg)](https://duckdb.org/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Tests: Pytest Passed](https://img.shields.io/badge/tests-16%20passed-green.svg)](https://pytest.org/)

一套面向未来 3—10 年周期的产业成长与非对称财富机会量化预测决策系统。

系统综合运用长期宏观经济周期、技术前沿演进、产业链利润沉淀、企业商业化质量、生存风险与金融合规数据，致力于在早期识别具备非线性成长潜力的产业赛道、核心标的与合法投资/经营权益。

---

## 🌟 系统核心特性

1. **真实数据与全链路血缘溯源**：
   - 直连 **OpenAlex API** 与 **arXiv API** 采集学术论文与高引引用网络；
   - 对接 **GitHub REST API** 追踪开源开发者活跃度与代码提交动能；
   - 集成 **世界银行 (World Bank API)** 与 **国家统计局/央行官方公报** 获取宏观长短期序列；
   - 提取典型硬科技上市公司真实披露财报（营收、毛利、经营现金流、ROIC、应收账款周转）。
2. **严防未来信息泄露 (Point-in-Time)**：
   - 物理记录 `as_of_date`（所属时间）、`published_at`（公开披露时间）与 `ingested_at`（抓取时间）；
   - 回测引擎在历史切片点 $T$ 时只能读取当时已公开披露的数据。
3. **区分四层转化逻辑**：
   $$\text{产业规模增长} \neq \text{企业营收转化} \neq \text{企业估值增长} \neq \text{投资者实际净收益}$$
4. **五级研究对象架构**：
   - Level 1: 技术主题 (Tech Topic)
   - Level 2: 新兴产业及细分赛道 (Emerging Industry)
   - Level 3: 产业链环节 (Value Chain Segment)
   - Level 4: 具体企业与项目 (Company Entity)
   - Level 5: 合法可参与资产/经营权益 (Investable Asset)
5. **多模型算法融合**：
   - 宏观：高斯隐马尔可夫模型 (Gaussian HMM) + 康波与基钦库存多周期分解
   - 扩散：Bass 创新扩散模型 + Logistic/Gompertz S曲线 + LightGBM 多期回归预测
   - 风险：Kaplan-Meier 生存估计 + Cox 比例风险回归 + 商业化四大红旗诊断
   - 机会：分位数回归 + 5,000次路径蒙特卡洛情景模拟
   - 回测：滚动步进回测，输出 Precision@K, Recall@K, Brier Score 概率校准优度
6. **14章节全景智能研究报告**：
   - 一键生成专业合规的深度产业研报，包含悲观/基准/乐观三情景推演与数据引用源。

---

## 📂 项目结构

```
预测模型/
├── configs/                       # 全局配置与五级产业字典定义
│   ├── system_config.yaml         # 系统存储、阈值、模型参数配置
│   └── taxonomy.yaml              # 产业与五级实体字典
├── data/                          # 数据持久化目录 (DuckDB + Parquet)
│   ├── storage.duckdb             # DuckDB 嵌入式高性能分析数据库
│   └── exports/                   # 研报与导出成果目录
├── src/                           # 核心系统实现源码
│   ├── core/                      # 数据模型规范(Pydantic)、配置加载、DuckDB管理器
│   ├── ingestion/                 # 多源数据采集流水线 (OpenAlex, arXiv, GitHub, Macro, 企业)
│   ├── macro/                     # 宏观 HMM 状态转换、多周期HP滤波与贝叶斯变点检测
│   ├── tech_discovery/            # 语义嵌入对齐、动态聚类与 CUSUM 拐点检测
│   ├── diffusion/                 # Bass/Logistic S曲线扩散拟合与 LightGBM 多期预测
│   ├── value_chain/               # 产业链图谱、利润池测算与商业化红旗诊断
│   ├── risk_survival/             # Cox 比例风险生存分析模型
│   ├── payoff_engine/             # 蒙特卡洛模拟与合法二级市场资产映射
│   ├── decision_engine/           # 多模型决策融合与非对称机会候选池构建
│   ├── monitoring/                # 动态监控巡检与预警审计
│   ├── backtest/                  # 严格 Point-in-Time 滚动回测与指标评估
│   ├── reporting/                 # 14章节智能研究报告生成器
│   ├── api/                       # FastAPI REST 微服务层
│   └── cli/                       # 命令行交互工具
├── ui/                            # 现代金融科技交互界面 (Streamlit)
│   ├── app.py                     # Streamlit 门户入口 (集成11大研究视图)
│   └── components/theme.py        # 深色专业金融工作站图表样式
├── docs/                          # 交付技术文档
│   ├── system_architecture.md     # 总体架构设计与拓扑
│   ├── data_dictionary.md         # 数据字典与表结构规范
│   ├── model_methodology.md       # 算法体系与回测数学推导
│   └── limitations_and_risks.md   # 模型局限性与合规风险说明
├── tests/                         # 全套自动化测试套件 (16/16 Passed)
├── scripts/                       # 运维与启动脚本
│   ├── setup_env.sh               # 依赖安装与虚拟环境配置
│   ├── run_pipeline.py            # 端到端一体化运行示范脚本
│   └── start_ui.sh                # Streamlit 前台一键启动
├── requirements.txt               # Python 锁定依赖清单
└── pytest.ini                     # Pytest 测试配置
```

---

## ⚡ 快速上手与运行指南

### 1. 环境初始化
系统完全支持 macOS Apple Silicon (arm64) 与 Linux/Windows，无需昂贵 GPU：
```bash
# 自动通过 uv 配置虚拟环境并安装依赖
bash scripts/setup_env.sh
```

### 2. 运行端到端完整业务流水线 (示范脚本)
执行数据采集、清洗、模型拟合、候选池生成、历史回测与研报导出：
```bash
.venv/bin/python scripts/run_pipeline.py
```

### 3. 一键启动交互式 Web 分析门户 (11 大功能页面)
```bash
bash scripts/start_ui.sh
```
在浏览器中打开：`http://localhost:8501` 即可体验专业金融科技交互工作站。

---

## 💻 命令行 CLI 工具使用

```bash
# 1. 运行数据采集流水线
.venv/bin/python -m src.cli.main ingest

# 2. 查看当前最新非对称机会候选池排名
.venv/bin/python -m src.cli.main pool

# 3. 运行商业化红旗与技术爆发异常巡检
.venv/bin/python -m src.cli.main scan

# 4. 执行 Point-in-Time 历史滚动回测 (2021 -> 2024)
.venv/bin/python -m src.cli.main backtest --cutoff-year 2021 --eval-year 2024

# 5. 一键生成指定技术的 14 章节结构化研报
.venv/bin/python -m src.cli.main report --topic TOPIC_EMBODIED_AI --output report.md
```

---

## 🧪 自动化测试验证

全套测试用例覆盖数据采集、时间审计隔离、HMM 状态转移、S 曲线拟合、生存分析、回测评估与端到端闭环：
```bash
.venv/bin/pytest tests/ -v
```
**测试结果**：16 passed, 0 failed.

---

## 🛣️ V1.0 交付清单与后续迭代路线

### V1.0 已交付功能：
- [x] 多源真实数据采集与 Point-in-Time 审计归档流水线
- [x] Gaussian HMM 宏观状态模型与康波/库存多周期分解
- [x] 语义聚类与 CUSUM 技术爆发拐点检测
- [x] Bass & Logistic 扩散模型与 LightGBM 多期增长回归
- [x] 产业链拓扑分析与四大商业化红旗缺陷诊断
- [x] Kaplan-Meier 与 Cox 比例风险企业生存模型
- [x] 5,000次路径蒙特卡洛非对称收益模拟与合法资产映射
- [x] 严格时间顺序步进回测（Precision@K、PR-AUC、Brier Score）
- [x] 14章节专业级结构化智能研报自动生成器
- [x] 11 大分析页面的 Streamlit 交互式金融级可视化界面
- [x] CLI 命令行控制中心与 REST API 接口定义

### 后续 V1.1~V2.0 迭代规划：
1. 接入企业级实时行情 Level-2 订单薄流动性特征；
2. 引入图神经网络 (GNN) 建模全球产业链断链传染效应；
3. 支持多租户权限隔离与 PostgreSQL 联邦查询扩展。
