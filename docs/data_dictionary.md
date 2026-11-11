# 《新兴产业与非对称财富机会智能预测系统》数据字典与溯源规范

## 一、持久化存储层架构 (DuckDB + Parquet)

数据库统一存储于 `data/storage.duckdb`，底层核心物理表结构规范如下：

### 1. `raw_records` (多源原始数据归档表)
| 字段名 | 类型 | 说明 | 示例 |
| :--- | :--- | :--- | :--- |
| `record_id` | VARCHAR | 主键，全局唯一数据记录标识 | `OPENALEX_embodied AI_2023` |
| `source_name` | VARCHAR | 数据来源名称 | `OpenAlex`, `arXiv`, `World Bank` |
| `source_url` | VARCHAR | 原始采集出处链接 | `https://api.openalex.org/works?...` |
| `as_of_date` | DATE | 统计所属时间 (数据反映的真实业务时点) | `2023-12-31` |
| `published_at`| TIMESTAMP | 首次公开发布时间 (Point-in-Time 物理切片基准) | `2023-12-31 23:59:59` |
| `ingested_at` | TIMESTAMP | 系统抓取入库时间 | `2026-11-11 08:00:00` |
| `quality_status`| VARCHAR | 数据质量状态 (`VERIFIED` / `UNVERIFIED` / `SYNTHETIC` / `MANUAL_REFERENCE` / `UNAVAILABLE`) | `MANUAL_REFERENCE` |
| `data_version`| VARCHAR | 数据规范版本 | `v1.1` |
| `raw_hash` | VARCHAR | 原始载荷 SHA-256 哈希防篡改指纹 | `e3b0c44298fc1c149afbf4c8...` |
| `payload` | JSON | 原始文本/结构化字典完整载荷 | `{"count": 1500, "citations": ...}` |

> **V1.1 数据质量状态规范说明**：
> - `VERIFIED`：真实且已经审计核实的数据（具备可追溯官方信源、合规公开时间戳且通过交叉验真）。
> - `UNVERIFIED`：真实来源声明但尚未完成人工/第三方逐项审计核实的数据（如开放API实时抓取未核验数据）。
> - `SYNTHETIC`：程序生成的模拟/合成数据（仅限测试模块使用，严禁进入生产预测）。
> - `MANUAL_REFERENCE`：系统内置或人工整理录入、待逐项核实原始公告出处的参考数据。
> - `UNAVAILABLE`：无法合法合规获取或缺失历史真实记录的数据。

---

### 2. `tech_metrics` (技术主题时序特征表)
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `topic_id` | VARCHAR | 技术主题唯一编码 (如 `TOPIC_EMBODIED_AI`) |
| `as_of_date` | DATE | 统计年度/季度截止日 |
| `published_at`| TIMESTAMP | 披露时间戳 |
| `papers_count`| BIGINT | 当期学术文献产出量 |
| `citations_count`| BIGINT | 当期高引次数 |
| `active_institutions`| INTEGER | 参与独立科研机构数量 |
| `github_stars`| BIGINT | 开源生态累计 Star 数 |
| `github_forks`| BIGINT | 开源生态 Fork 衍生数 |
| `entropy_score`| DOUBLE | 机构扩散香农熵 (0.0 ~ 1.0) |
| `novelty_score`| DOUBLE | 技术新颖度得分 (0.0 ~ 1.0) |

---

### 3. `macro_indicators` (宏观经济序列规范表)
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `indicator_code`| VARCHAR | 宏观指标编码 (`CN_PMI`, `CN_PPI_YOY`, `CN_M2_YOY`) |
| `as_of_date` | DATE | 统计所属月份/季度末 |
| `published_at`| TIMESTAMP | 官方官方发布时间戳 |
| `indicator_name`| VARCHAR | 指标中文全称 |
| `value` | DOUBLE | 指标实际数值 |
| `unit` | VARCHAR | 计量单位 (`点`, `%`) |
| `source` | VARCHAR | 数据来源机构 |

---

### 4. `company_financials` (企业历史财务数据表)
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `company_id` | VARCHAR | 企业唯一编码 (如 `300124_SZ`, `688017_SH`) |
| `as_of_date` | DATE | 报告期所属日期 (年报 12-31, 半年报 06-30) |
| `published_at`| TIMESTAMP | 财报披露法定日期 (如次年 04-26) |
| `revenue` | DOUBLE | 营业收入 (亿元) |
| `revenue_yoy` | DOUBLE | 营业收入同比增速 (%) |
| `gross_margin`| DOUBLE | 销售毛利率 (%) |
| `net_profit` | DOUBLE | 归母净利润 (亿元) |
| `ocf` | DOUBLE | 经营性现金流量净额 (亿元) |
| `rd_expense` | DOUBLE | 研发费用支出 (亿元) |
| `rd_ratio` | DOUBLE | 研发费用占营收比 (%) |
| `roic` | DOUBLE | 投入资本回报率 (%) |
| `ar_turnover_days`| DOUBLE| 应收账款周转天数 (天) |
| `debt_to_asset`| DOUBLE | 资产负债率 (%) |

---

### 5. `model_forecasts` (模型预测日志表 - 不可篡改)
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `forecast_id` | VARCHAR | 预测批次全局唯一ID |
| `target_id` | VARCHAR | 预测目标对象ID (如 `TOPIC_EMBODIED_AI`) |
| `target_level`| INTEGER | 对象层级 (1-5) |
| `model_name` | VARCHAR | 生成模型标识 (`ForesightAlpha-Synthesizer`) |
| `model_version`| VARCHAR| 模型算法版本 (`1.1.0-audit`) |
| `as_of_data_date`| DATE | 预测使用的数据截止日期 |
| `horizon_years`| INTEGER | 预测期限 (1, 3, 5 年) |
| `predicted_metric`| VARCHAR| 预测目标指标名称 |
| `point_estimate`| DOUBLE | 点估计值 (若模型未训练或数据不足，允许为 NULL，严禁使用虚假倍数填充) |
| `ci_low` | DOUBLE | 置信区间下界 (未训练为 NULL) |
| `ci_high` | DOUBLE | 置信区间上界 (未训练为 NULL) |
| `confidence_score`| DOUBLE | 贝叶斯校准置信度 (若无独立校准结果为 NULL，禁止硬编码伪置信度) |
| `prediction_status`| VARCHAR | 预测准入状态 (`PREDICTED` / `NOT_PREDICTABLE` / `MODEL_NOT_TRAINED` / `UNVERIFIED_RESEARCH`) |
| `is_verified_forecast`| BOOLEAN | 是否为经过核验的正式预测 (默认 FALSE) |
| `scenarios_json`| JSON | 悲观/基准/乐观三情景分布 |
| `drivers_json` | JSON | 特征重要性敏感度归因 |

---

### 6. `company_distress_cohort` (企业生存与真实违约事件历史队列表)
| 字段名 | 类型 | 说明 | 示例 |
| :--- | :--- | :--- | :--- |
| `company_id` | VARCHAR | 企业唯一股票代码与交易所后缀 | `300104_SZ` |
| `company_name` | VARCHAR | 企业法定简称 | `乐视网` |
| `industry_sector` | VARCHAR | 产业赛道分类 | `INTERNET_PLATFORM` |
| `duration_months` | DOUBLE | 存续月数 (上市或设立至违约/截断) | `118.0` |
| `is_distressed` | INTEGER | 是否发生终结性违约事件 (1=违约/退市/重整, 0=右删失正常存续) | `1` |
| `distress_type` | VARCHAR | 违约/存续具体类型 | `MAJOR_FINANCIAL_FRAUD_DELISTING` |
| `event_date` | DATE | 违约或司法重整受理发生日期 | `2020-05-14` |
| `published_at` | TIMESTAMP | 官方裁定或终止上市公开发布时间戳 | `2020-05-14 18:30:00` |
| `cash_burn_months` | DOUBLE | 危机发生时的现金消耗月数 (Runway) | `1.2` |
| `debt_ratio` | DOUBLE | 资产负债率 (%) | `285.6` |
| `gross_margin` | DOUBLE | 销售毛利率 (%) | `-32.4` |
| `competitor_count` | DOUBLE | 赛道竞争对手密度 | `15.0` |
| `official_announcement_id`| VARCHAR | 交易所决定书文号或法院破产裁定案号 | `深交所退市决议-[2020]078` |
| `source_url` | VARCHAR | 巨潮资讯或交易所官方公告直达链接 | `http://www.cninfo.com.cn/...` |
| `quality_status` | VARCHAR | 数据质量状态 (`MANUAL_REFERENCE` 待逐项核验) | `MANUAL_REFERENCE` |

---

### 7. `tech_training_samples` (前沿技术多期真实特征与多期前瞻标签训练表)
| 字段名 | 类型 | 说明 | 示例 |
| :--- | :--- | :--- | :--- |
| `sample_id` | VARCHAR | 样本唯一标识 | `TECH_SAMPLE_TRANSFORMER_2017` |
| `topic_id` | VARCHAR | 前沿技术主题编码 | `TOPIC_TRANSFORMER_LLM` |
| `base_date` | DATE | 特征所属基准时点 (Point-in-Time 历史截面) | `2017-12-31` |
| `published_at` | TIMESTAMP | 数据首次公开发布时间戳 | `2017-12-31 23:59:59` |
| `velocity_1y` | DOUBLE | 过去1年文献与专利复合增速 | `120.0` |
| `acceleration` | DOUBLE | 文献增长加速度 (二阶导数) | `45.0` |
| `active_institutions`| DOUBLE | 参与独立科研与工业机构数量 | `85.0` |
| `github_momentum` | DOUBLE | 开源社区关注与衍生热度 | `1200.0` |
| `macro_multiplier` | DOUBLE | 当期宏观经济环境成长阻尼乘数 | `1.15` |
| `entropy_score` | DOUBLE | 跨机构香农机构扩散熵 (0.0 ~ 1.0) | `0.52` |
| `novelty_score` | DOUBLE | 技术语义新颖度得分 (0.0 ~ 1.0) | `0.95` |
| `target_1y` | DOUBLE | 真实历史1年后实际学术与开源成果增长倍数 | `1.85` |
| `target_3y` | DOUBLE | 真实历史3年后实际学术与开源成果增长倍数 | `4.20` |
| `target_5y` | DOUBLE | 真实历史5年后实际学术与开源成果增长倍数 | `8.60` |
| `quality_status` | VARCHAR | 数据质量状态 | `MANUAL_REFERENCE` |
