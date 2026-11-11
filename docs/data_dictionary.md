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
| `quality_status`| VARCHAR | 数据质量状态 (`VALIDATED` / `PROVISIONAL`) | `VALIDATED` |
| `data_version`| VARCHAR | 数据规范版本 | `v1.0` |
| `raw_hash` | VARCHAR | 原始载荷 SHA-256 哈希防篡改指纹 | `e3b0c44298fc1c149afbf4c8...` |
| `payload` | JSON | 原始文本/结构化字典完整载荷 | `{"count": 1500, "citations": ...}` |

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
| `model_version`| VARCHAR| 模型算法版本 (`1.0.0`) |
| `as_of_data_date`| DATE | 预测使用的数据截止日期 |
| `horizon_years`| INTEGER | 预测期限 (1, 3, 5 年) |
| `predicted_metric`| VARCHAR| 预测目标指标名称 |
| `point_estimate`| DOUBLE | 点估计值 |
| `ci_low` | DOUBLE | 置信区间下界 |
| `ci_high` | DOUBLE | 置信区间上界 |
| `confidence_score`| DOUBLE | 贝叶斯校准置信度 (0.0 ~ 1.0) |
| `scenarios_json`| JSON | 悲观/基准/乐观三情景分布 |
| `drivers_json` | JSON | 特征重要性敏感度归因 |
