"""核心数据类型与契约定义 (Data Contracts & Schemas).

严格遵循 Point-in-Time 原则与五级研究对象架构。
"""

from datetime import datetime, date
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ObjectLevel(int, Enum):
    """五级研究对象层级"""
    TECH_TOPIC = 1           # 技术主题
    EMERGING_INDUSTRY = 2    # 新兴产业与细分赛道
    VALUE_CHAIN_NODE = 3     # 产业链环节
    COMPANY_ENTITY = 4       # 具体企业与项目
    INVESTABLE_ASSET = 5     # 可合法持有资产/经营权益


class DataQualityStatus(str, Enum):
    """数据质量状态 (符合 V1.1 真实性审计标准)"""
    VERIFIED = "VERIFIED"                   # 真实且已经核实的数据
    UNVERIFIED = "UNVERIFIED"               # 真实来源声明但尚未核实的数据
    SYNTHETIC = "SYNTHETIC"                 # 程序生成的模拟数据
    MANUAL_REFERENCE = "MANUAL_REFERENCE"   # 人工录入、待逐项核实的数据
    UNAVAILABLE = "UNAVAILABLE"             # 无法获取的数据

    # 兼容历史别名
    VALIDATED = "VERIFIED"                  # 兼容原有代码中的 VALIDATED 引用
    PROVISIONAL = "UNVERIFIED"              # 兼容原有代码中的 PROVISIONAL 引用
    INSUFFICIENT_HISTORY = "UNAVAILABLE"    # 兼容原有代码中的 INSUFFICIENT 引用


class PredictionStatus(str, Enum):
    """预测准入状态"""
    PREDICTED = "PREDICTED"                         # 经过验证模型生成的正式预测
    NOT_PREDICTABLE = "NOT_PREDICTABLE"             # 数据不足，标记为不可预测
    MODEL_NOT_TRAINED = "MODEL_NOT_TRAINED"         # 模型尚未训练，禁止输出正式预测
    UNVERIFIED_RESEARCH = "UNVERIFIED_RESEARCH"     # 仅供研究参考的未验证估计
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"         # 历史数据不足以支撑推演


class ModelTrainingStatus(str, Enum):
    """模型训练状态"""
    TRAINED = "TRAINED"                             # 已使用真实验证数据完成训练
    MODEL_NOT_TRAINED = "MODEL_NOT_TRAINED"         # 模型未训练
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"         # 真实样本不足无法训练


class MacroRegime(str, Enum):
    """宏观周期状态"""
    EXPANSION = "EXPANSION"       # 繁荣扩张 (高增长/温和通胀)
    SLOWDOWN = "SLOWDOWN"         # 过热放缓 (增长见顶/高通胀/紧缩)
    CONTRACTION = "CONTRACTION"   # 紧缩衰退 (负增长/去杠杆)
    RECOVERY = "RECOVERY"         # 筑底复苏 (流动性宽松/政策发力)


class ProvenanceMetadata(BaseModel):
    """数据溯源元数据 (Point-in-Time 审计追踪)"""
    record_id: str
    source_name: str
    source_url: Optional[str] = None
    as_of_date: date                         # 统计所属日期
    published_at: datetime                   # 首次公开发布时间戳 (禁止未来信息泄漏的关键时标)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)  # 采集入库时间戳
    quality_status: DataQualityStatus = DataQualityStatus.VALIDATED
    data_version: str = "v1.0"
    raw_hash: Optional[str] = None


class TechTopicEntity(BaseModel):
    """Level 1: 技术主题实体"""
    topic_id: str
    canonical_name: str
    aliases: List[str] = Field(default_factory=list)
    domain_category: str
    first_detected_date: date
    summary: str
    is_active: bool = True


class IndustrySectorEntity(BaseModel):
    """Level 2: 新兴产业及细分赛道实体"""
    sector_id: str
    name: str
    parent_topic_ids: List[str]
    market_capacity_potential_est: float     # 潜在市场容量 (亿元)
    maturity_stage: str                      # 萌芽期 / 早期扩散 / 快速成长 / 成熟期
    regulatory_stance: str                   # 鼓励 / 审慎 / 监管规范


class ValueChainNodeEntity(BaseModel):
    """Level 3: 产业链环节实体"""
    node_id: str
    name: str
    sector_id: str
    tier: str                                # UPSTREAM / MIDSTREAM / DOWNSTREAM
    gross_margin_avg: float                  # 平均毛利率 (%)
    capex_intensity: str                     # HIGH / MEDIUM / LOW
    pricing_power: str                       # STRONG / MEDIUM / WEAK
    entry_barriers: List[str] = Field(default_factory=list)


class CompanyEntity(BaseModel):
    """Level 4: 具体企业实体"""
    company_id: str
    name: str
    ticker: Optional[str] = None             # 证券代码 (若有)
    exchange: Optional[str] = None           # 交易所
    node_ids: List[str] = Field(default_factory=list)
    founded_year: int
    revenue_ttm: float                       # 最近TTM营收 (亿元)
    revenue_growth_yoy: float                # 营收同比增速 (%)
    gross_margin: float                      # 毛利率 (%)
    operating_cashflow: float                # 经营性现金流 (亿元)
    rd_ratio: float                          # 研发支出占营收比 (%)
    roic: float                              # 资本回报率 (%)
    customer_concentration: float            # 前五大客户占比 (%)
    is_operating: bool = True


class InvestableAssetEntity(BaseModel):
    """Level 5: 可合法持有资产/经营权益"""
    asset_id: str
    company_id: Optional[str] = None
    name: str
    asset_type: str                          # A_SHARE / HK_SHARE / US_SHARE / ETF / INDUSTRIAL_REIT / LICENSED_RIGHT
    is_legally_accessible: bool = True       # 是否合规合法可投资
    trading_liquidity_daily_avg: float       # 日均流动性 (万元)
    pe_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    regulatory_notes: str = ""


class ForecastRecord(BaseModel):
    """模型预测记录 (严格记录不可篡改)"""
    forecast_id: str
    target_id: str                               # 预测目标对象ID
    target_level: ObjectLevel
    model_name: str
    model_version: str
    as_of_data_date: date                        # 数据截止日期 (Point-in-Time)
    prediction_created_at: datetime = Field(default_factory=datetime.utcnow)
    horizon_years: int                           # 预测期限 (1, 3, 5 年)
    predicted_metric: str                        # 如 adoption_rate, revenue_growth, cagr_multiple
    point_estimate: Optional[float] = None       # 点估计值 (未训练或不可预测时为 None)
    confidence_interval_low: Optional[float] = None
    confidence_interval_high: Optional[float] = None
    confidence_score: Optional[float] = None     # 经过校准的置信度/概率 (未校准时为 None)
    is_calibrated: bool = False                  # 是否通过独立经验校准
    is_verified_forecast: bool = False           # 是否为经过真实验证的正式预测
    prediction_status: PredictionStatus = PredictionStatus.UNVERIFIED_RESEARCH
    scenarios: Dict[str, float] = Field(default_factory=dict)  # {"bear": x, "base": y, "bull": z}
    driver_features: Dict[str, float] = Field(default_factory=dict)


class AlertRecord(BaseModel):
    """预警记录"""
    alert_id: str
    target_id: str
    target_level: ObjectLevel
    severity: str                            # INFO / WARNING / CRITICAL
    rule_type: str                           # ACCELERATION_SPIKE / MARGIN_COLLAPSE / CASHFLOW_DIVERGENCE
    title: str
    description: str
    triggered_at: datetime
    data_evidence: Dict[str, Any]
    is_active: bool = True
    invalidation_notes: Optional[str] = None


class BacktestMetricRecord(BaseModel):
    """回测指标记录"""
    test_run_id: str
    backtest_cutoff_date: date
    evaluation_date: date
    target_horizon_years: int
    precision_at_k: Dict[int, float]         # {3: 0.67, 5: 0.60}
    recall_at_k: Dict[int, float]
    pr_auc: float
    brier_score: float                       # 概率校准误差 (越低越好)
    excess_hit_rate_vs_random: float         # 相对随机基准的超额命中率
    excess_hit_rate_vs_momentum: float       # 相对简单动量基准的超额命中率
