"""合法可持有资产映射与流动性准入引擎 (Legal Asset Mapping & Liquidity Filter)."""

from typing import Dict, Any, List, Optional
from src.core.types import InvestableAssetEntity


class LegalAssetMapper:
    """合法参与方式与资产映射器"""

    def __init__(self):
        # 预设合法可参与标的字典
        self.asset_registry: Dict[str, List[Dict[str, Any]]] = {
            "TOPIC_EMBODIED_AI": [
                {
                    "asset_id": "ASSET_300124_SZ",
                    "name": "汇川技术 (300124.SZ)",
                    "asset_type": "A_SHARE",
                    "is_legally_accessible": True,
                    "trading_liquidity_daily_avg": 85000.0,  # 万元
                    "pe_ratio": 32.5,
                    "ps_ratio": 5.8,
                    "regulatory_notes": "A股创业板上市标的，合规股票账户可直接参与，流动性充裕。"
                },
                {
                    "asset_id": "ASSET_688017_SH",
                    "name": "绿的谐波 (688017.SH)",
                    "asset_type": "A_SHARE",
                    "is_legally_accessible": True,
                    "trading_liquidity_daily_avg": 35000.0,
                    "pe_ratio": 85.0,
                    "ps_ratio": 18.2,
                    "regulatory_notes": "科创板上市标的，需满足科创板适当性投资者门槛 (50万资产+2年经验)。"
                },
                {
                    "asset_id": "ASSET_ROBOT_ETF_159770",
                    "name": "机器人产业ETF (159770.SZ)",
                    "asset_type": "ETF",
                    "is_legally_accessible": True,
                    "trading_liquidity_daily_avg": 42000.0,
                    "pe_ratio": 38.0,
                    "ps_ratio": 3.5,
                    "regulatory_notes": "场内公募ETF，门槛低(一手起购)，天然分散单一公司失败破产风险。"
                }
            ],
            "TOPIC_SOLID_STATE_BATT": [
                {
                    "asset_id": "ASSET_300750_SZ",
                    "name": "宁德时代 (300750.SZ)",
                    "asset_type": "A_SHARE",
                    "is_legally_accessible": True,
                    "trading_liquidity_daily_avg": 350000.0,
                    "pe_ratio": 19.5,
                    "ps_ratio": 2.8,
                    "regulatory_notes": "全球电池龙头，千亿市值，机构持仓与流动性极好。"
                }
            ],
            "TOPIC_AGENTIC_AI": [
                {
                    "asset_id": "ASSET_688256_SH",
                    "name": "寒武纪 (688256.SH)",
                    "asset_type": "A_SHARE",
                    "is_legally_accessible": True,
                    "trading_liquidity_daily_avg": 120000.0,
                    "pe_ratio": None,
                    "ps_ratio": 45.0,
                    "regulatory_notes": "算力芯片龙头，处于研发转商业化阶段，波动较大。"
                }
            ],
            "TOPIC_EVTOL_AERO": [
                {
                    "asset_id": "ASSET_EH_US",
                    "name": "亿航智能 (EH.US)",
                    "asset_type": "US_SHARE",
                    "is_legally_accessible": True,
                    "trading_liquidity_daily_avg": 15000.0,
                    "pe_ratio": None,
                    "ps_ratio": 12.0,
                    "regulatory_notes": "美股中概ADR，需合规境外账户参与，获得中国民航局TC适航认证。"
                }
            ]
        }

    def map_topic_to_assets(self, topic_id: str) -> List[Dict[str, Any]]:
        """将技术主题映射到境内外合法可投资标的"""
        assets = self.asset_registry.get(topic_id, [])
        if not assets:
            return [{
                "asset_id": "NO_PUBLIC_ASSET",
                "name": "当前尚无公开二级市场合规标的",
                "asset_type": "RESEARCH_ONLY",
                "is_legally_accessible": False,
                "trading_liquidity_daily_avg": 0.0,
                "pe_ratio": None,
                "ps_ratio": None,
                "regulatory_notes": "当前技术处于实验室或未上市天使创投阶段，仅输出产业研究结论，不生成虚假收益预测。"
            }]
        return assets
