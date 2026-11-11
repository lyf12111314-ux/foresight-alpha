"""产业链利润沉淀与壁垒分析引擎 (Value Chain Profit Pool & Margin Dynamics)."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.value_chain.graph import ValueChainGraph


class ProfitPoolAnalyzer:
    """利润池与产业剩余沉淀分析器"""

    def __init__(self):
        self.vc_graph = ValueChainGraph()

    def analyze_domain_profit_pool(self, domain_id: str) -> Dict[str, Any]:
        """深入分析某产业链各层级的利润沉淀特征与定价壁垒"""
        nodes = self.vc_graph.get_domain_nodes(domain_id)
        if not nodes:
            return {"error": f"Domain {domain_id} not found in taxonomy."}

        tiers_summary = {"UPSTREAM": [], "MIDSTREAM": [], "DOWNSTREAM": []}
        for node in nodes:
            tier = node["tier"]
            gm_low, gm_high = node["gross_margin_range"]
            tiers_summary[tier].append({
                "node_id": node["node_id"],
                "name": node["name"],
                "avg_margin": (gm_low + gm_high) / 2.0,
                "capex": node["capex_intensity"]
            })

        tier_metrics = {}
        for tier, items in tiers_summary.items():
            if items:
                avg_m = float(np.mean([it["avg_margin"] for it in items]))
                tier_metrics[tier] = {
                    "average_gross_margin": round(avg_m, 1),
                    "node_count": len(items),
                    "items": items
                }

        # 判断高利润与定价权卡位沉淀在哪个环节 (通常为高技术壁垒上游核心零部件或平台级下游)
        highest_margin_tier = max(tier_metrics.keys(), key=lambda t: tier_metrics[t]["average_gross_margin"])

        if highest_margin_tier == "UPSTREAM":
            bottleneck_desc = "利润主要沉淀在上游核心基础材料、高精密零部件与底层芯片环节，中下游属于重资产或组装代工，面临议价挤压。"
        elif highest_margin_tier == "DOWNSTREAM":
            bottleneck_desc = "利润主要沉淀在下游高黏性客户场景与数据运营平台，上游中游高度标准化竞争。"
        else:
            bottleneck_desc = "利润主要沉淀在中游核心设备与模块系统集成商，具备规模护城河。"

        return {
            "domain_id": domain_id,
            "tier_metrics": tier_metrics,
            "highest_margin_tier": highest_margin_tier,
            "profit_bottleneck_diagnosis": bottleneck_desc
        }
