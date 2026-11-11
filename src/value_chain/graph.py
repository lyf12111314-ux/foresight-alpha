"""产业链拓扑图谱与节点依赖关系 (Value Chain NetworkX Graph & Flow)."""

import networkx as nx
from typing import Dict, Any, List, Optional
from src.core.config import get_taxonomy_config


class ValueChainGraph:
    """产业链有向拓扑传导图谱"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_from_taxonomy()

    def _build_from_taxonomy(self):
        """根据标准产业字典构建产业链上下游拓扑图"""
        taxonomy = get_taxonomy_config()
        domains = taxonomy.get("core_domains", [])

        for d in domains:
            domain_id = d["domain_id"]
            domain_name = d["name"]
            nodes = d.get("value_chain_nodes", [])

            # 添加产业链环节节点
            upstream_nodes = []
            midstream_nodes = []
            downstream_nodes = []

            for node in nodes:
                nid = node["node_id"]
                tier = node["tier"]
                self.graph.add_node(
                    nid,
                    name=node["name"],
                    domain_id=domain_id,
                    domain_name=domain_name,
                    tier=tier,
                    gross_margin_range=node.get("gross_margin_range", [20, 40]),
                    capex_intensity=node.get("capex_intensity", "MEDIUM")
                )
                if tier == "UPSTREAM":
                    upstream_nodes.append(nid)
                elif tier == "MIDSTREAM":
                    midstream_nodes.append(nid)
                elif tier == "DOWNSTREAM":
                    downstream_nodes.append(nid)

            # 连接传导有向边 (上游 -> 中游 -> 下游)
            for up in upstream_nodes:
                for mid in midstream_nodes:
                    self.graph.add_edge(up, mid, relation="SUPPLIES_TO")

            for mid in midstream_nodes:
                for down in downstream_nodes:
                    self.graph.add_edge(mid, down, relation="INTEGRATED_INTO")

    def get_domain_nodes(self, domain_id: str) -> List[Dict[str, Any]]:
        """获取特定产业领域的产业链节点明细"""
        res = []
        for n, attr in self.graph.nodes(data=True):
            if attr.get("domain_id") == domain_id:
                res.append({
                    "node_id": n,
                    "name": attr.get("name"),
                    "tier": attr.get("tier"),
                    "gross_margin_range": attr.get("gross_margin_range"),
                    "capex_intensity": attr.get("capex_intensity")
                })
        return res

    def get_all_edges(self) -> List[Dict[str, Any]]:
        """获取全量拓扑边"""
        return [{"source": u, "target": v, "relation": d.get("relation", "")} for u, v, d in self.graph.edges(data=True)]
