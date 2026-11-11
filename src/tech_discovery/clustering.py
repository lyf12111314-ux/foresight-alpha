"""新兴技术自动发现与动态聚类引擎 (Emerging Tech Topic Discovery & Novelty)."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from src.tech_discovery.semantic import SemanticEntityResolver


class TechClusterDiscoveryEngine:
    """新技术聚类与候选池发现器"""

    def __init__(self, min_samples: int = 2, eps: float = 0.45):
        self.min_samples = min_samples
        self.eps = eps
        self.resolver = SemanticEntityResolver()

    def calculate_shannon_entropy(self, institution_counts: List[int]) -> float:
        """计算参与研究机构的香农熵 (衡量技术扩散的多样性与去中心化程度)"""
        arr = np.array(institution_counts, dtype=float)
        tot = arr.sum()
        if tot <= 0:
            return 0.0
        p = arr / tot
        # 过滤零概率
        p = p[p > 0]
        entropy = -np.sum(p * np.log2(p))
        # 归一化到 0.0 ~ 1.0 (以 log2(N) 为基底)
        max_entropy = np.log2(len(arr)) if len(arr) > 1 else 1.0
        return float(round(entropy / max(max_entropy, 1.0), 4))

    def discover_from_corpus(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """从文献/项目语料中自动聚类并提取新主题
        
        documents: [{"title": "...", "summary": "...", "published_at": "..."}]
        """
        if len(documents) < 3:
            return []

        texts = [f"{d.get('title', '')} {d.get('summary', '')}" for d in documents]
        vec = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
        try:
            X = vec.fit_transform(texts).toarray()
        except ValueError:
            return []

        # 密度聚类发现新主题簇
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        labels = clustering.fit_predict(X)

        discovered_topics = []
        feature_names = np.array(vec.get_feature_names_out())

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue  # 忽略孤立噪声

            indices = np.where(labels == cluster_id)[0]
            cluster_center = X[indices].mean(axis=0)
            top_term_indices = cluster_center.argsort()[-4:][::-1]
            top_keywords = feature_names[top_term_indices].tolist()
            topic_title = " ".join([k.title() for k in top_keywords[:3]])

            # 语义对齐检查
            mapped_id, canonical_name, score = self.resolver.resolve_topic(topic_title)

            # 计算新颖度得分 (基于簇内紧密度与历史已有概念偏离度)
            novelty = float(round(1.0 - score * 0.7, 3))

            discovered_topics.append({
                "cluster_id": int(cluster_id),
                "proposed_title": topic_title,
                "keywords": top_keywords,
                "sample_count": len(indices),
                "mapped_topic_id": mapped_id,
                "canonical_name": canonical_name,
                "similarity_score": score,
                "is_novel_topic": score < 0.65,
                "novelty_score": novelty
            })

        return discovered_topics
