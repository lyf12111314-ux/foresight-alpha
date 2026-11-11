"""语义分析与同义技术实体对齐引擎 (Semantic Embedding & Topic Resolution)."""

import re
from typing import List, Dict, Any, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticEntityResolver:
    """技术同义词与多名归一对齐器"""

    def __init__(self, similarity_threshold: float = 0.65):
        self.similarity_threshold = similarity_threshold
        # 内置核心技术标准别名词典
        self.canonical_map: Dict[str, Dict[str, Any]] = {
            "TOPIC_EMBODIED_AI": {
                "canonical_name": "具身智能与人形机器人",
                "aliases": ["embodied ai", "embodied intelligence", "humanoid robot", "具身智能", "物理AI", "physical ai", "robotics foundation model"],
                "domain": "ROBOTICS_EMBODIED"
            },
            "TOPIC_SOLID_STATE_BATT": {
                "canonical_name": "全固态电池与先进电解质",
                "aliases": ["solid state battery", "solid-state electrolyte", "全固态电池", "硫化物固态", "氧化物固态电解质", "all-solid-state lithium"],
                "domain": "ENERGY_STORAGE"
            },
            "TOPIC_NEUROMORPHIC": {
                "canonical_name": "神经拟态计算与类脑芯片",
                "aliases": ["neuromorphic computing", "spiking neural network", "类脑计算", "脉冲神经网络", "brain-inspired chip"],
                "domain": "AI_INFRA"
            },
            "TOPIC_AGENTIC_AI": {
                "canonical_name": "AI智能体与自主交易基础设施",
                "aliases": ["agentic ai", "autonomous agents", "ai agent", "智能体交易协议", "multi-agent coordination", "llm agent workflow"],
                "domain": "AI_INFRA"
            },
            "TOPIC_EVTOL_AERO": {
                "canonical_name": "低空经济与eVTOL飞行器",
                "aliases": ["evtol", "electric vertical takeoff", "低空经济", "城市空中交通", "uam", "urban air mobility"],
                "domain": "LOW_ALTITUDE_AERO"
            },
            "TOPIC_SYNTHETIC_BIO": {
                "canonical_name": "合成生物学与生物智造",
                "aliases": ["synthetic biology", "cell factory", "合成生物", "底盘细胞", "directed evolution", "bio-manufacturing"],
                "domain": "BIOTECH_SYNTHETIC"
            }
        }

    def normalize_text(self, text: str) -> str:
        """文本清洗与低频标点规范化"""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
        return " ".join(text.split())

    def resolve_topic(self, term_or_phrase: str) -> Tuple[str, str, float]:
        """将任意技术输入归一到已有标准主题，或标记为新增独立主题
        
        返回: (topic_id, canonical_name, match_confidence)
        """
        norm_input = self.normalize_text(term_or_phrase)

        # 1. 严格子串/别名全量匹配
        for topic_id, info in self.canonical_map.items():
            for alias in info["aliases"]:
                if alias.lower() in norm_input or norm_input in alias.lower():
                    return topic_id, info["canonical_name"], 1.0

        # 2. TF-IDF 模糊语义相似度推断
        corpus = [norm_input]
        labels = []
        for topic_id, info in self.canonical_map.items():
            corpus.append(" ".join(info["aliases"]))
            labels.append(topic_id)

        try:
            vec = TfidfVectorizer().fit_transform(corpus)
            sims = cosine_similarity(vec[0:1], vec[1:])[0]
            best_idx = int(sims.argmax())
            best_score = float(sims[best_idx])

            if best_score >= self.similarity_threshold:
                matched_id = labels[best_idx]
                return matched_id, self.canonical_map[matched_id]["canonical_name"], round(best_score, 3)
        except Exception:
            pass

        # 3. 确属未见过的全新技术主题
        new_topic_id = f"TOPIC_DISCOVERED_{abs(hash(term_or_phrase)) % 10000:04d}"
        return new_topic_id, term_or_phrase.strip().title(), 0.50
