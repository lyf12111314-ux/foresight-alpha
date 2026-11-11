"""技术发现与拐点检测测试 (Tech Discovery & Inflection Tests)."""

import pytest
import pandas as pd
from datetime import date
from src.tech_discovery.semantic import SemanticEntityResolver
from src.tech_discovery.clustering import TechClusterDiscoveryEngine
from src.tech_discovery.inflection import InflectionDetector


def test_semantic_entity_resolver():
    resolver = SemanticEntityResolver()
    # 测试同义词对齐
    tid, name, score = resolver.resolve_topic("Embodied AI Foundation Model")
    assert tid == "TOPIC_EMBODIED_AI"
    assert score >= 0.70

    tid2, name2, score2 = resolver.resolve_topic("Solid-state battery electrolyte")
    assert tid2 == "TOPIC_SOLID_STATE_BATT"
    assert score2 >= 0.70


def test_tech_cluster_discovery():
    engine = TechClusterDiscoveryEngine()
    docs = [
        {"title": "Embodied agent in humanoid robotics", "summary": "Physical AI and robot manipulation"},
        {"title": "Humanoid robotics with embodied AI", "summary": "Bipedal gait and manipulation agent"},
        {"title": "Novel embodied humanoid manipulation", "summary": "Robotics agent perception"},
        {"title": "Quantum key distribution protocol", "summary": "Photon quantum cryptography channel"}
    ]
    clusters = engine.discover_from_corpus(docs)
    assert isinstance(clusters, list)


def test_inflection_acceleration_detector():
    detector = InflectionDetector()
    df = pd.DataFrame({
        "as_of_date": [date(2020, 12, 31), date(2021, 12, 31), date(2022, 12, 31), date(2023, 12, 31)],
        "papers_count": [100, 250, 600, 1500],
        "citations_count": [50, 200, 800, 3000],
        "active_institutions": [20, 50, 150, 400]
    })
    res = detector.analyze_tech_acceleration(df)
    assert res["status"] == "VALIDATED"
    assert res["latest_velocity_yoy"] > 0
    assert res["latest_acceleration"] > 0
