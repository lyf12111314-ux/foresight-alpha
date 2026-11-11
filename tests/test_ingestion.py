"""数据采集与溯源单元测试 (Data Ingestion & Provenance Tests)."""

import pytest
from datetime import date, datetime
from src.core.types import DataQualityStatus
from src.ingestion.base import BaseCollector
from src.ingestion.macro import MacroCollector
from src.ingestion.enterprise import EnterpriseCollector


class DummyCollector(BaseCollector):
    def collect(self, **kwargs):
        return {"status": "ok"}


def test_base_collector_provenance_and_hash():
    collector = DummyCollector(source_name="TestCollector", base_url="https://example.com")
    raw_str = '{"metric": "gdp", "value": 100.5}'
    hash_val = collector.calculate_hash(raw_str)
    assert len(hash_val) == 64  # SHA-256

    prov = collector.build_provenance(
        record_id="REC_001",
        as_of_date=date(2023, 12, 31),
        published_at=datetime(2024, 1, 15, 10, 0, 0),
        quality_status=DataQualityStatus.VALIDATED,
        raw_hash=hash_val
    )
    assert prov.record_id == "REC_001"
    assert prov.as_of_date == date(2023, 12, 31)
    assert prov.raw_hash == hash_val


def test_macro_collector_official_series():
    collector = MacroCollector()
    series = collector.get_official_macro_series()
    assert len(series) > 0
    first = series[0]
    assert "indicator_code" in first
    assert "value" in first
    assert "provenance" in first
    # 严格数据质量审计：未逐项核实前，内置宏观参考数据标记为 MANUAL_REFERENCE，绝不可标记为 VERIFIED
    assert first["provenance"]["quality_status"] == DataQualityStatus.MANUAL_REFERENCE
    assert first["provenance"]["quality_status"] != DataQualityStatus.VERIFIED


def test_enterprise_financials_collection():
    collector = EnterpriseCollector()
    fins = collector.get_core_enterprise_financials()
    assert len(fins) >= 20
    # 检查字段完整性
    for item in fins[:5]:
        assert "revenue" in item
        assert "gross_margin" in item
        assert "roic" in item
        assert item["revenue"] > 0 or item["revenue"] == 0 or item["revenue"] is not None
