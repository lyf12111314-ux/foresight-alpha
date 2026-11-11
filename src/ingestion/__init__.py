"""Ingestion 模块初始化"""
from src.ingestion.base import BaseCollector
from src.ingestion.openalex import OpenAlexCollector
from src.ingestion.arxiv import ArxivCollector
from src.ingestion.github import GithubCollector
from src.ingestion.macro import MacroCollector
from src.ingestion.enterprise import EnterpriseCollector
from src.ingestion.pipeline import IngestionPipeline

__all__ = [
    "BaseCollector",
    "OpenAlexCollector",
    "ArxivCollector",
    "GithubCollector",
    "MacroCollector",
    "EnterpriseCollector",
    "IngestionPipeline"
]
