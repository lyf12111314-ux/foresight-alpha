"""配置加载器 (Config Loader)."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def get_project_root() -> Path:
    """获取项目根路径"""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml_config(relative_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    root = get_project_root()
    config_file = root / relative_path
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_system_config() -> Dict[str, Any]:
    """获取系统主配置"""
    return load_yaml_config("configs/system_config.yaml")


def get_taxonomy_config() -> Dict[str, Any]:
    """获取五级实体与产业字典配置"""
    return load_yaml_config("configs/taxonomy.yaml")
