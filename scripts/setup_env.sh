#!/usr/bin/env bash
# 环境依赖自动安装与初始化脚本
set -e

echo "=== [Foresight Alpha] 正在初始化 Python 虚拟环境与依赖 ==="

if ! command -v uv &> /dev/null; then
    echo "未检测到 uv，建议安装 uv 或使用 python3 -m venv 创建环境。"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "检测到 uv，使用 uv 极速配置环境..."
    uv venv --python 3.11 .venv
    uv pip install -r requirements.txt --python .venv/bin/python
fi

echo "=== 依赖安装完成！==="
