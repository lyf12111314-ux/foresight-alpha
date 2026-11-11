#!/usr/bin/env bash
# 一键启动 Streamlit 交互式金融科技研究门户
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"
export PYTHONPATH="$DIR:${PYTHONPATH}"

echo "=== [Foresight Alpha] 正在启动 Streamlit 交互式研究平台 ==="
echo "访问地址: http://localhost:8501"

exec "$DIR/.venv/bin/streamlit" run "$DIR/ui/app.py" --server.port=8501 --server.headless=true --browser.gatherUsageStats=false
