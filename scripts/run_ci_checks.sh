#!/usr/bin/env bash
# =============================================================================
# Foresight Alpha — 本地与持续集成统一质量门禁脚本 (CI Quality Gate Runner)
# 严格遵循 P0-03 规范：本地与 CI 使用完全一致的强阻断检查命令。
# =============================================================================
set -euo pipefail

echo "========================================================================"
echo "🛡️  [Foresight Alpha] 正在执行统一自动化质量门禁检查..."
echo "========================================================================"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="python"
FLAKE8_BIN="flake8"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi
if [ -f ".venv/bin/flake8" ]; then
    FLAKE8_BIN=".venv/bin/flake8"
fi

echo -e "\n[Gate 1/5] 🐍 Python 语法完整性与编译检查 (compileall)..."
"$PYTHON_BIN" -m compileall -q src scripts tests
echo " -> PASSED: 所有模块语法编译无异常。"

echo -e "\n[Gate 2/5] 🔍 代码静态质量检查 (flake8 强阻断模式)..."
if ! command -v "$FLAKE8_BIN" &> /dev/null && [ ! -x "$FLAKE8_BIN" ]; then
    echo "❌ ERROR: 未检测到 flake8 静态检查工具！请先执行: pip install -r requirements.txt" >&2
    exit 1
fi
"$FLAKE8_BIN" src scripts tests --count --select=E9,F63,F7,F82 --show-source --statistics
echo " -> PASSED: flake8 致命语法与未定义名称检查通过。"

echo -e "\n[Gate 3/5] 🗄️  空 DuckDB 数据库初始化与原生系统表检查..."
"$PYTHON_BIN" -c "
import tempfile, os
from src.core.database import DatabaseManager

with tempfile.TemporaryDirectory() as tmpdir:
    test_db = os.path.join(tmpdir, 'clean_init.duckdb')
    mgr = DatabaseManager(db_path=test_db)
    if hasattr(mgr, 'initialize_database'):
        mgr.initialize_database()
    else:
        mgr.init_schema()
    conn = mgr.get_connection()
    try:
        tables_df = conn.execute(
            \"SELECT table_name FROM information_schema.tables WHERE table_schema = 'main';\"
        ).df()
        tables = set(tables_df['table_name'].tolist())
        assert 'tech_training_samples' in tables, 'Missing tech_training_samples table'
        assert 'model_forecasts' in tables, 'Missing model_forecasts table'
        assert 'raw_records' in tables, 'Missing raw_records table'
    finally:
        conn.close()
print(' -> PASSED: 空 DuckDB 库隔离环境顺利完成 Schema 初始化及系统表校验。')
"

echo -e "\n[Gate 4/5] 🚀 无外部网络依赖的离线端到端集成测试 (test_offline_e2e.py)..."
"$PYTHON_BIN" -m pytest tests/test_offline_e2e.py -v
echo " -> PASSED: 离线端到端全链路集成测试通过。"

echo -e "\n[Gate 5/5] 🧪 全量自动化测试回归套件 (pytest)..."
"$PYTHON_BIN" -m pytest -v

echo -e "\n✅ [Quality Gate PASSED] 全套统一质量门禁检查全部通过！"
