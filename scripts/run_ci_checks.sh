#!/usr/bin/env bash
# =============================================================================
# Foresight Alpha — 本地与持续集成质量门禁脚本 (CI Quality Gate Runner)
# =============================================================================
set -e

echo "========================================================================"
echo "🛡️  [Foresight Alpha] 正在执行全套自动化质量门禁检查..."
echo "========================================================================"

# 获取脚本所在目录及项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="python"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

echo -e "\n[Gate 1/5] 🐍 Python 语法完整性与编译检查 (compileall)..."
$PYTHON_BIN -m compileall -q src scripts tests
echo " -> PASSED: 所有模块语法编译无异常。"

echo -e "\n[Gate 2/5] 🔍 代码静态质量与 AST 语法解析检查..."
if command -v flake8 &> /dev/null; then
    flake8 src scripts tests --count --select=E9,F63,F7,F82 --show-source --statistics
    echo " -> PASSED: flake8 致命语法与未定义名称检查通过。"
else
    # 环境未全局安装 flake8 时，使用内置 AST 解析器进行致命语法静态审查
    $PYTHON_BIN -c "
import ast, glob, sys
errors = 0
for pattern in ['src/**/*.py', 'scripts/**/*.py', 'tests/**/*.py']:
    for f in glob.glob(pattern, recursive=True):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                ast.parse(fp.read(), filename=f)
        except Exception as err:
            print(f'Syntax/AST Error in {f}: {err}', file=sys.stderr)
            errors += 1
if errors > 0:
    sys.exit(1)
print(f' -> PASSED: 内置 AST 静态审查通过，未发现语法致命错误。')
"
fi

echo -e "\n[Gate 3/5] 🗄️  空数据库初始化与隔离性测试..."
$PYTHON_BIN -c "
import tempfile, os, sys
from src.core.database import DatabaseManager

with tempfile.TemporaryDirectory() as tmpdir:
    test_db = os.path.join(tmpdir, 'clean_init.db')
    mgr = DatabaseManager(db_path=test_db)
    if hasattr(mgr, 'initialize_database'):
        mgr.initialize_database()
    else:
        mgr.init_schema()
    with mgr.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table';\")
        tables = [r[0] for r in cursor.fetchall()]
        assert 'tech_training_samples' in tables, 'Missing tech_training_samples table'
        assert 'model_forecasts' in tables, 'Missing model_forecasts table'
print(' -> PASSED: 空库隔离环境顺利完成 Schema 初始化。')
"

echo -e "\n[Gate 4/5] 🚀 端到端无网络集成验证与一键示范脚本 (run_pipeline.py)..."
$PYTHON_BIN scripts/run_pipeline.py > /dev/null
echo " -> PASSED: scripts/run_pipeline.py 一键入口顺利贯通全流程并输出研报。"

echo -e "\n[Gate 5/5] 🧪 全量自动化测试回归套件 (pytest)..."
set +e
$PYTHON_BIN -m pytest -v
PYTEST_EXIT_CODE=$?
set -e

if [ $PYTEST_EXIT_CODE -ne 0 ]; then
    echo -e "\n❌ [Quality Gate FAILED] pytest 回归测试套件存在失败用例 (Exit code: $PYTEST_EXIT_CODE)！"
    echo "⚠️  必须遵循 AGENTS.md 规范：测试失败必须优先解决，严禁带病合并代码。"
    exit $PYTEST_EXIT_CODE
else
    echo -e "\n✅ [Quality Gate PASSED] 全套质量门禁检查全部通过！"
fi
