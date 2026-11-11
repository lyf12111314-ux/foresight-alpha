"""端到端完整运行示范脚本 (End-to-End Pipeline Execution Demo).

运行全流程：
1. 多源真实数据采集与溯源归档
2. 宏观周期与状态转移研判
3. 技术爆发拐点检测与扩散S曲线拟合
4. 产业链利润池与企业商业化健康度审计
5. 生存分析与非对称蒙特卡洛模拟
6. 构建非对称机会候选池并进行不可篡改持久化
7. 运行严格 Point-in-Time 历史滚动回测并输出 Precision@K
8. 导出全套 14 章节智能研报
"""

import sys
import os
import argparse
from datetime import date

# 确保项目根路径位于 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.pipeline import IngestionPipeline
from src.macro.hmm_regime import MacroHMMRegimeModel
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.monitoring.monitor import SystemMonitor
from src.backtest.engine import PointInTimeBacktestEngine
from src.reporting.dossier import ResearchDossierGenerator
from src.core.database import db_manager


def run_full_pipeline_demo(force_refresh: bool = False):
    print("=" * 80)
    print("🚀 [Foresight Alpha] 新兴产业与非对称财富机会智能预测系统 - 端到端运行")
    print("=" * 80)

    # 1. 数据采集与清洗
    cnt_df = db_manager.query_df("SELECT COUNT(*) as c FROM raw_records")
    existing_cnt = cnt_df.iloc[0]["c"] if not cnt_df.empty else 0

    if force_refresh or existing_cnt == 0:
        print("\n[Step 1] 执行多源数据采集与 Point-in-Time 溯源归档 (全量网络刷新)...")
        pipeline = IngestionPipeline()
        stats = pipeline.run_all(verbose=True)
        print(f" -> 采集入库统计: {stats}")
    else:
        print(f"\n[Step 1] 数据持久层已具备 {existing_cnt} 条 Point-in-Time 审计记录，跳过重复全量网络拉取 (如需强制刷新请传 --refresh)")

    # 2. 宏观状态研判
    print("\n[Step 2] 宏观经济长中短多周期研判 (Gaussian HMM)...")
    macro_df = db_manager.query_df("SELECT * FROM macro_indicators ORDER BY as_of_date ASC")
    macro_model = MacroHMMRegimeModel()
    macro_status = macro_model.get_latest_regime(macro_df)
    print(f" -> 当前宏观状态: {macro_status['regime']} (增长乘数: {macro_status['macro_growth_multiplier']}x)")
    print(f" -> 宏观启示: {macro_status['macro_implication']}")

    # 3. 机会候选池生成与多模型融合
    print("\n[Step 3] 融合多模型构建非对称机会候选池...")
    pool_mgr = OpportunityPoolManager()
    pool = pool_mgr.build_opportunity_pool()
    print(f" -> 成功生成 {len(pool)} 个候选技术机会画像")
    print("-" * 75)
    print(f"{'排名':<4} {'技术主题':<22} {'综合评分':<10} {'置信度':<8} {'3年增长':<8} {'评级'}")
    print("-" * 75)
    for rank, c in enumerate(pool, start=1):
        print(f"{rank:<4} {c['topic_name']:<22} {c['composite_score']:<10.1f} {int(c['calibrated_confidence']*100)}%{'':<4} {c['growth_3y_multiple']}x{'':<5} {c['rating']}")
    print("-" * 75)

    # 4. 商业化红旗与技术爆发监控巡检
    print("\n[Step 4] 自动化巡检与红旗告警检测...")
    monitor = SystemMonitor()
    alerts = monitor.run_daily_scan()
    print(f" -> 巡检触发 {len(alerts)} 条预警记录:")
    for a in alerts:
        print(f"    * [{a['severity']}] {a['title']}")

    # 5. 严格 Point-in-Time 历史滚动回测
    print("\n[Step 5] 严格 Point-in-Time 历史步进滚动回测 (2021 -> 2024)...")
    bt_engine = PointInTimeBacktestEngine()
    bt_res = bt_engine.run_walk_forward_backtest(
        cutoff_date=date(2021, 12, 31),
        evaluation_date=date(2024, 12, 31)
    )
    m = bt_res["metrics"]
    print(f" -> Top-3 预测命中率 Precision@3: {int(m['precision_at_3']*100)}%")
    print(f" -> Top-5 预测命中率 Precision@5: {int(m['precision_at_5']*100)}%")
    print(f" -> Brier 概率校准误差: {m['brier_score']}")
    print(f" -> 相对随机基准超额命中: +{int(m['excess_vs_random_baseline']*100)}%")

    # 6. 导出 Top 1 机会 14 章节智能研报
    top_candidate = pool[0]
    print(f"\n[Step 6] 为顶级机会 [{top_candidate['topic_name']}] 自动生成14章节结构化智能研报...")
    dossier_gen = ResearchDossierGenerator()
    report_text = dossier_gen.generate_dossier(top_candidate)
    export_path = "data/exports/Foresight_Alpha_Sample_Dossier.md"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f" -> 完整深度研报已导出至: {export_path}")

    print("\n" + "=" * 80)
    print("✅ 系统全流程运行完毕！所有数据已成功持久化至 data/storage.duckdb。")
    print("👉 运行以下命令可一键启动交互式 Web 分析门户:")
    print("   bash scripts/start_ui.sh")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foresight Alpha 端到端运行示范脚本")
    parser.add_argument("--refresh", action="store_true", help="强制从公网重新抓取并更新最新数据")
    args = parser.parse_args()
    run_full_pipeline_demo(force_refresh=args.refresh)
