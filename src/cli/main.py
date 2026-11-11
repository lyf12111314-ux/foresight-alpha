"""Foresight Alpha 命令行交互控制中心 (Command Line Interface)."""

import argparse
import sys
from datetime import date

from src.ingestion.pipeline import IngestionPipeline
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.monitoring.monitor import SystemMonitor
from src.backtest.engine import PointInTimeBacktestEngine
from src.reporting.dossier import ResearchDossierGenerator


def main():
    parser = argparse.ArgumentParser(description="Foresight Alpha - 新兴产业与非对称机会智能预测系统 CLI")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 1. 采集子命令
    subparsers.add_parser("ingest", help="执行多源数据采集与清洗入库流水线")

    # 2. 候选池子命令
    subparsers.add_parser("pool", help="生成并展示最新非对称机会候选池排名")

    # 3. 监控巡检子命令
    subparsers.add_parser("scan", help="执行全系统商业化红旗与技术爆发预警巡检")

    # 4. 回测子命令
    bt_parser = subparsers.add_parser("backtest", help="执行严格 Point-in-Time 历史滚动回测")
    bt_parser.add_argument("--cutoff-year", type=int, default=2021, help="回测切片年份 (默认 2021)")
    bt_parser.add_argument("--eval-year", type=int, default=2024, help="事后核验年份 (默认 2024)")

    # 5. 研报生成子命令
    rep_parser = subparsers.add_parser("report", help="为指定技术主题生成14章节结构化智能研报")
    rep_parser.add_argument("--topic", type=str, default="TOPIC_EMBODIED_AI", help="目标技术主题ID")
    rep_parser.add_argument("--output", type=str, default=None, help="导出研报文件路径")

    args = parser.parse_args()

    if args.command == "ingest":
        print("Starting ingestion pipeline...")
        pipeline = IngestionPipeline()
        stats = pipeline.run_all(verbose=True)
        print("Ingestion completed successfully:", stats)

    elif args.command == "pool":
        print("Building Asymmetric Opportunity Pool...")
        manager = OpportunityPoolManager()
        pool = manager.build_opportunity_pool()
        print("\n=== 非对称机会候选池 (Top Opportunities) ===")
        print(f"{'排名':<4} {'主题名称':<20} {'综合评分':<10} {'置信度':<8} {'3年增长':<8} {'评级'}")
        print("-" * 75)
        for rank, c in enumerate(pool, start=1):
            print(f"{rank:<4} {c['topic_name']:<20} {c['composite_score']:<10.1f} {int(c['calibrated_confidence']*100)}%{'':<4} {c['growth_3y_multiple']}x{'':<5} {c['rating']}")

    elif args.command == "scan":
        print("Running system monitoring scan...")
        monitor = SystemMonitor()
        alerts = monitor.run_daily_scan()
        print(f"Monitoring scan finished. Triggered {len(alerts)} alerts.")
        for a in alerts:
            print(f"[{a['severity']}] {a['title']}")

    elif args.command == "backtest":
        print(f"Running Point-in-Time Backtest ({args.cutoff_year} -> {args.eval_year})...")
        engine = PointInTimeBacktestEngine()
        res = engine.run_walk_forward_backtest(
            cutoff_date=date(args.cutoff_year, 12, 31),
            evaluation_date=date(args.eval_year, 12, 31)
        )
        print("\n=== 回测结果指标 ===")
        for k, v in res["metrics"].items():
            print(f"{k}: {v}")

    elif args.command == "report":
        manager = OpportunityPoolManager()
        pool = manager.build_opportunity_pool()
        target = next((c for c in pool if c["topic_id"] == args.topic), pool[0] if pool else None)
        if not target:
            print(f"Topic {args.topic} not found.")
            sys.exit(1)
        gen = ResearchDossierGenerator()
        dossier_text = gen.generate_dossier(target)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(dossier_text)
            print(f"Report written to {args.output}")
        else:
            print(dossier_text[:1200] + "\n... [截断输出，请使用 --output 导出完整报告]")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
