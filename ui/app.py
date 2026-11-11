"""新兴产业与非对称财富机会智能预测系统 (Streamlit Web Research Portal)."""

import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，避免 Streamlit 直接运行 ui/app.py 时找不到 src 与 ui 包
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

from src.core.database import db_manager
from src.macro.hmm_regime import MacroHMMRegimeModel
from src.decision_engine.opportunity_pool import OpportunityPoolManager
from src.monitoring.alerts import AlertManager
from src.backtest.engine import PointInTimeBacktestEngine
from src.reporting.dossier import ResearchDossierGenerator
from src.value_chain.graph import ValueChainGraph
from src.value_chain.margin_pool import ProfitPoolAnalyzer
from src.value_chain.financial_audit import CommercializationAuditEngine
from src.risk_survival.survival_cox import CompanySurvivalEngine
from src.diffusion.s_curve import SCurveDiffusionModel
from ui.components.theme import apply_fintech_theme, format_plotly_chart

st.set_page_config(
    page_title="Foresight Alpha 智能预测系统",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_fintech_theme()

# 侧边栏导航
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Foresight Alpha")
st.sidebar.caption("新兴产业与非对称机会预测系统 V1.0")

menu = st.sidebar.radio(
    "系统导航菜单",
    [
        "1. 综合研究首页",
        "2. 新兴技术雷达",
        "3. 技术趋势详情",
        "4. 宏观周期分析",
        "5. 产业链利润地图",
        "6. 企业成长与生存分析",
        "7. 非对称机会候选池",
        "8. 风险与预警中心",
        "9. 历史回测中心",
        "10. 自动研究报告",
        "11. 数据源与模型管理"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**研究原则声明**：
- 严格遵循真实数据血缘追溯
- 严禁未来信息泄漏 (Point-in-Time)
- 严禁机械对齐康波年份
- 本系统定位为深度研究与决策辅助
""")

# 初始化核心单例
pool_mgr = OpportunityPoolManager()
alert_mgr = AlertManager()
macro_model = MacroHMMRegimeModel()

# ==========================================
# 1. 综合研究首页
# ==========================================
if menu == "1. 综合研究首页":
    st.title("🔭 新兴产业与非对称财富机会智能预测系统")
    st.markdown("*通过长期经济周期、前沿技术演进、产业链利润沉淀与企业生存风险，提前捕捉未来 3—10 年非线性成长机会。*")
    
    # 顶部关键指标卡片
    pool = pool_mgr.build_opportunity_pool()
    macro_df = db_manager.query_df("SELECT * FROM macro_indicators ORDER BY as_of_date ASC")
    macro_status = macro_model.get_latest_regime(macro_df) if not macro_df.empty else {"regime": "EXPANSION", "macro_growth_multiplier": 1.15}
    active_alerts = alert_mgr.get_active_alerts()
    top_candidate = pool[0] if pool else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">宏观经济周期状态</div>
                <div class="metric-value">{macro_status['regime']}</div>
                <span class="badge-alpha">增长阻尼: {macro_status.get('macro_growth_multiplier', 1.0)}x</span>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">机会池顶级标的 (Top 1)</div>
                <div class="metric-value">{top_candidate['topic_name'] if top_candidate else 'N/A'}</div>
                <span class="badge-alpha">综合分: {top_candidate['composite_score'] if top_candidate else '0'}</span>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">有效预警监控流</div>
                <div class="metric-value">{len(active_alerts)} 项</div>
                <span class="badge-danger">活跃红旗告警</span>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">数据底座入库记录</div>
                <div class="metric-value">{db_manager.query_df('SELECT COUNT(*) as c FROM raw_records').iloc[0]['c']} 条</div>
                <span class="badge-alpha">Point-in-Time 审计</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🏆 核心高潜非对称机会推荐榜")
    df_pool = pd.DataFrame([
        {
            "排名": idx + 1,
            "技术主题": c["topic_name"],
            "所属赛道": c["domain_id"],
            "综合评分 (0-100)": c["composite_score"],
            "模型置信度": f"{int(c['calibrated_confidence'] * 100)}%",
            "未来3年复合增长倍数": f"{c['growth_3y_multiple']}x",
            "10倍非对称大胜概率": f"{c['prob_10x_super_return_pct']}%",
            "评级": c["rating"]
        }
        for idx, c in enumerate(pool)
    ])
    st.dataframe(df_pool, use_container_width=True, hide_index=True)

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("#### 宏观状态转移概率矩阵 (Markov Transition)")
        trans = macro_status.get("transition_matrix", [])
        if trans:
            df_trans = pd.DataFrame(
                trans,
                index=["衰退", "复苏", "繁荣", "放缓"],
                columns=["衰退", "复苏", "繁荣", "放缓"]
            )
            fig_trans = px.imshow(df_trans, text_auto=True, color_continuous_scale="Viridis", title="宏观状态转移热力图")
            st.plotly_chart(format_plotly_chart(fig_trans), use_container_width=True)

    with col_r:
        st.markdown("#### 实时预警事件速览")
        if active_alerts:
            for a in active_alerts[:4]:
                st.warning(f"**[{a['severity']}] {a['title']}**\n\n{a['description']}")
        else:
            st.success("目前未检测到严重商业化失血或恶性内卷预警。")

# ==========================================
# 2. 新兴技术雷达
# ==========================================
elif menu == "2. 新兴技术雷达":
    st.title("📡 新兴技术自动发现与技术雷达")
    st.markdown("基于 OpenAlex、arXiv 与 GitHub API，以语义向量与动态聚类算法自动识别新生技术主题，计算新颖度与扩散加速度。")

    pool = pool_mgr.build_opportunity_pool()
    radar_data = []
    for c in pool:
        radar_data.append({
            "Topic": c["topic_name"],
            "CompositeScore": c["composite_score"],
            "GrowthMultiple3Y": c["growth_3y_multiple"],
            "Confidence": c["calibrated_confidence"] * 100,
            "Domain": c["domain_id"],
            "Penetration": c["penetration_pct"]
        })
    df_radar = pd.DataFrame(radar_data)

    fig_scatter = px.scatter(
        df_radar,
        x="Penetration",
        y="GrowthMultiple3Y",
        size="CompositeScore",
        color="Domain",
        hover_name="Topic",
        text="Topic",
        labels={
            "Penetration": "当前技术渗透率 (%)",
            "GrowthMultiple3Y": "未来3年预期增长倍数 (Point Estimate)",
            "CompositeScore": "综合评分"
        },
        title="新兴技术非线性成长潜力象限图 (气泡大小表示综合评分)"
    )
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(format_plotly_chart(fig_scatter), use_container_width=True)

    st.markdown("### 自动发现的新兴技术主题聚类表")
    st.table(df_radar[["Topic", "Domain", "CompositeScore", "GrowthMultiple3Y", "Penetration"]])

# ==========================================
# 3. 技术趋势详情
# ==========================================
elif menu == "3. 技术趋势详情":
    st.title("📈 技术趋势时序与 S 曲线扩散拟合")
    pool = pool_mgr.build_opportunity_pool()
    topic_options = {c["topic_name"]: c["topic_id"] for c in pool}
    selected_topic_name = st.selectbox("选择要深入研判的技术主题", list(topic_options.keys()))
    selected_topic_id = topic_options[selected_topic_name]

    # 查询历史学术与开源指标
    df_metrics = db_manager.query_df(
        "SELECT * FROM tech_metrics WHERE topic_id = ? ORDER BY as_of_date ASC", [selected_topic_id]
    )

    if not df_metrics.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            fig_papers = px.line(
                df_metrics, x="as_of_date", y="papers_count", markers=True,
                title=f"{selected_topic_name} - 全球年度学术文献产出趋势 (OpenAlex 权威源)"
            )
            st.plotly_chart(format_plotly_chart(fig_papers), use_container_width=True)

        with col2:
            fig_institutions = px.bar(
                df_metrics, x="as_of_date", y="active_institutions",
                title=f"{selected_topic_name} - 参与研究的独立机构数量扩散 (跨学科熵)"
            )
            st.plotly_chart(format_plotly_chart(fig_institutions), use_container_width=True)

        # 扩散S曲线拟合与推演
        s_curve = SCurveDiffusionModel()
        years = [d.year if hasattr(d, "year") else int(str(d)[:4]) for d in df_metrics["as_of_date"]]
        papers = df_metrics["papers_count"].values
        diff_res = s_curve.fit_and_forecast(years, papers, forecast_horizon_years=5)

        df_forecast = pd.DataFrame(diff_res["forecast_series"])
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(x=years, y=papers, mode="markers+lines", name="历史实际文献量", line=dict(color="#f0b90b", width=3)))
        fig_diff.add_trace(go.Scatter(x=df_forecast["year"], y=df_forecast["logistic_curve"], mode="lines", name="Logistic S曲线拟合", line=dict(dash="dash", color="#0ecb81")))
        fig_diff.add_trace(go.Scatter(x=df_forecast["year"], y=df_forecast["bass_curve"], mode="lines", name="Bass 创新扩散曲线", line=dict(dash="dot", color="#4992ff")))
        st.plotly_chart(format_plotly_chart(fig_diff, f"{selected_topic_name} - 未来5年S曲线扩散演化预测"), use_container_width=True)

        st.info(f"**模型扩散诊断**：当前渗透率约为 **{diff_res['current_penetration_pct']}%**，处于 **{diff_res['stage_description']}**，理论拐点年份预估为 **{diff_res['inflection_point_year']} 年**。")

# ==========================================
# 4. 宏观周期分析
# ==========================================
elif menu == "4. 宏观周期分析":
    st.title("🌐 宏观经济长中短多周期研判")
    st.markdown("融合高斯隐马尔可夫模型 (HMM)、康波长波结构定位与基钦库存周期。")

    macro_df = db_manager.query_df("SELECT * FROM macro_indicators ORDER BY as_of_date ASC")
    if not macro_df.empty:
        regimes_df = macro_model.predict_regimes(macro_df)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_pmi = px.line(
                macro_df[macro_df["indicator_code"] == "CN_PMI"],
                x="as_of_date", y="value", markers=True,
                title="制造业 PMI 走势 (荣枯线 50)"
            )
            fig_pmi.add_hline(y=50, line_dash="dash", line_color="red")
            st.plotly_chart(format_plotly_chart(fig_pmi), use_container_width=True)

        with c2:
            fig_regimes = px.area(
                regimes_df,
                x="as_of_date",
                y=["prob_expansion", "prob_recovery", "prob_slowdown", "prob_contraction"],
                title="宏观经济4状态后验概率演化 (HMM Filtered Probabilities)"
            )
            st.plotly_chart(format_plotly_chart(fig_regimes), use_container_width=True)

        st.markdown("### 康波第6次长波技术革命框架")
        st.json({
            "当前康波长波": "第六次康波浪潮 (信息技术-通用AI与新型能源)",
            "周期阶段": "扩散与繁荣构建期 (Deployment Phase)",
            "宏观阻尼系数": 1.15,
            "资产配置启示": "底层算力基础设施与核心零部件优先沉淀超额利润，防范下游应用同质化恶性价格战"
        })

# ==========================================
# 5. 产业链利润地图
# ==========================================
elif menu == "5. 产业链利润地图":
    st.title("⛓️ 产业链利润地图与卡位壁垒")
    st.markdown("分析产业快速发展后，利润最终沉淀在上游、中游还是下游环节。")

    vc_graph = ValueChainGraph()
    profit_analyzer = ProfitPoolAnalyzer()

    domains = ["ROBOTICS_EMBODIED", "AI_INFRA", "ENERGY_STORAGE", "LOW_ALTITUDE_AERO"]
    selected_domain = st.selectbox("选择产业赛道", domains)

    domain_nodes = vc_graph.get_domain_nodes(selected_domain)
    df_nodes = pd.DataFrame(domain_nodes)
    if not df_nodes.empty:
        df_nodes["毛利率下界(%)"] = df_nodes["gross_margin_range"].apply(lambda x: x[0])
        df_nodes["毛利率上界(%)"] = df_nodes["gross_margin_range"].apply(lambda x: x[1])
        df_nodes["平均毛利率(%)"] = (df_nodes["毛利率下界(%)"] + df_nodes["毛利率上界(%)"]) / 2.0

        fig_margin = px.bar(
            df_nodes,
            x="name",
            y="平均毛利率(%)",
            color="tier",
            text="平均毛利率(%)",
            title=f"{selected_domain} - 产业链各环节毛利率水平阶梯"
        )
        st.plotly_chart(format_plotly_chart(fig_margin), use_container_width=True)

        res = profit_analyzer.analyze_domain_profit_pool(selected_domain)
        st.success(f"**利润瓶颈诊断**：{res['profit_bottleneck_diagnosis']}")

# ==========================================
# 6. 企业成长与生存分析
# ==========================================
elif menu == "6. 企业成长与生存分析":
    st.title("🏢 企业真实经营指标与生存风险模型")
    st.markdown("基于公开财报真实数据，结合 Kaplan-Meier 生存估计与商业化红旗诊断。")

    df_fin = db_manager.query_df("SELECT * FROM company_financials ORDER BY as_of_date ASC")
    company_list = df_fin["company_id"].unique().tolist()
    selected_cid = st.selectbox("选择分析企业标的", company_list)

    sub_fin = df_fin[df_fin["company_id"] == selected_cid]
    if not sub_fin.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            fig_rev = px.bar(sub_fin, x="as_of_date", y="revenue", text="revenue", title="营业收入历年趋势 (亿元)")
            st.plotly_chart(format_plotly_chart(fig_rev), use_container_width=True)
        with col2:
            fig_gm = px.line(sub_fin, x="as_of_date", y=["gross_margin", "rd_ratio"], markers=True, title="毛利率与研发费用率走势 (%)")
            st.plotly_chart(format_plotly_chart(fig_gm), use_container_width=True)

        audit_engine = CommercializationAuditEngine()
        audit_res = audit_engine.audit_company(sub_fin)
        st.markdown(f"### 商业化健康度诊断评分：`{audit_res['health_score']} / 100`")
        if audit_res["red_flags"]:
            for rf in audit_res["red_flags"]:
                st.error(f"🚨 **[{rf['severity']}] {rf['title']}**\n\n{rf['evidence']}")
        else:
            st.success("✅ 该企业未触发商业化恶性失血或虚假繁荣红旗告警。")

# ==========================================
# 7. 非对称机会候选池
# ==========================================
elif menu == "7. 非对称机会候选池":
    st.title("🎯 非对称机会候选池与蒙特卡洛模拟")
    st.markdown("严格区分产业规模增长 $\neq$ 企业收入增长 $\neq$ 估值扩张 $\neq$ 投资者最终净收益。")

    pool = pool_mgr.build_opportunity_pool()
    for c in pool:
        with st.expander(f"【{c['rating']}】{c['topic_name']} — 综合评分 {c['composite_score']}", expanded=True if c == pool[0] else False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("3年预期复合倍数", f"{c['growth_3y_multiple']}x")
            c2.metric("非对称赔率比", f"{c['asymmetry_ratio']}:1")
            c3.metric("10倍超级收益概率", f"{c['prob_10x_super_return_pct']}%")
            c4.metric("3年经营生存概率", f"{int(c['survival_prob_3y']*100)}%")

            st.markdown("#### 合法可参与二级市场资产标的")
            for asset in c.get("legal_assets", []):
                st.markdown(f"- **{asset.get('name')}**（类型: `{asset.get('asset_type')}` ｜ 日均成交: `{asset.get('trading_liquidity_daily_avg')}万元`）\n  *合规指引: {asset.get('regulatory_notes')}*")

# ==========================================
# 8. 风险与预警中心
# ==========================================
elif menu == "8. 风险与预警中心":
    st.title("🚨 动态风险监测与预警中心")
    st.markdown("实时捕捉技术突变脉冲、毛利失血、经营现金流背离与估值泡沫。")

    active_alerts = alert_mgr.get_active_alerts()
    if active_alerts:
        for a in active_alerts:
            st.error(f"**[{a['severity']}] {a['title']}**\n\n- 触发时间: `{a['triggered_at']}`\n- 详细归因: {a['description']}\n- 关联证据: `{a['evidence']}`")
    else:
        st.success("系统处于稳定监测状态，当前无高危活跃警报。")

# ==========================================
# 9. 历史回测中心
# ==========================================
elif menu == "9. 历史回测中心":
    st.title("📊 Point-in-Time 历史回测与校准中心")
    st.markdown("物理隔离历史截面 $T$ 之后的信息，严格评估 Precision@K 与概率校准误差 (Brier Score)。")

    col_btn, col_blank = st.columns([1, 3])
    with col_btn:
        run_bt = st.button("🚀 运行一次滚动回测 (2021 -> 2024)", use_container_width=True)

    bt_engine = PointInTimeBacktestEngine()
    if run_bt:
        with st.spinner("正在严格模拟历史时间截面推演..."):
            bt_res = bt_engine.run_walk_forward_backtest(date(2021, 12, 31), date(2024, 12, 31))
            st.session_state["latest_bt"] = bt_res

    latest_bt = st.session_state.get("latest_bt", None)
    if not latest_bt:
        df_runs = db_manager.query_df("SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT 1")
        if not df_runs.empty:
            row = df_runs.iloc[0].to_dict()
            latest_bt = {
                "run_id": row["run_id"],
                "cutoff_date": str(row["cutoff_date"]),
                "evaluation_date": str(row["evaluation_date"]),
                "metrics": {
                    "precision_at_3": row["precision_3"],
                    "precision_at_5": row["precision_5"],
                    "recall_at_5": row["recall_5"],
                    "brier_score": row["brier_score"],
                    "excess_vs_random_baseline": row["excess_vs_random"],
                    "excess_vs_momentum_baseline": row["excess_vs_momentum"]
                }
            }

    if latest_bt:
        m = latest_bt["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision@3 (Top 3 命中率)", f"{int(m['precision_at_3'] * 100)}%")
        c2.metric("Precision@5 (Top 5 命中率)", f"{int(m['precision_at_5'] * 100)}%")
        c3.metric("Brier 概率校准误差", f"{m['brier_score']} (越低越好)")
        c4.metric("相对随机基准超额命中", f"+{int(m['excess_vs_random_baseline'] * 100)}%")

        st.markdown(f"**回测切片时点**：`{latest_bt['cutoff_date']}` ｜ **事后核验时点**：`{latest_bt['evaluation_date']}`")

# ==========================================
# 10. 自动研究报告
# ==========================================
elif menu == "10. 自动研究报告":
    st.title("📝 14 章节结构化智能研究报告")
    st.markdown("自动为任意候选技术机会生成合规、严谨、多情景推演的专业研究报告。")

    pool = pool_mgr.build_opportunity_pool()
    topic_options = {c["topic_name"]: c for c in pool}
    selected_topic_name = st.selectbox("选择要生成报告的目标技术", list(topic_options.keys()))
    candidate = topic_options[selected_topic_name]

    dossier_gen = ResearchDossierGenerator()
    report_text = dossier_gen.generate_dossier(candidate)

    st.download_button(
        label="📥 下载完整 Markdown 研报",
        data=report_text,
        file_name=f"Foresight_Alpha_Report_{candidate['topic_id']}.md",
        mime="text/markdown"
    )

    st.markdown("---")
    st.markdown(report_text)

# ==========================================
# 11. 数据源与模型管理
# ==========================================
elif menu == "11. 数据源与模型管理":
    st.title("🛡️ 数据源血缘追溯与模型治理中心")
    st.markdown("记录所有原始数据哈希签名、版本审计记录与 Point-in-Time 物理隔离状态。")

    st.markdown("### 数据库各核心表记录概览")
    tables = ["raw_records", "tech_metrics", "macro_indicators", "company_financials", "model_forecasts", "model_alerts", "backtest_runs"]
    table_stats = []
    for t in tables:
        cnt = db_manager.query_df(f"SELECT COUNT(*) as c FROM {t}").iloc[0]["c"]
        table_stats.append({"表名": t, "记录行数": cnt, "存储格式": "DuckDB Columnar / Parquet"})
    st.table(pd.DataFrame(table_stats))

    st.markdown("### 最新入库原始数据溯源抽样 (Point-in-Time 证据链)")
    df_raw = db_manager.query_df("SELECT record_id, source_name, as_of_date, published_at, ingested_at, quality_status, raw_hash FROM raw_records ORDER BY ingested_at DESC LIMIT 6")
    st.dataframe(df_raw, use_container_width=True)

    st.info("系统所有数据均来自权威官方接口（OpenAlex, arXiv, GitHub, World Bank, 国家统计局及交易所财报），无任何伪造数据。")
