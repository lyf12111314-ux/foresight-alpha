"""UI 主题与通用图表样式组件 (Fintech Theme & Plotly Styler)."""

import plotly.graph_objects as go
import streamlit as st


def apply_fintech_theme():
    """应用现代金融科技深色高对比度样式"""
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
        }
        .metric-card {
            background: linear-gradient(135deg, #1e222d 0%, #2a2e39 100%);
            border: 1px solid #363c4e;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .metric-title {
            color: #848e9c;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .metric-value {
            color: #f0b90b;
            font-size: 1.8rem;
            font-weight: 700;
        }
        .badge-alpha {
            background-color: #163c2c;
            color: #0ecb81;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-danger {
            background-color: #3e1b1b;
            color: #f6465d;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)


def format_plotly_chart(fig: go.Figure, title: str = "") -> go.Figure:
    """统一 Plotly 金融图表视觉风格"""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 15, "color": "#EAECEF"}
        },
        template="plotly_dark",
        paper_bgcolor="#161A25",
        plot_bgcolor="#161A25",
        font={"color": "#B7BDC6", "family": "Arial, sans-serif"},
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#2B313A")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#2B313A")
    return fig
