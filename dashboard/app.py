"""
dashboard/app.py – Production-grade Streamlit dashboard for NSE Quant AI Platform.

Run with:
    streamlit run dashboard/app.py

Features:
  - Home overview with market summary
  - Top Ranked Stocks with filtering
  - Portfolio Analytics with interactive charts
  - Model Performance comparison
  - Backtest Results with equity curves
  - Feature Importance (tree-map + bar)
  - Market Regime Analysis
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Page config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="NSE Quant AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ────────────────────────────────────────────────────────────
DARK_BG   = "#0a0e1a"
CARD_BG   = "#111827"
BORDER    = "#1f2937"
ACCENT    = "#00d4ff"
ACCENT2   = "#7c3aed"
GREEN     = "#10b981"
RED       = "#ef4444"
YELLOW    = "#f59e0b"
TEXT      = "#e2e8f0"
SUBTEXT   = "#94a3b8"

PLOTLY_THEME = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=DARK_BG,
    font=dict(color=TEXT, family="JetBrains Mono, monospace"),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Space Grotesk', sans-serif;
        background-color: {DARK_BG};
        color: {TEXT};
    }}
    .stApp {{ background-color: {DARK_BG}; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0d1320 0%, #111827 100%);
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: {ACCENT};
    }}

    /* Cards */
    .metric-card {{
        background: linear-gradient(135deg, {CARD_BG} 0%, #1a2235 100%);
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 20px 24px;
        position: relative;
        overflow: hidden;
    }}
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    }}
    .metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: {ACCENT};
        line-height: 1;
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: {SUBTEXT};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }}
    .metric-delta {{
        font-size: 0.85rem;
        margin-top: 6px;
    }}

    /* Signal badge */
    .signal-buy {{
        background: rgba(16,185,129,0.15);
        color: {GREEN};
        border: 1px solid {GREEN};
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 1px;
    }}
    .signal-sell {{
        background: rgba(239,68,68,0.15);
        color: {RED};
        border: 1px solid {RED};
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 1px;
    }}
    .signal-hold {{
        background: rgba(245,158,11,0.15);
        color: {YELLOW};
        border: 1px solid {YELLOW};
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 1px;
    }}

    /* Dividers */
    .section-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {ACCENT};
        letter-spacing: 0.5px;
        padding-bottom: 8px;
        border-bottom: 1px solid {BORDER};
        margin-bottom: 16px;
    }}

    /* Streamlit overrides */
    .stSelectbox > div > div {{ background-color: {CARD_BG}; border-color: {BORDER}; }}
    .stMultiSelect > div > div {{ background-color: {CARD_BG}; border-color: {BORDER}; }}
    div[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; color: {ACCENT}; }}
    .stDataFrame {{ border: 1px solid {BORDER}; border-radius: 8px; }}
    hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)


# ── Data loading ────────────────────────────────────────────────────────────

OUTPUTS = Path("outputs")

@st.cache_data(ttl=300)
def load_csv(filename: str) -> Optional[pd.DataFrame]:
    """Load a CSV from the outputs directory."""
    path = OUTPUTS / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def load_all_outputs() -> Dict[str, Optional[pd.DataFrame]]:
    return {
        "top_stocks":    load_csv("top_stocks.csv"),
        "portfolio":     load_csv("portfolio.csv"),
        "backtest":      load_csv("backtest_results.csv"),
        "metrics":       load_csv("model_metrics.csv"),
        "features":      load_csv("feature_importance.csv"),
        "trades":        load_csv("trades.csv"),
    }


# ── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center; padding: 20px 0'>
            <div style='font-size:2.5rem'>📈</div>
            <div style='font-size:1.2rem; font-weight:700; color:{ACCENT};'>NSE Quant AI</div>
            <div style='font-size:0.72rem; color:{SUBTEXT}; letter-spacing:2px;'>INSTITUTIONAL PLATFORM</div>
        </div>
        <hr/>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            [
                "🏠  Home",
                "🏆  Top Ranked Stocks",
                "💼  Portfolio Analytics",
                "🤖  Model Performance",
                "📊  Backtest Results",
                "🔍  Feature Importance",
                "🌐  Market Regime",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.72rem; color:{SUBTEXT};'>Auto-refresh</div>", unsafe_allow_html=True)
        auto_refresh = st.checkbox("Enable (5 min)", value=False)
        if auto_refresh:
            st.rerun()

        st.markdown(f"""
        <div style='margin-top:auto; padding-top:20px; font-size:0.68rem; color:{SUBTEXT}; text-align:center;'>
            Quant AI Platform v2.0<br/>NSE Indian Equities
        </div>
        """, unsafe_allow_html=True)

    return page.split("  ")[1]


# ── Metric card helper ──────────────────────────────────────────────────────

def metric_card(label: str, value: str, delta: str = "", delta_color: str = GREEN) -> str:
    delta_html = f"<div class='metric-delta' style='color:{delta_color}'>{delta}</div>" if delta else ""
    return f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{value}</div>
        {delta_html}
    </div>
    """


# ── Pages ───────────────────────────────────────────────────────────────────

def page_home(data: dict) -> None:
    st.markdown(f"""
    <h1 style='font-size:2rem; font-weight:700; color:{ACCENT};'>
        🏠 Platform Overview
    </h1>
    <p style='color:{SUBTEXT}; margin-bottom:2rem;'>
        Real-time Indian equities AI prediction platform · NSE Universe
    </p>
    """, unsafe_allow_html=True)

    # Summary metrics row
    top = data["top_stocks"]
    port = data["portfolio"]
    bt = data["backtest"]
    met = data["metrics"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n_stocks = len(top) if top is not None else 0
        st.markdown(metric_card("Ranked Stocks", str(n_stocks)), unsafe_allow_html=True)
    with c2:
        n_held = len(port) if port is not None else 0
        st.markdown(metric_card("Portfolio Positions", str(n_held)), unsafe_allow_html=True)
    with c3:
        cagr = "—"
        if bt is not None and "cagr" in bt.columns:
            v = bt["cagr"].dropna().iloc[-1] if len(bt) else None
            cagr = f"{v*100:.1f}%" if v else "—"
        st.markdown(metric_card("CAGR", cagr, delta_color=GREEN), unsafe_allow_html=True)
    with c4:
        sharpe = "—"
        if bt is not None and "sharpe" in bt.columns:
            v = bt["sharpe"].dropna().iloc[-1] if len(bt) else None
            sharpe = f"{v:.2f}" if v else "—"
        st.markdown(metric_card("Sharpe Ratio", sharpe), unsafe_allow_html=True)
    with c5:
        mdd = "—"
        if bt is not None and "max_drawdown" in bt.columns:
            v = bt["max_drawdown"].dropna().iloc[-1] if len(bt) else None
            mdd = f"{v*100:.1f}%" if v else "—"
            color = RED if v and v < -0.15 else YELLOW
        else:
            color = SUBTEXT
        st.markdown(metric_card("Max Drawdown", mdd, delta_color=color), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Equity curve (if available)
    if bt is not None and "portfolio_value" in bt.columns and "date" in bt.columns:
        st.markdown("<div class='section-title'>📈 Equity Curve</div>", unsafe_allow_html=True)
        bt_plot = bt.copy()
        bt_plot["date"] = pd.to_datetime(bt_plot["date"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bt_plot["date"], y=bt_plot["portfolio_value"],
            name="Portfolio",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.07)",
        ))
        if "benchmark_value" in bt_plot.columns:
            fig.add_trace(go.Scatter(
                x=bt_plot["date"], y=bt_plot["benchmark_value"],
                name="Nifty 50 (Benchmark)",
                line=dict(color=SUBTEXT, width=1.5, dash="dot"),
            ))
        fig.update_layout(**PLOTLY_THEME, title="Portfolio vs Benchmark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Run the pipeline first to populate output data. See `python main.py`")

    # Model summary table
    if met is not None:
        st.markdown("<div class='section-title'>🤖 Model Summary</div>", unsafe_allow_html=True)
        st.dataframe(
            met.style.format(precision=4).background_gradient(
                subset=[c for c in met.columns if "auc" in c.lower() or "sharpe" in c.lower()],
                cmap="Blues",
            ),
            use_container_width=True,
        )


def page_top_stocks(data: dict) -> None:
    st.markdown(f"<h1 style='color:{ACCENT}'>🏆 Top Ranked Stocks</h1>", unsafe_allow_html=True)
    top = data["top_stocks"]

    if top is None:
        st.warning("No top_stocks.csv found. Run the pipeline to generate rankings.")
        return

    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        sectors = ["All"] + sorted(top["sector"].unique().tolist()) if "sector" in top.columns else ["All"]
        sel_sector = st.selectbox("Filter by Sector", sectors)
    with col2:
        if "score" in top.columns:
            min_score = st.slider("Minimum Score", 0.0, 1.0, 0.5, 0.01)
        else:
            min_score = 0.0
    with col3:
        top_n = st.number_input("Show Top N", 5, 100, 20)

    filtered = top.copy()
    if sel_sector != "All" and "sector" in filtered.columns:
        filtered = filtered[filtered["sector"] == sel_sector]
    if "score" in filtered.columns:
        filtered = filtered[filtered["score"] >= min_score]
    filtered = filtered.head(int(top_n))

    # Signal column
    if "score" in filtered.columns:
        def _signal(s):
            if s >= 0.65: return "<span class='signal-buy'>BUY</span>"
            elif s >= 0.45: return "<span class='signal-hold'>HOLD</span>"
            else: return "<span class='signal-sell'>SELL</span>"
        filtered["signal"] = filtered["score"].apply(_signal)

    # Render table with HTML signals
    if "signal" in filtered.columns:
        sig_col = filtered.pop("signal")
        filtered.insert(1, "signal", sig_col)

    st.markdown("<div class='section-title'>Ranked Universe</div>", unsafe_allow_html=True)
    st.write(filtered.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Scatter: score vs momentum
    if "score" in top.columns and "mom_21d" in top.columns:
        st.markdown("<div class='section-title'>Score vs 1-Month Momentum</div>", unsafe_allow_html=True)
        fig = px.scatter(
            top.head(50),
            x="mom_21d",
            y="score",
            color="sector" if "sector" in top.columns else None,
            text="ticker" if "ticker" in top.columns else None,
            size_max=12,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(**PLOTLY_THEME, height=420, title="Model Score vs Momentum")
        st.plotly_chart(fig, use_container_width=True)


def page_portfolio(data: dict) -> None:
    st.markdown(f"<h1 style='color:{ACCENT}'>💼 Portfolio Analytics</h1>", unsafe_allow_html=True)
    port = data["portfolio"]

    if port is None:
        st.warning("No portfolio.csv found. Run the pipeline first.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='section-title'>Portfolio Weights</div>", unsafe_allow_html=True)
        if "weight" in port.columns and "ticker" in port.columns:
            fig = px.pie(
                port,
                values="weight",
                names="ticker",
                color_discrete_sequence=px.colors.qualitative.Dark24,
                template="plotly_dark",
                hole=0.45,
            )
            fig.update_layout(**PLOTLY_THEME, height=380, showlegend=True)
            fig.update_traces(textposition="inside", textinfo="label+percent")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Sector Exposure</div>", unsafe_allow_html=True)
        if "sector" in port.columns and "weight" in port.columns:
            sec_df = port.groupby("sector")["weight"].sum().reset_index()
            sec_df = sec_df.sort_values("weight", ascending=True)
            fig = px.bar(
                sec_df, x="weight", y="sector", orientation="h",
                color="weight",
                color_continuous_scale=[[0, ACCENT2], [1, ACCENT]],
                template="plotly_dark",
            )
            fig.update_layout(**PLOTLY_THEME, height=380, coloraxis_showscale=False,
                              title="Sector Weight (%)")
            fig.update_xaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)

    # Holdings table
    st.markdown("<div class='section-title'>Holdings</div>", unsafe_allow_html=True)
    display_cols = [c for c in ["ticker", "sector", "weight", "expected_return",
                                "expected_volatility", "expected_sharpe"] if c in port.columns]
    if display_cols:
        st.dataframe(
            port[display_cols].style.format({
                "weight": "{:.2%}",
                "expected_return": "{:.2%}",
                "expected_volatility": "{:.2%}",
                "expected_sharpe": "{:.3f}",
            }),
            use_container_width=True,
        )


def page_model_performance(data: dict) -> None:
    st.markdown(f"<h1 style='color:{ACCENT}'>🤖 Model Performance</h1>", unsafe_allow_html=True)
    met = data["metrics"]

    if met is None:
        st.warning("No model_metrics.csv found. Run the pipeline first.")
        return

    # Classification metrics radar chart
    clf_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc"] if c in met.columns]
    if clf_cols and "model" in met.columns:
        st.markdown("<div class='section-title'>Classification Metrics – Radar</div>", unsafe_allow_html=True)
        fig = go.Figure()
        colors = [ACCENT, ACCENT2, GREEN, YELLOW, RED]
        for i, (_, row) in enumerate(met.iterrows()):
            vals = [row[c] for c in clf_cols]
            vals += [vals[0]]  # close radar
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=clf_cols + [clf_cols[0]],
                name=str(row.get("model", f"Model {i}")),
                fill="toself",
                line=dict(color=colors[i % len(colors)]),
                fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)},{int(colors[i % len(colors)][3:5], 16)},{int(colors[i % len(colors)][5:], 16)},0.08)",
            ))
        fig.update_layout(
            **PLOTLY_THEME,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=BORDER, color=SUBTEXT),
                angularaxis=dict(gridcolor=BORDER, color=TEXT),
                bgcolor=DARK_BG,
            ),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Finance metrics bar comparison
    fin_cols = [c for c in ["sharpe", "cagr", "win_rate", "max_drawdown"] if c in met.columns]
    if fin_cols and "model" in met.columns:
        st.markdown("<div class='section-title'>Finance Metrics Comparison</div>", unsafe_allow_html=True)
        cols = st.columns(len(fin_cols))
        for i, col_name in enumerate(fin_cols):
            with cols[i]:
                fig = px.bar(
                    met, x="model", y=col_name,
                    color="model",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    template="plotly_dark",
                    title=col_name.replace("_", " ").title(),
                )
                fig.update_layout(**PLOTLY_THEME, height=280, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # Full metrics table
    st.markdown("<div class='section-title'>Full Metrics Table</div>", unsafe_allow_html=True)
    st.dataframe(met.style.format(precision=4), use_container_width=True)


def page_backtest(data: dict) -> None:
    st.markdown(f"<h1 style='color:{ACCENT}'>📊 Backtest Results</h1>", unsafe_allow_html=True)
    bt = data["backtest"]
    trades = data["trades"]

    if bt is None:
        st.warning("No backtest_results.csv found. Run the pipeline first.")
        return

    bt = bt.copy()
    if "date" in bt.columns:
        bt["date"] = pd.to_datetime(bt["date"])

    # Key stats
    if all(c in bt.columns for c in ["cagr", "sharpe", "max_drawdown"]):
        latest = bt.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            v = latest.get("cagr", 0)
            st.markdown(metric_card("CAGR", f"{v*100:.1f}%" if v else "—",
                                    delta_color=GREEN if v > 0 else RED), unsafe_allow_html=True)
        with c2:
            v = latest.get("sharpe", 0)
            st.markdown(metric_card("Sharpe", f"{v:.2f}" if v else "—",
                                    delta_color=GREEN if v > 1 else YELLOW), unsafe_allow_html=True)
        with c3:
            v = latest.get("max_drawdown", 0)
            st.markdown(metric_card("Max Drawdown", f"{v*100:.1f}%" if v else "—",
                                    delta_color=RED if v and v < -0.1 else YELLOW), unsafe_allow_html=True)
        with c4:
            v = latest.get("win_rate", 0)
            st.markdown(metric_card("Win Rate", f"{v*100:.1f}%" if v else "—",
                                    delta_color=GREEN if v > 0.5 else YELLOW), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Equity curve + drawdown
    if "date" in bt.columns and "portfolio_value" in bt.columns:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
        )
        fig.add_trace(go.Scatter(
            x=bt["date"], y=bt["portfolio_value"],
            name="Portfolio", line=dict(color=ACCENT, width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
        ), row=1, col=1)

        if "benchmark_value" in bt.columns:
            fig.add_trace(go.Scatter(
                x=bt["date"], y=bt["benchmark_value"],
                name="Nifty 50", line=dict(color=SUBTEXT, width=1.5, dash="dash"),
            ), row=1, col=1)

        if "drawdown" in bt.columns:
            fig.add_trace(go.Scatter(
                x=bt["date"], y=bt["drawdown"] * 100,
                name="Drawdown (%)", line=dict(color=RED, width=1.5),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
            ), row=2, col=1)

        fig.update_layout(
            **PLOTLY_THEME, height=550,
            title="Portfolio Equity Curve & Drawdown",
        )
        fig.update_yaxes(title_text="Value (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # Rolling Sharpe
    if "rolling_sharpe" in bt.columns and "date" in bt.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bt["date"], y=bt["rolling_sharpe"],
            name="Rolling Sharpe (252d)",
            line=dict(color=ACCENT2, width=2),
        ))
        fig.add_hline(y=1.0, line_dash="dot", line_color=GREEN, annotation_text="Sharpe=1")
        fig.add_hline(y=0.0, line_dash="dot", line_color=RED, annotation_text="Sharpe=0")
        fig.update_layout(**PLOTLY_THEME, height=280, title="Rolling Sharpe Ratio")
        st.plotly_chart(fig, use_container_width=True)

    # Trade log
    if trades is not None:
        st.markdown("<div class='section-title'>Trade Log</div>", unsafe_allow_html=True)
        st.dataframe(trades, use_container_width=True)


def page_feature_importance(data: dict) -> None:
    st.markdown(f"<h1 style='color:{ACCENT}'>🔍 Feature Importance</h1>", unsafe_allow_html=True)
    feat = data["features"]

    if feat is None:
        st.warning("No feature_importance.csv found. Run the pipeline first.")
        return

    # Model selector
    model_cols = [c for c in feat.columns if c not in ["feature", "rank"]]
    if not model_cols:
        st.warning("Feature importance file is missing model columns.")
        return

    sel_model = st.selectbox("Select Model", model_cols)
    top_n = st.slider("Top N Features", 10, min(50, len(feat)), 25)

    feat_sorted = feat.nlargest(top_n, sel_model)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure(go.Bar(
            x=feat_sorted[sel_model],
            y=feat_sorted["feature"],
            orientation="h",
            marker=dict(
                color=feat_sorted[sel_model],
                colorscale=[[0, ACCENT2], [0.5, ACCENT], [1, GREEN]],
                showscale=True,
            ),
        ))
        fig.update_layout(**PLOTLY_THEME, height=600, title=f"Top {top_n} Features – {sel_model}")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Treemap
        fig = px.treemap(
            feat_sorted,
            path=["feature"],
            values=sel_model,
            color=sel_model,
            color_continuous_scale=[[0, ACCENT2], [1, ACCENT]],
            template="plotly_dark",
        )
        fig.update_layout(**PLOTLY_THEME, height=600, title="Feature Importance Treemap")
        st.plotly_chart(fig, use_container_width=True)

    # Feature group summary
    st.markdown("<div class='section-title'>Feature Group Analysis</div>", unsafe_allow_html=True)
    feat_copy = feat.copy()
    feat_copy["group"] = feat_copy["feature"].apply(_classify_feature_group)
    group_agg = feat_copy.groupby("group")[sel_model].sum().sort_values(ascending=False)
    fig = px.pie(
        group_agg.reset_index(), values=sel_model, names="group",
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Dark24,
        template="plotly_dark",
    )
    fig.update_layout(**PLOTLY_THEME, height=350)
    st.plotly_chart(fig, use_container_width=True)


def _classify_feature_group(name: str) -> str:
    n = name.lower()
    if "mom" in n:    return "Momentum"
    if "ma_" in n or "price_ma" in n: return "Moving Average"
    if "vol_" in n or "atr" in n or "bb" in n: return "Volatility"
    if "rsi" in n or "macd" in n or "cci" in n or "stoch" in n: return "Oscillators"
    if "ret_" in n or "log_ret" in n: return "Returns"
    if "obv" in n or "vwap" in n or "dollar" in n: return "Volume"
    if "adx" in n:    return "Trend Strength"
    if "52w" in n or "zscore" in n: return "Statistical"
    if "lag" in n:    return "Lag Features"
    if "cs_rank" in n: return "Cross-Sectional"
    return "Other"


def page_market_regime(data: dict) -> None:
    st.markdown(f"<h1 style='color:{ACCENT}'>🌐 Market Regime Analysis</h1>", unsafe_allow_html=True)
    bt = data["backtest"]

    if bt is None or "date" not in bt.columns:
        st.warning("No backtest data found. Run the pipeline first.")
        return

    bt = bt.copy()
    bt["date"] = pd.to_datetime(bt["date"])

    if "portfolio_value" not in bt.columns:
        st.warning("portfolio_value column required for regime analysis.")
        return

    # Compute rolling return & volatility
    pv = bt.set_index("date")["portfolio_value"]
    ret = pv.pct_change()

    regimes = pd.DataFrame(index=bt["date"])
    regimes["ret_63d"]   = ret.rolling(63).mean() * 252
    regimes["vol_63d"]   = ret.rolling(63).std() * np.sqrt(252)
    regimes["sharpe_63d"]= regimes["ret_63d"] / (regimes["vol_63d"] + 1e-10)

    # Regime classification
    def classify(row):
        if row["ret_63d"] > 0.10 and row["vol_63d"] < 0.20: return "Bull"
        elif row["ret_63d"] < -0.05 and row["vol_63d"] > 0.25: return "Bear"
        elif row["vol_63d"] > 0.30: return "High Volatility"
        else: return "Sideways"

    regimes["regime"] = regimes.apply(classify, axis=1)
    regimes = regimes.reset_index()

    # Regime distribution
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<div class='section-title'>Regime Distribution</div>", unsafe_allow_html=True)
        dist = regimes["regime"].value_counts().reset_index()
        dist.columns = ["regime", "count"]
        colors_map = {"Bull": GREEN, "Bear": RED, "Sideways": YELLOW, "High Volatility": ACCENT2}
        fig = px.pie(
            dist, values="count", names="regime", hole=0.45,
            color="regime",
            color_discrete_map=colors_map,
            template="plotly_dark",
        )
        fig.update_layout(**PLOTLY_THEME, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Rolling Sharpe & Regime Timeline</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=regimes["date"],
            y=regimes["sharpe_63d"],
            name="Rolling Sharpe (63d)",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.05)",
        ))
        # Add regime shading
        regime_color_map = {"Bull": "rgba(16,185,129,0.08)", "Bear": "rgba(239,68,68,0.08)",
                            "Sideways": "rgba(245,158,11,0.05)", "High Volatility": "rgba(124,58,237,0.08)"}
        prev_regime = None
        start_x = None
        for _, row in regimes.iterrows():
            if row["regime"] != prev_regime:
                if prev_regime and start_x is not None:
                    fig.add_vrect(
                        x0=start_x, x1=row["date"],
                        fillcolor=regime_color_map.get(prev_regime, "rgba(0,0,0,0)"),
                        layer="below", line_width=0,
                        annotation_text=prev_regime if False else "",
                    )
                prev_regime = row["regime"]
                start_x = row["date"]

        fig.add_hline(y=1.0, line_dash="dot", line_color=GREEN, line_width=1)
        fig.add_hline(y=0.0, line_dash="dot", line_color=RED, line_width=1)
        fig.update_layout(**PLOTLY_THEME, height=320, title="63-Day Rolling Sharpe")
        st.plotly_chart(fig, use_container_width=True)

    # Return/volatility scatter by regime
    st.markdown("<div class='section-title'>Return vs Volatility by Regime</div>", unsafe_allow_html=True)
    clean = regimes.dropna(subset=["ret_63d", "vol_63d"])
    fig = px.scatter(
        clean, x="vol_63d", y="ret_63d", color="regime",
        color_discrete_map=colors_map,
        template="plotly_dark",
        opacity=0.65,
        title="Risk/Return by Regime",
    )
    fig.update_layout(**PLOTLY_THEME, height=380)
    fig.update_xaxes(tickformat=".0%", title="Rolling 63d Volatility")
    fig.update_yaxes(tickformat=".0%", title="Rolling 63d Return")
    st.plotly_chart(fig, use_container_width=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    page = render_sidebar()
    data = load_all_outputs()

    dispatch = {
        "Home":                page_home,
        "Top Ranked Stocks":   page_top_stocks,
        "Portfolio Analytics": page_portfolio,
        "Model Performance":   page_model_performance,
        "Backtest Results":    page_backtest,
        "Feature Importance":  page_feature_importance,
        "Market Regime":       page_market_regime,
    }

    handler = dispatch.get(page, page_home)
    handler(data)


if __name__ == "__main__":
    main()