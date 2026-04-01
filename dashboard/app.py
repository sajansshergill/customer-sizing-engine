# dashboard/app.py
"""
Streamlit dashboard — Adobe Propensity & Headroom Sizing Engine
Run: streamlit run dashboard/app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from typing import Optional

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Adobe Propensity & Headroom Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme constants ───────────────────────────────────────────────────────────

SEGMENT_COLORS = {
    "High-Value Engaged":      "#1D9E75",
    "Growth-Stage Expanding":  "#378ADD",
    "At-Risk Dormant":         "#E24B4A",
    "Loyal Mid-Market":        "#BA7517",
    "Early-Stage Potential":   "#534AB7",
}

TIER_COLORS = {
    "Tier 1 — Critical":    "#E24B4A",
    "Tier 2 — Prioritize":  "#BA7517",
    "Tier 3 — Develop":     "#378ADD",
    "Tier 4 — Nurture":     "#1D9E75",
    "Tier 5 — Monitor":     "#888780",
}

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: #F8F8F6;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 0.5px solid #D3D1C7;
    }
    .metric-label { font-size: 12px; color: #5F5E5A; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; line-height: 1; }
    .metric-sub   { font-size: 11px; color: #888780; margin-top: 4px; }
    .section-title {
        font-size: 15px; font-weight: 600;
        color: #2C2C2A; margin: 1.5rem 0 0.75rem;
        border-left: 3px solid #378ADD;
        padding-left: 10px;
    }
    .tag {
        display: inline-block;
        font-size: 10px; padding: 2px 8px;
        border-radius: 20px; margin: 1px;
        background: #E6F1FB; color: #185FA5;
        border: 0.5px solid #B5D4F4;
    }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Plotly helpers ────────────────────────────────────────────────────────────
# Streamlit's Plotly bridge can throw JSON parse errors if NaN/Inf or invalid
# marker sizes reach the frontend (e.g. "Invalid digits after decimal point").

def _sanitize_scatter_for_plotly(
    df: pd.DataFrame,
    x: str,
    y: str,
    size_col: Optional[str] = None,
    string_cols: tuple = (),
) -> pd.DataFrame:
    d = df.copy()
    for col in (x, y, size_col):
        if col and col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
            d[col] = d[col].replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[x, y])
    if size_col and size_col in d.columns:
        med = d[size_col].median()
        if not np.isfinite(med) or med <= 0:
            med = d[size_col].mean()
        if not np.isfinite(med) or med <= 0:
            med = 1.0
        d[size_col] = d[size_col].fillna(med).clip(lower=1e-6)
    for sc in string_cols:
        if sc in d.columns:
            d[sc] = d[sc].fillna("").astype(str).str.replace("\n", " ", regex=False)
    return d


def _sanitize_dashboard_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf, coerce numerics — avoids Plotly/Streamlit JSON edge cases."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out


def _safe_money_m(x: float) -> str:
    if pd.isna(x) or not np.isfinite(float(x)):
        return ""
    return f"${float(x) / 1e6:.1f}M"


def _safe_money_k(x: float) -> str:
    if pd.isna(x) or not np.isfinite(float(x)):
        return ""
    return f"${float(x) / 1e3:.0f}K"


def _safe_pct(x: float) -> str:
    if pd.isna(x) or not np.isfinite(float(x)):
        return ""
    return f"{float(x):.1%}"


def st_plotly_safe(fig_in) -> None:
    """
    Round-trip through Plotly JSON so only JSON-safe scalars reach Streamlit's frontend
    (fixes Safari/React 'Invalid digits after decimal point' on some numpy/NaN paths).
    """
    fig_out = fig_in
    try:
        fig_out = pio.from_json(pio.to_json(fig_in, validate=False))
    except Exception:
        pass
    st.plotly_chart(fig_out, use_container_width=True)


# ── Data loader ───────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    base = Path("data")
    accounts  = pd.read_parquet(base / "prioritized_accounts.parquet")
    headroom  = pd.read_parquet(base / "headroom.parquet")
    survival  = pd.read_parquet(base / "survival.parquet")
    features  = pd.read_parquet(base / "features.parquet")
    segments  = pd.read_parquet(base / "segments.parquet")

    # prioritized_accounts already has churn_prob_6m / risk_adj_clv; merging those
    # again from survival would create *_x / *_y columns and break KPIs.
    surv_cols = ["rmst_months", "churn_prob_6m", "base_clv", "risk_adj_clv"]
    s = survival[["account_id"] + [c for c in surv_cols if c in survival.columns]].copy()
    for c in surv_cols:
        if c in accounts.columns and c in s.columns:
            s = s.drop(columns=[c])
    df = accounts.merge(s, on="account_id", how="left")
    feat_cols = [
        "account_id", "num_products", "num_users",
        "tenure_months", "avg_feature_adoption", "avg_monthly_logins",
        "upsell_label",
    ]
    feat_cols = [c for c in feat_cols if c in features.columns]
    f = features[feat_cols].copy()
    for c in feat_cols:
        if c != "account_id" and c in df.columns and c in f.columns:
            f = f.drop(columns=[c])
    df = df.merge(f, on="account_id", how="left")
    df = _sanitize_dashboard_df(df)

    return df, headroom, survival, segments


def check_data_ready() -> bool:
    required = [
        "data/prioritized_accounts.parquet",
        "data/headroom.parquet",
        "data/survival.parquet",
        "data/features.parquet",
        "data/segments.parquet",
    ]
    return all(Path(p).exists() for p in required)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Adobe_Corporate_Logo.png/320px-Adobe_Corporate_Logo.png",
                     width=120)
    st.sidebar.markdown("### Filters")

    segments = st.sidebar.multiselect(
        "Customer segment",
        options=sorted(df["segment_name"].dropna().unique()),
        default=sorted(df["segment_name"].dropna().unique()),
    )

    tiers = st.sidebar.multiselect(
        "Priority tier",
        options=sorted(df["final_tier"].dropna().astype(str).unique()),
        default=sorted(df["final_tier"].dropna().astype(str).unique()),
    )

    min_propensity = st.sidebar.slider(
        "Min propensity score", 0.0, 1.0, 0.0, step=0.05
    )
    min_rsam = st.sidebar.slider(
        "Min rSAM score ($)", 0, 200_000, 0, step=5_000
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline**")
    if st.sidebar.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("""
    <small>
    Built by <b>Sajan Singh Shergill</b><br>
    MS Data Science · Pace University<br>
    <a href="https://linkedin.com/in/sajanshergill">linkedin.com/in/sajanshergill</a>
    </small>
    """, unsafe_allow_html=True)

    filtered = df[
        df["segment_name"].isin(segments) &
        df["final_tier"].astype(str).isin(tiers) &
        (df["propensity_score"] >= min_propensity) &
        (df["rsam_score"] >= min_rsam)
    ]
    return filtered


# ── KPI row ───────────────────────────────────────────────────────────────────

def render_kpis(df: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    def kpi(col, label, value, sub, color="#2C2C2A"):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    kpi(c1, "Accounts",          f"{len(df):,}",
        "in current filter", "#2C2C2A")
    rsam_sum = float(df["rsam_score"].sum())
    kpi(c2, "Total rSAM Opportunity",
        _safe_money_m(rsam_sum) if np.isfinite(rsam_sum) else "—",
        "pipeline headroom", "#378ADD")
    kpi(c3, "Critical Accounts",
        f"{(df['final_tier'].astype(str).str.startswith('Tier 1')).sum():,}",
        "Tier 1 priority", "#E24B4A")
    p_mean = float(df["propensity_score"].mean())
    kpi(c4, "Avg Propensity",
        f"{p_mean:.2f}" if np.isfinite(p_mean) else "—",
        "mean upsell score", "#1D9E75")
    clv_sum = float(df["risk_adj_clv"].sum())
    kpi(c5, "Portfolio CLV",
        _safe_money_m(clv_sum) if np.isfinite(clv_sum) else "—",
        "risk-adjusted", "#534AB7")
    ch_mean = float(df["churn_prob_6m"].mean())
    kpi(c6, "Avg Churn Risk",
        f"{ch_mean:.1%}" if np.isfinite(ch_mean) else "—",
        "6-month horizon", "#BA7517")


# ── Tab 1 — Overview ──────────────────────────────────────────────────────────

def render_overview(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Revenue Opportunity by Segment</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        seg_summary = (
            df.groupby("segment_name")
            .agg(total_rsam=("rsam_score", "sum"),
                 account_count=("account_id", "count"))
            .reset_index()
            .sort_values("total_rsam", ascending=True)
        )
        fig = px.bar(
            seg_summary,
            x="total_rsam", y="segment_name",
            orientation="h",
            color="segment_name",
            color_discrete_map=SEGMENT_COLORS,
            text=seg_summary["total_rsam"].apply(_safe_money_m),
            labels={"total_rsam": "Total rSAM ($)", "segment_name": ""},
        )
        fig.update_traces(textposition="outside", textfont_size=10)
        fig.update_layout(
            title="Total rSAM Opportunity by Segment",
            showlegend=False,
            plot_bgcolor="#F8F8F6", paper_bgcolor="#F8F8F6",
            height=320, margin=dict(l=10, r=40, t=40, b=10),
            xaxis=dict(tickformat="$,.0f", showgrid=True,
                       gridcolor="#D3D1C7", gridwidth=0.5),
        )
        st_plotly_safe(fig)

    with col2:
        tier_counts = df["final_tier"].astype(str).value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        tier_counts = tier_counts.sort_values("tier")
        fig2 = px.pie(
            tier_counts, values="count", names="tier",
            color="tier",
            color_discrete_map=TIER_COLORS,
            hole=0.45,
        )
        fig2.update_traces(textposition="outside", textinfo="label+percent",
                           textfont_size=9)
        fig2.update_layout(
            title="Priority Tier Distribution",
            showlegend=False,
            plot_bgcolor="#F8F8F6", paper_bgcolor="#F8F8F6",
            height=320, margin=dict(l=10, r=10, t=40, b=10),
        )
        st_plotly_safe(fig2)

    # Propensity vs headroom scatter
    st.markdown('<div class="section-title">Propensity vs Revenue Headroom</div>',
                unsafe_allow_html=True)

    raw_sample = df.sample(n=min(3000, len(df)), random_state=42)
    sample = _sanitize_scatter_for_plotly(
        raw_sample,
        x="propensity_score",
        y="rsam_score",
        size_col="risk_adj_clv",
        string_cols=("action_tags",),
    )
    for col in ("arr", "composite_score"):
        if col in sample.columns:
            sample[col] = pd.to_numeric(sample[col], errors="coerce")
    if sample.empty:
        st.warning("No rows with valid propensity and rSAM scores to plot.")
        return
    fig3 = px.scatter(
        sample,
        x="propensity_score",
        y="rsam_score",
        color="segment_name",
        color_discrete_map=SEGMENT_COLORS,
        size="risk_adj_clv",
        size_max=18,
        hover_data=["account_id", "arr", "composite_score", "action_tags"],
        opacity=0.65,
        labels={
            "propensity_score": "Propensity Score",
            "rsam_score": "rSAM Headroom ($)",
            "segment_name": "Segment",
        },
    )
    y_med = float(sample["rsam_score"].median()) if len(sample) else 0.0
    y_hi = float(sample["rsam_score"].quantile(0.92)) if len(sample) else 0.0
    fig3.add_vline(x=0.55, line_dash="dot", line_color="#888780", line_width=1)
    fig3.add_hline(y=y_med, line_dash="dot",
                   line_color="#888780", line_width=1)
    fig3.add_annotation(x=0.78, y=y_hi,
                        text="Priority Targets", showarrow=False,
                        font=dict(size=10, color="#1D9E75"))
    fig3.add_annotation(x=0.15, y=y_hi,
                        text="High Headroom / Low Intent", showarrow=False,
                        font=dict(size=10, color="#BA7517"))
    fig3.update_layout(
        plot_bgcolor="#F8F8F6", paper_bgcolor="#F8F8F6",
        height=420, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(tickformat="$,.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0),
    )
    st_plotly_safe(fig3)


# ── Tab 2 — Account List ──────────────────────────────────────────────────────

def render_account_list(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Ranked Account List</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("🔍 Search account ID", placeholder="ACC-00001")
    with col2:
        top_n = st.selectbox("Show top", [50, 100, 250, 500, 1000], index=0)
    with col3:
        sort_by = st.selectbox("Sort by",
            ["composite_score", "rsam_score", "propensity_score", "risk_adj_clv"],
            index=0)

    display = df.copy()
    if search:
        display = display[display["account_id"].str.contains(search, case=False)]

    display = display.sort_values(sort_by, ascending=False).head(top_n)

    show_cols = {
        "rank":             "Rank",
        "account_id":       "Account",
        "segment_name":     "Segment",
        "arr":              "ARR ($)",
        "rsam_score":       "rSAM ($)",
        "propensity_score": "Propensity",
        "composite_score":  "Score",
        "final_tier":       "Tier",
        "action_tags":      "Actions",
        "shap_driver_1":    "Driver 1",
        "shap_driver_2":    "Driver 2",
        "churn_prob_6m":    "Churn Risk",
        "contract_end_days":"Days to Renewal",
    }

    tbl = display[list(show_cols.keys())].rename(columns=show_cols)
    tbl["ARR ($)"]   = tbl["ARR ($)"].apply(lambda x: f"${x:,.0f}")
    tbl["rSAM ($)"]  = tbl["rSAM ($)"].apply(lambda x: f"${x:,.0f}")
    tbl["Propensity"]= tbl["Propensity"].apply(lambda x: f"{x:.3f}")
    tbl["Score"]     = tbl["Score"].apply(lambda x: f"{x:.4f}")
    tbl["Churn Risk"]= tbl["Churn Risk"].apply(lambda x: f"{x:.1%}")

    st.dataframe(tbl, use_container_width=True, height=480)

    # download
    csv = display[list(show_cols.keys())].to_csv(index=False)
    st.download_button(
        "⬇️ Download filtered list",
        data=csv,
        file_name="prioritized_accounts_filtered.csv",
        mime="text/csv",
    )


# ── Tab 3 — Segment Deep Dive ─────────────────────────────────────────────────

def render_segment_deep_dive(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Segment Deep Dive</div>',
                unsafe_allow_html=True)

    seg = st.selectbox(
        "Select segment",
        options=sorted(df["segment_name"].dropna().unique()),
        index=0,
    )
    seg_df = df[df["segment_name"] == seg]
    color  = SEGMENT_COLORS.get(seg, "#378ADD")

    m1, m2, m3, m4 = st.columns(4)
    def mini_kpi(col, label, val):
        col.metric(label, val)

    arr_med = float(seg_df["arr"].median())
    rsam_sum = float(seg_df["rsam_score"].sum())
    prop_m = float(seg_df["propensity_score"].mean())
    mini_kpi(m1, "Accounts",          f"{len(seg_df):,}")
    mini_kpi(m2, "Median ARR",        f"${arr_med:,.0f}" if np.isfinite(arr_med) else "—")
    mini_kpi(m3, "Total Opportunity", _safe_money_m(rsam_sum) if np.isfinite(rsam_sum) else "—")
    mini_kpi(m4, "Avg Propensity",    f"{prop_m:.3f}" if np.isfinite(prop_m) else "—")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            seg_df, x="propensity_score", nbins=30,
            color_discrete_sequence=[color],
            labels={"propensity_score": "Propensity Score"},
            title=f"Propensity Distribution — {seg}",
        )
        fig.update_layout(
            plot_bgcolor="#F8F8F6", paper_bgcolor="#F8F8F6",
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st_plotly_safe(fig)

    with col2:
        fig2 = px.histogram(
            seg_df, x="rsam_score", nbins=30,
            color_discrete_sequence=[color],
            labels={"rsam_score": "rSAM Score ($)"},
            title=f"Headroom Distribution — {seg}",
        )
        fig2.update_layout(
            plot_bgcolor="#F8F8F6", paper_bgcolor="#F8F8F6",
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            xaxis=dict(tickformat="$,.0f"),
        )
        st_plotly_safe(fig2)

    # SHAP drivers for this segment
    st.markdown('<div class="section-title">Top SHAP Drivers in Segment</div>',
                unsafe_allow_html=True)

    driver_counts = pd.concat([
        seg_df["shap_driver_1"].dropna(),
        seg_df["shap_driver_2"].dropna(),
        seg_df["shap_driver_3"].dropna(),
    ]).value_counts().head(15).reset_index()
    driver_counts.columns = ["feature", "count"]

    fig3 = px.bar(
        driver_counts.sort_values("count"),
        x="count", y="feature", orientation="h",
        color_discrete_sequence=[color],
        labels={"count": "Frequency as Top Driver", "feature": ""},
        title="Most Frequent SHAP Drivers",
    )
    fig3.update_layout(
        plot_bgcolor="#F8F8F6", paper_bgcolor="#F8F8F6",
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    st_plotly_safe(fig3)


# ── Tab 4 — Survival & CLV ────────────────────────────────────────────────────

def render_survival(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Survival & Lifetime Value</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        clv_seg = (
            df.groupby("segment_name")["risk_adj_clv"]
            .median().reset_index()
            .sort_values("risk_adj_clv", ascending=True)
        )
        fig = px.bar(
            clv_seg,
            x="risk_adj_clv", y="segment_name",
            orientation="h",
            color="segment_name",
            color_discrete_map=SEGMENT_COLORS,
            text=clv_seg["risk_adj_clv"].apply(_safe_money_k),
            labels={"risk_adj_clv": "Median Risk-Adj CLV ($)", "segment_name": ""},
            title="Median Risk-Adjusted CLV by Segment",
        )
        fig.update_traces(textposition="outside", textfont_size=9)
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="#F8F8F6", paper_bgcolor="#F8F8F6",
            height=320, margin=dict(l=10, r=50, t=40, b=10),
            xaxis=dict(tickformat="$,.0f"),
        )
        st_plotly_safe(fig)

    with col2:
        churn_seg = (
            df.groupby("segment_name")["churn_prob_6m"]
            .mean().reset_index()
            .sort_values("churn_prob_6m", ascending=False)
        )
        fig2 = px.bar(
            churn_seg,
            x="segment_name", y="churn_prob_6m",
            color="segment_name",
            color_discrete_map=SEGMENT_COLORS,
            text=churn_seg["churn_prob_6m"].apply(_safe_pct),
            labels={"churn_prob_6m": "Avg 6-Month Churn Probability",
                    "segment_name": ""},
            title="Churn Risk by Segment (6-Month Horizon)",
        )
        fig2.update_traces(textposition="outside", textfont_size=9)
        fig2.update_layout(
            showlegend=False,
            plot_bgcolor="#F8F8F6", paper_bgcolor="#F8F8F6",
            height=320, margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(tickformat=".0%"),
            xaxis=dict(tickangle=15, tickfont_size=9),
        )
        st_plotly_safe(fig2)

    # CLV vs churn risk scatter
    st.markdown('<div class="section-title">CLV vs Churn Risk — Account Landscape</div>',
                unsafe_allow_html=True)

    raw_s2 = df.sample(n=min(2500, len(df)), random_state=42)
    sample = _sanitize_scatter_for_plotly(
        raw_s2,
        x="churn_prob_6m",
        y="risk_adj_clv",
        size_col="arr",
    )
    for col in ("propensity_score", "rsam_score"):
        if col in sample.columns:
            sample[col] = pd.to_numeric(sample[col], errors="coerce")
    if sample.empty:
        st.warning("No rows with valid churn and CLV values to plot.")
        return
    fig3 = px.scatter(
        sample,
        x="churn_prob_6m",
        y="risk_adj_clv",
        color="segment_name",
        color_discrete_map=SEGMENT_COLORS,
        size="arr",
        size_max=16,
        opacity=0.6,
        hover_data=["account_id", "propensity_score", "rsam_score"],
        labels={
            "churn_prob_6m": "Churn Probability (6M)",
            "risk_adj_clv":  "Risk-Adjusted CLV ($)",
            "segment_name":  "Segment",
        },
    )
    y92 = float(sample["risk_adj_clv"].quantile(0.92)) if len(sample) else 0.0
    fig3.add_vline(x=0.40, line_dash="dot", line_color="#888780", line_width=1)
    fig3.add_annotation(x=0.05, y=y92,
                        text="Protect & Grow", showarrow=False,
                        font=dict(size=10, color="#1D9E75"))
    fig3.add_annotation(x=0.55, y=y92,
                        text="Save Now", showarrow=False,
                        font=dict(size=10, color="#E24B4A"))
    fig3.update_layout(
        plot_bgcolor="#F8F8F6", paper_bgcolor="#F8F8F6",
        height=420, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat="$,.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0),
    )
    st_plotly_safe(fig3)


# ── Tab 5 — Model Health ──────────────────────────────────────────────────────

def render_model_health(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Model Health & Score Distributions</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x="propensity_score", nbins=40,
            color_discrete_sequence=["#378ADD"],
            title="Propensity Score Distribution",
            labels={"propensity_score": "Propensity Score"},
        )
        fig.update_layout(
            plot_bgcolor="#F8F8F6", paper_bgcolor="#F8F8F6",
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st_plotly_safe(fig)

    with col2:
        fig2 = px.histogram(
            df, x="composite_score", nbins=40,
            color_discrete_sequence=["#534AB7"],
            title="Composite Priority Score Distribution",
            labels={"composite_score": "Composite Score"},
        )
        fig2.update_layout(
            plot_bgcolor="#F8F8F6", paper_bgcolor="#F8F8F6",
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st_plotly_safe(fig2)

    # Score decile lift table
    st.markdown('<div class="section-title">Propensity Decile Lift Analysis</div>',
                unsafe_allow_html=True)

    lift_src = df.copy()
    lift_src["decile"] = pd.qcut(
        lift_src["propensity_score"], q=10, labels=False, duplicates="drop"
    ) + 1
    agg_spec = {
        "accounts":       ("account_id",      "count"),
        "avg_propensity": ("propensity_score", "mean"),
        "total_rsam":     ("rsam_score",       "sum"),
        "avg_clv":        ("risk_adj_clv",     "mean"),
    }
    if "upsell_label" in lift_src.columns:
        agg_spec["upsell_rate"] = ("upsell_label", "mean")
    lift_df = (
        lift_src.groupby("decile")
        .agg(**agg_spec)
        .round(3)
        .reset_index()
        .sort_values("decile", ascending=False)
    )
    if "upsell_rate" not in lift_df.columns:
        lift_df["upsell_rate"] = np.nan
    lift_df["total_rsam"]    = lift_df["total_rsam"].apply(lambda x: f"${x:,.0f}")
    lift_df["avg_clv"]       = lift_df["avg_clv"].apply(lambda x: f"${x:,.0f}")
    lift_df["avg_propensity"]= lift_df["avg_propensity"].apply(lambda x: f"{x:.3f}")
    lift_df["upsell_rate"]   = lift_df["upsell_rate"].apply(
        lambda x: "—" if pd.isna(x) else f"{float(x):.1%}"
    )
    lift_df.columns = ["Decile", "Accounts", "Avg Propensity",
                       "Total rSAM", "Avg CLV", "Upsell Rate"]
    st.dataframe(lift_df, use_container_width=True, hide_index=True)


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.title("📊 Adobe Propensity & Headroom Sizing Engine")
    st.caption("Customer upsell scoring · rSAM headroom · CLV · Campaign prioritization")

    if not check_data_ready():
        st.error("Pipeline outputs not found. Run `make pipeline` first.")
        st.code("make pipeline", language="bash")
        st.stop()

    df, headroom, survival, segments = load_data()

    filtered = render_sidebar(df)

    if filtered.empty:
        st.warning("No accounts match current filters. Adjust sidebar settings.")
        st.stop()

    render_kpis(filtered)

    tabs = st.tabs([
        "📈 Overview",
        "🎯 Account List",
        "🔍 Segment Deep Dive",
        "⏳ Survival & CLV",
        "🩺 Model Health",
    ])

    with tabs[0]: render_overview(filtered)
    with tabs[1]: render_account_list(filtered)
    with tabs[2]: render_segment_deep_dive(filtered)
    with tabs[3]: render_survival(filtered)
    with tabs[4]: render_model_health(filtered)


if __name__ == "__main__":
    main()