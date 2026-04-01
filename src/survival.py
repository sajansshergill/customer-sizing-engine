# src/survival.py
"""
Customer lifetime value via survival analysis.
Reads data/features.parquet + data/segments.parquet
Fits Kaplan-Meier survival curves per segment and computes:
  - median survival time (months to churn)
  - restricted mean survival time (RMST)
  - CLV estimate per account

Outputs:
  data/survival.parquet                — per-account survival + CLV scores
  outputs/km_curves.png                — KM survival curves by segment
  outputs/clv_by_segment.png           — CLV distribution boxplot
  outputs/survival_profiles.csv        — segment-level survival summary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42

SEGMENT_COLORS = {
    "High-Value Engaged":      "#1D9E75",
    "Growth-Stage Expanding":  "#378ADD",
    "At-Risk Dormant":         "#E24B4A",
    "Loyal Mid-Market":        "#BA7517",
    "Early-Stage Potential":   "#534AB7",
}

# Horizon for RMST calculation (months)
RMST_HORIZON = 36


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    features = pd.read_parquet("data/features.parquet")
    segments = pd.read_parquet("data/segments.parquet")[
        ["account_id", "segment_name"]
    ]
    headroom = pd.read_parquet("data/headroom.parquet")[
        ["account_id", "rsam_score", "propensity_score", "expected_revenue_90d"]
    ]

    df = (
        features
        .merge(segments,  on="account_id", how="left")
        .merge(headroom,  on="account_id", how="left")
    )
    print(f"✓ Loaded — {len(df):,} accounts")
    print(f"  Churn rate     : {df['churned'].mean():.1%}")
    print(f"  Median tenure  : {df['observed_months'].median():.0f} months")
    return df


# ── 2. Kaplan-Meier per segment ───────────────────────────────────────────────

def fit_km_curves(df: pd.DataFrame) -> dict[str, KaplanMeierFitter]:
    """Fit one KM curve per segment. Returns dict of fitted KMF objects."""
    kmf_dict = {}
    for seg in df["segment_name"].unique():
        mask = df["segment_name"] == seg
        kmf  = KaplanMeierFitter(label=seg)
        kmf.fit(
            durations  = df.loc[mask, "observed_months"],
            event_observed = df.loc[mask, "churned"],
        )
        kmf_dict[seg] = kmf
        median_surv = kmf.median_survival_time_
        print(f"  {seg:<30} median survival: {median_surv:.1f} months")
    return kmf_dict


# ── 3. Log-rank test ──────────────────────────────────────────────────────────

def run_logrank_test(df: pd.DataFrame) -> None:
    result = multivariate_logrank_test(
        df["observed_months"],
        df["segment_name"],
        df["churned"],
    )
    print(f"\n── Multivariate Log-Rank Test ───────────────────────")
    print(f"  Test statistic : {result.test_statistic:.4f}")
    print(f"  p-value        : {result.p_value:.6f}")
    print(f"  Segments differ significantly: {result.p_value < 0.05}")


# ── 4. RMST — restricted mean survival time ───────────────────────────────────

def compute_rmst(kmf: KaplanMeierFitter, horizon: int = RMST_HORIZON) -> float:
    """
    RMST = area under KM curve up to horizon.
    Approximated via trapezoidal integration on the step function.
    """
    timeline = kmf.survival_function_.index.values
    sf       = kmf.survival_function_.iloc[:, 0].values

    # clip to horizon
    mask     = timeline <= horizon
    t_clip   = np.append(timeline[mask], horizon)
    sf_clip  = np.append(sf[mask], sf[mask][-1] if mask.any() else 0)

    rmst = np.trapz(sf_clip, t_clip)
    return round(rmst, 2)


# ── 5. Per-account CLV estimate ───────────────────────────────────────────────

def compute_clv(df: pd.DataFrame, kmf_dict: dict) -> pd.DataFrame:
    """
    CLV formula:
        monthly_revenue = arr / 12
        survival_months = RMST(segment) — expected months remaining
        base_clv        = monthly_revenue × survival_months
        expansion_clv   = base_clv + expected_revenue_90d × (survival_months / 3)
        risk_adjusted   = expansion_clv × (1 - churn_probability_proxy)

    churn_probability_proxy = 1 - KM survival at median tenure + 6 months
    """
    rmst_map = {seg: compute_rmst(kmf) for seg, kmf in kmf_dict.items()}
    df["rmst_months"] = df["segment_name"].map(rmst_map)

    # monthly ARR
    df["monthly_arr"] = (df["arr"] / 12).round(2)

    # base CLV
    df["base_clv"] = (df["monthly_arr"] * df["rmst_months"]).round(2)

    # expansion uplift from rSAM
    df["expansion_clv"] = (
        df["base_clv"]
        + df["expected_revenue_90d"] * (df["rmst_months"] / 3)
    ).round(2)

    # churn probability proxy — inverse of KM survival at (tenure + 6 months)
    def churn_prob(row):
        kmf = kmf_dict.get(row["segment_name"])
        if kmf is None:
            return 0.3
        horizon = min(row["observed_months"] + 6, RMST_HORIZON)
        sf_val  = kmf.predict(horizon)
        return round(float(1 - sf_val), 4)

    df["churn_prob_6m"] = df.apply(churn_prob, axis=1)

    # risk-adjusted CLV
    df["risk_adj_clv"] = (
        df["expansion_clv"] * (1 - df["churn_prob_6m"])
    ).round(2)

    print(f"\n── CLV Summary ──────────────────────────────────────")
    print(f"  Median base CLV         : ${df['base_clv'].median():>12,.0f}")
    print(f"  Median expansion CLV    : ${df['expansion_clv'].median():>12,.0f}")
    print(f"  Median risk-adj CLV     : ${df['risk_adj_clv'].median():>12,.0f}")
    print(f"  Total portfolio CLV     : ${df['risk_adj_clv'].sum():>12,.0f}")
    return df


# ── 6. KM curves plot ─────────────────────────────────────────────────────────

def plot_km_curves(kmf_dict: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#F8F8F6")

    for seg, kmf in kmf_dict.items():
        color = SEGMENT_COLORS.get(seg, "#888780")
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=color,
            linewidth=2,
            label=seg,
        )
        # median line
        med = kmf.median_survival_time_
        if not np.isinf(med):
            ax.axvline(x=med, color=color, lw=0.8, linestyle=":", alpha=0.6)

    ax.set_xlabel("Months", fontsize=10)
    ax.set_ylabel("Survival Probability", fontsize=10)
    ax.set_title("Kaplan-Meier Survival Curves by Customer Segment", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    # RMST annotation box
    rmst_lines = [f"{s[:20]:<20} RMST: {compute_rmst(k):.1f}mo" for s, k in kmf_dict.items()]
    ax.text(
        0.98, 0.55,
        "\n".join(rmst_lines),
        transform=ax.transAxes,
        fontsize=7, family="monospace",
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7, edgecolor="#D3D1C7")
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "km_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ outputs/km_curves.png saved")


# ── 7. CLV distribution boxplot ───────────────────────────────────────────────

def plot_clv_boxplot(df: pd.DataFrame) -> None:
    seg_order = (
        df.groupby("segment_name")["risk_adj_clv"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#F8F8F6")

    data_by_seg = [
        df.loc[df["segment_name"] == seg, "risk_adj_clv"].clip(upper=df["risk_adj_clv"].quantile(0.97))
        for seg in seg_order
    ]
    colors = [SEGMENT_COLORS.get(s, "#888780") for s in seg_order]

    bp = ax.boxplot(
        data_by_seg,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1, color="#888780"),
        capprops=dict(linewidth=1, color="#888780"),
        flierprops=dict(marker="o", markersize=2, alpha=0.3, linestyle="none"),
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("none")

    for i, (seg, color) in enumerate(zip(seg_order, colors)):
        med = df.loc[df["segment_name"] == seg, "risk_adj_clv"].median()
        ax.text(i + 1, med + df["risk_adj_clv"].std() * 0.05,
                f"${med/1e3:.0f}K", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=color)

    ax.set_xticks(range(1, len(seg_order) + 1))
    ax.set_xticklabels([s.replace(" ", "\n") for s in seg_order], fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    ax.set_ylabel("Risk-Adjusted CLV", fontsize=10)
    ax.set_title("Customer Lifetime Value Distribution by Segment", fontsize=13, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "clv_by_segment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ outputs/clv_by_segment.png saved")


# ── 8. Segment survival profiles ──────────────────────────────────────────────

def build_survival_profiles(df: pd.DataFrame, kmf_dict: dict) -> pd.DataFrame:
    profiles = df.groupby("segment_name").agg(
        account_count       = ("account_id",      "count"),
        churn_rate          = ("churned",          "mean"),
        median_tenure       = ("observed_months",  "median"),
        avg_churn_prob_6m   = ("churn_prob_6m",    "mean"),
        median_base_clv     = ("base_clv",         "median"),
        median_expansion_clv= ("expansion_clv",    "median"),
        median_risk_adj_clv = ("risk_adj_clv",     "median"),
        total_portfolio_clv = ("risk_adj_clv",     "sum"),
    ).round(2).reset_index()

    profiles["rmst_months"] = profiles["segment_name"].map(
        {seg: compute_rmst(kmf) for seg, kmf in kmf_dict.items()}
    )
    profiles["median_survival_months"] = profiles["segment_name"].map(
        {seg: kmf.median_survival_time_ for seg, kmf in kmf_dict.items()}
    )
    profiles = profiles.sort_values("median_risk_adj_clv", ascending=False)
    profiles.to_csv(OUTPUT_DIR / "survival_profiles.csv", index=False)
    print(f"✓ outputs/survival_profiles.csv saved")
    return profiles


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    df = load_data()

    print(f"\n── Kaplan-Meier Fit by Segment ──────────────────────")
    kmf_dict = fit_km_curves(df)

    run_logrank_test(df)
    df = compute_clv(df, kmf_dict)

    plot_km_curves(kmf_dict)
    plot_clv_boxplot(df)

    profiles = build_survival_profiles(df, kmf_dict)

    # save
    out_cols = [
        "account_id", "segment_name",
        "observed_months", "churned",
        "rmst_months", "churn_prob_6m",
        "monthly_arr", "base_clv",
        "expansion_clv", "risk_adj_clv",
    ]
    df[out_cols].to_parquet("data/survival.parquet", index=False)
    print(f"✓ data/survival.parquet saved")

    print(f"\n── Survival Profiles ────────────────────────────────")
    print(profiles[[
        "segment_name", "account_count", "churn_rate",
        "rmst_months", "avg_churn_prob_6m",
        "median_risk_adj_clv", "total_portfolio_clv"
    ]].to_string(index=False))

    return df


if __name__ == "__main__":
    run()