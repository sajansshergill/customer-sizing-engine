# src/prioritize.py
"""
Campaign prioritization — final ranked account list.
Reads data/headroom.parquet + data/survival.parquet + data/segments.parquet
Computes composite priority score and outputs seller-ready account list.

Outputs:
  data/prioritized_accounts.parquet     — full ranked list
  outputs/prioritized_accounts.csv      — seller-ready CSV
  outputs/priority_distribution.png     — tier breakdown chart
  outputs/top50_heatmap.png             — top 50 accounts heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")
SURVIVAL_PATH = DATA_DIR / "survival.parquet"
SURVIVAL_MERGE_COLS = [
    "account_id",
    "rmst_months",
    "churn_prob_6m",
    "base_clv",
    "expansion_clv",
    "risk_adj_clv",
]


def _survival_stub(headroom: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic CLV columns when survival.parquet is missing and lifelines/survival.py cannot run.
    Mirrors the rough structure of survival.compute_clv without Kaplan-Meier.
    """
    feat = pd.read_parquet("data/features.parquet")[
        ["account_id", "churned", "observed_months"]
    ]
    m = headroom[["account_id", "arr", "expected_revenue_90d"]].merge(
        feat, on="account_id", how="left"
    )
    monthly = m["arr"] / 12
    obs = m["observed_months"].fillna(12).clip(1, 120)
    ch = m["churned"].fillna(0).astype(float)
    rmst = np.clip(24 + 0.2 * obs * (1 - ch), 6, 36)
    churn_prob = np.clip(0.12 + 0.4 * ch, 0.05, 0.95)
    base = (monthly * rmst).round(2)
    exp = (base + m["expected_revenue_90d"] * (rmst / 3)).round(2)
    risk = (exp * (1 - churn_prob)).round(2)
    return pd.DataFrame(
        {
            "account_id": m["account_id"],
            "rmst_months": np.round(rmst, 2),
            "churn_prob_6m": np.round(churn_prob, 4),
            "base_clv": base,
            "expansion_clv": exp,
            "risk_adj_clv": risk,
        }
    )


def write_survival_stub_parquet(path: Path | str = SURVIVAL_PATH) -> pd.DataFrame:
    """Write heuristic survival/CLV columns to disk (e.g. when lifelines is unavailable)."""
    headroom = pd.read_parquet("data/headroom.parquet")
    out = _survival_stub(headroom)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"✓ {path} (heuristic CLV stub)")
    return out


def _survival_merge_df(headroom: pd.DataFrame) -> pd.DataFrame:
    """Prefer data/survival.parquet; otherwise run survival.py; otherwise heuristic stub."""
    if SURVIVAL_PATH.is_file():
        return pd.read_parquet(SURVIVAL_PATH)[SURVIVAL_MERGE_COLS]
    try:
        import survival
    except ModuleNotFoundError:
        print(
            "⚠ lifelines not installed — using heuristic CLV proxy "
            "(pip install lifelines && python src/survival.py for Kaplan-Meier outputs)."
        )
        return _survival_stub(headroom)
    print("⚠ data/survival.parquet not found — running survival.py to generate it...")
    try:
        survival.run()
    except Exception as exc:
        print(f"⚠ survival.py failed ({exc}); using heuristic CLV proxy.")
        return _survival_stub(headroom)
    if not SURVIVAL_PATH.is_file():
        return _survival_stub(headroom)
    return pd.read_parquet(SURVIVAL_PATH)[SURVIVAL_MERGE_COLS]

SEED = 42

SEGMENT_COLORS = {
    "High-Value Engaged":      "#1D9E75",
    "Growth-Stage Expanding":  "#378ADD",
    "At-Risk Dormant":         "#E24B4A",
    "Loyal Mid-Market":        "#BA7517",
    "Early-Stage Potential":   "#534AB7",
}

# ── Composite score weights ───────────────────────────────────────────────────
# Tunable — mirrors what a sales ops team would negotiate with leadership
WEIGHTS = {
    "propensity_score":    0.35,   # model signal — strongest predictor
    "rsam_score_norm":     0.30,   # revenue opportunity size
    "risk_adj_clv_norm":   0.20,   # long-term value
    "health_score_norm":   0.10,   # relationship health
    "renewal_urgency":     0.05,   # contract timing boost
}


# ── 1. Load & merge all upstream outputs ─────────────────────────────────────

def load_data() -> pd.DataFrame:
    headroom = pd.read_parquet("data/headroom.parquet")
    survival = _survival_merge_df(headroom)
    features = pd.read_parquet("data/features.parquet")[
        ["account_id", "health_score", "renewal_window_flag",
         "contract_end_days", "nps_score", "num_products",
         "tenure_months", "seg_enterprise", "seg_midmarket"]
    ]

    df = (
        headroom
        .merge(survival, on="account_id", how="left")
        .merge(features, on="account_id", how="left")
    )
    print(f"✓ Loaded — {len(df):,} accounts across {df['segment_name'].nunique()} segments")
    return df


# ── 2. Composite priority score ───────────────────────────────────────────────

def compute_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max normalize each component to [0, 1] then apply weights.
    composite_score = Σ weight_i × normalized_component_i
    """
    def minmax(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        return ((series - mn) / (mx - mn + 1e-9)).round(6)

    df["rsam_score_norm"]    = minmax(df["rsam_score"])
    df["risk_adj_clv_norm"]  = minmax(df["risk_adj_clv"])
    df["health_score_norm"]  = minmax(df["health_score"])

    # renewal urgency — higher score when contract ends sooner
    df["renewal_urgency"] = minmax(1 / (df["contract_end_days"] + 1))

    df["composite_score"] = (
        WEIGHTS["propensity_score"]  * df["propensity_score"]   +
        WEIGHTS["rsam_score_norm"]   * df["rsam_score_norm"]    +
        WEIGHTS["risk_adj_clv_norm"] * df["risk_adj_clv_norm"]  +
        WEIGHTS["health_score_norm"] * df["health_score_norm"]  +
        WEIGHTS["renewal_urgency"]   * df["renewal_urgency"]
    ).round(6)

    # final priority tier on composite score
    df["final_tier"] = pd.cut(
        df["composite_score"],
        bins=[0, 0.25, 0.45, 0.65, 0.80, 1.0],
        labels=["Tier 5 — Monitor", "Tier 4 — Nurture",
                "Tier 3 — Develop", "Tier 2 — Prioritize",
                "Tier 1 — Critical"],
        include_lowest=True,
    )

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    print(f"\n── Composite Score Distribution ─────────────────────")
    print(f"  Mean   : {df['composite_score'].mean():.4f}")
    print(f"  Median : {df['composite_score'].median():.4f}")
    print(f"  Max    : {df['composite_score'].max():.4f}")
    print(f"\n── Tier Breakdown ───────────────────────────────────")
    print(df["final_tier"].value_counts().sort_index().to_string())
    return df


# ── 3. Seller-ready action tags ───────────────────────────────────────────────

def assign_action_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based action tags that tell sellers exactly what to do.
    Multiple tags can apply per account.
    """
    tags = []
    for _, row in df.iterrows():
        account_tags = []

        if row["propensity_score"] >= 0.75:
            account_tags.append("🔥 Hot Upsell")
        if row["renewal_window_flag"] == 1:
            account_tags.append("📅 Renewal Window")
        if row["churn_prob_6m"] >= 0.45:
            account_tags.append("⚠️ Churn Risk")
        if row["rsam_score"] >= 60_000:
            account_tags.append("💰 High Headroom")
        if row["nps_score"] >= 9:
            account_tags.append("⭐ Promoter")
        if row["num_products"] == 1:
            account_tags.append("➕ Cross-Sell")
        if row["health_score"] >= 75 and row["propensity_score"] >= 0.60:
            account_tags.append("✅ Ready to Expand")
        if not account_tags:
            account_tags.append("👀 Monitor")

        tags.append(" | ".join(account_tags))

    df["action_tags"] = tags
    return df


# ── 4. Priority distribution chart ───────────────────────────────────────────

def plot_priority_distribution(df: pd.DataFrame) -> None:
    tier_counts = df["final_tier"].value_counts().sort_index()
    tier_labels = [t.split("—")[1].strip() for t in tier_counts.index]
    tier_colors = ["#E24B4A", "#BA7517", "#378ADD", "#1D9E75", "#888780"][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#F8F8F6")
    for ax in (ax1, ax2):
        ax.set_facecolor("#F8F8F6")

    # bar chart — tier counts
    bars = ax1.bar(tier_labels, tier_counts.values,
                   color=tier_colors, edgecolor="none", width=0.55)
    for bar, val in zip(bars, tier_counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30,
                 f"{val:,}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#2C2C2A")
    ax1.set_title("Account Count by Priority Tier", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Accounts", fontsize=9)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.tick_params(axis="x", labelsize=8, rotation=15)

    # stacked bar — tier × segment composition
    tier_seg = (
        df.groupby(["final_tier", "segment_name"])
        .size()
        .unstack(fill_value=0)
    )
    tier_seg.index = [t.split("—")[1].strip() for t in tier_seg.index]
    bottom = np.zeros(len(tier_seg))
    for seg in tier_seg.columns:
        color = SEGMENT_COLORS.get(seg, "#888780")
        ax2.bar(tier_seg.index, tier_seg[seg],
                bottom=bottom, label=seg,
                color=color, edgecolor="none", width=0.55, alpha=0.85)
        bottom += tier_seg[seg].values

    ax2.set_title("Tier Composition by Segment", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accounts", fontsize=9)
    ax2.legend(fontsize=7, framealpha=0.85, loc="upper right")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(axis="x", labelsize=8, rotation=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "priority_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ outputs/priority_distribution.png saved")


# ── 5. Top-50 heatmap ─────────────────────────────────────────────────────────

def plot_top50_heatmap(df: pd.DataFrame) -> None:
    top50 = df.head(50).copy()

    metrics = [
        "propensity_score", "rsam_score_norm",
        "risk_adj_clv_norm", "health_score_norm", "renewal_urgency",
    ]
    labels = [
        "Propensity", "Headroom\n(norm)",
        "CLV\n(norm)", "Health\n(norm)", "Renewal\nUrgency",
    ]

    heat_data = top50[metrics].values.T   # shape (5, 50)

    fig, ax = plt.subplots(figsize=(18, 3.5))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#F8F8F6")

    im = ax.imshow(heat_data, aspect="auto", cmap="YlGn", vmin=0, vmax=1)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(50))
    ax.set_xticklabels(
        [f"#{r}" for r in top50["rank"].values],
        fontsize=6, rotation=90
    )
    ax.set_title("Top 50 Accounts — Signal Heatmap", fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, orientation="vertical",
                 fraction=0.015, pad=0.01, label="Normalized Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top50_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ outputs/top50_heatmap.png saved")


# ── 6. Save outputs ───────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame) -> pd.DataFrame:
    seller_cols = [
        "rank", "account_id", "segment_name",
        "arr", "rsam_score", "risk_adj_clv",
        "propensity_score", "churn_prob_6m",
        "composite_score", "final_tier", "action_tags",
        "shap_driver_1", "shap_driver_2", "shap_driver_3",
        "contract_end_days", "health_score", "nps_score",
    ]
    out = df[seller_cols].copy()
    out.to_parquet("data/prioritized_accounts.parquet", index=False)
    out.to_csv(OUTPUT_DIR / "prioritized_accounts.csv",  index=False)
    print(f"✓ data/prioritized_accounts.parquet saved")
    print(f"✓ outputs/prioritized_accounts.csv saved")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    df  = load_data()
    df  = compute_priority_score(df)
    df  = assign_action_tags(df)

    plot_priority_distribution(df)
    plot_top50_heatmap(df)

    out = save_outputs(df)

    print(f"\n── Top 15 Accounts ──────────────────────────────────")
    print(out[[
        "rank", "account_id", "segment_name",
        "arr", "rsam_score", "propensity_score",
        "composite_score", "final_tier", "action_tags"
    ]].head(15).to_string(index=False))

    return out


if __name__ == "__main__":
    run()