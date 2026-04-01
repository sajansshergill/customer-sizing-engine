# src/headroom.py
"""
Headroom sizing — rSAM model.
Reads data/features.parquet + data/propensity_scores.parquet
Computes per-account and per-segment revenue opportunity.

Outputs:
  data/headroom.parquet                 — per-account headroom scores
  outputs/headroom_by_segment.csv       — segment-level opportunity summary
  outputs/headroom_waterfall.png        — waterfall chart by segment
  outputs/headroom_scatter.png          — propensity vs headroom scatter
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

# ── Offering-level ARR ceilings ───────────────────────────────────────────────
# In a real Adobe deployment these come from market sizing / analyst inputs
OFFERING_CEILING = {
    "Creative Cloud":    {"Enterprise": 600_000, "Mid-Market": 180_000, "SMB": 50_000,  "Startup": 25_000},
    "Experience Cloud":  {"Enterprise": 800_000, "Mid-Market": 250_000, "SMB": 80_000,  "Startup": 40_000},
    "Document Cloud":    {"Enterprise": 300_000, "Mid-Market": 100_000, "SMB": 30_000,  "Startup": 15_000},
    "Acrobat":           {"Enterprise": 150_000, "Mid-Market": 60_000,  "SMB": 20_000,  "Startup": 10_000},
}

SEGMENT_COLORS = {
    "High-Value Engaged":      "#1D9E75",
    "Growth-Stage Expanding":  "#378ADD",
    "At-Risk Dormant":         "#E24B4A",
    "Loyal Mid-Market":        "#BA7517",
    "Early-Stage Potential":   "#534AB7",
}

# features.parquet stores offering as one-hots; decode for ceiling + export
_OFFERING_OH = ["offering_cc", "offering_ec", "offering_dc", "offering_acrobat"]
_OFFERING_NAMES = np.array(
    ["Creative Cloud", "Experience Cloud", "Document Cloud", "Acrobat"]
)


def ensure_primary_offering(df: pd.DataFrame) -> pd.DataFrame:
    if "primary_offering" in df.columns:
        return df
    if all(c in df.columns for c in _OFFERING_OH):
        idx = df[_OFFERING_OH].to_numpy().argmax(axis=1)
        out = df.copy()
        out["primary_offering"] = _OFFERING_NAMES[idx]
        return out
    out = df.copy()
    out["primary_offering"] = "Creative Cloud"
    return out


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    features  = pd.read_parquet("data/features.parquet")
    propensity = pd.read_parquet("data/propensity_scores.parquet")[
        ["account_id", "propensity_score", "priority_tier", "segment_name",
         "shap_driver_1", "shap_driver_2", "shap_driver_3"]
    ]
    df = features.merge(propensity, on="account_id", how="left")
    df = ensure_primary_offering(df)
    print(f"✓ Loaded — {len(df):,} accounts")
    return df


# ── 2. rSAM headroom formula ──────────────────────────────────────────────────

def compute_headroom(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core rSAM formula:
        raw_headroom      = offering_ceiling(segment, offering) - current_ARR
        weighted_headroom = raw_headroom × propensity_score
        adjusted_headroom = weighted_headroom × health_weight × tenure_weight

    health_weight  — accounts with high health score are more likely to expand
    tenure_weight  — accounts in renewal window get a boost
    """

    # ── offering-level ceiling lookup ────────────────────────────────────────
    def get_ceiling(row):
        seg_map = {
            "High-Value Engaged":      "Enterprise",
            "Growth-Stage Expanding":  "Mid-Market",
            "At-Risk Dormant":         "SMB",
            "Loyal Mid-Market":        "Mid-Market",
            "Early-Stage Potential":   "Startup",
        }
        broad_seg = seg_map.get(row["segment_name"], "SMB")
        offering  = row.get("primary_offering", "Creative Cloud")
        return OFFERING_CEILING.get(offering, OFFERING_CEILING["Creative Cloud"]).get(broad_seg, 50_000)

    df["offering_ceiling"] = df.apply(get_ceiling, axis=1)

    # ── raw headroom (dollar gap to ceiling) ─────────────────────────────────
    df["raw_headroom"] = (df["offering_ceiling"] - df["arr"]).clip(lower=0)

    # ── health weight  (0.5 → 1.3 scale) ─────────────────────────────────────
    df["health_weight"] = (
        0.5 + (df["health_score"] / 100.0) * 0.8
    ).round(4)

    # ── tenure weight (renewal window = 1.15 boost, else 1.0) ────────────────
    df["tenure_weight"] = np.where(df["renewal_window_flag"] == 1, 1.15, 1.0)

    # ── rSAM score — core formula ─────────────────────────────────────────────
    df["rsam_score"] = (
        df["raw_headroom"]
        * df["propensity_score"]
        * df["health_weight"]
        * df["tenure_weight"]
    ).round(2)

    # ── opportunity tier ──────────────────────────────────────────────────────
    df["opportunity_tier"] = pd.cut(
        df["rsam_score"],
        bins=[0, 5_000, 20_000, 60_000, 150_000, np.inf],
        labels=["Minimal", "Low", "Medium", "High", "Strategic"],
    )

    # ── expected revenue (conservative: 25% of rsam_score realized) ──────────
    df["expected_revenue_90d"] = (df["rsam_score"] * 0.25).round(2)

    print(f"✓ rSAM scores computed")
    print(f"  Total pipeline opportunity : ${df['rsam_score'].sum():>15,.0f}")
    print(f"  Expected 90-day revenue    : ${df['expected_revenue_90d'].sum():>15,.0f}")
    print(f"  Median rSAM score          : ${df['rsam_score'].median():>15,.0f}")
    return df


# ── 3. Segment-level summary ──────────────────────────────────────────────────

def build_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("segment_name").agg(
        account_count          = ("account_id",           "count"),
        total_rsam_opportunity = ("rsam_score",            "sum"),
        median_rsam_score      = ("rsam_score",            "median"),
        avg_propensity         = ("propensity_score",      "mean"),
        avg_raw_headroom       = ("raw_headroom",          "mean"),
        avg_health_weight      = ("health_weight",         "mean"),
        total_expected_rev_90d = ("expected_revenue_90d",  "sum"),
        strategic_accounts     = ("opportunity_tier",
                                  lambda x: (x == "Strategic").sum()),
        high_accounts          = ("opportunity_tier",
                                  lambda x: (x == "High").sum()),
    ).round(2).reset_index()

    summary = summary.sort_values("total_rsam_opportunity", ascending=False)
    summary["opportunity_share"] = (
        summary["total_rsam_opportunity"] / summary["total_rsam_opportunity"].sum()
    ).round(4)

    summary.to_csv(OUTPUT_DIR / "headroom_by_segment.csv", index=False)
    print(f"✓ outputs/headroom_by_segment.csv saved")
    return summary


# ── 4. Waterfall chart ────────────────────────────────────────────────────────

def plot_waterfall(summary: pd.DataFrame) -> None:
    seg_names = summary["segment_name"].tolist()
    values    = summary["total_rsam_opportunity"].tolist()
    colors    = [SEGMENT_COLORS.get(s, "#888780") for s in seg_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#F8F8F6")

    bars = ax.bar(seg_names, values, color=colors, edgecolor="none", width=0.55)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"${val/1e6:.1f}M",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2C2C2A"
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.0f}M"))
    ax.set_ylabel("Total rSAM Opportunity", fontsize=10)
    ax.set_title("Revenue Headroom by Customer Segment", fontsize=13, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "headroom_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ outputs/headroom_waterfall.png saved")


# ── 5. Propensity vs headroom scatter ─────────────────────────────────────────

def plot_scatter(df: pd.DataFrame) -> None:
    sample = df.sample(n=min(3000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#F8F8F6")

    for seg_name, color in SEGMENT_COLORS.items():
        mask = sample["segment_name"] == seg_name
        ax.scatter(
            sample.loc[mask, "propensity_score"],
            sample.loc[mask, "raw_headroom"],
            c=color, alpha=0.45, s=14, linewidths=0, label=seg_name
        )

    # quadrant lines
    ax.axvline(x=0.55, color="#888780", lw=1, linestyle="--", alpha=0.6)
    ax.axhline(y=df["raw_headroom"].median(), color="#888780",
               lw=1, linestyle="--", alpha=0.6)

    # quadrant labels
    ymax  = sample["raw_headroom"].quantile(0.97)
    ax.text(0.57, ymax * 0.92, "Priority\nTargets",
            fontsize=8, color="#1D9E75", fontweight="bold")
    ax.text(0.05, ymax * 0.92, "High Headroom\nLow Intent",
            fontsize=8, color="#BA7517", fontweight="bold")
    ax.text(0.57, df["raw_headroom"].median() * 0.1, "Quick Wins",
            fontsize=8, color="#378ADD", fontweight="bold")
    ax.text(0.05, df["raw_headroom"].median() * 0.1, "Low Priority",
            fontsize=8, color="#888780", fontweight="bold")

    ax.set_xlabel("Propensity Score", fontsize=10)
    ax.set_ylabel("Raw Headroom ($)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    ax.set_title("Propensity vs Revenue Headroom — Account Landscape", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.85, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "headroom_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ outputs/headroom_scatter.png saved")


# ── 6. Save final headroom table ──────────────────────────────────────────────

def save_headroom(df: pd.DataFrame) -> None:
    out_cols = [
        "account_id", "segment_name", "primary_offering",
        "arr", "offering_ceiling", "raw_headroom",
        "propensity_score", "health_weight", "tenure_weight",
        "rsam_score", "expected_revenue_90d",
        "opportunity_tier", "priority_tier",
        "shap_driver_1", "shap_driver_2", "shap_driver_3",
    ]
    out = df[out_cols].sort_values("rsam_score", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    out.to_parquet("data/headroom.parquet", index=False)
    print(f"✓ data/headroom.parquet saved")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    df      = load_data()
    df      = compute_headroom(df)
    summary = build_segment_summary(df)
    out     = save_headroom(df)

    plot_waterfall(summary)
    plot_scatter(df)

    print(f"\n── Segment Opportunity Summary ──────────────────────")
    print(summary[[
        "segment_name", "account_count",
        "total_rsam_opportunity", "avg_propensity",
        "strategic_accounts", "opportunity_share"
    ]].to_string(index=False))

    print(f"\n── Top 10 Accounts by rSAM Score ────────────────────")
    print(out[[
        "rank", "account_id", "segment_name",
        "arr", "raw_headroom", "propensity_score",
        "rsam_score", "opportunity_tier"
    ]].head(10).to_string(index=False))

    return out


if __name__ == "__main__":
    run()