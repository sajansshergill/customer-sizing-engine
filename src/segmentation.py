# src/segmentation.py
"""
Customer segmentation using K-Means clustering.
Reads data/features.parquet, fits K-Means (k=5),
assigns named segment labels, and outputs:
  data/segments.parquet       — account_id + cluster metadata
  outputs/segment_profiles.csv — summary stats per segment
  outputs/segmentation_pca.png — 2D PCA scatter plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

SEED       = 42
K          = 5
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Features used for clustering (no leakage — no labels) ───────────────────
CLUSTER_FEATURES = [
    "arr",
    "arr_utilization",
    "arr_headroom",
    "num_products",
    "num_users",
    "tenure_months",
    "nps_score",
    "health_score",
    "avg_feature_adoption",
    "avg_monthly_logins",
    "total_api_calls",
    "support_tickets_l90d",
    "contract_end_days",
    "executive_sponsor",
    "qbr_completed",
    "is_promoter",
    "is_detractor",
    "logins_per_user",
    "arr_per_tenure_month",
    "renewal_window_flag",
]

# ── Segment name mapping (assigned after inspecting cluster centroids) ────────
# Order matches cluster index sorted by median ARR descending
SEGMENT_NAMES = {
    0: "High-Value Engaged",       # high ARR, high adoption, promoters
    1: "Growth-Stage Expanding",   # mid ARR, low utilization, headroom opportunity
    2: "At-Risk Dormant",          # low engagement, detractors, high tickets
    3: "Loyal Mid-Market",         # stable ARR, long tenure, passive NPS
    4: "Early-Stage Potential",    # low ARR, short tenure, high headroom ratio
}

SEGMENT_COLORS = {
    "High-Value Engaged":      "#1D9E75",
    "Growth-Stage Expanding":  "#378ADD",
    "At-Risk Dormant":         "#E24B4A",
    "Loyal Mid-Market":        "#BA7517",
    "Early-Stage Potential":   "#534AB7",
}


def load_features(path: str = "data/features.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"✓ Loaded features — {len(df):,} rows × {len(df.columns)} columns")
    return df


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> None:
    """Elbow + silhouette plot — run once to validate k=5."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels, sample_size=2000, random_state=SEED))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#F8F8F6")
    for ax in (ax1, ax2):
        ax.set_facecolor("#F8F8F6")

    ax1.plot(list(k_range), inertias, marker="o", color="#378ADD", linewidth=2)
    ax1.set_title("Elbow — Inertia vs K", fontsize=12)
    ax1.set_xlabel("K"); ax1.set_ylabel("Inertia")

    ax2.plot(list(k_range), silhouettes, marker="o", color="#1D9E75", linewidth=2)
    ax2.set_title("Silhouette Score vs K", fontsize=12)
    ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ outputs/elbow_silhouette.png saved")


def fit_kmeans(X_scaled: np.ndarray) -> KMeans:
    km = KMeans(n_clusters=K, random_state=SEED, n_init=20, max_iter=500)
    km.fit(X_scaled)
    sil = silhouette_score(X_scaled, km.labels_, sample_size=3000, random_state=SEED)
    print(f"✓ K-Means fitted  — k={K}, inertia={km.inertia_:,.0f}, silhouette={sil:.3f}")
    return km


def assign_segment_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort raw cluster indices by median ARR so naming is deterministic
    across reruns, then map to human-readable segment names.
    """
    cluster_arr_rank = (
        df.groupby("cluster")["arr"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    rank_to_name = {cluster: SEGMENT_NAMES[rank] for rank, cluster in enumerate(cluster_arr_rank)}
    df["segment_name"] = df["cluster"].map(rank_to_name)
    return df


def plot_pca(df: pd.DataFrame, X_scaled: np.ndarray) -> None:
    pca    = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X_scaled)
    var    = pca.explained_variance_ratio_

    df = df.copy()
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#F8F8F6")

    for seg_name, color in SEGMENT_COLORS.items():
        mask = df["segment_name"] == seg_name
        ax.scatter(
            df.loc[mask, "pc1"],
            df.loc[mask, "pc2"],
            c=color, alpha=0.45, s=12, linewidths=0, label=seg_name
        )

    # centroids
    for seg_name, color in SEGMENT_COLORS.items():
        mask = df["segment_name"] == seg_name
        cx, cy = df.loc[mask, "pc1"].mean(), df.loc[mask, "pc2"].mean()
        ax.scatter(cx, cy, c=color, s=180, marker="D",
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.annotate(seg_name, (cx, cy),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, fontweight="bold", color=color)

    ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)", fontsize=10)
    ax.set_title("Customer Segments — PCA Projection", fontsize=13, fontweight="bold")

    patches = [mpatches.Patch(color=c, label=n) for n, c in SEGMENT_COLORS.items()]
    ax.legend(handles=patches, fontsize=8, framealpha=0.85, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segmentation_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ outputs/segmentation_pca.png saved")


def build_segment_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profile = df.groupby("segment_name").agg(
        account_count      = ("account_id",          "count"),
        median_arr         = ("arr",                  "median"),
        median_headroom    = ("arr_headroom",          "median"),
        avg_utilization    = ("arr_utilization",       "mean"),
        avg_health_score   = ("health_score",          "mean"),
        avg_tenure_months  = ("tenure_months",         "mean"),
        avg_nps            = ("nps_score",             "mean"),
        avg_feature_adopt  = ("avg_feature_adoption",  "mean"),
        upsell_rate        = ("upsell_label",          "mean"),
        churn_rate         = ("churned",               "mean"),
    ).round(3).reset_index()

    profile = profile.sort_values("median_arr", ascending=False)
    profile.to_csv(OUTPUT_DIR / "segment_profiles.csv", index=False)
    print(f"✓ outputs/segment_profiles.csv saved")
    return profile


def run(elbow_plot: bool = False) -> pd.DataFrame:
    df   = load_features()
    X    = df[CLUSTER_FEATURES].fillna(0).values
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if elbow_plot:
        find_optimal_k(X_scaled)

    km = fit_kmeans(X_scaled)
    df["cluster"] = km.labels_
    df = assign_segment_names(df)

    plot_pca(df, X_scaled)

    # save segments
    out = df[["account_id", "cluster", "segment_name",
              "arr", "arr_headroom", "health_score", "upsell_label", "churned",
              "observed_months"]].copy()
    out.to_parquet("data/segments.parquet", index=False)
    print(f"✓ data/segments.parquet saved")

    profiles = build_segment_profiles(df)
    print(f"\nSegment profiles:")
    print(profiles[["segment_name", "account_count", "median_arr",
                     "median_headroom", "avg_health_score",
                     "upsell_rate", "churn_rate"]].to_string(index=False))
    return df


if __name__ == "__main__":
    run(elbow_plot=True)