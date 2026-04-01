# src/propensity.py
"""
Propensity model: P(upsell within 90 days) per account.
Reads data/features.parquet + data/segments.parquet
Outputs:
  data/propensity_scores.parquet     — per-account scores + SHAP
  outputs/propensity_scores.csv      — seller-ready ranked list
  outputs/shap_summary.png           — SHAP beeswarm plot
  outputs/feature_importance.png     — top-20 feature importance
  mlflow run logged under experiment: adobe-propensity
"""

import os
import numpy as np
import pandas as pd
import shap
import mlflow
import mlflow.xgboost
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    RocCurveDisplay
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Make matplotlib usable in restricted environments (no writable home cache).
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))

import matplotlib.pyplot as plt  # noqa: E402

SEED = 42

# ── Features (same as clustering minus survival/label cols) ──────────────────
DROP_COLS = [
    "account_id", "upsell_label", "churned",
    "observed_months", "nps_category",
    "cluster", "segment_name",
    "arr_headroom", "max_potential_arr",     # leakage guard — headroom is the output
]

PARAMS = {
    "n_estimators":       400,
    "max_depth":          5,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   3,
    "gamma":              0.1,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "scale_pos_weight":   1.8,      # mild correction for class imbalance
    "eval_metric":        "aucpr",
    "use_label_encoder":  False,
    "random_state":       SEED,
    "n_jobs":             -1,
}


# ── 1. Load & prep ────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    features = pd.read_parquet("data/features.parquet")
    segments = pd.read_parquet("data/segments.parquet")[
        ["account_id", "cluster", "segment_name"]
    ]
    df = features.merge(segments, on="account_id", how="left")

    y = df["upsell_label"]
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].fillna(0)

    print(f"✓ Dataset loaded — {len(X):,} rows × {len(feature_cols)} features")
    print(f"  Upsell rate: {y.mean():.1%}  |  Class ratio: {(y==0).sum()}:{(y==1).sum()}")
    return df, X, y, feature_cols


# ── 2. Cross-validation ───────────────────────────────────────────────────────

def cross_validate_model(X: pd.DataFrame, y: pd.Series) -> dict:
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model  = XGBClassifier(**PARAMS)
    scores = cross_validate(
        model, X, y, cv=skf,
        scoring=["roc_auc", "average_precision"],
        return_train_score=True,
        n_jobs=1,  # avoid joblib multiprocessing restrictions in some environments
    )
    print(f"\n── 5-Fold Cross-Validation ──────────────────────────")
    print(f"  ROC-AUC    : {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}")
    print(f"  PR-AUC     : {scores['test_average_precision'].mean():.4f} ± {scores['test_average_precision'].std():.4f}")
    print(f"  Train AUC  : {scores['train_roc_auc'].mean():.4f}  (overfit gap check)")
    return scores


# ── 3. Train final model ──────────────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
    base  = XGBClassifier(**PARAMS)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X, y)
    print(f"✓ Final model trained + isotonic calibration applied")
    return model


# ── 4. SHAP explainability ────────────────────────────────────────────────────

def compute_shap(
    model: CalibratedClassifierCV,
    df: pd.DataFrame,
    X: pd.DataFrame,
    feature_cols: list[str],
    sample_n: int = 2000,
) -> tuple[np.ndarray, pd.DataFrame]:
    # extract underlying XGBoost estimator from calibrated wrapper
    xgb_estimator = model.calibrated_classifiers_[0].estimator

    if len(X) == 0:
        raise ValueError("X is empty; cannot compute SHAP values.")

    sample_n = min(sample_n, len(X))
    sample_idx = np.random.default_rng(SEED).choice(len(X), size=sample_n, replace=False)
    X_sample = X.iloc[sample_idx].reset_index(drop=True)
    account_ids = df["account_id"].iloc[sample_idx].astype(str).to_numpy()

    explainer   = shap.TreeExplainer(xgb_estimator)
    shap_values = explainer.shap_values(X_sample)   # shape (n, features)

    # top-3 SHAP drivers per account
    abs_shap    = np.abs(shap_values)
    top3_idx    = np.argsort(abs_shap, axis=1)[:, -3:][:, ::-1]
    feature_arr = np.array(feature_cols)

    top3_df = pd.DataFrame({
        "account_id": account_ids,
        "shap_driver_1": feature_arr[top3_idx[:, 0]],
        "shap_value_1":  shap_values[np.arange(sample_n), top3_idx[:, 0]].round(4),
        "shap_driver_2": feature_arr[top3_idx[:, 1]],
        "shap_value_2":  shap_values[np.arange(sample_n), top3_idx[:, 1]].round(4),
        "shap_driver_3": feature_arr[top3_idx[:, 2]],
        "shap_value_3":  shap_values[np.arange(sample_n), top3_idx[:, 2]].round(4),
    })

    # SHAP beeswarm plot
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                      show=False, max_display=20)
    plt.title("SHAP Feature Impact — Upsell Propensity", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ outputs/shap_summary.png saved")

    return shap_values, top3_df


# ── 5. Feature importance bar chart ──────────────────────────────────────────

def plot_feature_importance(
    model: CalibratedClassifierCV,
    feature_cols: list[str],
    top_n: int = 20,
) -> None:
    xgb_est    = model.calibrated_classifiers_[0].estimator
    importances = xgb_est.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#F8F8F6")

    bars = ax.barh(fi_df["feature"], fi_df["importance"], color="#378ADD",
                   edgecolor="none", height=0.65)
    ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=3, color="#444441")
    ax.set_xlabel("Gain Importance", fontsize=10)
    ax.set_title(f"Top {top_n} Feature Importances — XGBoost", fontsize=12, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ outputs/feature_importance.png saved")


# ── 6. ROC + calibration plots ────────────────────────────────────────────────

def plot_diagnostics(model, X: pd.DataFrame, y: pd.Series) -> None:
    y_prob = model.predict_proba(X)[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#F8F8F6")
    for ax in (ax1, ax2):
        ax.set_facecolor("#F8F8F6")

    # ROC curve
    RocCurveDisplay.from_predictions(y, y_prob, ax=ax1, color="#1D9E75",
                                     name=f"XGB calibrated")
    ax1.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
    ax1.set_title("ROC Curve", fontsize=11, fontweight="bold")

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10, strategy="uniform")
    ax2.plot(prob_pred, prob_true, marker="o", color="#534AB7", linewidth=2, label="Model")
    ax2.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Fraction of positives")
    ax2.set_title("Calibration Curve", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ outputs/model_diagnostics.png saved")


# ── 7. Score all accounts ─────────────────────────────────────────────────────

def score_accounts(
    df: pd.DataFrame,
    model: CalibratedClassifierCV,
    X: pd.DataFrame,
    top3_df: pd.DataFrame,
) -> pd.DataFrame:
    probs = model.predict_proba(X)[:, 1]

    scores = df[["account_id", "segment_name", "arr", "arr_headroom",
                 "health_score", "upsell_label"]].copy()
    scores["propensity_score"]  = probs.round(4)
    scores["propensity_decile"] = pd.qcut(probs, q=10, labels=False, duplicates="drop") + 1
    scores["priority_tier"] = pd.cut(
        probs,
        bins=[0, 0.30, 0.55, 0.75, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
    )

    # merge top-3 SHAP drivers (available for sampled accounts)
    scores = scores.merge(top3_df, on="account_id", how="left")
    scores = scores.sort_values("propensity_score", ascending=False).reset_index(drop=True)
    scores["rank"] = scores.index + 1

    scores.to_parquet("data/propensity_scores.parquet", index=False)
    scores.to_csv(OUTPUT_DIR / "propensity_scores.csv", index=False)
    print(f"✓ data/propensity_scores.parquet saved")
    print(f"✓ outputs/propensity_scores.csv saved")
    return scores


# ── 8. MLflow logging ─────────────────────────────────────────────────────────

def log_to_mlflow(model, X, y, scores, cv_scores, params):
    mlflow.set_experiment("adobe-propensity")
    with mlflow.start_run(run_name="xgb-calibrated-v1"):
        mlflow.log_params(params)

        y_prob = model.predict_proba(X)[:, 1]
        mlflow.log_metrics({
            "roc_auc":          round(roc_auc_score(y, y_prob), 4),
            "pr_auc":           round(average_precision_score(y, y_prob), 4),
            "cv_roc_auc_mean":  round(cv_scores["test_roc_auc"].mean(), 4),
            "cv_pr_auc_mean":   round(cv_scores["test_average_precision"].mean(), 4),
            "critical_accounts": int((scores["priority_tier"] == "Critical").sum()),
            "high_accounts":     int((scores["priority_tier"] == "High").sum()),
        })

        mlflow.log_artifact(str(OUTPUT_DIR / "shap_summary.png"))
        mlflow.log_artifact(str(OUTPUT_DIR / "feature_importance.png"))
        mlflow.log_artifact(str(OUTPUT_DIR / "model_diagnostics.png"))
        mlflow.xgboost.log_model(
            model.calibrated_classifiers_[0].estimator,
            artifact_path="xgb_model"
        )
    print(f"✓ MLflow run logged — experiment: adobe-propensity")


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    df, X, y, feature_cols = load_data()

    cv_scores = cross_validate_model(X, y)
    model     = train_model(X, y)

    shap_values, top3_df = compute_shap(model, df, X, feature_cols)
    plot_feature_importance(model, feature_cols)
    plot_diagnostics(model, X, y)

    scores = score_accounts(df, model, X, top3_df)

    y_prob = model.predict_proba(X)[:, 1]
    print(f"\n── Final Model Metrics ──────────────────────────────")
    print(f"  ROC-AUC : {roc_auc_score(y, y_prob):.4f}")
    print(f"  PR-AUC  : {average_precision_score(y, y_prob):.4f}")
    print(f"\n── Priority Tier Distribution ───────────────────────")
    print(scores["priority_tier"].value_counts().to_string())
    print(f"\n── Top 10 Accounts by Propensity ────────────────────")
    print(scores[["rank", "account_id", "segment_name", "arr",
                  "propensity_score", "priority_tier",
                  "shap_driver_1", "shap_driver_2"]].head(10).to_string(index=False))

    log_to_mlflow(model, X, y, scores, cv_scores, PARAMS)
    return scores


if __name__ == "__main__":
    run()