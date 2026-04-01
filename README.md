# Customer Propensity & Headroom Sizing Engine

> Score every account. Size every dollar. Ship every insight.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://python.org)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.10-teal?style=flat-square)](https://duckdb.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

An end-to-end revenue intelligence pipeline that replicates the core modeling stack used by enterprise B2B data science teams. Given a synthetic 10,000-account dataset mimicking Adobe's book of business, the system:

- Segments customers into named cohorts using K-Means clustering
- Scores upsell propensity per account using XGBoost + SHAP explainability
- Quantifies headroom opportunity via a rSAM-style formula
- Estimates customer lifetime value using Kaplan-Meier survival analysis
- Surfaces a ranked, seller-ready account prioritization list
- Delivers all outputs through an interactive Streamlit dashboard

**Built to directly mirror the Data Science Engineer role at Adobe** — rSAM modeling, propensity scoring, segmentation, CLV, pipeline productionization, and stakeholder-facing insights.

---

## Architecture
```
Raw Data (DuckDB)
      ↓
Feature Engineering (SQL — features.py)
      ↓
Customer Segmentation (K-Means — segmentation.py)
      ↓
Propensity Model (XGBoost + SHAP — propensity.py)
      ↓
Headroom Sizing / rSAM (headroom.py)
      ↓
CLV + Survival Analysis (Kaplan-Meier — survival.py)
      ↓
Campaign Prioritization + Dashboard (prioritize.py + app.py)
```

---

## Model Modules

### 01 — Customer Segmentation
K-Means clustering on usage signals and firmographic features into 4–5 named segments (e.g., "High-Spend Dormant", "Growth-Stage Engaged", "Churn-Risk Mid-Market"). PCA used for 2D visualization.

**Stack:** scikit-learn, DuckDB, Matplotlib

---

### 02 — Propensity Model
XGBoost binary classifier that outputs `P(upsell within 90 days)` per account. SHAP values expose the top 3 feature drivers per prediction, enabling sellers to have informed customer conversations.

**Stack:** XGBoost, SHAP, scikit-learn pipeline

---

### 03 — Headroom Sizing (rSAM)
Per-account and per-segment headroom calculation:
```
headroom_score = (max_potential_ARR − current_ARR) × propensity_score
```

Aggregated by segment and offering to produce a pipeline opportunity view for senior stakeholders.

**Stack:** DuckDB SQL, Python

---

### 04 — CLV + Survival Analysis
Kaplan-Meier survival curves by customer segment. Estimated months-to-churn feeds into lifetime value scoring and influences campaign prioritization weights.

**Stack:** lifelines, Matplotlib

---

### 05 — Campaign Prioritization
Ranked account list sorted by composite score `(propensity × headroom × survival_months)` with top SHAP drivers surfaced per account. Automated model refresh script for weekly pipeline runs.

**Stack:** pandas, Streamlit, Makefile / cron

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | DuckDB · pandas · NumPy |
| ML | scikit-learn · XGBoost · SHAP · lifelines |
| Dashboard | Streamlit · Plotly · Matplotlib |
| Dev tooling | pytest · MLflow · Docker · Makefile |
| CI | GitHub Actions (lint → test → build) |

---

## Repo Structure
```
adobe-propensity-rsam/
├── data/
│   ├── generate_synthetic.py     # 10K synthetic B2B account dataset
│   └── schema.sql
├── src/
│   ├── features.py               # DuckDB SQL feature engineering
│   ├── segmentation.py           # K-Means + PCA visualization
│   ├── propensity.py             # XGBoost classifier + SHAP
│   ├── headroom.py               # rSAM scoring logic
│   ├── survival.py               # Kaplan-Meier CLV
│   └── prioritize.py             # Ranked account list output
├── dashboard/
│   └── app.py                    # Streamlit stakeholder dashboard
├── tests/
│   └── test_*.py
├── notebooks/
│   └── 01_eda.ipynb
├── .github/
│   └── workflows/ci.yml
├── Makefile
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quickstart
```bash
# Clone and set up environment
git clone https://github.com/sajanshergill/adobe-propensity-rsam
cd adobe-propensity-rsam
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic dataset
python data/generate_synthetic.py

# Run full pipeline
make pipeline

# Launch Streamlit dashboard
streamlit run dashboard/app.py

# Run tests
pytest tests/ -v
```

---

## Key Outputs

| File | Description |
|---|---|
| `segment_profiles.csv` | Named segments with ARR, usage, and churn risk summaries |
| `propensity_scores.csv` | Per-account upsell probability + top-3 SHAP drivers |
| `headroom_by_segment.csv` | rSAM headroom ($ARR) per segment and offering |
| `prioritized_accounts.csv` | Ranked accounts by composite score — seller-ready |

---

## JD Coverage Map

| Adobe JD Requirement | Implementation |
|---|---|
| rSAM / headroom sizing | `headroom.py` — propensity × ARR gap formula |
| Propensity modeling | XGBoost + SHAP in `propensity.py` |
| Customer segmentation | K-Means + PCA in `segmentation.py` |
| CLV + survival analysis | Kaplan-Meier by segment in `survival.py` |
| SQL + Databricks-style queries | DuckDB SQL throughout `features.py` |
| Productionize pipelines | Makefile + Docker + GitHub Actions CI |
| Communicate insights to stakeholders | Streamlit dashboard with plain-English cards |
| Model refresh automation | Makefile pipeline target + cron-ready script |

---

## Author

**Sajan Singh Shergill**
MS Data Science — Pace University (May 2026)
[linkedin.com/in/sajanshergill](https://linkedin.com/in/sajanshergill) · [sajansshergill.github.io](https://sajansshergill.github.io)
