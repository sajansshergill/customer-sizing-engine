# data/generate_synthetic.py
"""
Generates a synthetic 10,000-account B2B SaaS dataset
mimicking Adobe's enterprise book of business.

Outputs:
  data/accounts.parquet   — raw account table
  data/events.parquet     — monthly usage events per account
"""

import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

SEED = 42
N_ACCOUNTS = 10_000
rng = np.random.default_rng(SEED)

# ── 1. Account master ────────────────────────────────────────────────────────

SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Startup"]
INDUSTRIES = ["Financial Services", "Healthcare", "Retail", "Media", "Tech", "Education", "Manufacturing"]
OFFERINGS = ["Creative Cloud", "Experience Cloud", "Document Cloud", "Acrobat"]
REGIONS = ["NA", "EMEA", "APAC", "LATAM"]
CHANNELS = ["Direct", "Partner", "Self-Serve", "Marketplace"]

segment_weights    = [0.15, 0.30, 0.35, 0.20]
segment_arr_params = {          # (mean, std) log-normal ARR by segment
    "Enterprise":  (11.5, 0.6),
    "Mid-Market":  (10.2, 0.5),
    "SMB":         (8.8,  0.6),
    "Startup":     (7.5,  0.7),
}

account_ids = [f"ACC-{i:05d}" for i in range(1, N_ACCOUNTS + 1)]
segments    = rng.choice(SEGMENTS, size=N_ACCOUNTS, p=segment_weights)

arr = np.array([
    np.exp(rng.normal(*segment_arr_params[s]))
    for s in segments
])

tenure_months = rng.integers(1, 73, size=N_ACCOUNTS)           # 1–72 months
num_products  = rng.integers(1, 5,  size=N_ACCOUNTS)
num_users     = (arr / 500).astype(int) + rng.integers(1, 50, size=N_ACCOUNTS)
num_users     = np.clip(num_users, 1, 5000)

accounts = pd.DataFrame({
    "account_id":     account_ids,
    "segment":        segments,
    "industry":       rng.choice(INDUSTRIES, size=N_ACCOUNTS),
    "region":         rng.choice(REGIONS,    size=N_ACCOUNTS),
    "channel":        rng.choice(CHANNELS,   size=N_ACCOUNTS),
    "primary_offering": rng.choice(OFFERINGS, size=N_ACCOUNTS),
    "arr":            arr.round(2),
    "num_products":   num_products,
    "num_users":      num_users,
    "tenure_months":  tenure_months,
    "contract_end_days": rng.integers(30, 365, size=N_ACCOUNTS),
    "support_tickets_l90d": rng.poisson(lam=2.5, size=N_ACCOUNTS),
    "nps_score":      rng.integers(0, 11, size=N_ACCOUNTS),
    "executive_sponsor": rng.choice([True, False], size=N_ACCOUNTS, p=[0.3, 0.7]),
    "qbr_completed":     rng.choice([True, False], size=N_ACCOUNTS, p=[0.4, 0.6]),
})

# ── 2. Derived signals ───────────────────────────────────────────────────────

# Max potential ARR — segment ceiling used in headroom sizing
segment_arr_ceiling = {
    "Enterprise": 500_000,
    "Mid-Market": 150_000,
    "SMB":        40_000,
    "Startup":    20_000,
}
accounts["max_potential_arr"] = accounts["segment"].map(segment_arr_ceiling)
accounts["arr_headroom"]      = (accounts["max_potential_arr"] - accounts["arr"]).clip(lower=0)
accounts["arr_utilization"]   = (accounts["arr"] / accounts["max_potential_arr"]).round(4)

# ── 3. Upsell label (ground truth) ──────────────────────────────────────────
# P(upsell) driven by real signals — not random

def upsell_probability(row):
    p = 0.10
    if row["segment"] == "Enterprise":   p += 0.15
    if row["segment"] == "Mid-Market":   p += 0.10
    if row["arr_utilization"] < 0.4:     p += 0.12   # low utilization = headroom
    if row["num_products"] < 3:          p += 0.08
    if row["tenure_months"] > 24:        p += 0.06
    if row["nps_score"] >= 8:            p += 0.07
    if row["executive_sponsor"]:         p += 0.05
    if row["qbr_completed"]:             p += 0.04
    if row["support_tickets_l90d"] > 5:  p -= 0.05   # churn risk
    if row["contract_end_days"] < 90:    p += 0.06   # renewal window
    return min(p, 0.95)

accounts["upsell_prob_true"] = accounts.apply(upsell_probability, axis=1)
accounts["upsell_label"]     = (
    rng.uniform(size=N_ACCOUNTS) < accounts["upsell_prob_true"]
).astype(int)

# ── 4. Monthly usage events ──────────────────────────────────────────────────

event_rows = []
for _, acc in accounts.iterrows():
    base_logins = int(acc["num_users"] * rng.uniform(0.3, 1.0))
    for month_offset in range(min(int(acc["tenure_months"]), 12)):
        trend = 1 + (0.02 * month_offset)                          # slight growth trend
        logins      = int(base_logins * trend * rng.uniform(0.7, 1.3))
        api_calls   = int(logins * rng.uniform(5, 50))
        storage_gb  = round(rng.uniform(1, 500), 2)
        feature_adoption = round(rng.uniform(0.1, 1.0), 3)
        event_rows.append({
            "account_id":       acc["account_id"],
            "month_offset":     month_offset,
            "logins":           logins,
            "api_calls":        api_calls,
            "storage_gb":       storage_gb,
            "feature_adoption": feature_adoption,
        })

events = pd.DataFrame(event_rows)

# ── 5. Churn / survival data ─────────────────────────────────────────────────
# For Kaplan-Meier: observed duration + event flag

accounts["churned"] = (
    (accounts["nps_score"] <= 4) &
    (rng.uniform(size=N_ACCOUNTS) < 0.35)
).astype(int)

accounts["observed_months"] = np.where(
    accounts["churned"] == 1,
    (accounts["tenure_months"] * rng.uniform(0.3, 0.9, size=N_ACCOUNTS)).astype(int).clip(1),
    accounts["tenure_months"],
)

# ── 6. Save ──────────────────────────────────────────────────────────────────

Path("data").mkdir(exist_ok=True)

accounts.to_parquet("data/accounts.parquet", index=False)
events.to_parquet("data/events.parquet",     index=False)

# Also load into DuckDB for downstream SQL modules
con = duckdb.connect("data/adobe_rsam.duckdb")
con.execute("CREATE OR REPLACE TABLE accounts AS SELECT * FROM read_parquet('data/accounts.parquet')")
con.execute("CREATE OR REPLACE TABLE events   AS SELECT * FROM read_parquet('data/events.parquet')")
con.close()

print(f"✓ accounts.parquet  — {len(accounts):,} rows")
print(f"✓ events.parquet    — {len(events):,} rows")
print(f"✓ adobe_rsam.duckdb — tables: accounts, events")
print(f"\nUpsell rate: {accounts['upsell_label'].mean():.1%}")
print(f"Churn rate:  {accounts['churned'].mean():.1%}")
print(accounts[["segment","arr","arr_utilization","upsell_label"]].groupby("segment").agg({
    "arr":             "median",
    "arr_utilization": "mean",
    "upsell_label":    "mean",
}).round(3).to_string())