# src/features.py
"""
Feature engineering via DuckDB SQL.
Reads raw tables from adobe_rsam.duckdb and produces
a model-ready feature matrix saved as:
  data/features.parquet
"""

import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = "data/adobe_rsam.duckdb"


def build_features(db_path: str = DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(db_path)

    # ── 1. Usage aggregates from events table ────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE usage_agg AS
        SELECT
            account_id,

            -- volume signals
            SUM(logins)                         AS total_logins,
            AVG(logins)                         AS avg_monthly_logins,
            MAX(logins)                         AS peak_logins,
            STDDEV(logins)                      AS logins_stddev,

            -- API engagement
            SUM(api_calls)                      AS total_api_calls,
            AVG(api_calls)                      AS avg_monthly_api_calls,

            -- storage
            MAX(storage_gb)                     AS max_storage_gb,
            AVG(storage_gb)                     AS avg_storage_gb,

            -- feature adoption
            AVG(feature_adoption)               AS avg_feature_adoption,
            MAX(feature_adoption)               AS peak_feature_adoption,

            -- growth trend (last month vs first month logins)
            LAST(logins ORDER BY month_offset)
              - FIRST(logins ORDER BY month_offset) AS login_growth,

            COUNT(*)                            AS months_active

        FROM events
        GROUP BY account_id
    """)

    # ── 2. Join accounts + usage, engineer final features ────────────────────
    features_df = con.execute("""
        SELECT
            -- identifiers (drop before training)
            a.account_id,
            a.upsell_label,
            a.churned,
            a.observed_months,

            -- ── ARR & headroom signals ──────────────────────────────────────
            a.arr,
            a.arr_headroom,
            a.arr_utilization,
            a.max_potential_arr,
            LOG(a.arr + 1)                          AS log_arr,

            -- ── firmographic features ────────────────────────────────────────
            a.num_products,
            a.num_users,
            a.tenure_months,
            a.contract_end_days,
            a.support_tickets_l90d,
            a.nps_score,
            CAST(a.executive_sponsor AS INT)         AS executive_sponsor,
            CAST(a.qbr_completed AS INT)             AS qbr_completed,

            -- ── segment / categorical (one-hot) ─────────────────────────────
            CASE WHEN a.segment = 'Enterprise'  THEN 1 ELSE 0 END AS seg_enterprise,
            CASE WHEN a.segment = 'Mid-Market'  THEN 1 ELSE 0 END AS seg_midmarket,
            CASE WHEN a.segment = 'SMB'         THEN 1 ELSE 0 END AS seg_smb,
            CASE WHEN a.segment = 'Startup'     THEN 1 ELSE 0 END AS seg_startup,

            CASE WHEN a.region = 'NA'           THEN 1 ELSE 0 END AS region_na,
            CASE WHEN a.region = 'EMEA'         THEN 1 ELSE 0 END AS region_emea,
            CASE WHEN a.region = 'APAC'         THEN 1 ELSE 0 END AS region_apac,
            CASE WHEN a.region = 'LATAM'        THEN 1 ELSE 0 END AS region_latam,

            CASE WHEN a.channel = 'Direct'      THEN 1 ELSE 0 END AS channel_direct,
            CASE WHEN a.channel = 'Partner'     THEN 1 ELSE 0 END AS channel_partner,
            CASE WHEN a.channel = 'Self-Serve'  THEN 1 ELSE 0 END AS channel_selfserve,

            CASE WHEN a.primary_offering = 'Creative Cloud'    THEN 1 ELSE 0 END AS offering_cc,
            CASE WHEN a.primary_offering = 'Experience Cloud'  THEN 1 ELSE 0 END AS offering_ec,
            CASE WHEN a.primary_offering = 'Document Cloud'    THEN 1 ELSE 0 END AS offering_dc,
            CASE WHEN a.primary_offering = 'Acrobat'          THEN 1 ELSE 0 END AS offering_acrobat,

            -- ── usage signals (from aggregated events) ───────────────────────
            COALESCE(u.total_logins,           0) AS total_logins,
            COALESCE(u.avg_monthly_logins,     0) AS avg_monthly_logins,
            COALESCE(u.peak_logins,            0) AS peak_logins,
            COALESCE(u.logins_stddev,          0) AS logins_stddev,
            COALESCE(u.total_api_calls,        0) AS total_api_calls,
            COALESCE(u.avg_monthly_api_calls,  0) AS avg_monthly_api_calls,
            COALESCE(u.max_storage_gb,         0) AS max_storage_gb,
            COALESCE(u.avg_storage_gb,         0) AS avg_storage_gb,
            COALESCE(u.avg_feature_adoption,   0) AS avg_feature_adoption,
            COALESCE(u.peak_feature_adoption,  0) AS peak_feature_adoption,
            COALESCE(u.login_growth,           0) AS login_growth,
            COALESCE(u.months_active,          0) AS months_active,

            -- ── engineered ratio features ────────────────────────────────────
            CASE
                WHEN COALESCE(u.avg_monthly_logins, 0) = 0 THEN 0
                ELSE a.num_users / u.avg_monthly_logins
            END                                     AS users_per_login,

            CASE
                WHEN a.num_users = 0 THEN 0
                ELSE COALESCE(u.avg_monthly_logins, 0) / a.num_users
            END                                     AS logins_per_user,

            CASE
                WHEN a.tenure_months = 0 THEN 0
                ELSE a.arr / a.tenure_months
            END                                     AS arr_per_tenure_month,

            CASE
                WHEN a.contract_end_days <= 90  THEN 1 ELSE 0
            END                                     AS renewal_window_flag,

            CASE
                WHEN a.nps_score >= 9 THEN 'promoter'
                WHEN a.nps_score >= 7 THEN 'passive'
                ELSE 'detractor'
            END                                     AS nps_category,

            CASE
                WHEN a.nps_score >= 9 THEN 1 ELSE 0
            END                                     AS is_promoter,

            CASE
                WHEN a.nps_score <= 4 THEN 1 ELSE 0
            END                                     AS is_detractor,

            -- ── health score (composite, 0–100) ─────────────────────────────
            ROUND(
                (a.nps_score / 10.0)               * 30 +   -- NPS weight
                COALESCE(u.avg_feature_adoption, 0) * 25 +   -- feature adoption
                (1 - a.arr_utilization)             * 20 +   -- headroom proxy
                CASE WHEN a.qbr_completed   THEN 1 ELSE 0 END * 15 +
                CASE WHEN a.executive_sponsor THEN 1 ELSE 0 END * 10
            , 2)                                    AS health_score

        FROM accounts a
        LEFT JOIN usage_agg u USING (account_id)
    """).df()

    con.close()
    return features_df


def save_features(df: pd.DataFrame, path: str = "data/features.parquet") -> None:
    Path("data").mkdir(exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✓ features.parquet — {len(df):,} rows × {len(df.columns)} columns")
    print(f"\nFeature groups:")
    print(f"  ARR / headroom signals  : arr, log_arr, arr_headroom, arr_utilization")
    print(f"  Firmographic            : num_products, num_users, tenure_months, ...")
    print(f"  Segment one-hots        : seg_enterprise, seg_midmarket, seg_smb, seg_startup")
    print(f"  Region / channel        : region_*, channel_*")
    print(f"  Usage (from events)     : total_logins, avg_monthly_api_calls, ...")
    print(f"  Engineered ratios       : logins_per_user, arr_per_tenure_month, ...")
    print(f"  Composite               : health_score (0–100)")
    print(f"\nLabel distribution:")
    print(df["upsell_label"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    df = build_features()
    save_features(df)
    print(df[["account_id", "arr", "arr_utilization", "health_score",
              "avg_feature_adoption", "upsell_label"]].head(10).to_string(index=False))