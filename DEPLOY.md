# Deploy the Streamlit dashboard (Streamlit Community Cloud)

## Prerequisites

- Repository pushed to **GitHub** (this repo).
- Pipeline outputs committed under **`data/`** (`prioritized_accounts.parquet`, `headroom.parquet`, `survival.parquet`, `features.parquet`, `segments.parquet`).  
  Regenerate locally with: `make pipeline`

## Steps (Streamlit Community Cloud)

1. Sign in at [https://share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **New app** → pick this repository and branch (e.g. `main`).
3. **Main file path:** `dashboard/app.py`
4. **App URL (optional):** choose a subdomain.
5. **Python dependencies:** leave default **`requirements.txt`** at the repo root (or point to `dashboard/requirements.txt` — same packages).
6. Deploy. Wait for the build; open the app URL.

## Configuration on the server

- **Theme / UI:** `.streamlit/config.toml` at the repo root is picked up automatically.
- **Working directory:** Cloud uses the **repository root**, so the app resolves **`data/`** via `dashboard/app.py` → repo root (see `_DATA_DIR` in `app.py`).

## Updating the app

Push to `main` (or your connected branch); Cloud redeploys on new commits.

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| `ModuleNotFoundError` | `requirements.txt` committed; redeploy. |
| “Pipeline outputs not found” | All required `data/*.parquet` files exist on the branch you deploy. Run `make pipeline` and commit `data/`. |
| Wrong Python version | `runtime.txt` in repo root (e.g. `python-3.11`). |
