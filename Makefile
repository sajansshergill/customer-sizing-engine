# Run from repo root: `make pipeline`
# Produces data/*.parquet + outputs/ for the Streamlit dashboard.

# Use your conda/venv interpreter (system `python3` often lacks deps).
PYTHON ?= python

.PHONY: pipeline

pipeline:
	@echo "==> Synthetic data"
	$(PYTHON) data/generate_synthetic.py
	@echo "==> Features"
	$(PYTHON) src/features.py
	@echo "==> Segmentation"
	$(PYTHON) src/segmentation.py
	@echo "==> Propensity"
	$(PYTHON) src/propensity.py
	@echo "==> Headroom"
	$(PYTHON) src/headroom.py
	@echo "==> Survival (Kaplan-Meier; falls back to heuristic stub if needed)"
	@$(PYTHON) src/survival.py || $(PYTHON) scripts/write_survival_stub.py
	@echo "==> Prioritize"
	$(PYTHON) src/prioritize.py
	@echo "✓ Pipeline complete. Dashboard: streamlit run dashboard/app.py"
