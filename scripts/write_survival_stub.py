#!/usr/bin/env python3
"""Called from Makefile when src/survival.py fails (e.g. lifelines not installed)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritize import write_survival_stub_parquet  # noqa: E402

if __name__ == "__main__":
    write_survival_stub_parquet(ROOT / "data" / "survival.parquet")
