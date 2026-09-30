"""Central configuration. All paths are relative to the repo root; override via env vars."""
from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = Path(os.getenv("YC_DATA_DIR", ROOT / "data"))
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"
REPORTS_DIR: Path = ROOT / "reports"
DB_DIR: Path = ROOT / "database"

RAW_COMPANIES: Path = RAW_DIR / "yc_companies_all.json"
RAW_META: Path = RAW_DIR / "yc_meta.json"
FETCH_MANIFEST: Path = RAW_DIR / "fetch_manifest.json"

CLEAN_COMPANIES: Path = PROCESSED_DIR / "companies_clean.parquet"
FEATURES_COMPANIES: Path = PROCESSED_DIR / "companies_features.parquet"
DATASET_META: Path = PROCESSED_DIR / "dataset_meta.json"
CLEANING_LOG: Path = REPORTS_DIR / "cleaning_log.json"
QUALITY_JSON: Path = REPORTS_DIR / "data_quality.json"
QUALITY_MD: Path = REPORTS_DIR / "data_quality.md"

DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{PROCESSED_DIR / 'yc.db'}")
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# Primary: git mirror of the yc-oss/api build output (raw file). Fallback: the GitHub Pages endpoint.
SOURCE_URLS: list[str] = [
    "https://raw.githubusercontent.com/yc-oss/api/main/companies/all.json",
    "https://yc-oss.github.io/api/companies/all.json",
]
META_URLS: list[str] = [
    "https://raw.githubusercontent.com/yc-oss/api/main/meta.json",
    "https://yc-oss.github.io/api/meta.json",
]

EXPERIMENTS_LOG: Path = ROOT / "experiments" / "runs.jsonl"
FORECAST_BACKTEST: Path = REPORTS_DIR / "forecast_backtest.csv"
MODEL_RESULTS: Path = REPORTS_DIR / "model_results.json"
MODEL_PREDICTIONS: Path = REPORTS_DIR / "model_test_predictions.csv"
