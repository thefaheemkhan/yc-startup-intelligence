"""Pipeline stages. Each stage reads the previous stage's file, so stages can run independently."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime

import pandas as pd

from src import config
from src.cleaning.clean import clean_companies
from src.cleaning.quality import data_quality_report, write_quality_report
from src.cleaning.validate import validate_raw
from src.database.load import load_database
from src.features.build import build_features
from src.ingestion.fetch import fetch_raw

log = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")


def _load_raw() -> list[dict]:
    if not config.RAW_COMPANIES.exists():
        raise FileNotFoundError(f"{config.RAW_COMPANIES} not found. Run scripts/fetch_data.py first.")
    return json.loads(config.RAW_COMPANIES.read_text())


def as_of_date() -> date:
    """Analysis reference date = retrieval date of the raw data (keeps reruns deterministic)."""
    if config.FETCH_MANIFEST.exists():
        ts = json.loads(config.FETCH_MANIFEST.read_text())["retrieved_at_utc"]
        return datetime.fromisoformat(ts).date()
    return date.today()


def run_fetch(force: bool = False) -> None:
    fetch_raw(force=force)


def run_validate() -> dict:
    result = validate_raw(_load_raw())
    for w in result["warnings"]:
        log.warning("validation: %s", w)
    if not result["ok"]:
        raise ValueError(f"Raw data failed validation: {result['errors']}")
    log.info("Raw data valid (%d records, %d warnings)", result["n_records"], len(result["warnings"]))
    return result


def run_clean() -> pd.DataFrame:
    records = _load_raw()
    df, steps = clean_companies(records, as_of=as_of_date())
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.CLEAN_COMPANIES, index=False)
    write_quality_report(data_quality_report(records, df, steps), steps)
    return df


def run_features() -> pd.DataFrame:
    df = pd.read_parquet(config.CLEAN_COMPANIES)
    out = build_features(df, as_of=as_of_date())
    out.to_parquet(config.FEATURES_COMPANIES, index=False)
    manifest = json.loads(config.FETCH_MANIFEST.read_text()) if config.FETCH_MANIFEST.exists() else {}
    config.DATASET_META.write_text(json.dumps({
        "dataset_version": manifest.get("sha256", "unknown")[:12], "as_of": as_of_date().isoformat(),
        "retrieved_at_utc": manifest.get("retrieved_at_utc"), "source_url": manifest.get("url"),
        "n_companies": len(out)}, indent=2))
    return out


def run_load() -> dict[str, int]:
    return load_database(pd.read_parquet(config.FEATURES_COMPANIES))


def run_all(fetch: bool = True, force: bool = False) -> None:
    if fetch:
        run_fetch(force=force)
    run_validate()
    run_clean()
    run_features()
    log.info("Database row counts: %s", run_load())
