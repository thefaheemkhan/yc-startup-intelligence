"""Run forecasting backtests and the status model; write reports and log every run as an experiment."""
from __future__ import annotations

import logging

import pandas as pd

from src import config
from src.forecasting import models as fm
from src.modeling import tracking
from src.modeling.status_model import run_status_experiments

log = logging.getLogger(__name__)
GENERIC_SERIES = ["All companies", "AI companies (broad, derived)", "AI companies (YC tags only)"]


def run_models() -> None:
    df = pd.read_parquet(config.FEATURES_COMPANIES)
    version = pd.read_json(config.DATASET_META, typ="series").get("dataset_version", "unknown")
    inds = [i for i, c in df["industry"].value_counts().items() if i != "Unspecified" and c >= 100]
    frames = []
    for sel in GENERIC_SERIES + inds:
        tbl, _ = fm.compare_models(fm.annual_counts(df, sel))
        frames.append(tbl.assign(series=sel))
        for r in tbl.to_dict("records"):
            tracking.log_run("forecast", r["model"], {"series": sel, "horizon": 1, "min_train": 8, "validation": "rolling-origin"}, [],
                             {k: v for k, v in r.items() if k != "model"}, "annual counts, complete years only", version)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(config.FORECAST_BACKTEST, index=False)
    res = run_status_experiments(df, dataset_version=str(version))
    log.info("Forecast series: %d | status model best-by-CV: %s | test PR-AUC table written to %s", len(frames), res["best_model_by_cv"], config.MODEL_RESULTS)
