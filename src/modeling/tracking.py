"""Lightweight experiment tracking: append-only JSON Lines log (one record per run)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config


def log_run(kind: str, model: str, params: dict, features: list[str], metrics: dict, notes: str = "",
            dataset_version: str = "unknown", path: Path | None = None) -> str:
    """Append one experiment record; returns its id."""
    path = Path(path or config.EXPERIMENTS_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"experiment_id": uuid.uuid4().hex[:10], "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "kind": kind, "dataset_version": dataset_version, "model": model, "params": params, "features": features,
           "metrics": {k: (None if v != v else round(float(v), 4)) for k, v in metrics.items() if isinstance(v, (int, float))},
           "notes": notes}
    with path.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec["experiment_id"]


def load_runs(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or config.EXPERIMENTS_LOG)
    if not path.exists():
        return pd.DataFrame()
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    flat = [{**{k: v for k, v in r.items() if k != "metrics"}, **{f"m_{k}": v for k, v in r["metrics"].items()}} for r in recs]
    return pd.DataFrame(flat)
