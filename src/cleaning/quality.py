"""Data-quality report: what is in the data, what is missing, what was invalid."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src import config


def data_quality_report(raw_records: list[dict], df: pd.DataFrame, cleaning_log: list[dict]) -> dict[str, Any]:
    """Summarise the raw payload and the cleaned frame. All numbers are computed, none hard-coded."""
    n = len(df)
    scalar_cols = [c for c in df.columns if not isinstance(df[c].iloc[0], (list, tuple))] if n else []
    missing = {c: round(float(df[c].isna().mean() * 100), 2) for c in df.columns
               if not isinstance(df[c].iloc[0], (list, tuple))}
    # empty lists count as "missing" for list columns
    for c in df.columns:
        if n and isinstance(df[c].iloc[0], (list, tuple)):
            missing[c] = round(float(df[c].map(len).eq(0).mean() * 100), 2)
    return {
        "raw_rows": len(raw_records),
        "clean_rows": n,
        "columns": df.shape[1],
        "missing_pct": dict(sorted(missing.items(), key=lambda kv: -kv[1])),
        "duplicates": {
            "company_id": int(df["company_id"].duplicated().sum()),
            "slug": int(df["slug"].duplicated().sum()),
            "name_case_insensitive_extra_rows": int(df["name"].astype("string").str.lower().duplicated().sum()),
        },
        "unique_counts": {c: int(df[c].nunique()) for c in
                          ["batch", "industry", "subindustry", "status", "stage", "primary_country", "primary_city"]},
        "invalid_values": {s["step"]: s["rows_affected"] for s in cleaning_log},
        "date_ranges": {
            "batch_year": [int(df["batch_year"].min()), int(df["batch_year"].max())],
            "launched_at": [str(df["launched_at"].min().date()), str(df["launched_at"].max().date())],
        },
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "scalar_columns_checked": len(scalar_cols),
    }


def write_quality_report(report: dict[str, Any], cleaning_log: list[dict]) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.QUALITY_JSON.write_text(json.dumps(report, indent=2, default=str))
    config.CLEANING_LOG.write_text(json.dumps(cleaning_log, indent=2))
    lines = ["# Data Quality Report", "",
             f"- Raw rows: **{report['raw_rows']:,}**; clean rows: **{report['clean_rows']:,}**; columns: **{report['columns']}**",
             f"- Duplicates: {report['duplicates']}",
             f"- Batch years: {report['date_ranges']['batch_year']}; launched_at: {report['date_ranges']['launched_at']}", "",
             "## Cleaning decisions", "", "| Step | Rows | Decision |", "|---|---:|---|"]
    lines += [f"| {s['step']} | {s['rows_affected']:,} | {s['decision']} |" for s in cleaning_log]
    lines += ["", "## Missing values (%, top 12)", "", "| Column | Missing % |", "|---|---:|"]
    lines += [f"| {c} | {p} |" for c, p in list(report["missing_pct"].items())[:12]]
    config.QUALITY_MD.write_text("\n".join(lines) + "\n")
