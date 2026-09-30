"""Cleaning: raw YC JSON -> tidy DataFrame. Every non-trivial decision is written to a cleaning log."""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Approximate batch start month: used ONLY for ordering and for flagging future/in-progress batches.
SEASONS: dict[str, tuple[str, int]] = {"Winter": ("W", 1), "Spring": ("X", 4), "Summer": ("S", 6), "Fall": ("F", 9)}
BATCH_RE = re.compile(r"^(Winter|Spring|Summer|Fall)\s+(\d{4})$")
COUNTRY_ALIASES = {"USA": "United States", "US": "United States", "U.S.": "United States",
                   "United States of America": "United States", "UK": "United Kingdom"}
KNOWN_STATUSES = {"Active", "Inactive", "Acquired", "Public"}
OPERATING_STATUSES = {"Active", "Public"}


def clean_text(value: object) -> object:
    """Collapse whitespace; None/NaN/empty -> pd.NA."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if text else pd.NA


def parse_batch(raw: object) -> dict:
    """'Winter 2012' -> season/year/code/approx start. Unparseable -> Nones."""
    m = BATCH_RE.match(str(raw).strip()) if raw is not None else None
    if not m:
        return {"batch_season": None, "batch_year": None, "batch_code": None,
                "batch_start": pd.NaT, "batch_order": None}
    season, year = m.group(1), int(m.group(2))
    code, month = SEASONS[season]
    return {"batch_season": season, "batch_year": year, "batch_code": f"{code}{str(year)[2:]}",
            "batch_start": pd.Timestamp(year=year, month=month, day=1), "batch_order": year * 100 + month}


def parse_location(entry: object) -> dict:
    """Parse one 'City, Region, Country' entry. 'Remote' is flagged, not treated as a place."""
    place = clean_text(entry)
    if place is pd.NA:
        return {"place_raw": None, "city": None, "region": None, "country": None, "is_remote": False}
    if place.lower() == "remote":
        return {"place_raw": place, "city": None, "region": None, "country": None, "is_remote": True}
    parts = [p.strip() for p in place.split(",") if p.strip()]
    country = COUNTRY_ALIASES.get(parts[-1], parts[-1])
    if len(parts) == 1:  # bare country name (e.g. 'India'): no city information
        return {"place_raw": place, "city": None, "region": None, "country": country, "is_remote": False}
    region = ", ".join(parts[1:-1]) or None
    return {"place_raw": place, "city": parts[0], "region": region, "country": country, "is_remote": False}


def split_locations(all_locations: object) -> list[str]:
    text = clean_text(all_locations)
    return [] if text is pd.NA else [e.strip() for e in text.split(";") if e.strip()]


def _iqr_upper_fence(series: pd.Series, k: float = 3.0) -> float:
    logv = np.log1p(series.dropna().astype(float))
    q1, q3 = logv.quantile([0.25, 0.75])
    return float(np.expm1(q3 + k * (q3 - q1)))


def clean_companies(records: list[dict], as_of: date | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Clean raw records. Returns (dataframe, cleaning_log). Nothing is dropped silently."""
    as_of = as_of or datetime.now(timezone.utc).date()
    steps: list[dict] = []

    def note(step: str, n: object, decision: str) -> None:
        steps.append({"step": step, "rows_affected": int(n), "decision": decision})

    raw = pd.DataFrame.from_records(records)
    n_raw = len(raw)
    raw = raw.drop_duplicates(subset="id", keep="last").reset_index(drop=True)
    note("duplicate_ids", n_raw - len(raw), "Dropped repeated company ids (kept last).")

    df = pd.DataFrame({"company_id": raw["id"].astype("int64"), "slug": raw["slug"].map(clean_text)})
    df["name"] = raw["name"].map(clean_text)
    for src, dst in [("website", "website"), ("one_liner", "one_liner"),
                     ("long_description", "long_description"), ("url", "yc_url"), ("stage", "stage")]:
        df[dst] = raw[src].map(clean_text)
    df["website"] = df["website"].astype("string").str.lower()

    # --- status / industry ---
    status = raw["status"].map(clean_text).astype("string").str.title()
    bad = ~status.isin(KNOWN_STATUSES)
    note("unknown_status", bad.sum(), "Status outside {Active, Inactive, Acquired, Public} recoded to 'Unknown'.")
    df["status"] = status.where(~bad, "Unknown")
    df["is_operating"] = df["status"].isin(OPERATING_STATUSES)  # Active + Public

    df["industry"] = raw["industry"].map(clean_text)
    note("industry_unspecified", df["industry"].eq("Unspecified").sum(),
         "Kept as its own category 'Unspecified' (source value); exclude from industry rate metrics.")
    sub = raw["subindustry"].map(clean_text).astype("string")
    df["subindustry"] = sub.str.split("->").str[-1].str.strip().where(sub.str.contains("->", na=False), other=pd.NA)

    # --- batch ---
    parsed = pd.DataFrame([parse_batch(b) for b in raw["batch"]])
    df["batch"] = raw["batch"].map(clean_text)
    df = pd.concat([df, parsed], axis=1)
    df["batch_year"] = df["batch_year"].astype("Int64")
    df["batch_order"] = df["batch_order"].astype("Int64")
    df["batch_start"] = pd.to_datetime(df["batch_start"])
    note("unparsed_batch", df["batch_season"].isna().sum(),
         "Batch not '<Season> <YYYY>' (e.g. 'Unspecified'): season/year/code null, row retained.")

    # --- team size: CURRENT headcount recorded by YC. NOT a founder count. ---
    ts = pd.to_numeric(raw["team_size"], errors="coerce")
    invalid = ts.le(0)
    note("team_size_invalid", invalid.sum(), "team_size <= 0 set to null (invalid headcount).")
    ts = ts.mask(invalid)
    fence = _iqr_upper_fence(ts)
    df["team_size"] = ts.round().astype("Int64")
    df["team_size_outlier"] = ts.gt(fence).fillna(False)
    note("team_size_outlier", df["team_size_outlier"].sum(),
         f"team_size > {fence:.0f} (3xIQR fence, log1p scale) flagged and retained; use robust statistics.")
    note("team_size_missing", df["team_size"].isna().sum(), "Missing team_size left null (not imputed).")

    # --- dates ---
    launched = pd.to_datetime(raw["launched_at"], unit="s", utc=True, errors="coerce")
    oob = launched.lt(pd.Timestamp("2005-01-01", tz="UTC")) | launched.gt(pd.Timestamp(as_of, tz="UTC") + pd.Timedelta(days=1))
    note("launched_at_out_of_range", oob.sum(), "launched_at outside [2005, as_of+1d] set to null.")
    df["launched_at"] = launched.mask(oob).dt.tz_localize(None)

    # --- flags, tags, regions ---
    df["is_hiring"] = raw["isHiring"].fillna(False).astype(bool)
    df["is_nonprofit"] = raw["nonprofit"].fillna(False).astype(bool)
    df["is_top_company"] = raw["top_company"].fillna(False).astype(bool)

    def _clean_list(v: object) -> list[str]:
        return list(dict.fromkeys(str(x).strip() for x in v if str(x).strip())) if isinstance(v, (list, tuple)) else []
    df["tags"] = raw["tags"].map(_clean_list)
    df["regions"] = raw["regions"].map(_clean_list)

    # --- locations ---
    df["all_locations"] = raw["all_locations"].map(clean_text)
    df["location_entries"] = raw["all_locations"].map(split_locations)
    parsed_locs = df["location_entries"].map(lambda es: [parse_location(e) for e in es])

    def _primary(locs: list[dict]) -> dict:
        for loc in locs:
            if not loc["is_remote"]:
                return loc
        return {"city": None, "region": None, "country": None}
    prim = parsed_locs.map(_primary)
    df["primary_city"] = prim.map(lambda d: d["city"])
    df["primary_region"] = prim.map(lambda d: d["region"])
    df["primary_country"] = prim.map(lambda d: d["country"])
    df["location_count"] = parsed_locs.map(len)
    df["is_remote"] = parsed_locs.map(lambda ls: any(l["is_remote"] for l in ls)) | df["regions"].map(lambda r: "Remote" in r)
    note("no_physical_location", df["primary_country"].isna().sum(),
         "No non-remote location (missing or remote-only): primary_city/country left null, not imputed.")

    # --- shared names: reported, NOT dropped (distinct companies can share a name) ---
    dup = df["name"].astype("string").str.lower().duplicated(keep=False).sum()
    note("shared_names", dup, "Rows sharing a case-insensitive name kept: distinct company_id/slug.")

    df = df.sort_values("company_id").reset_index(drop=True)
    log.info("Cleaned %d -> %d rows, %d columns", n_raw, len(df), df.shape[1])
    return df, steps
