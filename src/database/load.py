"""Load the feature table into a normalised relational DB (SQLite by default, PostgreSQL via DATABASE_URL)."""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine, text

from src import config
from src.cleaning.clean import parse_location

log = logging.getLogger(__name__)


def _run_schema(engine) -> None:
    sql = (config.DB_DIR / "schema.sql").read_text()
    with engine.begin() as conn:
        for stmt in (s.strip() for s in sql.split(";")):
            if stmt and not all(line.strip().startswith("--") or not line.strip() for line in stmt.splitlines()):
                conn.execute(text(stmt))


def _iso(s: pd.Series) -> pd.Series:
    return s.dt.strftime("%Y-%m-%d").astype("object").where(s.notna(), None)


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the wide feature frame into normalised tables with surrogate keys."""
    batches = (df[["batch", "batch_season", "batch_year", "batch_code", "batch_order", "batch_start", "batch_status"]]
               .drop_duplicates("batch").sort_values(["batch_order", "batch"], na_position="last").reset_index(drop=True))
    batches.insert(0, "batch_id", range(1, len(batches) + 1))
    batches = batches.rename(columns={"batch_season": "season", "batch_year": "year", "batch_code": "code",
                                      "batch_start": "approx_start_date"})
    batches["approx_start_date"] = _iso(pd.to_datetime(batches["approx_start_date"]))

    inds = df[["industry", "subindustry"]].drop_duplicates().sort_values(["industry", "subindustry"], na_position="first").reset_index(drop=True)
    inds.insert(0, "industry_id", range(1, len(inds) + 1))

    entries = (df[["company_id", "location_entries"]].explode("location_entries").dropna(subset=["location_entries"]))
    parsed = pd.DataFrame([parse_location(e) for e in entries["location_entries"]], index=entries.index)
    entries = pd.concat([entries[["company_id"]], parsed], axis=1)
    locations = entries.drop_duplicates("place_raw")[["place_raw", "city", "region", "country", "is_remote"]].reset_index(drop=True)
    locations.insert(0, "location_id", range(1, len(locations) + 1))
    loc_id = dict(zip(locations["place_raw"], locations["location_id"]))

    company_locations = entries[["company_id", "place_raw"]].drop_duplicates().copy()
    company_locations["location_id"] = company_locations["place_raw"].map(loc_id)
    first_physical = (entries[~entries["is_remote"]].drop_duplicates("company_id").set_index("company_id")["place_raw"].map(loc_id))
    company_locations["is_primary"] = [int(first_physical.get(c) == l) for c, l in zip(company_locations["company_id"], company_locations["location_id"])]
    company_locations = company_locations[["company_id", "location_id", "is_primary"]]

    comp = df.merge(batches[["batch_id", "batch"]], on="batch", how="left")
    comp = comp.merge(inds, on=["industry", "subindustry"], how="left")
    comp["primary_location_id"] = comp["company_id"].map(first_physical).astype("Int64")
    comp["launched_at"] = _iso(comp["launched_at"])
    bool_cols = ["is_operating", "team_size_outlier", "is_hiring", "is_nonprofit", "is_top_company",
                 "ai_tag", "ai_keyword", "is_ai", "is_remote"]
    comp[bool_cols] = comp[bool_cols].astype(int)
    companies = comp[["company_id", "slug", "name", "website", "yc_url", "one_liner", "long_description", "batch_id",
                      "industry_id", "status", "is_operating", "stage", "team_size", "team_size_outlier", "is_hiring",
                      "is_nonprofit", "is_top_company", "ai_tag", "ai_keyword", "is_ai", "is_remote", "launched_at",
                      "primary_location_id"]]

    tags = df[["company_id", "tags"]].explode("tags").dropna(subset=["tags"]).rename(columns={"tags": "tag"}).drop_duplicates()
    return {"batches": batches.drop(columns=[]), "industries": inds, "locations": locations, "companies": companies,
            "company_locations": company_locations, "company_tags": tags}


def load_database(df: pd.DataFrame, url: str | None = None) -> dict[str, int]:
    """Create schema and load all tables. Idempotent (schema drops and recreates). Returns row counts."""
    url = url or config.DATABASE_URL
    if url.startswith("sqlite:///"):
        from pathlib import Path
        Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)
    _run_schema(engine)
    tables = build_tables(df)
    # Object columns from pandas 3 'string' dtype -> plain object for driver compatibility
    counts: dict[str, int] = {}
    for name in ["batches", "industries", "locations", "companies", "company_locations", "company_tags"]:
        t = tables[name].copy()
        for c in t.columns:
            if str(t[c].dtype) in ("string", "str"):
                t[c] = t[c].astype("object").where(t[c].notna(), None)
        t.to_sql(name, engine, if_exists="append", index=False, chunksize=500)
        counts[name] = len(t)
        log.info("Loaded %-18s %6d rows", name, len(t))
    return counts
