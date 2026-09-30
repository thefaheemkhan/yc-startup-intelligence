"""Pure analytics functions used by the dashboard. No Streamlit imports, so they are unit-testable."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Filters:
    """Empty tuple = no restriction on that dimension."""
    years: tuple[int, int] = (2005, 2100)
    industries: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    countries: tuple[str, ...] = ()
    ai: str = "All"                    # All | AI | Non-AI
    ai_definition: str = "broad"       # broad (tags + keywords, derived) | tag (YC tags only)
    include_incomplete: bool = False   # in-progress / future batches
    team_range: tuple[int, int] | None = None
    include_unknown_team: bool = True


def ai_column(definition: str) -> str:
    return "is_ai" if definition == "broad" else "ai_tag"


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """Apply sidebar filters. Rows with an unknown batch year are excluded by the year filter."""
    m = df["batch_year"].between(f.years[0], f.years[1]).fillna(False).astype(bool)
    if not f.include_incomplete:
        m &= df["batch_status"].isin(["complete", "unknown"])
    if f.industries:
        m &= df["industry"].isin(f.industries)
    if f.statuses:
        m &= df["status"].isin(f.statuses)
    if f.countries:
        m &= df["primary_country"].isin(f.countries)
    if f.ai != "All":
        flag = df[ai_column(f.ai_definition)].astype(bool)
        m &= flag if f.ai == "AI" else ~flag
    if f.team_range is not None:
        in_range = df["team_size"].between(*f.team_range).fillna(False).astype(bool)
        m &= in_range | (df["team_size"].isna() & f.include_unknown_team)
    return df[m]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def kpis(d: pd.DataFrame, ai_col: str = "is_ai") -> dict[str, float]:
    n = len(d)
    if n == 0:
        return {}
    return {
        "total": n,
        "operating_pct": 100 * d["is_operating"].mean(),
        "acquired": int((d["status"] == "Acquired").sum()),
        "industries": d.loc[d["industry"] != "Unspecified", "industry"].nunique(),
        "countries": d["primary_country"].nunique(),
        "cities": d["primary_city"].nunique(),
        "ai_share_pct": 100 * d[ai_col].mean(),
        "median_team_size": float(d["team_size"].median()) if d["team_size"].notna().any() else float("nan"),
    }


def per_batch(d: pd.DataFrame) -> pd.DataFrame:
    x = d.dropna(subset=["batch_order"])
    out = x.groupby(["batch", "batch_order", "batch_season"]).size().reset_index(name="n")
    return out.sort_values("batch_order").reset_index(drop=True)


def year_status(d: pd.DataFrame) -> pd.DataFrame:
    x = d.dropna(subset=["batch_year"]).assign(year=lambda t: t["batch_year"].astype(int))
    return x.groupby(["year", "status"]).size().reset_index(name="n")


def top_counts(d: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    vc = d[col].dropna().value_counts().head(n)
    return vc.rename_axis(col).reset_index(name="n")


def crosstab_share(d: pd.DataFrame, row: str, col: str, top_n: int = 10, normalize: str = "columns") -> pd.DataFrame:
    """Percent matrix. normalize='columns': each column sums to 100 (e.g. industry mix within a year)."""
    x = d.dropna(subset=[row, col])
    if col == "batch_year":
        x = x.assign(batch_year=x["batch_year"].astype(int))
    ct = pd.crosstab(x[row], x[col])
    share = ct.div(ct.sum(axis=0 if normalize == "columns" else 1), axis=1 if normalize == "columns" else 0) * 100
    top = x[row].value_counts().head(top_n).index
    return share.loc[[g for g in top if g in share.index]]


def rate_by_group(d: pd.DataFrame, group: str, flag: str, min_n: int = 30) -> pd.DataFrame:
    """Share of rows where boolean column `flag` is True, per group, with Wilson 95% CI."""
    x = d.dropna(subset=[group])
    g = x.groupby(group)[flag].agg(k="sum", n="size").reset_index()
    g = g[g["n"] >= min_n].copy()
    ci = [wilson_ci(int(k), int(n)) for k, n in zip(g["k"], g["n"])]
    g["rate"] = 100 * g["k"] / g["n"]
    g["lo"] = [100 * c[0] for c in ci]
    g["hi"] = [100 * c[1] for c in ci]
    return g.sort_values("rate", ascending=False).reset_index(drop=True)


def ai_share_by_year(d: pd.DataFrame) -> pd.DataFrame:
    x = d.dropna(subset=["batch_year"]).assign(year=lambda t: t["batch_year"].astype(int))
    g = x.groupby("year").agg(n=("company_id", "size"), tag_based=("ai_tag", "mean"), broad=("is_ai", "mean")).reset_index()
    g[["tag_based", "broad"]] *= 100
    return g


def momentum(d: pd.DataFrame, level: str = "industry", window: int = 3, min_recent: int = 20) -> pd.DataFrame:
    """Industry Growth Momentum: change in a group's share of companies between the last `window`
    batch years in the data and the `window` years before. Descriptive only; NOT a success predictor."""
    cols = [level, "recent_n", "prior_n", "recent_share_pct", "prior_share_pct", "share_change_pp", "count_growth_pct"]
    x = d.dropna(subset=[level, "batch_year"]).assign(y=lambda t: t["batch_year"].astype(int))
    years = sorted(x["y"].unique())
    if len(years) < 2 * window:
        return pd.DataFrame(columns=cols)
    recent, prior = years[-window:], years[-2 * window:-window]
    r = x[x["y"].isin(recent)][level].value_counts()
    p = x[x["y"].isin(prior)][level].value_counts()
    t = pd.DataFrame({"recent_n": r, "prior_n": p}).fillna(0).astype(int)
    t["recent_share_pct"] = 100 * t["recent_n"] / t["recent_n"].sum()
    t["prior_share_pct"] = 100 * t["prior_n"] / t["prior_n"].sum()
    t["share_change_pp"] = t["recent_share_pct"] - t["prior_share_pct"]
    t["count_growth_pct"] = np.where(t["prior_n"] > 0, 100 * (t["recent_n"] / t["prior_n"].replace(0, np.nan) - 1), np.nan)
    t = t[t["recent_n"] >= min_recent].rename_axis(level).reset_index()
    t.attrs["recent_years"], t.attrs["prior_years"] = [int(y) for y in recent], [int(y) for y in prior]
    return t.sort_values("share_change_pp", ascending=False)[cols].reset_index(drop=True)


def insights(d: pd.DataFrame, ai_col: str = "is_ai") -> list[str]:
    """Findings computed from the CURRENT filtered data. Nothing is hard-coded."""
    if d.empty:
        return []
    out: list[str] = []
    pb = per_batch(d)
    if not pb.empty:
        top = pb.loc[pb["n"].idxmax()]
        out.append(f"Largest batch in this view: **{top['batch']}** with {int(top['n']):,} companies.")
    ind = top_counts(d[d["industry"] != "Unspecified"], "industry", 1)
    if not ind.empty:
        out.append(f"Most common industry: **{ind.iloc[0]['industry']}** ({100 * ind.iloc[0]['n'] / len(d):.1f}% of companies).")
    ctry = top_counts(d, "primary_country", 1)
    known = d["primary_country"].notna().sum()
    if not ctry.empty and known:
        out.append(f"**{ctry.iloc[0]['primary_country']}** hosts {100 * ctry.iloc[0]['n'] / known:.1f}% of companies with a known physical location.")
    out.append(f"**{100 * d['is_operating'].mean():.1f}%** are operating (Active or Public); {100 * (d['status'] == 'Acquired').mean():.1f}% were acquired.")
    a = d.dropna(subset=["batch_year"]).assign(y=lambda t: t["batch_year"].astype(int)).groupby("y")[ai_col].mean() * 100
    if len(a) >= 2:
        y1, y0 = a.index.max(), a.index.max() - 3
        if y0 in a.index:
            out.append(f"AI share (selected definition): {a[y0]:.1f}% in {y0} vs {a[y1]:.1f}% in {y1}.")
    return out
