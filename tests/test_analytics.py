import numpy as np
import pandas as pd
import pytest

from src.analytics import core
from src.analytics.core import Filters


@pytest.fixture
def d():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        y = 2016 + i % 8
        rows.append({"company_id": i, "batch_year": y, "batch_order": y * 100 + 1, "batch": f"Winter {y}", "batch_season": "Winter",
                     "batch_status": "complete", "industry": "B2B" if i % 3 else ("Fintech" if y >= 2021 else "Consumer"),
                     "subindustry": "x", "status": ["Active", "Inactive", "Acquired"][i % 3], "primary_country": "United States" if i % 4 else "India",
                     "primary_city": "SF", "team_size": 1 + i % 20 if i % 10 else np.nan, "ai_tag": y >= 2022 and i % 2 == 0,
                     "is_ai": y >= 2022, "tags": ["AI"]})
    df = pd.DataFrame(rows)
    df["is_operating"] = df["status"].eq("Active")
    df["batch_year"] = df["batch_year"].astype("Int64")
    df["batch_order"] = df["batch_order"].astype("Int64")
    df["team_size"] = df["team_size"].astype("Int64")
    return df


def test_filters_combine(d):
    out = core.apply_filters(d, Filters(years=(2018, 2020), industries=("B2B",), countries=("India",)))
    assert len(out) and out["batch_year"].between(2018, 2020).all() and (out["industry"] == "B2B").all()
    assert (out["primary_country"] == "India").all()


def test_filter_ai_definitions(d):
    broad = core.apply_filters(d, Filters(ai="AI", ai_definition="broad"))
    tag = core.apply_filters(d, Filters(ai="AI", ai_definition="tag"))
    assert len(tag) < len(broad) and broad["is_ai"].all() and tag["ai_tag"].all()


def test_incomplete_batches_excluded_by_default(d):
    d2 = d.copy(); d2.loc[d2.index[:10], "batch_status"] = "in_progress"
    assert len(core.apply_filters(d2, Filters())) == len(d) - 10
    assert len(core.apply_filters(d2, Filters(include_incomplete=True))) == len(d)


def test_team_filter_unknown_toggle(d):
    kept = core.apply_filters(d, Filters(team_range=(1, 5), include_unknown_team=True))
    dropped = core.apply_filters(d, Filters(team_range=(1, 5), include_unknown_team=False))
    assert kept["team_size"].isna().any() and dropped["team_size"].notna().all() and len(kept) > len(dropped)


def test_wilson_ci_properties():
    lo, hi = core.wilson_ci(50, 100)
    assert lo < 0.5 < hi and 0.40 < lo < 0.42 and 0.58 < hi < 0.60
    assert core.wilson_ci(0, 10)[0] == 0.0 and np.isnan(core.wilson_ci(0, 0)[0])


def test_kpis_and_rates(d):
    k = core.kpis(d)
    assert k["total"] == 400 and k["countries"] == 2 and 0 < k["operating_pct"] < 100
    r = core.rate_by_group(d, "industry", "is_operating", min_n=1)
    assert (r["lo"] <= r["rate"] + 1e-9).all() and (r["rate"] <= r["hi"] + 1e-9).all()   # tolerance: float error at p=0/1
    assert core.rate_by_group(d, "industry", "is_operating", min_n=10_000).empty


def test_crosstab_columns_sum_to_100(d):
    m = core.crosstab_share(d, "industry", "batch_year", top_n=10)
    assert np.allclose(m.sum(axis=0), 100)
    m2 = core.crosstab_share(d, "primary_country", "industry", top_n=5, normalize="index")
    assert np.allclose(m2.sum(axis=1), 100)


def test_momentum_detects_rising_group(d):
    m = core.momentum(d, "industry", window=3, min_recent=1)
    assert m.attrs["recent_years"] == [2021, 2022, 2023] or len(m.attrs["recent_years"]) == 3
    assert "Fintech" in set(m["industry"]) and m.set_index("industry").loc["Fintech", "share_change_pp"] > 0
    assert core.momentum(d.head(5), window=3).empty


def test_insights_are_data_driven(d):
    assert len(core.insights(d)) >= 4 and core.insights(d.iloc[0:0]) == []
