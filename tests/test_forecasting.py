import numpy as np
import pandas as pd
import pytest

from src.forecasting import models as fm


def _series(vals, start=2006):
    return pd.Series(vals, index=range(start, start + len(vals)), dtype=float)


def test_naive_on_linear_series_has_mae_equal_slope():
    y = _series(np.arange(10, 40, 2))                   # slope 2
    bt = fm.rolling_origin(y, fm.naive, min_train=8)
    assert np.allclose((bt["actual"] - bt["pred"]).abs(), 2.0)


def test_rolling_origin_never_uses_future():
    """A model that peeks would beat a random walk; here the model sees only train and must fail on a jump."""
    y = _series([10] * 10 + [100] * 4)
    seen = []
    def spy(train, h):
        seen.append(len(train)); return np.repeat(train[-1], h)
    bt = fm.rolling_origin(y, spy, min_train=8)
    assert seen == [8, 9, 10, 11, 12, 13]                # expanding window, strictly past
    assert bt.loc[bt["year"] == 2016, "pred"].iloc[0] == 10 and bt.loc[bt["year"] == 2016, "actual"].iloc[0] == 100


def test_compare_models_table_and_metrics():
    y = _series(np.arange(20) * 3 + 50 + np.random.default_rng(1).normal(0, 2, 20))
    tbl, bts = fm.compare_models(y)
    assert set(tbl["model"]) == set(fm.MODELS) and (tbl["n_eval"] == 12).all()
    assert tbl["rmse"].is_monotonic_increasing and tbl.loc[tbl["model"] == "naive", "rmse_vs_naive"].iloc[0] == pytest.approx(1.0)
    assert (tbl["mape_pct"] >= 0).all()


def test_short_series_rejected():
    with pytest.raises(ValueError):
        fm.compare_models(_series([1, 2, 3, 4, 5]))


def test_forecast_interval_widens_and_is_clipped():
    y = _series(np.arange(20) * 3 + 5.0 + np.random.default_rng(2).normal(0, 3, 20))
    _, bts = fm.compare_models(y)
    f = fm.forecast(y, "naive", 4, bts["naive"], 0.8)
    assert list(f["year"]) == [2026, 2027, 2028, 2029]
    assert ((f["hi"] - f["lo"]).diff().dropna() >= 0).all() and (f["lo"] >= 0).all() and (f["lo"] <= f["forecast"]).all()
    wide = fm.forecast(y, "naive", 2, bts["naive"], 0.95)
    assert (wide["hi"] - wide["lo"] > (f["hi"] - f["lo"]).iloc[:2].to_numpy()).all()


def test_failed_fit_returns_nan_not_exception():
    assert np.isnan(fm._safe(lambda t, h: (_ for _ in ()).throw(ValueError("boom")), np.array([1.0, 2.0]), 1)).all()


def test_full_years_drop_incomplete_and_partial_first_year():
    df = pd.DataFrame({"batch_year": pd.array([2005, 2006, 2006, 2007, 2007, 2008], dtype="Int64"),
                       "batch_status": ["complete", "complete", "complete", "complete", "in_progress", "future"],
                       "industry": "B2B", "is_ai": False, "ai_tag": False})
    assert fm.full_years(df) == [2006]                   # 2005 < FIRST_FULL_YEAR; 2007/2008 have incomplete batches
    assert fm.annual_counts(df).tolist() == [2.0]
