"""Annual time-series forecasting with rolling-origin (expanding-window) evaluation.

Time is never shuffled: every forecast in the backtest is made using only earlier years.
Prediction intervals are EMPIRICAL (from backtest errors) and therefore approximate.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

ModelFn = Callable[[np.ndarray, int], np.ndarray]
FIRST_FULL_YEAR = 2006   # 2005 had a single batch (Summer 2005)
Z = {0.8: 1.2816, 0.95: 1.96}


def naive(train: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(train[-1], h)


def moving_average_3(train: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(train[-3:].mean(), h)


def exp_smoothing(train: np.ndarray, h: int) -> np.ndarray:
    return np.asarray(SimpleExpSmoothing(train, initialization_method="estimated").fit().forecast(h))


def holt_damped(train: np.ndarray, h: int) -> np.ndarray:
    m = ExponentialSmoothing(train, trend="add", damped_trend=True, initialization_method="estimated")
    return np.asarray(m.fit().forecast(h))


def arima_110(train: np.ndarray, h: int) -> np.ndarray:
    return np.asarray(ARIMA(train, order=(1, 1, 0), trend="t").fit().forecast(h))


MODELS: dict[str, ModelFn] = {"naive": naive, "moving_avg_3": moving_average_3, "exp_smoothing": exp_smoothing,
                              "holt_damped": holt_damped, "arima_110": arima_110}


def _safe(fn: ModelFn, train: np.ndarray, h: int) -> np.ndarray:
    """Run a model; on failure return NaNs (counted as a fit failure, never silently replaced)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.asarray(fn(train, h), dtype=float)
        return out if np.isfinite(out).all() else np.full(h, np.nan)
    except Exception:  # statsmodels raises many error types on tiny samples
        return np.full(h, np.nan)


# ---------------------------------------------------------------- series construction
def full_years(df: pd.DataFrame) -> list[int]:
    """Years where every batch is complete (drops in-progress/future years) and >= FIRST_FULL_YEAR."""
    x = df.dropna(subset=["batch_year"])
    incomplete = set(x.loc[x["batch_status"] != "complete", "batch_year"].astype(int))
    return sorted(y for y in set(x["batch_year"].astype(int)) - incomplete if y >= FIRST_FULL_YEAR)


def series_mask(df: pd.DataFrame, selection: str) -> pd.Series:
    if selection == "All companies":
        return pd.Series(True, index=df.index)
    if selection.startswith("AI companies (broad"):
        return df["is_ai"].astype(bool)
    if selection.startswith("AI companies (YC tags"):
        return df["ai_tag"].astype(bool)
    return df["industry"].eq(selection)


def annual_counts(df: pd.DataFrame, selection: str = "All companies") -> pd.Series:
    """Companies per batch year for complete years only (zero-filled)."""
    years = full_years(df)
    sub = df[series_mask(df, selection)].dropna(subset=["batch_year"])
    counts = sub["batch_year"].astype(int).value_counts()
    return counts.reindex(years, fill_value=0).sort_index().astype(float).rename(selection)


def partial_year_count(df: pd.DataFrame, selection: str = "All companies") -> tuple[int, int] | None:
    """(year, count so far) for the first year that contains incomplete batches, if any."""
    x = df.dropna(subset=["batch_year"])
    bad = sorted(set(x.loc[x["batch_status"].isin(["in_progress"]), "batch_year"].astype(int)))
    if not bad:
        return None
    y = bad[0]
    return y, int(series_mask(df, selection)[x.index][x["batch_year"].astype(int) == y].sum())


# ---------------------------------------------------------------- evaluation
def rolling_origin(y: pd.Series, fn: ModelFn, min_train: int = 8) -> pd.DataFrame:
    """One-step-ahead expanding-window backtest. Each prediction uses only years before it."""
    vals = y.to_numpy(float)
    rows = [{"year": int(y.index[t]), "actual": vals[t], "pred": _safe(fn, vals[:t], 1)[0]} for t in range(min_train, len(vals))]
    return pd.DataFrame(rows)


def _metrics(bt: pd.DataFrame) -> dict[str, float]:
    ok = bt.dropna(subset=["pred"])
    if ok.empty:
        return {"mae": np.nan, "rmse": np.nan, "mape_pct": np.nan, "n_eval": 0, "fit_failures": len(bt)}
    err = ok["actual"] - ok["pred"]
    pos = ok["actual"] > 0
    return {"mae": float(err.abs().mean()), "rmse": float(np.sqrt((err**2).mean())),
            "mape_pct": float((err[pos].abs() / ok.loc[pos, "actual"]).mean() * 100) if pos.any() else np.nan,
            "n_eval": int(len(ok)), "fit_failures": int(len(bt) - len(ok))}


def compare_models(y: pd.Series, min_train: int = 8) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Backtest every model. Returns (comparison table sorted by RMSE, per-model backtest frames)."""
    if len(y) < min_train + 4:
        raise ValueError(f"Series too short ({len(y)} points) for a backtest with min_train={min_train}")
    backtests = {name: rolling_origin(y, fn, min_train) for name, fn in MODELS.items()}
    tbl = pd.DataFrame({name: _metrics(bt) for name, bt in backtests.items()}).T.rename_axis("model").reset_index()
    tbl["rmse_vs_naive"] = tbl["rmse"] / tbl.loc[tbl["model"] == "naive", "rmse"].iloc[0]
    return tbl.sort_values("rmse").reset_index(drop=True), backtests


def forecast(y: pd.Series, model: str, horizon: int, backtest: pd.DataFrame, level: float = 0.8) -> pd.DataFrame:
    """Point forecast + approximate interval = point +/- z * backtest RMSE * sqrt(h). Clipped at 0."""
    point = _safe(MODELS[model], y.to_numpy(float), horizon)
    ok = backtest.dropna(subset=["pred"])
    rmse = float(np.sqrt(((ok["actual"] - ok["pred"]) ** 2).mean())) if len(ok) else np.nan
    h = np.arange(1, horizon + 1)
    half = Z[level] * rmse * np.sqrt(h)
    return pd.DataFrame({"year": int(y.index[-1]) + h, "forecast": point, "lo": np.maximum(point - half, 0), "hi": point + half,
                         "model": model, "interval_level": level})
