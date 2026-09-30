import json

import numpy as np
import pandas as pd
import pytest

from src.modeling import status_model as sm
from src.modeling import tracking


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    rows = []
    for y in range(2005, 2024):
        for i in range(45):
            ind = rng.choice(["B2B", "Consumer", "Fintech"])
            p = 0.15 + 0.25 * (ind == "Consumer") + 0.02 * max(0, 2016 - y)      # signal: consumer + older
            rows.append({"company_id": len(rows), "name": f"c{len(rows)}", "batch": f"Winter {y}", "batch_year": y, "batch_season": "Winter",
                         "batch_status": "complete" if y < 2023 else "in_progress", "industry": ind, "subindustry": ind + "-x",
                         "primary_country": rng.choice(["United States", "India", None]), "is_ai": rng.random() < 0.3,
                         "status": "Inactive" if rng.random() < p else rng.choice(["Active", "Acquired", "Public"]),
                         "team_size": 5, "is_hiring": True, "stage": "Early"})
    d = pd.DataFrame(rows)
    d["batch_year"] = d["batch_year"].astype("Int64")
    return d


def test_no_post_admission_features_used():
    assert not sm.FORBIDDEN & set(sm.FEATURES)
    assert {"team_size", "is_hiring", "stage", "status"}.isdisjoint(sm.FEATURES)


def test_dataset_excludes_censored_and_incomplete_cohorts(synth):
    X, y, meta = sm.build_dataset(synth)
    assert X["batch_year"].min() >= sm.FIRST_YEAR and X["batch_year"].max() <= sm.LAST_YEAR
    assert list(X.columns) == sm.FEATURES and set(y.unique()) <= {0, 1} and len(X) == len(y) == len(meta)
    assert not X.isna().any().any()


def test_forward_folds_are_strictly_temporal(synth):
    X, _, _ = sm.build_dataset(synth)
    folds = sm.forward_folds(X["batch_year"].to_numpy()[X["batch_year"] <= sm.TRAIN_END])
    assert len(folds) >= 2
    yrs = X["batch_year"].to_numpy()[X["batch_year"] <= sm.TRAIN_END]
    for tr, va in folds:
        assert yrs[tr].max() < yrs[va].min() and not set(tr) & set(va)


def test_threshold_and_score_report():
    y = np.array([0, 0, 0, 1, 1, 1]); s = np.array([.1, .2, .3, .6, .7, .9])
    thr = sm.best_f1_threshold(y, s)
    r = sm.score_report(y, s, thr)
    assert r["roc_auc"] == 1.0 and r["f1"] == 1.0 and r["tp"] == 3 and r["fp"] == 0 and r["prevalence"] == 0.5


def test_end_to_end_is_deterministic_and_beats_chance(synth, tmp_path):
    kw = dict(seed=7, dataset_version="t", write=False, track=False)
    a = sm.run_status_experiments(synth, **kw); b = sm.run_status_experiments(synth, **kw)
    ta, tb = pd.DataFrame(a["table"]), pd.DataFrame(b["table"])
    pd.testing.assert_frame_equal(ta, tb)
    assert set(ta["model"]) == set(sm.MODEL_NAMES) and a["best_model_by_cv"] != "baseline_prior"
    assert ta.loc[ta["model"] == "baseline_prior", "test_roc_auc"].iloc[0] == pytest.approx(0.5)
    assert ta.loc[ta["model"] == "logistic_regression", "test_roc_auc"].iloc[0] > 0.55       # planted signal is detectable
    c = a["confusion"]; assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == a["cohort"]["n_test"]
    assert a["cohort"]["train_years"][1] < a["cohort"]["test_years"][0]


def test_tracker_appends_and_loads(tmp_path):
    p = tmp_path / "runs.jsonl"
    i1 = tracking.log_run("classification", "m1", {"a": 1}, ["f"], {"auc": 0.6, "bad": float("nan")}, "n", "v1", p)
    i2 = tracking.log_run("forecast", "m2", {}, [], {"rmse": 3.0}, "", "v1", p)
    assert i1 != i2 and len(p.read_text().splitlines()) == 2
    runs = tracking.load_runs(p)
    assert list(runs["model"]) == ["m1", "m2"] and runs.loc[0, "m_auc"] == 0.6 and pd.isna(runs.loc[0, "m_bad"])
    assert json.loads(p.read_text().splitlines()[0])["dataset_version"] == "v1"
    assert tracking.load_runs(tmp_path / "none.jsonl").empty
