"""Predict whether a company is currently 'Inactive' from information known AT ADMISSION.

Design decisions (all deliberate, all documented in the app):
* Target = status == 'Inactive' vs everything else (Active, Public, Acquired). 'Acquired' is not a shutdown.
* Cohort = complete batches 2006-2021 only. Later batches have had too little time to fail (right-censoring).
* Split is TEMPORAL: train <= 2017, test 2018-2021. Cross-validation uses forward-chaining folds, never random folds.
* Features exclude everything measured after admission (team_size, is_hiring, stage, top_company, launched_at,
  description length, tag count). Remaining features are still CURRENT snapshots (industry, tags-derived AI flag,
  location), so some leakage risk remains and is stated in the results.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.modeling import tracking

log = logging.getLogger(__name__)

FIRST_YEAR, TRAIN_END, LAST_YEAR = 2006, 2017, 2021
CV_CUTOFFS = (2011, 2013, 2015, 2017)          # folds: train<=a, validate a<year<=b (consecutive pairs)
CAT = ["batch_season", "industry", "subindustry", "country_group"]
NUM = ["batch_year"]
BIN = ["is_ai"]
FEATURES = CAT + NUM + BIN
FORBIDDEN = {"team_size", "team_size_outlier", "is_hiring", "stage", "is_top_company", "launched_at", "description_length",
             "tag_count", "status", "is_operating", "website", "years_since_batch"}
assert not FORBIDDEN & set(FEATURES)


def country_group(s: pd.Series, top: list[str]) -> pd.Series:
    out = s.astype(object).where(s.isin(top), "Other")
    return out.where(s.notna(), "Unknown")


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, meta) for the modelling cohort. Top countries are chosen from TRAIN years only."""
    x = df[(df["batch_status"] == "complete") & df["batch_year"].between(FIRST_YEAR, LAST_YEAR).fillna(False).astype(bool)].copy()
    x["batch_year"] = x["batch_year"].astype(int)
    top = x.loc[x["batch_year"] <= TRAIN_END, "primary_country"].value_counts().head(8).index.tolist()
    x["country_group"] = country_group(x["primary_country"], top)
    for c in ["batch_season", "industry", "subindustry"]:
        x[c] = x[c].astype(object).where(x[c].notna(), "Unknown")
    x["is_ai"] = x["is_ai"].astype(int)
    y = (x["status"] == "Inactive").astype(int)
    return x[FEATURES].reset_index(drop=True), y.reset_index(drop=True), x[["company_id", "name", "batch", "industry"]].reset_index(drop=True)


def forward_folds(years: np.ndarray, cutoffs: tuple[int, ...] = CV_CUTOFFS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining folds as (train_idx, val_idx); validation years are always later than training years."""
    folds = []
    for a, b in zip(cutoffs[:-1], cutoffs[1:]):
        tr, va = np.where(years <= a)[0], np.where((years > a) & (years <= b))[0]
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


def make_model(name: str, seed: int) -> Pipeline:
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=15, sparse_output=False), CAT),
                             ("num", StandardScaler(), NUM), ("bin", "passthrough", BIN)])
    clf = {
        "baseline_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=seed),
        "decision_tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=15, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=seed),
    }[name]
    return Pipeline([("pre", pre), ("clf", clf)])


MODEL_NAMES = ["baseline_prior", "logistic_regression", "decision_tree", "random_forest", "gradient_boosting"]


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))]) if len(t) else 0.5


def score_report(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, float]:
    pred = (s >= thr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)), "threshold": thr,
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "prevalence": float(y.mean()), "n": int(len(y))}


def run_status_experiments(df: pd.DataFrame, seed: int = config.RANDOM_SEED, dataset_version: str = "unknown",
                           log_path=None, write: bool = True, track: bool = True) -> dict[str, Any]:
    """Train/evaluate all models. Model choice uses forward-CV only; the test set is used once for reporting."""
    X, y, meta = build_dataset(df)
    yrs = X["batch_year"].to_numpy()
    tr, te = np.where(yrs <= TRAIN_END)[0], np.where(yrs > TRAIN_END)[0]
    Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr].to_numpy(), X.iloc[te], y.iloc[te].to_numpy()
    folds = forward_folds(yrs[tr])
    rows, fitted, oof_thr = [], {}, {}
    for name in MODEL_NAMES:
        oof_y, oof_s, fold_pr, fold_roc = [], [], [], []
        for a, b in folds:
            m = make_model(name, seed).fit(Xtr.iloc[a], ytr[a])
            s = m.predict_proba(Xtr.iloc[b])[:, 1]
            oof_y.append(ytr[b]); oof_s.append(s)
            fold_pr.append(average_precision_score(ytr[b], s)); fold_roc.append(roc_auc_score(ytr[b], s) if len(set(ytr[b])) > 1 else np.nan)
        thr = best_f1_threshold(np.concatenate(oof_y), np.concatenate(oof_s))
        final = make_model(name, seed).fit(Xtr, ytr)
        rep = score_report(yte, final.predict_proba(Xte)[:, 1], thr)
        fitted[name], oof_thr[name] = final, thr
        rows.append({"model": name, "cv_pr_auc_mean": float(np.mean(fold_pr)), "cv_pr_auc_std": float(np.std(fold_pr)),
                     "cv_roc_auc_mean": float(np.nanmean(fold_roc)), **{f"test_{k}": v for k, v in rep.items() if k not in ("prevalence", "n")}})
        if track:
            tracking.log_run("classification", name, {"target": "inactive", "train_end": TRAIN_END, "seed": seed, "cv_cutoffs": list(CV_CUTOFFS)},
                             FEATURES, {k: v for k, v in rows[-1].items() if k != "model"}, "temporal split; threshold from forward-CV OOF F1",
                             dataset_version, log_path)
    table = pd.DataFrame(rows)
    best = table[table["model"] != "baseline_prior"].sort_values("cv_pr_auc_mean", ascending=False).iloc[0]["model"]   # chosen on CV only
    pi = permutation_importance(fitted[best], Xte, yte, scoring="average_precision", n_repeats=20, random_state=seed)
    imp = pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean, "std": pi.importances_std}).sort_values("importance", ascending=False)
    scores = fitted[best].predict_proba(Xte)[:, 1]
    preds = meta.iloc[te].assign(y_true=yte, score=scores, predicted_inactive=(scores >= oof_thr[best]).astype(int))
    b = table[table["model"] == best].iloc[0]
    result = {
        "cohort": {"years": [FIRST_YEAR, LAST_YEAR], "train_years": [FIRST_YEAR, TRAIN_END], "test_years": [TRAIN_END + 1, LAST_YEAR],
                   "n_train": int(len(tr)), "n_test": int(len(te)), "train_inactive_rate": float(ytr.mean()), "test_inactive_rate": float(yte.mean())},
        "best_model_by_cv": best, "features": FEATURES, "table": table.to_dict("records"),
        "confusion": {k: int(b[f"test_{k}"]) for k in ("tp", "fp", "fn", "tn")}, "importance": imp.to_dict("records"),
        "dataset_version": dataset_version, "seed": seed}
    if write:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        import json
        config.MODEL_RESULTS.write_text(json.dumps(result, indent=2, default=float))
        preds.to_csv(config.MODEL_PREDICTIONS, index=False)
    result["predictions"] = preds
    return result
