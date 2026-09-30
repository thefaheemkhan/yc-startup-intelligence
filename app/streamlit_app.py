"""YC Startup Intelligence: interactive dashboard.  Run:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.analytics import core  # noqa: E402
from src.forecasting import models as fm  # noqa: E402
from src.modeling import tracking  # noqa: E402

st.set_page_config(page_title="YC Startup Intelligence", page_icon="📊", layout="wide")
STATUS_COLORS = {"Active": "#2a9d8f", "Public": "#264653", "Acquired": "#e9c46a", "Inactive": "#e76f51", "Unknown": "#adb5bd"}
TEMPLATE = "plotly_white"


@st.cache_data(show_spinner="Loading data...")
def load() -> tuple[pd.DataFrame, dict, list]:
    df = pd.read_parquet(config.FEATURES_COMPANIES)
    meta = json.loads(config.DATASET_META.read_text()) if config.DATASET_META.exists() else {}
    clog = json.loads(config.CLEANING_LOG.read_text()) if config.CLEANING_LOG.exists() else []
    return df, meta, clog


def show(fig: go.Figure, height: int = 420) -> None:
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


def download(df: pd.DataFrame, label: str, name: str) -> None:
    st.download_button(f"Download {label} (CSV)", df.to_csv(index=False).encode(), f"{name}.csv", "text/csv", key=f"dl_{name}")


if not config.FEATURES_COMPANIES.exists():
    st.error("Processed data not found. Run `python scripts/run_pipeline.py` first.")
    st.stop()

df, meta, clog = load()
complete_years = df.loc[df["batch_status"] == "complete", "batch_year"].dropna().astype(int)
all_years = df["batch_year"].dropna().astype(int)

# ------------------------------------------------------------------ sidebar filters
with st.sidebar:
    st.header("Filters")
    include_incomplete = st.checkbox("Include in-progress / future batches", False,
                                     help="Recent batches are incomplete: companies are still being added and tagged.")
    years = st.slider("Batch year", int(all_years.min()), int(all_years.max()), (int(all_years.min()), int(complete_years.max())))
    industries = st.multiselect("Industry", sorted(df["industry"].dropna().unique()))
    statuses = st.multiselect("Status", ["Active", "Public", "Acquired", "Inactive"])
    countries = st.multiselect("Country", df["primary_country"].value_counts().head(40).index.tolist())
    ai_def = st.radio("AI definition", ["broad", "tag"], format_func={"broad": "Broad (YC tags + keywords, derived)", "tag": "YC tags only"}.get,
                      help="Broad also matches AI keywords in company descriptions. It is a derived classification, not a YC field.")
    ai_sel = st.radio("AI filter", ["All", "AI", "Non-AI"], horizontal=True)
    sizes = [1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000, 10000]
    team = st.select_slider("Team size (current headcount)", sizes, value=(1, 10000))
    unk_team = st.checkbox("Include unknown team size", True)
    top_n = st.slider("Top N", 5, 25, 10)

f = core.Filters(years=years, industries=tuple(industries), statuses=tuple(statuses), countries=tuple(countries), ai=ai_sel,
                 ai_definition=ai_def, include_incomplete=include_incomplete,
                 team_range=None if team == (1, 10000) else team, include_unknown_team=unk_team)
d = core.apply_filters(df, f)
ai_col = core.ai_column(ai_def)

st.title("YC Startup Intelligence")
st.caption(f"{len(d):,} of {len(df):,} companies match the filters · dataset {meta.get('dataset_version', '?')} · as of {meta.get('as_of', '?')} · "
           "source: yc-oss/api (YC public directory, publicly launched companies only)")
if d.empty:
    st.warning("No companies match these filters. Relax a filter in the sidebar.")
    st.stop()

k = core.kpis(d, ai_col)
c = st.columns(6)
c[0].metric("Companies", f"{k['total']:,}")
c[1].metric("Operating", f"{k['operating_pct']:.1f}%", help="Active + Public")
c[2].metric("AI share", f"{k['ai_share_pct']:.1f}%", help="Derived; depends on the AI definition")
c[3].metric("Industries", k["industries"])
c[4].metric("Countries", k["countries"])
c[5].metric("Median team size", f"{k['median_team_size']:.0f}" if not np.isnan(k["median_team_size"]) else "n/a", help="Current headcount, not founders")

tabs = st.tabs(["Overview", "Industries", "Geography", "AI", "Status & Team", "Explorer", "Forecasting", "Predictive Model", "Data & Method"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    st.subheader("What is happening in the YC ecosystem?")
    with st.container(border=True):
        st.markdown("**Key insights (computed from the current filter)**")
        for line in core.insights(d, ai_col):
            st.markdown(f"- {line}")
    pb = core.per_batch(d)
    fig = px.bar(pb, x="batch", y="n", color="batch_season", title="Companies per batch",
                 category_orders={"batch": pb["batch"].tolist()}, labels={"n": "Companies", "batch": "", "batch_season": "Season"})
    show(fig)
    st.caption("Batch cadence changed from 2 to 4 per year during 2024-2025, so yearly totals are not comparable across that change.")
    ys = core.year_status(d)
    show(px.bar(ys, x="year", y="n", color="status", color_discrete_map=STATUS_COLORS, title="Companies per year by current status",
                labels={"n": "Companies", "year": "Batch year"}))
    download(pb, "batch counts", "batch_counts")

# ------------------------------------------------------------------ Industries
with tabs[1]:
    lvl = st.radio("Level", ["industry", "subindustry"], horizontal=True)
    base = d[d[lvl].ne("Unspecified")]
    left, right = st.columns(2)
    with left:
        tc = core.top_counts(base, lvl, top_n).sort_values("n")
        show(px.bar(tc, x="n", y=lvl, orientation="h", title=f"Top {top_n} {lvl} groups", labels={"n": "Companies", lvl: ""}))
    with right:
        hm = core.crosstab_share(base, lvl, "batch_year", top_n)
        show(px.imshow(hm.round(1), aspect="auto", color_continuous_scale="Blues", text_auto=".0f",
                       title=f"{lvl.title()} mix by batch year (% of that year's companies)", labels={"color": "%"}))
    st.markdown("#### Outcome rates by group")
    min_n = st.slider("Minimum companies per group", 10, 200, 30, key="min_n_ind")
    metric = st.radio("Outcome", ["Operating (Active/Public)", "Acquired"], horizontal=True)
    flag = "is_operating" if metric.startswith("Operating") else "is_acquired"
    rt = core.rate_by_group(base.assign(is_acquired=base["status"].eq("Acquired")), lvl, flag, min_n).head(top_n * 2)
    if rt.empty:
        st.info("No groups meet the minimum size.")
    else:
        fig = go.Figure(go.Bar(x=rt[lvl], y=rt["rate"], error_y=dict(type="data", symmetric=False, array=rt["hi"] - rt["rate"], arrayminus=rt["rate"] - rt["lo"]),
                               marker_color="#457b9d", customdata=rt["n"], hovertemplate="%{x}<br>%{y:.1f}% (n=%{customdata})<extra></extra>"))
        fig.update_layout(title=f"{metric} rate with 95% Wilson interval", yaxis_title="%")
        show(fig)
        st.caption("Rates reflect current status only. Older batches have had longer to fail or be acquired, so compare groups within similar batch years. Not causal.")
    st.markdown("#### Industry Growth Momentum")
    w1, w2 = st.columns(2)
    window = w1.slider("Window (batch years)", 2, 5, 3)
    min_recent = w2.slider("Min companies in recent window", 5, 100, 20)
    mo = core.momentum(base, lvl, window, min_recent)
    if mo.empty:
        st.info("Not enough batch years in the current filter for this window.")
    else:
        st.caption(f"Share change (percentage points) of {min(mo.attrs['recent_years'])}-{max(mo.attrs['recent_years'])} vs "
                   f"{min(mo.attrs['prior_years'])}-{max(mo.attrs['prior_years'])}. Descriptive only; not a forecast or a measure of success.")
        m2 = pd.concat([mo.head(top_n // 2 + 1), mo.tail(top_n // 2 + 1)]).drop_duplicates(lvl).sort_values("share_change_pp")
        show(px.bar(m2, x="share_change_pp", y=lvl, orientation="h", color="share_change_pp", color_continuous_scale="RdBu",
                    color_continuous_midpoint=0, title="Biggest share gainers and losers", labels={"share_change_pp": "Share change (pp)", lvl: ""}))
        st.dataframe(mo.round(1), width="stretch", hide_index=True)
        download(mo, "momentum table", "industry_momentum")

# ------------------------------------------------------------------ Geography
with tabs[2]:
    g = core.top_counts(d, "primary_country", 300)
    g["log10 companies"] = np.log10(g["n"])
    fig = px.choropleth(g, locations="primary_country", locationmode="country names", color="log10 companies", hover_name="primary_country",
                        hover_data={"n": True, "log10 companies": False}, color_continuous_scale="Blues", title="Companies by country (log scale)")
    fig.update_layout(coloraxis_colorbar_title="log10")
    show(fig, 460)
    a, b = st.columns(2)
    with a:
        show(px.bar(g.head(top_n).sort_values("n"), x="n", y="primary_country", orientation="h", title=f"Top {top_n} countries", labels={"n": "Companies", "primary_country": ""}))
    with b:
        ci = core.top_counts(d, "primary_city", top_n).sort_values("n")
        show(px.bar(ci, x="n", y="primary_city", orientation="h", title=f"Top {top_n} cities", labels={"n": "Companies", "primary_city": ""}))
    mix = core.crosstab_share(d[d["industry"] != "Unspecified"], "primary_country", "industry", min(top_n, 12), "index")
    show(px.imshow(mix.round(1), aspect="auto", color_continuous_scale="Blues", text_auto=".0f", title="Industry mix within each country (% of country's companies)"))
    st.caption(f"{d['primary_country'].isna().sum():,} companies in this view have no physical location (remote-only or blank) and are excluded from country/city charts.")
    download(g[["primary_country", "n"]], "country counts", "country_counts")

# ------------------------------------------------------------------ AI
with tabs[3]:
    st.caption("AI status is a DERIVED classification (YC tags, plus keywords in current descriptions for the broad version). It is not a field provided by YC.")
    ay = core.ai_share_by_year(d)
    fig = go.Figure()
    fig.add_scatter(x=ay["year"], y=ay["broad"], name="Broad (tags + keywords)", mode="lines+markers")
    fig.add_scatter(x=ay["year"], y=ay["tag_based"], name="YC tags only", mode="lines+markers")
    fig.update_layout(title="AI share of companies by batch year", yaxis_title="% of companies", xaxis_title="Batch year")
    show(fig)
    st.caption("The two definitions diverge in recent years. One possible reason is that newer companies are tagged later; this has not been tested. Treat the newest year with caution.")
    a, b = st.columns(2)
    with a:
        ai_ind = core.rate_by_group(d[d["industry"] != "Unspecified"], "industry", ai_col, 20)
        show(px.bar(ai_ind.sort_values("rate"), x="rate", y="industry", orientation="h", title="AI share by industry (selected definition)", labels={"rate": "% AI", "industry": ""}))
    with b:
        tags = d[d[ai_col]].explode("tags")["tags"].value_counts().head(top_n).rename_axis("tag").reset_index(name="n")
        show(px.bar(tags.sort_values("n"), x="n", y="tag", orientation="h", title="Most common tags among AI companies", labels={"n": "Companies", "tag": ""}))
    download(ay, "AI share by year", "ai_share_by_year")

# ------------------------------------------------------------------ Status & Team
with tabs[4]:
    a, b = st.columns(2)
    with a:
        sc = d["status"].value_counts().rename_axis("status").reset_index(name="n")
        show(px.pie(sc, names="status", values="n", hole=0.5, color="status", color_discrete_map=STATUS_COLORS, title="Current status"))
    with b:
        ct = pd.crosstab(d["batch_year"].dropna().astype(int), d.loc[d["batch_year"].notna(), "status"], normalize="index") * 100
        show(px.bar(ct.reset_index().melt("batch_year", var_name="status", value_name="pct"), x="batch_year", y="pct", color="status",
                    color_discrete_map=STATUS_COLORS, title="Status mix by batch year (%)", labels={"batch_year": "Batch year", "pct": "%"}))
    st.caption("Survivorship/age caveat: older cohorts have had more time to become inactive or acquired, and inactive companies may be missing from the source. Differences by year are not causal.")
    st.markdown("#### Team size (current headcount as recorded by YC; NOT founder count)")
    ts = d.dropna(subset=["team_size"]).assign(team_size=lambda t: t["team_size"].astype(int))
    a, b = st.columns(2)
    with a:
        show(px.histogram(ts, x="team_size", nbins=40, log_x=True, title="Team size distribution (log scale)", labels={"team_size": "Team size"}))
    with b:
        med = ts.dropna(subset=["batch_year"]).groupby(ts["batch_year"].astype("Int64").astype(float).astype(int))["team_size"].median().reset_index()
        med.columns = ["year", "median"]
        show(px.line(med, x="year", y="median", markers=True, title="Median current team size by batch year", labels={"median": "Median headcount"}))
    st.caption(f"{d['team_size'].isna().sum():,} companies have unknown team size (missing or invalid in the source).")

# ------------------------------------------------------------------ Explorer
with tabs[5]:
    q = st.text_input("Search name or one-liner")
    ex = d
    if q:
        hay = ex["name"].fillna("").str.lower() + " " + ex["one_liner"].fillna("").str.lower()
        ex = ex[hay.str.contains(q.lower(), regex=False)]
    cols = ["name", "batch", "industry", "subindustry", "primary_city", "primary_country", "status", "team_size", "is_ai", "one_liner", "website", "yc_url"]
    view = ex[cols].sort_values("name", key=lambda s: s.str.lower())
    st.caption(f"{len(view):,} companies. Public directory fields only.")
    st.dataframe(view, width="stretch", hide_index=True, height=480,
                 column_config={"website": st.column_config.LinkColumn("website"), "yc_url": st.column_config.LinkColumn("YC page")})
    download(view, "filtered companies", "yc_companies_filtered")


@st.cache_data(show_spinner="Backtesting models...")
def backtest(y: pd.Series):
    return fm.compare_models(y)


# ------------------------------------------------------------------ Forecasting
with tabs[6]:
    st.subheader("Forecasting: companies per batch year")
    st.caption("Uses all complete years (2006 onward) and ignores the sidebar filters. Forecasts are ESTIMATES from simple statistical models, not predictions of any company's success. "
               "Each model is scored by rolling-origin backtesting: it forecasts one year ahead using only earlier years.")
    inds = [i for i, n_ in df["industry"].value_counts().items() if i != "Unspecified" and n_ >= 100]
    sel = st.selectbox("Series", ["All companies", "AI companies (broad, derived)", "AI companies (YC tags only)"] + inds)
    c1, c2, c3 = st.columns(3)
    horizon = c1.slider("Forecast horizon (years)", 1, 5, 3)
    level = c2.radio("Prediction interval", [0.8, 0.95], format_func=lambda v: f"{int(v * 100)}%", horizontal=True)
    y = fm.annual_counts(df, sel)
    tbl, bts = backtest(y)
    choice = c3.selectbox("Model", ["Best by backtest RMSE"] + list(fm.MODELS))
    model = tbl.iloc[0]["model"] if choice.startswith("Best") else choice
    fc = fm.forecast(y, model, horizon, bts[model], level)
    fig = go.Figure()
    fig.add_scatter(x=y.index, y=y.values, name="Observed (complete years)", mode="lines+markers", line=dict(color="#264653"))
    fig.add_scatter(x=fc["year"], y=fc["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fc["year"], y=fc["lo"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(69,123,157,0.2)", name=f"{int(level * 100)}% interval (approx.)")
    fig.add_scatter(x=fc["year"], y=fc["forecast"], name=f"Forecast ({model})", mode="lines+markers", line=dict(color="#457b9d", dash="dash"))
    pc = fm.partial_year_count(df, sel)
    if pc:
        fig.add_scatter(x=[pc[0]], y=[pc[1]], mode="markers", name=f"{pc[0]} so far (incomplete)", marker=dict(symbol="diamond", size=11, color="#e76f51"))
    fig.update_layout(title=f"{sel}: companies per batch year", xaxis_title="Batch year", yaxis_title="Companies")
    show(fig, 450)
    best = tbl.iloc[0]
    if best["model"] == "naive" or best["rmse_vs_naive"] >= 0.95:
        st.warning("No model clearly beats the naive 'same as last year' forecast in the backtest for this series, so these forecasts add little information.")
    else:
        st.info(f"{best['model']} lowered backtest RMSE by {100 * (1 - best['rmse_vs_naive']):.0f}% versus naive over {int(best['n_eval'])} backtest years. "
                "That is a small sample, and the same backtest chose the model and set the interval, so treat the gain as optimistic.")
    st.markdown("**Model comparison (one-step-ahead rolling-origin backtest)**")
    st.dataframe(tbl.round(2), width="stretch", hide_index=True)
    st.caption("MAE/RMSE are in companies per year; MAPE in %; rmse_vs_naive < 1 means better than naive. Intervals = point +/- z x backtest RMSE x sqrt(horizon), clipped at 0.")
    d1, d2 = st.columns(2)
    with d1:
        download(fc, "forecast", "forecast")
    with d2:
        download(tbl, "model comparison", "forecast_model_comparison")

# ------------------------------------------------------------------ Predictive Model
with tabs[7]:
    st.subheader("Predicting which companies are Inactive")
    if not config.MODEL_RESULTS.exists():
        st.info("Run `python scripts/run_models.py` to generate model results.")
    else:
        res = json.loads(config.MODEL_RESULTS.read_text())
        co = res["cohort"]
        st.caption("Descriptive modelling exercise, not investment or hiring advice. Associations only; nothing here is causal.")
        with st.expander("Setup, leakage controls and limitations", expanded=True):
            st.markdown(f"""
- **Target:** current status is *Inactive* (1) vs Active / Public / Acquired (0). Acquired is not treated as a shutdown.
- **Cohort:** complete batches {co['years'][0]}-{co['years'][1]} only ({co['n_train'] + co['n_test']:,} companies). Later batches are excluded because they have not had time to fail (right-censoring).
- **Temporal split:** train on {co['train_years'][0]}-{co['train_years'][1]} ({co['n_train']:,}), test on {co['test_years'][0]}-{co['test_years'][1]} ({co['n_test']:,}). Model selection uses forward-chaining CV on the training years only.
- **Class balance:** {100 * co['train_inactive_rate']:.1f}% inactive in train vs {100 * co['test_inactive_rate']:.1f}% in test. The gap is expected: younger cohorts have had less time to become inactive.
- **Features (at-admission only):** {", ".join(res['features'])}. Team size, hiring, stage, top-company flag, launch date and description/tag counts are deliberately excluded because they are measured after the outcome.
- **Remaining leakage risk:** industry, AI flag and country are current snapshots and may have been updated after a company changed course.
- **Threshold:** chosen to maximise F1 on cross-validated training predictions, then applied unchanged to the test set.
""")
        t = pd.DataFrame(res["table"])
        show_cols = ["model", "cv_pr_auc_mean", "cv_pr_auc_std", "test_roc_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1"]
        st.markdown(f"**Model comparison** (test PR-AUC of a no-skill model = test prevalence = {co['test_inactive_rate']:.3f}; ROC-AUC no-skill = 0.5)")
        st.dataframe(t[show_cols].round(3), width="stretch", hide_index=True)
        best_row = t[t["model"] == res["best_model_by_cv"]].iloc[0]
        flagged = (best_row["test_tp"] + best_row["test_fp"]) / co["n_test"]
        st.warning(f"Best by cross-validation: **{res['best_model_by_cv']}** (test ROC-AUC {best_row['test_roc_auc']:.2f}). "
                   f"Discrimination is weak, and at the F1-optimal threshold it flags {100 * flagged:.0f}% of test companies as inactive. "
                   "Batch, industry and location alone carry little information about who shuts down.")
        a, b = st.columns(2)
        with a:
            cm = res["confusion"]
            show(px.imshow(np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]), text_auto=True, color_continuous_scale="Blues",
                           x=["Predicted not inactive", "Predicted inactive"], y=["Actually not inactive", "Actually inactive"],
                           title=f"Test confusion matrix ({res['best_model_by_cv']})"), 360)
        with b:
            imp = pd.DataFrame(res["importance"]).sort_values("importance")
            fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h", error_x=dict(type="data", array=imp["std"]), marker_color="#457b9d"))
            fig.update_layout(title="Permutation importance on test set (drop in PR-AUC)", xaxis_title="Mean decrease in PR-AUC (+/- sd)")
            show(fig, 360)
        st.caption("Importance shows how much the model's ranking relies on a feature, not that the feature causes shutdown. Values near zero or negative mean no usable signal. SHAP is not included.")
        if config.MODEL_PREDICTIONS.exists():
            download(pd.read_csv(config.MODEL_PREDICTIONS), "test-set predictions", "status_model_test_predictions")
        runs = tracking.load_runs()
        if not runs.empty:
            st.markdown("**Experiment log** (all runs, newest first)")
            cols = [c for c in ["experiment_id", "timestamp_utc", "kind", "model", "dataset_version", "m_rmse", "m_mae", "m_test_roc_auc", "m_test_pr_auc", "m_cv_pr_auc_mean"] if c in runs.columns]
            st.dataframe(runs.sort_values("timestamp_utc", ascending=False)[cols], width="stretch", hide_index=True, height=260)
            download(runs.drop(columns=["params", "features"], errors="ignore"), "experiment log", "experiments")

# ------------------------------------------------------------------ Data & Method
with tabs[8]:
    st.markdown("""
**Observed vs derived.** Observed (from YC's directory): batch, status, industry/subindustry, tags, location text, team size, website.
Derived by this project: parsed city/country, AI classification, batch completeness, all rates, shares and momentum scores.

**Known limitations**
- Only *publicly launched* companies with YC pages are included; this creates selection bias and may undercount recent batches.
- Status is a single current snapshot: no history, and *acquired* is not the same as *successful*.
- No founder data exists in the source; team size is current headcount.
- The AI keyword layer was not audited for precision. Descriptions are current, so companies that later pivoted to AI are labelled AI for earlier batches.
- Batch start months are approximations used only for ordering and completeness flags.
- Forecasts use ~20 annual points; intervals are empirical and approximate. Annual counts also reflect YC's own capacity decisions.
- The status model uses a temporal split on 2006-2021 batches, current-snapshot features only, and shows weak discrimination.
- All comparisons are descriptive; nothing here is causal.
""")
    st.markdown("**Cleaning log**")
    if clog:
        st.dataframe(pd.DataFrame(clog), width="stretch", hide_index=True)
    st.json(meta)
