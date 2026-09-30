"""Feature engineering, incl. the transparent (derived) AI classification."""
from __future__ import annotations

import re
from datetime import date

import pandas as pd

# Layer 1 (closest to observed): YC's own tags. Layer 2 (derived): keywords in current descriptions.
AI_TAG_RE = re.compile(r"\bai\b|artificial intelligence|machine learning|deep learning|computer vision|\bnlp\b|generative|\bllm", re.I)
AI_KEYWORD_RE = re.compile(
    r"\bai\b|\ba\.i\.|artificial intelligence|machine learning|deep learning|\bllms?\b|large language model|"
    r"\bgpt\b|generative|neural network|computer vision|\bnlp\b", re.I)
IN_PROGRESS_DAYS = 120  # a YC batch runs roughly 3 months; approximate.
TEAM_BINS = [0, 1, 2, 5, 10, 50, 200, float("inf")]
TEAM_LABELS = ["1", "2", "3-5", "6-10", "11-50", "51-200", "200+"]


def is_ai_tag(tags: list[str]) -> bool:
    return any(AI_TAG_RE.search(t) for t in tags)


def is_ai_keyword(text: str | None) -> bool:
    return bool(text) and bool(AI_KEYWORD_RE.search(text))


def batch_status(start: pd.Timestamp, as_of: date) -> str:
    """'future' | 'in_progress' | 'complete' | 'unknown', relative to as_of (approximate batch dates)."""
    if pd.isna(start):
        return "unknown"
    as_of_ts = pd.Timestamp(as_of)
    if start > as_of_ts:
        return "future"
    return "in_progress" if (as_of_ts - start).days < IN_PROGRESS_DAYS else "complete"


def build_features(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Add derived columns. Derived columns are prefixed/labelled so they are never mistaken for source data."""
    out = df.copy()
    out["ai_tag"] = out["tags"].map(is_ai_tag)
    text = (out["one_liner"].fillna("") + " " + out["long_description"].fillna(""))
    out["ai_keyword"] = text.map(is_ai_keyword)
    out["is_ai"] = out["ai_tag"] | out["ai_keyword"]  # broad, DERIVED
    out["batch_status"] = out["batch_start"].map(lambda s: batch_status(s, as_of))
    out["years_since_batch"] = (as_of.year - out["batch_year"]).astype("Int64")
    out["tag_count"] = out["tags"].map(len)
    out["team_size_bucket"] = pd.cut(out["team_size"].astype("float"), bins=TEAM_BINS, labels=TEAM_LABELS, right=True)
    out["team_size_bucket"] = out["team_size_bucket"].astype("string")
    out["description_length"] = out["long_description"].fillna("").str.len()
    return out
