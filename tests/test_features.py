from datetime import date

import pandas as pd

from src.features.build import batch_status, build_features, is_ai_keyword, is_ai_tag
from tests.conftest import AS_OF


def test_ai_tag_and_keyword_rules():
    assert is_ai_tag(["Generative AI"]) and is_ai_tag(["AI-powered Drug Discovery"]) and is_ai_tag(["Machine Learning"])
    assert not is_ai_tag(["SaaS", "Fintech", "Paid"])          # 'paid' must not match 'ai'
    assert is_ai_keyword("An LLM-based assistant") and not is_ai_keyword("Sells tailored suits")
    assert not is_ai_keyword(None)


def test_batch_status_relative_to_as_of():
    assert batch_status(pd.Timestamp("2027-01-01"), AS_OF) == "future"
    assert batch_status(pd.Timestamp("2026-09-01"), AS_OF) == "in_progress"
    assert batch_status(pd.Timestamp("2012-01-01"), AS_OF) == "complete"
    assert batch_status(pd.NaT, AS_OF) == "unknown"


def test_build_features_columns(cleaned):
    out = build_features(cleaned[0], AS_OF).set_index("company_id")
    assert out.loc[4, "is_ai"] and out.loc[4, "ai_tag"]
    assert not out.loc[1, "is_ai"]
    assert out.loc[5, "batch_status"] == "future"
    assert out.loc[1, "team_size_bucket"] == "3-5" and out.loc[1, "years_since_batch"] == 14
    assert len(out) == len(cleaned[0])
