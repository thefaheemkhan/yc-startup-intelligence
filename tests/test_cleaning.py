import pandas as pd

from tests.conftest import AS_OF, _rec
from src.cleaning.clean import clean_companies, clean_text, parse_batch, parse_location, split_locations
from src.cleaning.validate import validate_raw


def test_parse_batch_variants():
    b = parse_batch("Winter 2012")
    assert (b["batch_season"], b["batch_year"], b["batch_code"]) == ("Winter", 2012, "W12")
    assert parse_batch("Spring 2025")["batch_code"] == "X25"
    assert parse_batch("Unspecified")["batch_year"] is None


def test_parse_location():
    assert parse_location("San Francisco, CA, USA") == {
        "place_raw": "San Francisco, CA, USA", "city": "San Francisco", "region": "CA", "country": "United States", "is_remote": False}
    assert parse_location("London, England, United Kingdom")["country"] == "United Kingdom"
    assert parse_location("Remote")["is_remote"] is True
    assert parse_location("India")["city"] is None and parse_location("India")["country"] == "India"


def test_split_locations_and_clean_text():
    assert split_locations("A, B; Remote ;") == ["A, B", "Remote"]
    assert split_locations("") == [] and split_locations(None) == []
    assert clean_text("  a   b ") == "a b" and clean_text("   ") is pd.NA


def test_duplicate_ids_dropped_and_logged(cleaned):
    df, log = cleaned
    assert df["company_id"].is_unique and len(df) == 6
    assert df.set_index("company_id").loc[1, "name"] == "Duplicate id"   # keep='last' is the documented rule
    assert {s["step"]: s["rows_affected"] for s in log}["duplicate_ids"] == 1


def test_invalid_and_missing_team_size_become_null(cleaned):
    df, _ = cleaned
    ts = df.set_index("company_id")["team_size"]
    assert pd.isna(ts[2]) and pd.isna(ts[3]) and ts[1] == 5


def test_outlier_flagged_not_dropped():
    # The fence is data-driven, so it needs a realistic sample (not a 4-row fixture).
    recs = [_rec(i, team_size=2 + i % 6) for i in range(10, 60)] + [_rec(6, team_size=100000)]
    df, log = clean_companies(recs, as_of=AS_OF)
    flagged = df.set_index("company_id")["team_size_outlier"]
    assert flagged[6] and flagged.sum() == 1 and len(df) == 51


def test_status_normalised_and_unknown_recoded(cleaned):
    df, _ = cleaned
    s = df.set_index("company_id")["status"]
    assert s[1] == "Active" and s[3] == "Unknown"
    assert bool(df.set_index("company_id").loc[1, "is_operating"]) is True


def test_locations_primary_skips_remote(cleaned):
    df = cleaned[0].set_index("company_id")
    assert df.loc[4, "primary_country"] == "United Kingdom" and df.loc[4, "is_remote"]
    assert pd.isna(df.loc[3, "primary_country"]) and df.loc[3, "is_remote"]
    assert pd.isna(df.loc[5, "primary_country"])


def test_subindustry_and_website_normalised(cleaned):
    df = cleaned[0].set_index("company_id")
    assert df.loc[1, "subindustry"] == "Engineering" and df.loc[1, "website"] == "https://example.com"
    assert df.loc[4, "name"] == "Co 4"   # whitespace trimmed


def test_validate_raw_flags_missing_required_field():
    assert validate_raw([])["ok"] is False
    assert validate_raw([{"id": 1}])["ok"] is False
