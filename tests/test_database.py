import pandas as pd
from sqlalchemy import create_engine, text

from src.database.load import build_tables, load_database
from src.features.build import build_features
from tests.conftest import AS_OF


def test_load_database_integrity(cleaned, tmp_path):
    df = build_features(cleaned[0], AS_OF)
    url = f"sqlite:///{tmp_path / 't.db'}"
    counts = load_database(df, url=url)
    assert counts["companies"] == len(df)
    eng = create_engine(url)
    with eng.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM companies WHERE batch_id IS NULL")).scalar() == 0
        assert c.execute(text("SELECT COUNT(*) FROM companies c LEFT JOIN industries i USING(industry_id) WHERE i.industry_id IS NULL")).scalar() == 0
        # company 4 has a UK location plus Remote; exactly one primary
        assert c.execute(text("SELECT SUM(is_primary) FROM company_locations WHERE company_id=4")).scalar() == 1


def test_normalised_tables_have_unique_keys(cleaned):
    t = build_tables(build_features(cleaned[0], AS_OF))
    assert t["batches"]["batch"].is_unique and t["locations"]["place_raw"].is_unique
    assert not t["company_tags"].duplicated().any()
