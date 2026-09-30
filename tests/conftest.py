from datetime import date

import pytest

from src.cleaning.clean import clean_companies

AS_OF = date(2026, 9, 29)


def _rec(i, **kw):
    base = {"id": i, "name": f"  Co {i} ", "slug": f"co-{i}", "website": "HTTPS://Example.com", "all_locations": "San Francisco, CA, USA",
            "long_description": "We sell widgets.", "one_liner": "Widgets", "team_size": 5, "industry": "B2B",
            "subindustry": "B2B -> Engineering", "launched_at": 1322045523, "tags": ["SaaS"], "isHiring": False,
            "nonprofit": False, "top_company": False, "batch": "Winter 2012", "status": "active",
            "regions": ["United States of America"], "stage": "Early", "url": f"https://yc/{i}"}
    base.update(kw)
    return base


@pytest.fixture
def records():
    return [
        _rec(1),
        _rec(1, name="Duplicate id"),                                   # duplicate id -> dropped
        _rec(2, team_size=-1, batch="Unspecified", industry="Unspecified"),
        _rec(3, team_size=None, all_locations="Remote", status="Weird"),
        _rec(4, all_locations="London, England, United Kingdom; Remote", tags=["AI", "Generative AI"]),
        _rec(5, all_locations="", launched_at=None, batch="Summer 2027"),
        _rec(6, team_size=100000),
    ]


@pytest.fixture
def cleaned(records):
    return clean_companies(records, as_of=AS_OF)
