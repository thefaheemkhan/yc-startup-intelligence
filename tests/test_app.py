import pytest
from streamlit.testing.v1 import AppTest

from src import config

APP = str(config.ROOT / "app" / "streamlit_app.py")

pytestmark = pytest.mark.skipif(not config.FEATURES_COMPANIES.exists(), reason="run the pipeline first")


def test_app_renders_without_exception():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, at.exception
    assert len(at.metric) == 6 and len(at.tabs) == 9


def test_filters_change_results():
    at = AppTest.from_file(APP, default_timeout=90).run()
    before = at.metric[0].value
    at.sidebar.multiselect[0].set_value(["Fintech"]).run()   # Industry
    assert not at.exception, at.exception
    assert at.metric[0].value != before
    at.sidebar.radio[1].set_value("AI").run()                # AI filter
    assert not at.exception, at.exception


def test_empty_filter_shows_warning_not_crash():
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.multiselect[0].set_value(["Government"]).run()
    at.sidebar.multiselect[2].set_value(["Public"]).run()    # may be empty or tiny; must not crash
    assert not at.exception, at.exception


def test_forecast_and_model_tabs_respond_to_controls():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    series = [s for s in at.selectbox if s.label == "Series"][0]
    series.set_value("AI companies (broad, derived)").run()
    assert not at.exception, at.exception
    [s for s in at.slider if s.label == "Forecast horizon (years)"][0].set_value(5).run()
    assert not at.exception, at.exception
    assert any("Model comparison" in m.value for m in at.markdown)
