"""The Ask 311 page, run headless with Streamlit's AppTest against the fixtures.

Offline is the default; the Claude path is exercised with a FakeLLM in place
of ClaudeClient, so no test touches the network or needs a key.
"""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from analyst.llm import FakeLLM, LLMResponse

PAGE = "dashboard/ask.py"
SECRETS = {"ANTHROPIC_API_KEY": "sk-ant-test-0123456789", "ASK_PASSCODE": "correct horse battery staple"}
# hand-checked in tests/test_agent.py
QUEENS_FEB = "Queens recorded 925 complaints in Feb 2026, down 120 (-11.5%) from Feb 2025 and down 300 (-24.5%) from Jan 2026."


@pytest.fixture
def page(fixture_dir, monkeypatch):
    monkeypatch.setenv("ASK311_DATA_DIR", str(fixture_dir))
    monkeypatch.setenv("ASK311_FORECAST_DIR", str(fixture_dir / "forecasts"))
    st.cache_resource.clear()   # the shared budget and client must not leak between tests
    return AppTest.from_file(PAGE, default_timeout=30)


def last_answer(at: AppTest) -> str:
    return at.chat_message[-1].markdown[0].value


def all_text(at: AppTest) -> str:
    return " ".join(e.value for kind in ("markdown", "caption", "title") for e in at.get(kind))


def test_offline_by_default(page):
    at = page.run()
    assert not at.exception
    assert "Offline mode." in at.sidebar.markdown[0].value
    assert len(at.get("text_input")) == 0 and len(at.toggle) == 0          # no passcode box without secrets
    assert "Data from Jan 2025 through Feb 2026" in at.caption[0].value

    at.chat_input[0].set_value("How was Queens in February 2026?").run()
    assert not at.exception
    assert last_answer(at).startswith(QUEENS_FEB)
    labels = [e.label for e in at.expander]
    assert {"How this was calculated", "Method notes used", "Warnings (2)"} <= set(labels)
    assert any("Read by rules · written by template" in c.value for c in at.caption)
    assert len(at.get("download_button")) == 1


def test_example_buttons_ask_once(page):
    at = page.run()
    at.button(key="example_2").click().run()                              # the forecast example
    assert last_answer(at).startswith("The latest frozen forecasts were made on Mar 5, 2026")
    at.run()                                                              # a rerun doesn't ask again
    assert len(at.chat_message) == 2
    assert not [b for b in at.button if (b.key or "").startswith("example_")]   # examples only on a blank page


def test_the_form_runs_a_tool_without_the_model(page):
    at = page.run()
    at.selectbox(key="form_tool").set_value("One-month briefing").run()
    at.selectbox(key="form_place").set_value("Queens")
    at.selectbox(key="form_month").set_value("2026-02")
    at.button(key="form_run").click().run()
    assert not at.exception
    assert at.chat_message[0].markdown[0].value == "Briefing, Queens: Feb 2026"
    assert last_answer(at).startswith(QUEENS_FEB)
    assert any("Read by form" in c.value for c in at.caption)


def test_the_form_reports_invalid_periods(page):
    at = page.run()
    at.selectbox(key="form_end_b").set_value("2025-06")
    at.selectbox(key="form_end_a").set_value("2026-02")                    # "earlier" ends after "later"
    at.button(key="form_run").click().run()
    assert not at.exception
    assert last_answer(at).startswith("I couldn't run that as asked")


def test_wrong_passcodes_lock_the_model_out(page):
    page.secrets.update(SECRETS)
    at = page.run()
    for attempt in range(5):
        at.text_input(key="passcode").input(f"guess {attempt}").run()
    assert any("Too many wrong passcodes" in c.value for c in at.sidebar.caption)
    assert len(at.toggle) == 0 and len(at.get("text_input")) == 0


def test_the_passcode_turns_on_the_model_within_its_limits(page, monkeypatch):
    calls = []

    def fake_claude(**kwargs):
        calls.append(kwargs)
        return FakeLLM(lambda system, messages, tools: LLMResponse(
            "Queens recorded 925 complaints in Feb 2026, 120 fewer than in Feb 2025.", [], "end_turn",
            "fake", cost_usd=0.0123), budget=kwargs["budget"])

    monkeypatch.setattr("analyst.llm.ClaudeClient", fake_claude)
    page.secrets.update(SECRETS | {"ASK_MAX_CALLS_PER_DAY": 3})
    at = page.run()
    at.text_input(key="passcode").input("not it").run()
    assert any("tries left" in c.value for c in at.sidebar.caption)
    at.text_input(key="passcode").input(SECRETS["ASK_PASSCODE"]).run()
    assert at.toggle(key="use_model").value is True
    assert any("0 of 3 model calls" in c.value for c in at.sidebar.caption)

    at.chat_input[0].set_value("How was Queens in February 2026?").run()
    assert not at.exception
    assert last_answer(at) == "Queens recorded 925 complaints in Feb 2026, 120 fewer than in Feb 2025."
    assert any("written by model · 1 model call, about \\$0.0123" in c.value for c in at.caption)  # $ escaped
    assert calls[0]["api_key"] == SECRETS["ANTHROPIC_API_KEY"]               # read from secrets, server side
    assert SECRETS["ANTHROPIC_API_KEY"] not in all_text(at)                 # and never shown

    at.toggle(key="use_model").set_value(False).run()
    at.chat_input[0].set_value("How was Queens in February 2026?").run()
    assert last_answer(at).startswith(QUEENS_FEB)                           # offline again


def test_the_entry_point_serves_both_pages():
    """The deployed app: dashboard/app.py on the committed data in data/live/."""
    st.cache_resource.clear()
    at = AppTest.from_file("dashboard/app.py", default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "NYC 311 Civic Analytics and Complaint Forecasting"
    at.switch_page("ask.py").run()
    assert not at.exception and at.title[0].value == "Ask 311"
