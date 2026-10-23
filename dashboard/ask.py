"""Ask 311: questions about recorded NYC 311 complaints, answered by checked calculations.

Offline by default: rules read the question and templates write the answer,
with no model and no key. With ANTHROPIC_API_KEY and ASK_PASSCODE in the
app's secrets, someone with the passcode can turn on Claude for questions
the rules can't place and for the wording. Every model answer is checked
against the tool result, and app-wide daily limits cap calls and spending
(see .streamlit/secrets.toml.example).
"""

from __future__ import annotations

import hmac
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analyst.agent import Analyst, Answer, freshness_line
from analyst.briefing import briefing_markdown
from analyst.catalog import BOROUGHS
from analyst.charts import chart_for
from analyst.data import month_name, shift_month
from analyst.llm import DEFAULT_MODEL, BudgetGuard, ClaudeClient
from analyst.retrieval import KnowledgeBase
from analyst.router import RouterContext
from analyst.writer import period_label
from config.settings import FORECAST_DIR, LIVE_DIR

# the tests point the page at the synthetic fixtures
DATA_DIR = Path(os.environ.get("ASK311_DATA_DIR", LIVE_DIR))
FORECASTS = Path(os.environ.get("ASK311_FORECAST_DIR", FORECAST_DIR))
EXAMPLES = [
    "Which complaint types drove the change in Queens between last summer and the summer before?",
    "Prepare a briefing on last month in Manhattan.",
    "What does the forecast predict for next month, and how accurate has it been?",
    "How have noise complaints in Brooklyn changed over time?",
    "Why don't borough totals add up to the citywide total?",
]
MAX_PASSCODE_TRIES = 5
FORM_TOOLS = {"Compare two periods": "compare_periods", "Monthly trend": "monthly_trend",
              "One-month briefing": "month_summary", "Forecast and its track record": "forecast_status"}
PLACES = {"Citywide": "CITYWIDE", **{b.title(): b for b in BOROUGHS}}


def secret(name: str, default: Any = None) -> Any:
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:  # no secrets file: offline mode
        return default


@st.cache_resource(ttl=3600)
def shared_parts(data_dir: str) -> tuple[RouterContext, KnowledgeBase]:
    """Read-only pieces every session can share."""
    return RouterContext.load(Path(data_dir)), KnowledgeBase.load()


@st.cache_resource
def shared_budget() -> BudgetGuard:
    """One guard for the whole app, so the limits hold across every visitor."""
    return BudgetGuard(max_requests=int(secret("ASK_MAX_CALLS_PER_DAY", 20)),
                       max_usd=float(secret("ASK_MAX_USD_PER_DAY", 1.00)))


@st.cache_resource
def shared_claude() -> ClaudeClient:
    return ClaudeClient(api_key=secret("ANTHROPIC_API_KEY"), model=secret("ASK_MODEL", DEFAULT_MODEL),
                        effort=secret("ASK_EFFORT", "low"), budget=shared_budget())


def make_analyst(use_model: bool) -> Analyst:
    """A fresh Analyst per question: cheap, and no state shared between visitors."""
    context, knowledge = shared_parts(str(DATA_DIR))
    return Analyst(DATA_DIR, FORECASTS, knowledge=knowledge, llm=shared_claude() if use_model else None,
                   context=context)


def safe_markdown(text: str) -> str:
    return text.replace("$", "\\$")  # "$" would start LaTeX in st.markdown


# ------------------------------------------------------------------ answer mode

def check_passcode() -> None:
    entered = st.session_state.get("passcode", "")
    st.session_state["passcode"] = ""
    if entered and hmac.compare_digest(entered.encode(), str(secret("ASK_PASSCODE")).encode()):
        st.session_state["unlocked"] = True
    elif entered:
        st.session_state["tries"] = st.session_state.get("tries", 0) + 1


def answer_mode() -> bool:
    """True when this session may use Claude. Offline unless a key and a passcode are set."""
    offline_note = ("**Offline mode.** Rules read each question and fixed templates write the answer from the "
                    "calculation, so no AI model is used and nothing leaves this app.")
    with st.sidebar:
        st.subheader("Answer mode")
        if not (secret("ANTHROPIC_API_KEY") and secret("ASK_PASSCODE")):
            st.markdown(offline_note)
            return False
        if not st.session_state.get("unlocked"):
            st.markdown(offline_note)
            tries = st.session_state.get("tries", 0)
            if tries >= MAX_PASSCODE_TRIES:
                st.caption("Too many wrong passcodes; Claude stays off for this session.")
                return False
            st.text_input("Passcode to turn on Claude", type="password", key="passcode", on_change=check_passcode)
            if tries:
                st.caption(f"That passcode isn't right ({MAX_PASSCODE_TRIES - tries} tries left).")
            return False
        use_model = st.toggle("Use Claude", value=True, key="use_model")
        budget = shared_budget()
        st.caption(f"Claude ({secret('ASK_MODEL', DEFAULT_MODEL)}) reads questions the rules can't place and "
                   "words the answers. Any answer with a number that isn't in the calculation is replaced by the "
                   f"template. Today, for the whole app: {budget.requests} of {budget.max_requests} model calls and "
                   f"\\${budget.usd:.2f} of \\${budget.max_usd:.2f} used.")
        return use_model


# ------------------------------------------------------------------ showing an answer

def show_answer(entry: dict[str, Any], index: int) -> None:
    answer: Answer = entry["answer"]
    st.markdown(safe_markdown(answer.text))
    if answer.kind == "clarify" and answer.options:
        st.caption("Options: " + ", ".join(answer.options) + ". Ask again with one of them, or build the query below.")
    if entry.get("model_error"):
        st.caption(f"Claude wasn't used for this answer: {entry['model_error']}")
    result = answer.result
    if result is not None:
        figure = chart_for(answer.tool, result)
        if figure is not None:
            st.plotly_chart(figure, key=f"chart_{index}")
        if result.warnings:
            with st.expander(f"Warnings ({len(result.warnings)})", expanded=False):
                st.markdown("\n".join(f"- {safe_markdown(w)}" for w in result.warnings))
        with st.expander("How this was calculated"):
            p = result.provenance
            st.markdown(f"**Tool:** `{p.tool}`  \n**Parameters:** `{json.dumps(answer.params)}`  \n"
                        f"**Unit:** {p.unit}  \n**Calculation:** {safe_markdown(p.calculation)}")
            st.markdown("\n".join(f"- {safe_markdown(f)}" for f in p.filters))
            if result.rows:
                st.dataframe(pd.DataFrame([row.model_dump(exclude={"labels"}) for row in result.rows]),
                             hide_index=True, key=f"rows_{index}")
            st.caption(" · ".join(f"{s.path} (SHA-256 {s.sha256[:12]}…)" for s in p.sources))
        st.download_button("Download this answer as a briefing (.md)", briefing_markdown(entry["question"], answer),
                           file_name=f"nyc311_{answer.tool}_{p.computed_at[:10]}.md", mime="text/markdown",
                           key=f"briefing_{index}", on_click="ignore")
    if answer.notes:
        with st.expander("Method notes used"):
            for chunk, _ in answer.notes:
                st.markdown(f"**{chunk.title}: {chunk.section}**  \n{safe_markdown(chunk.text)}")
    footer = [f"Read by {answer.routed_by}", f"written by {answer.written_by}"]
    if answer.model_calls:
        calls = f"{answer.model_calls} model call{'s' if answer.model_calls != 1 else ''}"
        footer.append(f"{calls}, about \\${answer.cost_usd:.4f}")
    if result is not None:
        footer.insert(0, freshness_line(result))
    st.caption(" · ".join(footer))


def ask(question: str, use_model: bool) -> None:
    analyst = make_analyst(use_model)
    with st.spinner("Working it out…"):
        answer = analyst.ask(question)
    st.session_state["history"].append({"question": question, "answer": answer,
                                        "model_error": analyst.last_model_error})


def run_form(tool: str, params: dict[str, Any], description: str) -> None:
    analyst = make_analyst(False)  # the form needs no model to read it; templates write the answer
    st.session_state["history"].append({"question": description,
                                        "answer": analyst.run_tool(tool, params, question=description)})


def queue(question: str) -> None:
    st.session_state["queued"] = question


# ------------------------------------------------------------------ the form

def query_form(context: RouterContext) -> None:
    months = []
    month = context.complete_through
    while month >= context.first_month:
        months.append(month)
        month = shift_month(month, -1)
    with st.expander("Or build a query from lists"):
        label = st.selectbox("Analysis", list(FORM_TOOLS), key="form_tool")
        tool = FORM_TOOLS[label]
        params: dict[str, Any] = {}
        if tool != "forecast_status":
            place = st.selectbox("Place", list(PLACES), key="form_place")
            params["geography"] = PLACES[place]
        if tool == "compare_periods":
            left, middle, right = st.columns(3)
            length = left.selectbox("Months in each period", [1, 3, 6, 12], index=1, key="form_length")
            end_b = middle.selectbox("Later period ends", months, format_func=month_name, key="form_end_b")
            end_a = right.selectbox("Earlier period ends", months, index=min(12, len(months) - 1),
                                    format_func=month_name, key="form_end_a")
            params["period_a"] = {"start": shift_month(end_a, -(length - 1)), "end": end_a}
            params["period_b"] = {"start": shift_month(end_b, -(length - 1)), "end": end_b}
            categories = st.multiselect("Complaint types (leave empty for all)", context.categories,
                                        max_selections=25, key="form_categories")
            description = (f"Compare {place}: {period_label(params['period_a'])} vs "
                           f"{period_label(params['period_b'])}")
        elif tool == "monthly_trend":
            ascending = months[::-1]
            start, end = st.select_slider("Months", ascending, value=(ascending[max(0, len(ascending) - 12)],
                                                                      ascending[-1]),
                                          format_func=month_name, key="form_range")
            params["period"] = {"start": start, "end": end}
            categories = st.multiselect("Complaint types (up to 10; leave empty for all)", context.categories,
                                        max_selections=10, key="form_trend_categories")
            description = f"Monthly trend, {place}: {period_label(params['period'])}"
        elif tool == "month_summary":
            params["month"] = st.selectbox("Month", months, format_func=month_name, key="form_month")
            categories = []
            description = f"Briefing, {place}: {month_name(params['month'])}"
        else:
            categories = []
            description = "Forecast and its track record"
        if categories:
            params["categories"] = categories
            description += f" ({', '.join(categories)})"
        if st.button("Run", key="form_run", type="primary"):
            run_form(tool, params, description)


# ------------------------------------------------------------------ the page

st.title("Ask 311")
st.markdown("Ask about recorded NYC 311 complaints by borough, month, and complaint type. Each answer comes "
            "from one checked calculation on the same data as the dashboard, with its chart, warnings, and "
            "exactly how it was calculated. Counts measure who reports, not only conditions.")
context, _ = shared_parts(str(DATA_DIR))
st.caption(f"Data from {month_name(context.first_month)} through {month_name(context.complete_through)}, "
           "the latest complete month.")
st.session_state.setdefault("history", [])
use_model = answer_mode()
with st.sidebar:
    if st.session_state["history"] and st.button("Clear the conversation"):
        st.session_state["history"] = []

query_form(context)
typed = st.chat_input("Ask about recorded 311 complaints…", max_chars=500)
question = typed or st.session_state.pop("queued", None)
if question:
    ask(question, use_model)

for index, entry in enumerate(st.session_state["history"]):
    with st.chat_message("user"):
        st.markdown(safe_markdown(entry["question"]))
    with st.chat_message("assistant"):
        show_answer(entry, index)

if not st.session_state["history"]:
    st.markdown("**Try one of these:**")
    for i, example in enumerate(EXAMPLES):
        st.button(example, key=f"example_{i}", on_click=queue, args=(example,))
