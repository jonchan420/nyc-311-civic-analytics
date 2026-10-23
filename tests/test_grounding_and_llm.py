"""The grounding check, the budget guard, the strict tool schemas, and the Claude client
(against a stand-in SDK object; no network, no key)."""

import datetime as dt
import threading
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from analyst.agent import strict_schema, tool_definitions
from analyst.grounding import check, numbers_in, percents_in
from analyst.llm import (BudgetExceeded, BudgetGuard, ClaudeClient, OllamaClient, ProviderError, estimate_cost,
                         ollama_messages)
from analyst.schemas import ComparePeriodsParams

RESULT = {"summary": {"total_a": 244972, "total_b": 261150, "change": 16178, "pct_change": 6.6},
          "rows": [{"category": "Drug Activity", "change": -11479, "pct_change": -93.44}],
          "warnings": ["Jul 2025 recorded 5,497, more than 3 times its typical month (1,389)."]}
PARAMS = {"period_a": {"start": "2025-06", "end": "2025-08"}}


def test_numbers_in_reads_leaves_and_strings():
    assert numbers_in(RESULT) >= {244972.0, 16178.0, 6.6, -11479.0, 5497.0, 1389.0, 3.0}
    assert numbers_in(PARAMS) == {2025.0, 6.0, 8.0}


def test_percents_in_reads_percent_fields_and_percent_signs():
    assert percents_in(RESULT) == {6.6, -93.44}
    assert percents_in({"wape_seasonal_naive": 5.97, "mae_seasonal_naive": 98.0, "note": "53% fewer"}) == {5.97, 53.0}
    assert percents_in(PARAMS) == set()


@pytest.mark.parametrize("text, unsupported", [
    ("Queens recorded 261,150 complaints, up 16,178 (6.6%) from 244,972.", []),
    ("Drug Activity fell by 11,479, or about 93%.", []),                 # sign dropped, rounded
    ("Jun–Aug 2025 had 244,972 complaints.", []),                         # month and year from params
    ("Complaints rose about 7%.", []),                                    # 6.6 rounded to a whole percent
    ("Complaints rose about 8%.", ["8%"]),                                # the 8 in "2025-08" isn't a percent
    ("Complaints rose 1,389%.", ["1,389%"]),                              # a count isn't a percent
    ("Complaints rose 6.7%.", ["6.7%"]),                                  # not within 0.05 of 6.6
    ("Complaints rose 900%.", ["900%"]),
    ("About 16k more complaints were filed.", ["16k"]),                   # abbreviations are rejected
    ("Three categories made up 70% of the change.", ["70%"]),             # a computed share
])
def test_check(text, unsupported):
    report = check(text, RESULT, PARAMS)
    assert report.unsupported == unsupported
    assert report.grounded == (not unsupported)


def test_structural_numbers_are_allowed():
    assert check("The top 3 of the 5 boroughs, per 100,000 residents, over 12 months.", {}).grounded


def test_budget_guard_blocks_before_the_limit():
    guard = BudgetGuard(max_requests=2, max_usd=1.0, max_tokens_per_call=1000)
    guard.reserve(1000)
    guard.record(0.4)
    guard.reserve(500)
    guard.record(0.7)                     # $1.10 spent
    assert (guard.requests, guard.usd) == (2, pytest.approx(1.1))
    with pytest.raises(BudgetExceeded, match="model calls"):
        guard.reserve(500)
    with pytest.raises(BudgetExceeded, match="tokens"):
        BudgetGuard(max_tokens_per_call=100).reserve(101)
    with pytest.raises(BudgetExceeded, match="spending limit"):
        BudgetGuard(max_usd=1.0, usd=1.0).reserve(10)


def test_budget_guard_resets_daily():
    guard = BudgetGuard(max_requests=1, requests=1, usd=5.0, day=dt.date(2026, 1, 1))
    guard.reserve(10)                     # a new day: the counters reset, then this call is counted
    assert (guard.requests, guard.usd, guard.day) == (1, 0.0, dt.date.today())


def test_budget_guard_is_shared_safely_across_threads():
    guard = BudgetGuard(max_requests=50)
    outcomes = []

    def worker():
        try:
            guard.reserve(10)
            outcomes.append("ran")
        except BudgetExceeded:
            outcomes.append("blocked")

    threads = [threading.Thread(target=worker) for _ in range(200)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("ran") == 50 and guard.requests == 50


def test_estimate_cost():
    # 1,000 input tokens at $4/M plus 500 output tokens at $20/M
    assert estimate_cost("claude-opus-5-5", {"input_tokens": 1000, "output_tokens": 500}) == pytest.approx(0.014)


def test_strict_schema_uses_only_the_supported_subset():
    schema = strict_schema(ComparePeriodsParams)
    unsupported = {"minimum", "maximum", "minLength", "maxLength", "pattern", "minItems", "maxItems", "default"}

    def walk(node):
        if isinstance(node, dict):
            assert not unsupported & node.keys(), node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node.get("properties", {}))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    top_n = schema["properties"]["top_n"]
    assert {"type": "null"} in top_n["anyOf"] and "from 1 to 25" in top_n["description"]
    assert "YYYY-MM" in schema["$defs"]["MonthRange"]["properties"]["start"]["description"]
    assert [t["name"] for t in tool_definitions()] == ["compare_periods", "monthly_trend", "month_summary",
                                                       "forecast_status"]
    assert all(t["strict"] for t in tool_definitions())


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.requests = response, error, []

    def create(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.response


def sdk(response=None, error=None):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(response, error)))


def message(content, stop_reason="end_turn"):
    usage = SimpleNamespace(input_tokens=1200, output_tokens=300, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0)
    return SimpleNamespace(content=content, stop_reason=stop_reason, model="claude-opus-5-5", usage=usage)


def test_claude_client_request_and_response():
    tool_use = SimpleNamespace(type="tool_use", id="toolu_1", name="forecast_status", input={})
    thinking = SimpleNamespace(type="thinking", thinking="", signature="sig")
    fake = sdk(message([thinking, SimpleNamespace(type="text", text="Checking."), tool_use], "tool_use"))
    client = ClaudeClient(client=fake, budget=BudgetGuard())
    response = client.complete(system="rules", messages=[{"role": "user", "content": "q"}],
                               tools=tool_definitions(), max_tokens=1000)
    request = fake.beta.messages.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["system"] == [{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}]
    assert request["output_config"] == {"effort": "low"}
    assert (request["betas"], request["fallbacks"]) == (["server-side-fallback-2026-07-01"], "default")
    assert "tool_choice" not in request              # forcing a tool is rejected on this model
    assert response.text == "Checking."
    assert [(c.id, c.name) for c in response.tool_calls] == [("toolu_1", "forecast_status")]
    assert response.raw_content == [thinking, SimpleNamespace(type="text", text="Checking."), tool_use]
    assert response.cost_usd == pytest.approx((1200 * 4 + 300 * 20) / 1e6)
    assert client.budget.requests == 1


def test_claude_client_refusal_and_errors():
    refused = ClaudeClient(client=sdk(message([], "refusal"))).complete(system="s", messages=[])
    assert (refused.stop_reason, refused.text, refused.tool_calls) == ("refusal", "", [])
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    with pytest.raises(ProviderError, match="APIConnectionError"):
        ClaudeClient(client=sdk(error=error)).complete(system="s", messages=[])
    blocked = ClaudeClient(client=sdk(message([])), budget=BudgetGuard(max_requests=0))
    with pytest.raises(BudgetExceeded):
        blocked.complete(system="s", messages=[])
    assert blocked._client.beta.messages.requests == []   # nothing was sent


class FakeSession:
    """Stands in for requests.Session: records the body and returns a canned Ollama reply."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.bodies = reply, error, []

    def post(self, url, json, timeout):
        self.bodies.append((url, json))
        if self.error:
            raise self.error
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: self.reply)


def test_ollama_client_request_and_response():
    reply = {"model": "qwen3:8b", "done_reason": "stop", "prompt_eval_count": 780, "eval_count": 42,
             "message": {"role": "assistant", "content": "",
                         "tool_calls": [{"id": "call_1", "function": {"name": "forecast_status", "arguments": {}}}]}}
    session = FakeSession(reply)
    response = OllamaClient(session=session).complete(system="rules", messages=[{"role": "user", "content": "q"}],
                                                      tools=tool_definitions(), max_tokens=900)
    url, body = session.bodies[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert body["messages"] == [{"role": "system", "content": "rules"}, {"role": "user", "content": "q"}]
    assert (body["model"], body["stream"], body["think"]) == ("qwen3:8b", False, False)
    assert body["options"]["temperature"] == 0 and body["options"]["num_predict"] == 900
    first = body["tools"][0]
    assert first["type"] == "function" and first["function"]["parameters"] == tool_definitions()[0]["input_schema"]
    assert [(c.id, c.name, c.input) for c in response.tool_calls] == [("call_1", "forecast_status", {})]
    assert (response.stop_reason, response.cost_usd, response.usage) == (
        "tool_use", 0.0, {"input_tokens": 780, "output_tokens": 42})


def test_ollama_messages_translate_a_retry():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": "s"},
                                          {"type": "tool_use", "id": "toolu_1", "name": "month_summary",
                                           "input": {"geography": "LONG ISLAND"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True,
                                      "content": "Invalid parameters: geography"}]},
    ]
    assert ollama_messages(messages) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "month_summary", "arguments": {"geography": "LONG ISLAND"}}}]},
        {"role": "tool", "content": "Invalid parameters: geography", "tool_name": "month_summary"},
    ]


def test_ollama_client_string_arguments_and_errors():
    reply = {"message": {"content": "", "tool_calls": [{"function": {"name": "month_summary",
                                                                     "arguments": '{"geography": "QUEENS"}'}}]}}
    response = OllamaClient(session=FakeSession(reply)).complete(system="s", messages=[])
    assert response.tool_calls[0].input == {"geography": "QUEENS"} and response.tool_calls[0].id == "call_0"
    with pytest.raises(ProviderError, match="ConnectionError"):
        OllamaClient(session=FakeSession(error=ConnectionError("refused"))).complete(system="s", messages=[])
