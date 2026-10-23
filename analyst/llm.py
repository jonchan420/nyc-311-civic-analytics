"""The model layer: one small interface, two implementations, and a spending guard.

LLMClient is all agent.py depends on, so the provider can change without
touching the rest. FakeLLM returns scripted responses for tests and the
offline demo; ClaudeClient calls the Anthropic API. BudgetGuard refuses a
call before it would pass the request or dollar limit.
"""

from __future__ import annotations

import copy
import datetime as dt
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

DEFAULT_MODEL = "claude-opus-5-5"
# USD per million tokens. Input/output prices and the Opus/Sonnet cache-read prices are
# from Anthropic's SDK guidance bundled with Claude Code (Oct 2026); cache writes are
# assumed at 1.25x input and Haiku's cache read at 0.1x input. These feed the budget
# guard's estimate only; check https://www.anthropic.com/pricing for billing.
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
}
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    # the provider's own content blocks, passed back unchanged when a conversation
    # continues (Claude requires its thinking blocks to be returned as they came)
    raw_content: list[Any] = field(default_factory=list)


class LLMClient(Protocol):
    name: str

    def complete(self, *, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]] | None = None, max_tokens: int = 2000) -> LLMResponse: ...


class BudgetExceeded(RuntimeError):
    """The next call would go over a configured limit; no call was made."""


class ProviderError(RuntimeError):
    """The provider failed (network, auth, rate limit, server error) after its own retries."""


@dataclass
class BudgetGuard:
    """Daily request and spending caps, checked before every call.

    One guard can be shared by every session of the app: reserve() counts the
    request before the call is made, under a lock, so concurrent sessions
    can't both take the last slot. Spending can overshoot max_usd by at most
    the calls already in flight. Counters live in memory and reset when the
    app restarts; a monthly limit set in the Anthropic Console is the backstop.
    """

    max_requests: int = 20
    max_usd: float = 2.00
    max_tokens_per_call: int = 2000
    requests: int = 0
    usd: float = 0.0
    day: dt.date = field(default_factory=dt.date.today)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def reserve(self, max_tokens: int) -> None:
        """Raise BudgetExceeded, or count one request for the call about to be made."""
        with self._lock:
            if dt.date.today() != self.day:
                self.requests, self.usd, self.day = 0, 0.0, dt.date.today()
            if max_tokens > self.max_tokens_per_call:
                raise BudgetExceeded(f"A call asked for {max_tokens} tokens; the limit is {self.max_tokens_per_call}.")
            if self.requests >= self.max_requests:
                raise BudgetExceeded(f"The limit of {self.max_requests} model calls a day has been reached.")
            if self.usd >= self.max_usd:
                raise BudgetExceeded(f"The ${self.max_usd:.2f} daily spending limit has been reached.")
            self.requests += 1

    def record(self, cost_usd: float) -> None:
        with self._lock:
            self.usd += cost_usd


def estimate_cost(model: str, usage: dict[str, int]) -> float:
    price = PRICES.get(model)
    if price is None:  # unknown model: count it at the most expensive listed rate
        price = max(PRICES.values(), key=lambda p: p["output"])
    return (usage.get("input_tokens", 0) * price["input"]
            + usage.get("output_tokens", 0) * price["output"]
            + usage.get("cache_creation_input_tokens", 0) * price["cache_write"]
            + usage.get("cache_read_input_tokens", 0) * price["cache_read"]) / 1_000_000


class FakeLLM:
    """Scripted responses for tests and the offline demo; records every request.

    script is either a list of LLMResponses returned in order, or a function
    (system, messages, tools) -> LLMResponse.
    """

    name = "fake"

    def __init__(self, script: list[LLMResponse] | Callable[..., LLMResponse], budget: BudgetGuard | None = None):
        self.script = script
        self.budget = budget
        self.requests: list[dict[str, Any]] = []

    def complete(self, *, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]] | None = None, max_tokens: int = 2000) -> LLMResponse:
        if self.budget:
            self.budget.reserve(max_tokens)
        self.requests.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        if callable(self.script):
            response = self.script(system, messages, tools)
        else:
            if not self.script:
                raise AssertionError("FakeLLM ran out of scripted responses")
            response = self.script.pop(0)
        if self.budget:
            self.budget.record(response.cost_usd)
        return response


class ClaudeClient:
    """Claude through the official anthropic SDK (imported only when used)."""

    name = "claude"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL, effort: str = "low",
                 budget: BudgetGuard | None = None, client: Any = None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=60.0)
        self._client = client
        self.model = model
        self.effort = effort
        self.budget = budget or BudgetGuard()

    def complete(self, *, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]] | None = None, max_tokens: int = 2000) -> LLMResponse:
        self.budget.reserve(max_tokens)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            # the system prompt is identical across calls, so it is cached
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "output_config": {"effort": self.effort},
            # if a safety classifier declines, the API retries on a fallback model it picks
            "betas": [REFUSAL_FALLBACK_BETA],
            "fallbacks": "default",
        }
        if tools:
            request["tools"] = tools  # tool_choice stays "auto": forcing a tool is rejected on this model
        import anthropic
        try:
            response = self._client.beta.messages.create(**request)
        except anthropic.APIError as error:  # the SDK has already retried 429s, 5xx, and connection errors
            raise ProviderError(f"{type(error).__name__}: {error}") from error
        usage = {name: getattr(response.usage, name, 0) or 0 for name in
                 ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
        cost = estimate_cost(response.model, usage)
        self.budget.record(cost)
        if response.stop_reason == "refusal":
            return LLMResponse("", [], "refusal", response.model, usage, cost, list(response.content))
        return LLMResponse(
            text="".join(block.text for block in response.content if block.type == "text"),
            tool_calls=[ToolCall(block.id, block.name, dict(block.input))
                        for block in response.content if block.type == "tool_use"],
            stop_reason=response.stop_reason,
            model=response.model,
            usage=usage,
            cost_usd=cost,
            raw_content=list(response.content),
        )
