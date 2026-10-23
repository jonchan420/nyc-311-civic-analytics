"""The MCP server returns exactly what the app computes, through a real MCP client.

Every dev eval case that should be answered with a tool is run both ways on the
fixtures; apart from the clock time in the provenance, the results must be identical.
"""

import asyncio
import json
from pathlib import Path

import pytest
from mcp import Client

from analyst.agent import TOOLS, Analyst
from analyst.mcp_server import mcp

ROOT = Path(__file__).resolve().parents[1]
DEV = [json.loads(line) for line in (ROOT / "eval" / "dev_cases.jsonl").read_text().splitlines()]
TOOL_CASES = [case for case in DEV if case["expect"]["kind"] == "answer"]


def over_mcp(method: str, *args):
    async def go():
        async with Client(mcp) as client:
            return await getattr(client, method)(*args)
    return asyncio.run(go())


@pytest.fixture
def fixtures(fixture_dir, monkeypatch):
    monkeypatch.setenv("ASK311_DATA_DIR", str(fixture_dir))
    monkeypatch.setenv("ASK311_FORECAST_DIR", str(fixture_dir / "forecasts"))
    return fixture_dir


def without_clock(result: dict) -> dict:
    return {**result, "provenance": {k: v for k, v in result["provenance"].items() if k != "computed_at"}}


@pytest.mark.parametrize("case", TOOL_CASES, ids=[case["id"] for case in TOOL_CASES])
def test_mcp_matches_the_app(fixtures, case):
    tool, params = case["expect"]["tool"], case["expect"]["params"]
    in_app = Analyst(fixtures, fixtures / "forecasts").run_tool(tool, params)
    remote = over_mcp("call_tool", tool, params)
    assert not remote.is_error
    expected = {"answer": in_app.text, **in_app.result.model_dump(mode="json"),
                "method_notes": [chunk.id for chunk, _ in in_app.notes]}
    assert without_clock(remote.structured_content) == without_clock(expected)


def test_the_tools_are_the_app_tools_and_read_only(fixtures):
    listed = {tool.name: tool for tool in over_mcp("list_tools").tools}
    assert set(listed) == set(TOOLS) | {"search_method_notes"}
    assert all(tool.annotations.read_only_hint and not tool.annotations.open_world_hint for tool in listed.values())
    for name, tool in TOOLS.items():                     # the MCP signatures can't drift from the app's models
        assert set(listed[name].input_schema.get("properties", {})) == set(tool.params.model_fields)
        assert listed[name].description == tool.description


@pytest.mark.parametrize("tool, arguments, message", [
    ("month_summary", {"geography": "QUEENS", "month": "2026-03"}, "complete only through Feb 2026"),
    ("compare_periods", {"geography": "QUEENS", "period_a": {"start": "2026-01", "end": "2026-01"},
                         "period_b": {"start": "2025-01", "end": "2025-01"}}, "period_a must end before period_b starts"),
    ("monthly_trend", {"geography": "LONG ISLAND", "period": {"start": "2025-01", "end": "2025-02"}}, "geography"),
    ("monthly_trend", {"geography": "QUEENS", "period": {"start": "2025-01", "end": "2025-02"},
                       "categories": ["Bike Lane"]}, "Bike Lane"),
])
def test_errors_come_back_as_messages(fixtures, tool, arguments, message):
    remote = over_mcp("call_tool", tool, arguments)
    assert remote.is_error and message in remote.content[0].text


def test_method_notes_search(fixtures):
    remote = over_mcp("call_tool", "search_method_notes", {"query": "what does last month mean"})
    assert remote.structured_content["result"][0]["id"] == "complete-months#what-last-month-means"
