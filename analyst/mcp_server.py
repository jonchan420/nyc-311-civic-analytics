"""The analysis tools as an MCP server, so MCP clients (Claude Desktop, Claude Code,
other agents) use the same checked calculations as the app.

    python -m analyst.mcp_server            # stdio, from the repository root

Each tool validates its arguments with the same Pydantic model as the app, runs
the same allowlisted query, and returns the same ToolResult (numbers, warnings,
provenance) with the template answer. A client's model can word its own answer,
but the numbers come from here. Every tool only reads; no model is called.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analyst.agent import TOOLS, Analyst  # noqa: E402
from analyst.retrieval import KnowledgeBase  # noqa: E402
from analyst.router import RouterContext  # noqa: E402
from analyst.schemas import CategoryName, Geography, MonthRange, YearMonth  # noqa: E402
from config.settings import FORECAST_DIR, LIVE_DIR  # noqa: E402

INSTRUCTIONS = """Monthly counts of recorded NYC 311 service requests by borough and complaint type, \
from NYC Open Data. Places are CITYWIDE or one of BRONX, BROOKLYN, MANHATTAN, QUEENS, STATEN ISLAND; \
months are written YYYY-MM and only complete months are available. Counts measure reporting, not \
conditions: say "recorded complaints", don't claim causes, and quote numbers from the tool results \
as given. Each result includes its warnings and provenance; mention the warnings."""
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)

mcp = MCPServer("nyc-311-analyst", instructions=INSTRUCTIONS)
_shared: dict[tuple[Path, Path], tuple[RouterContext, KnowledgeBase]] = {}


def analyst() -> Analyst:
    """An Analyst with no model: these tools only calculate."""
    data_dir = Path(os.environ.get("ASK311_DATA_DIR", LIVE_DIR))
    forecast_dir = Path(os.environ.get("ASK311_FORECAST_DIR", FORECAST_DIR))
    if (data_dir, forecast_dir) not in _shared:
        _shared[data_dir, forecast_dir] = RouterContext.load(data_dir), KnowledgeBase.load()
    context, knowledge = _shared[data_dir, forecast_dir]
    return Analyst(data_dir, forecast_dir, knowledge=knowledge, context=context)


def run(tool: str, params: dict[str, Any]) -> dict[str, Any]:
    params = {name: value.model_dump() if isinstance(value, MonthRange) else value
              for name, value in params.items() if value is not None}
    answer = analyst().run_tool(tool, params)
    if answer.result is None:  # invalid parameters or months outside the data: the message says which
        raise ToolError(answer.text)
    return {"answer": answer.text, **answer.result.model_dump(mode="json"),
            "method_notes": [chunk.id for chunk, _ in answer.notes]}


@mcp.tool(description=TOOLS["compare_periods"].description, annotations=READ_ONLY)
def compare_periods(
    geography: Geography,
    period_a: Annotated[MonthRange, Field(description="The earlier, baseline period")],
    period_b: Annotated[MonthRange, Field(description="The later period, the same length as period_a")],
    categories: Annotated[list[CategoryName] | None,
                          Field(description="Up to 25 complaint types; omit for all of them")] = None,
    top_n: Annotated[int, Field(ge=1, le=25, description="How many complaint types to return")] = 10,
) -> dict[str, Any]:
    return run("compare_periods", {"geography": geography, "period_a": period_a, "period_b": period_b,
                                   "categories": categories, "top_n": top_n})


@mcp.tool(description=TOOLS["monthly_trend"].description, annotations=READ_ONLY)
def monthly_trend(
    geography: Geography,
    period: MonthRange,
    categories: Annotated[list[CategoryName] | None,
                          Field(description="Up to 10 complaint types, one series each; omit for all")] = None,
) -> dict[str, Any]:
    return run("monthly_trend", {"geography": geography, "period": period, "categories": categories})


@mcp.tool(description=TOOLS["month_summary"].description, annotations=READ_ONLY)
def month_summary(
    geography: Geography,
    month: YearMonth,
    top_n: Annotated[int, Field(ge=1, le=20, description="How many top complaint types to list")] = 8,
) -> dict[str, Any]:
    return run("month_summary", {"geography": geography, "month": month, "top_n": top_n})


@mcp.tool(description=TOOLS["forecast_status"].description, annotations=READ_ONLY)
def forecast_status() -> dict[str, Any]:
    return run("forecast_status", {})


@mcp.tool(annotations=READ_ONLY)
def search_method_notes(
    query: Annotated[str, Field(max_length=500, description="What to look up, in plain words")],
    k: Annotated[int, Field(ge=1, le=5)] = 3,
) -> list[dict[str, str]]:
    """The project's method notes on what the data means, how numbers are calculated, and their limits."""
    return [{"id": chunk.id, "title": chunk.title, "section": chunk.section, "text": chunk.text}
            for chunk, _ in analyst().knowledge.search(query, k=k)]


if __name__ == "__main__":
    mcp.run()
