"""A Markdown briefing for one answer: the text, the numbers behind it, the
warnings, and how it was calculated. Built only from the Answer, so every
number in it comes from the tool result or its provenance."""

from __future__ import annotations

import json
from typing import Any

from analyst.agent import Answer
from analyst.data import month_name
from analyst.writer import CLOSING

# column titles for the row fields the tools return; anything else keeps its field name
COLUMNS = {
    "category": "Complaint type", "count": "Count", "count_a": "First period", "count_b": "Second period",
    "change": "Change", "pct_change": "% change", "share_of_change": "Share of change (%)",
    "prior_month": "Month before", "last_year": "A year before", "change_vs_last_year": "Change vs a year before",
    "pct_vs_last_year": "% vs a year before", "share_of_month": "Share of month (%)", "month": "Month",
    "model": "Model", "status": "Status", "predicted": "Predicted", "actual": "Actual", "error": "Error",
    "pct_error": "% error", "flags": "Flags",
}
LEFT_OUT = {"labels", "trained_through", "made_on", "forecast_file"}  # in the provenance instead, or too long


def _cell(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".") if value % 1 else f"{value:,.0f}"
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value).replace("|", "\\|")


def rows_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields = [name for name in rows[0] if name not in LEFT_OUT]
    lines = ["| " + " | ".join(COLUMNS.get(name, name) for name in fields) + " |",
             "|" + "|".join("---" for _ in fields) + "|"]
    lines += ["| " + " | ".join(_cell(row[name]) for name in fields) + " |" for row in rows]
    return "\n".join(lines)


def briefing_markdown(question: str, answer: Answer) -> str:
    out = ["# NYC 311 briefing", "", f"**Question:** {question}", "", answer.text, ""]
    if answer.result is None:
        return "\n".join(out)
    result, p = answer.result, answer.result.provenance
    rows = [row.model_dump() for row in result.rows]
    if rows:
        out += ["## The numbers", "", rows_table(rows), ""]
    if result.warnings:
        out += ["## Warnings", ""] + [f"- {w}" for w in result.warnings] + [""]
    if answer.assumptions:
        out += ["## Assumptions", ""] + [f"- {a}" for a in answer.assumptions] + [""]
    out += [
        "## How this was calculated", "",
        f"- Tool: `{p.tool}` with parameters `{json.dumps(answer.params)}`",
        f"- Unit: {p.unit}",
        f"- Calculation: {p.calculation}",
        *[f"- Filter: {f}" for f in p.filters],
        *[f"- Source: `{s.path}` (SHA-256 {s.sha256[:12]}…)" for s in p.sources],
        f"- Data complete through {month_name(p.data_complete_through)}; refreshed {p.data_refreshed_at[:10]}; "
        f"computed {p.computed_at[:19].replace('T', ' ')} UTC",
        f"- Question read by: {answer.routed_by}; answer written by: {answer.written_by}",
        "",
    ]
    if answer.notes:
        out += ["## Method notes", ""] + [f"- {chunk.title}: {chunk.section}" for chunk, _ in answer.notes] + [""]
    if CLOSING not in answer.text:
        out += [CLOSING, ""]
    return "\n".join(out)
