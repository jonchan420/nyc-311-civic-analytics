"""Plotly charts built from tool results. The model never draws or edits a chart."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analyst.schemas import ToolResult
from analyst.writer import period_label

INCREASE, DECREASE, NEUTRAL = "#c0392b", "#2c7fb8", "#7f8c8d"


def chart_for(tool: str, result: ToolResult) -> go.Figure | None:
    builder = {"compare_periods": _compare, "monthly_trend": _trend,
               "month_summary": _summary, "forecast_status": _forecast}.get(tool)
    return builder(result) if builder and result.rows else None


def _compare(result: ToolResult) -> go.Figure:
    p = result.provenance.params
    rows = sorted(result.rows, key=lambda r: r.change)
    fig = go.Figure(go.Bar(
        x=[r.change for r in rows], y=[r.category for r in rows], orientation="h",
        marker_color=[INCREASE if r.change > 0 else DECREASE if r.change < 0 else NEUTRAL for r in rows],
        customdata=[[r.count_a, r.count_b] for r in rows],
        hovertemplate="%{y}<br>change %{x:+,}<br>%{customdata[0]:,} → %{customdata[1]:,}<extra></extra>",
    ))
    fig.update_layout(title=f"Change in recorded complaints, {period_label(p['period_a'])} to "
                            f"{period_label(p['period_b'])}",
                      xaxis_title="Change in complaints", yaxis_title=None, height=120 + 32 * len(rows))
    return fig


def _trend(result: ToolResult) -> go.Figure:
    frame = pd.DataFrame([r.model_dump() for r in result.rows])
    frame["month"] = pd.to_datetime(frame["month"])
    fig = px.line(frame, x="month", y="count", color="category", markers=True,
                  labels={"count": "Recorded complaints", "month": "", "category": ""})
    spikes = frame[frame["flags"].apply(lambda flags: "spike" in flags)]
    if not spikes.empty:
        fig.add_scatter(x=spikes["month"], y=spikes["count"], mode="markers", name="unusual month",
                        marker=dict(size=13, symbol="circle-open", color=INCREASE, line=dict(width=2)))
    fig.update_xaxes(dtick="M1" if len(frame["month"].unique()) <= 18 else "M3", tickformat="%b %Y")
    fig.update_layout(title="Recorded complaints per month", legend_title=None)
    return fig


def _summary(result: ToolResult) -> go.Figure:
    rows = sorted(result.rows, key=lambda r: r.count)
    month = result.summary["month"]
    fig = go.Figure([
        go.Bar(y=[r.category for r in rows], x=[r.last_year or 0 for r in rows], orientation="h",
               name=f"{result.summary['last_year_month']}", marker_color="#bdc3c7"),
        go.Bar(y=[r.category for r in rows], x=[r.count for r in rows], orientation="h",
               name=month, marker_color="#34495e"),
    ])
    fig.update_layout(barmode="group", title=f"Top complaint types in {month}, against a year before",
                      xaxis_title="Recorded complaints", height=140 + 44 * len(rows))
    return fig


def _forecast(result: ToolResult) -> go.Figure:
    frame = pd.DataFrame([r.model_dump() for r in result.rows])
    frame["month"] = pd.to_datetime(frame["month"])
    fig = px.line(frame, x="month", y="predicted", color="model", line_dash="status", markers=True,
                  labels={"predicted": "Recorded complaints (citywide)", "month": "", "model": "",
                          "status": ""})
    actual = frame.dropna(subset=["actual"]).drop_duplicates("month")
    fig.add_scatter(x=actual["month"], y=actual["actual"], mode="lines+markers", name="actual",
                    line=dict(color="black", width=3))
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    fig.update_layout(title="Frozen forecasts and actual counts")
    return fig
