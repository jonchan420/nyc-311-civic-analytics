"""Streamlit dashboard for NYC 311 Civic Analytics.

Run from the project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import LIVE_DIR, PROCESSED_DIR
from src.forecast_registry import load_forecasts, score, summarize
from src.monthly_data import load_monthly_counts

st.set_page_config(page_title="NYC 311 Civic Analytics", layout="wide")
st.title("NYC 311 Civic Analytics and Complaint Forecasting")


def prefer_live(live_name: str, processed_name: str) -> Path:
    # data/live/ is committed and refreshed monthly by the GitHub Action;
    # data/processed/ only exists on a machine that ran the full local pipeline
    live = LIVE_DIR / live_name
    return live if live.exists() else PROCESSED_DIR / processed_name


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_monthly() -> pd.DataFrame:
    return load_monthly_counts(verbose=False)


@st.cache_data(ttl=3600)
def load_borough_category() -> pd.DataFrame:
    df = read_table(prefer_live("monthly_borough_category.parquet",
                                "monthly_borough_category.parquet"))
    df["month"] = pd.to_datetime(df["month"])
    return df


@st.cache_data(ttl=3600)
def load_per_capita() -> pd.DataFrame | None:
    path = prefer_live("borough_per_capita.csv", "borough_per_capita.parquet")
    return read_table(path) if path.exists() else None


@st.cache_data(ttl=3600)
def load_refresh_info() -> dict | None:
    path = LIVE_DIR / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else None


monthly = load_monthly()
borough_cat = load_borough_category()
refresh_info = load_refresh_info()
if refresh_info:
    st.caption(
        f"Data complete through {pd.Timestamp(refresh_info['complete_through']):%B %Y} · "
        f"refreshed automatically from NYC Open Data on {refresh_info['refreshed_at'][:10]}"
    )

tab_overview, tab_trends, tab_forecast, tab_ethics = st.tabs(
    ["Overview", "Trends", "Forecast vs actual", "Data responsibility"]
)

with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total complaints", f"{int(monthly['complaint_count'].sum()):,}")
    top_type = (
        borough_cat.groupby("complaint_type")["complaint_count"].sum().idxmax()
    )
    col2.metric("Most common complaint", top_type)
    top_borough = (
        borough_cat.groupby("borough")["complaint_count"].sum().idxmax()
    )
    col3.metric("Highest-volume borough (raw)", top_borough.title())

    per_capita = load_per_capita()
    if per_capita is not None:
        st.subheader("The Equity Flip: Raw Volumes vs. Population-Adjusted Metrics")
        
        # Split layout into side-by-side view to emphasize the structural data flip
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Total Raw Volume (Brooklyn Leads)")
            fig_raw = px.bar(
                per_capita.sort_values("complaint_count"),
                x="complaint_count", y="borough", orientation="h",
                labels={"complaint_count": "Total Raw Complaints Initiated"},
                color="borough", color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_raw.update_layout(showlegend=False)
            st.plotly_chart(fig_raw, use_container_width=True)
            
        with chart_col2:
            st.markdown("#### Normalized Per 100k Population (Bronx Leads)")
            fig_pc = px.bar(
                per_capita.sort_values("complaints_per_100k"),
                x="complaints_per_100k", y="borough", orientation="h",
                labels={"complaints_per_100k": "Complaints Per 100,000 Residents"},
                color="borough", color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_pc.update_layout(showlegend=False)
            st.plotly_chart(fig_pc, use_container_width=True)

with tab_trends:
    st.subheader("Monthly complaint volume")
    boroughs = sorted(borough_cat["borough"].unique())
    selected_boroughs = st.multiselect("Boroughs", boroughs, default=boroughs)

    top_types = (
        borough_cat.groupby("complaint_type")["complaint_count"]
        .sum().nlargest(15).index.tolist()
    )
    selected_types = st.multiselect(
        "Complaint types (top 15 shown)", top_types, default=top_types[:4]
    )

    filtered = borough_cat[
        borough_cat["borough"].isin(selected_boroughs)
        & borough_cat["complaint_type"].isin(selected_types)
    ]
    trend = (
        filtered.groupby(["month", "complaint_type"])["complaint_count"]
        .sum().reset_index()
    )
    fig = px.line(trend, x="month", y="complaint_count", color="complaint_type")
    st.plotly_chart(fig, use_container_width=True)

with tab_forecast:
    st.subheader("Forecast track record")
    forecasts = load_forecasts()
    if forecasts.empty:
        st.info("No forecasts yet. Run src/forecast_2026.py first.")
    else:
        st.caption(
            "Every forecast is frozen with a timestamp before its months happen and is "
            "never edited. For each month, the track record uses the most recent "
            "forecast that existed in advance (superseded buggy files excluded)."
        )
        scored = score(forecasts, monthly)
        track = scored[scored["in_track_record"]]
        last_actual = monthly["month"].max()

        active = forecasts[~forecasts["superseded"]]
        newest = active.groupby("model_name")["training_end_month"].transform("max")
        upcoming = active[(active["training_end_month"] == newest)
                          & (active["month"] > last_actual)]

        predicted_rows = pd.concat([
            track[["month", "model_name", "predicted_count"]],
            upcoming[["month", "model_name", "predicted_count"]],
        ]).rename(columns={"model_name": "series", "predicted_count": "complaints"})
        actual_rows = monthly[monthly["month"] >= predicted_rows["month"].min()].rename(
            columns={"complaint_count": "complaints"})
        actual_rows["series"] = "actual"
        plot_df = pd.concat([predicted_rows, actual_rows[["month", "complaints", "series"]]])

        fig = px.line(plot_df.sort_values("month"), x="month", y="complaints",
                      color="series", markers=True)
        fig.update_xaxes(dtick="M1", tickformat="%b %Y")
        if not upcoming.empty:
            fig.add_vline(x=last_actual + pd.Timedelta(days=15), line_dash="dot",
                          line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        if track.empty:
            st.info("No forecast months have actual counts yet.")
        else:
            st.markdown("#### Accuracy so far")
            summary = summarize(track)
            for column in ["MAPE", "WAPE"]:
                summary[column] = summary[column].map(lambda v: f"{v:.2f}%")
            st.dataframe(summary, hide_index=True, use_container_width=True)

            st.markdown("#### Month by month: which forecast was closer")
            by_month = track.pivot(index="month", columns="model_name", values="pct_error")
            by_month["closer_model"] = by_month.abs().idxmin(axis=1)
            by_month["actual"] = track.groupby("month")["actual_count"].first()
            by_month.index = by_month.index.strftime("%b %Y")
            st.dataframe(
                by_month.reset_index().style.format(
                    {c: "{:+.1f}%" for c in by_month.columns if c not in ("closer_model", "actual")}
                    | {"actual": "{:,.0f}"}),
                hide_index=True, use_container_width=True,
            )
            st.caption("Percent error: negative means the forecast was too low.")

        if not upcoming.empty:
            st.markdown("#### Next months (frozen, not yet scorable)")
            next_table = upcoming.pivot(index="month", columns="model_name",
                                        values="predicted_count")
            next_table.index = next_table.index.strftime("%b %Y")
            st.dataframe(next_table.reset_index().style.format(
                {c: "{:,.0f}" for c in next_table.columns}),
                hide_index=True, use_container_width=True)

        with st.expander("Every frozen forecast file, including superseded ones"):
            files = (
                forecasts.groupby("source_file")
                .agg(model=("model_name", "first"),
                     training_end_month=("training_end_month", "first"),
                     created_at=("forecast_created_at", "first"),
                     superseded_because=("reason", "first"))
                .reset_index()
                .sort_values("created_at")
            )
            files["training_end_month"] = files["training_end_month"].dt.strftime("%Y-%m")
            files["superseded_because"] = files["superseded_because"].fillna("")
            st.dataframe(files, hide_index=True, use_container_width=True)
            st.caption("Superseded files are kept unedited; the README's 'Frozen "
                       "forecasts' section explains each bug.")

with tab_ethics:
    st.subheader("What this data can and cannot tell you")
    st.markdown(
        """
- **311 measures reporting, not conditions.** Complaint volume reflects who
  knows about 311, who has smartphone access, who speaks English comfortably,
  and who trusts city systems — not just where problems exist.
- **High volume does not equal high severity.** Civically engaged neighborhoods
  file more complaints for comparable conditions.
- **The forecast predicts recorded complaints, not neighborhood problems.**
  A model that predicts complaint volume is predicting reporting behavior.
- **The 2020 dip reflects COVID lockdowns**, not improved conditions —
  fewer people outside meant fewer observations and reports.
- **Duplicates and category inconsistency** inflate some totals; complaint
  categories are dispatcher-assigned and have changed over the years.
        """
    )
