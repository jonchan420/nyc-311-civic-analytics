"""Streamlit dashboard for NYC 311 Civic Analytics.

Run from the project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import FORECAST_DIR, PROCESSED_DIR

st.set_page_config(page_title="NYC 311 Civic Analytics", layout="wide")
st.title("NYC 311 Civic Analytics and Complaint Forecasting")


@st.cache_data
def load_monthly() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "monthly_citywide_counts.parquet")
    df["month"] = pd.to_datetime(df["month"])
    return df.sort_values("month")


@st.cache_data
def load_borough_category() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "monthly_borough_category.parquet")
    df["month"] = pd.to_datetime(df["month"])
    return df


monthly = load_monthly()
borough_cat = load_borough_category()

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

    per_capita_path = PROCESSED_DIR / "borough_per_capita.parquet"
    if per_capita_path.exists():
        per_capita = pd.read_parquet(per_capita_path)
        st.subheader("Raw totals vs complaints per 100k residents")
        fig = px.bar(
            per_capita.sort_values("complaints_per_100k"),
            x="complaints_per_100k", y="borough", orientation="h",
            labels={"complaints_per_100k": "Complaints per 100k residents"},
        )
        st.plotly_chart(fig, use_container_width=True)

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
    st.subheader("Frozen forecasts vs actual 2026 data")
    forecast_files = sorted(FORECAST_DIR.glob("forecast_2026_*.csv"))
    if not forecast_files:
        st.info("No forecasts yet. Run src/forecast_2026.py first.")
    else:
        chosen = st.selectbox("Forecast version", forecast_files,
                              format_func=lambda p: p.name)
        forecast = pd.read_csv(chosen, parse_dates=["month"])
        actual = monthly.rename(columns={"complaint_count": "actual_count"})
        merged = forecast.merge(actual, on="month", how="left")

        plot_df = merged.melt(
            id_vars="month",
            value_vars=["predicted_count", "actual_count"],
            var_name="series", value_name="complaints",
        ).dropna()
        fig = px.line(plot_df, x="month", y="complaints", color="series", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        scored = merged.dropna(subset=["actual_count"])
        if len(scored) > 0:
            wape = (
                (scored["actual_count"] - scored["predicted_count"]).abs().sum()
                / scored["actual_count"].sum() * 100
            )
            st.metric("WAPE on months with actuals", f"{wape:.1f}%")

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
