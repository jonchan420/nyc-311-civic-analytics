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
            st.markdown("#### Normalized Per 100k Population (BronX Leads)")
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
    st.subheader("Frozen forecasts vs actual 2026 data")
    forecast_files = sorted(FORECAST_DIR.glob("forecast_2026_*.csv"))
    if not forecast_files:
        st.info("No forecasts yet. Run src/forecast_2026.py first.")
    else:
        all_forecasts = []
        for f in forecast_files:
            df = pd.read_csv(f, parse_dates=["month"])
            df["source_file"] = f.name
            all_forecasts.append(df)
        all_forecasts = pd.concat(all_forecasts, ignore_index=True)

        # The active comparison uses each model's most recently frozen
        # file. Older files (superseded runs, e.g. the lag-1 contamination
        # fix) stay on disk per the never-overwrite policy and are listed
        # separately below, not silently dropped.
        latest_stamp = all_forecasts.groupby("model_name")["forecast_created_at"].transform("max")
        current = all_forecasts[all_forecasts["forecast_created_at"] == latest_stamp].copy()

        # Same incomplete-month guard used throughout the pipeline: the most
        # recent month in monthly_citywide_counts.parquet is often a partial
        # snapshot, not a finished month. Scoring a forecast against a
        # partial actual would misleadingly show every model "missing badly."
        actual_history = monthly.sort_values("month").reset_index(drop=True)
        trailing_avg = actual_history["complaint_count"].iloc[-13:-1].mean()
        incomplete_month = None
        if actual_history["complaint_count"].iloc[-1] < 0.85 * trailing_avg:
            incomplete_month = actual_history["month"].iloc[-1]
            actual_history = actual_history.iloc[:-1]
        actual = actual_history.rename(columns={"complaint_count": "actual_count"})

        if incomplete_month is not None:
            st.caption(
                f"{incomplete_month:%b %Y} excluded from the actuals below — "
                "that month's data snapshot is still partial, so it isn't a "
                "fair comparison against a full-month forecast yet."
            )

        predicted_rows = current[["month", "predicted_count", "model_name"]].rename(
            columns={"predicted_count": "complaints", "model_name": "series"}
        )
        actual_rows = actual[actual["month"] >= current["month"].min()][
            ["month", "actual_count"]
        ].rename(columns={"actual_count": "complaints"})
        actual_rows["series"] = "actual"
        plot_df = pd.concat(
            [predicted_rows[["month", "complaints", "series"]], actual_rows],
            ignore_index=True,
        ).dropna(subset=["complaints"])

        fig = px.line(plot_df, x="month", y="complaints", color="series", markers=True)
        fig.update_xaxes(dtick="M1", tickformat="%b %Y")
        st.plotly_chart(fig, use_container_width=True)

        scored = current.merge(actual, on="month", how="inner").dropna(subset=["actual_count"])
        if len(scored) > 0:
            scored["abs_error"] = (scored["actual_count"] - scored["predicted_count"]).abs()
            scored["pct_error"] = (
                (scored["predicted_count"] - scored["actual_count"]) / scored["actual_count"] * 100
            )

            st.markdown("#### Per-model accuracy on completed months")
            summary = (
                scored.groupby("model_name")
                .apply(lambda g: pd.Series({
                    "months_scored": len(g),
                    "MAE": round(g["abs_error"].mean()),
                    "MAPE": f"{g['pct_error'].abs().mean():.1f}%",
                    "WAPE": f"{g['abs_error'].sum() / g['actual_count'].sum() * 100:.1f}%",
                }), include_groups=False)
                .reset_index()
                .rename(columns={"model_name": "model"})
            )
            st.dataframe(summary, hide_index=True, use_container_width=True)

            st.markdown("#### Month-by-month: which forecast was closer")
            month_pivot = scored.pivot(index="month", columns="model_name", values="abs_error")
            month_pivot["closer_model"] = month_pivot.idxmin(axis=1)
            month_pivot.index = month_pivot.index.strftime("%b %Y")
            st.dataframe(month_pivot.reset_index(), hide_index=True, use_container_width=True)
        else:
            st.info("No completed months yet to score the active forecasts against.")

        st.markdown("#### Active forecast metadata")
        meta = (
            current[["model_name", "training_end_month", "forecast_created_at", "source_file"]]
            .drop_duplicates()
            .sort_values("model_name")
        )
        st.dataframe(meta, hide_index=True, use_container_width=True)

        with st.expander("Superseded forecast versions (kept per frozen-forecast policy)"):
            older = all_forecasts[~all_forecasts["source_file"].isin(current["source_file"])]
            if len(older) > 0:
                st.dataframe(
                    older[["source_file", "model_name", "training_end_month", "forecast_created_at"]]
                    .drop_duplicates()
                    .sort_values(["model_name", "forecast_created_at"]),
                    hide_index=True, use_container_width=True,
                )
                st.caption(
                    "These files are never edited or deleted — see the README's "
                    "'Frozen forecasts' section for why each was superseded."
                )
            else:
                st.caption("No superseded versions on disk yet.")

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
