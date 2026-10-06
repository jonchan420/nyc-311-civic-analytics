"""Load every frozen forecast and score it against actual complaint counts.

The headline "track record" uses, for each model and each month with an
actual count, only the forecast that was:
  - not superseded (see data/forecasts/superseded.csv),
  - trained on data ending before that month, and
  - created before that month's data could have been complete,
picking the one with the most recent training data. That is the best
prediction that genuinely existed in advance, with no hindsight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import FORECAST_DIR

SUPERSEDED_FILE = FORECAST_DIR / "superseded.csv"


def load_superseded() -> pd.DataFrame:
    if not SUPERSEDED_FILE.exists():
        return pd.DataFrame(columns=["source_file", "reason"])
    return pd.read_csv(SUPERSEDED_FILE)


def load_forecasts() -> pd.DataFrame:
    frames = []
    for path in sorted(FORECAST_DIR.glob("forecast_*.csv")):
        df = pd.read_csv(path, parse_dates=["month"])
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    forecasts = pd.concat(frames, ignore_index=True)
    # older files carry microseconds, the seasonal baseline's doesn't
    forecasts["forecast_created_at"] = pd.to_datetime(
        forecasts["forecast_created_at"], format="ISO8601")
    forecasts["training_end_month"] = pd.to_datetime(
        forecasts["training_end_month"], format="%Y-%m")
    superseded = load_superseded()
    forecasts = forecasts.merge(superseded, on="source_file", how="left")
    forecasts["superseded"] = forecasts["reason"].notna()
    return forecasts


def already_frozen(model_name: str, training_end_month: pd.Timestamp) -> str | None:
    """Name of an active forecast file for this model and anchor, if one exists."""
    forecasts = load_forecasts()
    if forecasts.empty:
        return None
    match = forecasts[
        (forecasts["model_name"] == model_name)
        & (forecasts["training_end_month"] == training_end_month)
        & ~forecasts["superseded"]
    ]
    return None if match.empty else match["source_file"].iloc[0]


def score(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Every (forecast file, month) pair that has an actual count."""
    scored = forecasts.merge(
        actuals[["month", "complaint_count"]].rename(columns={"complaint_count": "actual_count"}),
        on="month", how="inner",
    )
    scored["error"] = scored["predicted_count"] - scored["actual_count"]
    scored["pct_error"] = scored["error"] / scored["actual_count"] * 100
    scored["months_ahead"] = (
        (scored["month"].dt.year - scored["training_end_month"].dt.year) * 12
        + (scored["month"].dt.month - scored["training_end_month"].dt.month)
    )
    eligible = (
        ~scored["superseded"]
        & (scored["training_end_month"] < scored["month"])
        & (scored["forecast_created_at"] < scored["month"] + pd.offsets.MonthBegin(1))
    )
    best = (
        scored[eligible]
        .sort_values(["training_end_month", "forecast_created_at"])
        .groupby(["model_name", "month"])
        .tail(1)
        .index
    )
    scored["in_track_record"] = scored.index.isin(best)
    return scored.sort_values(["month", "model_name", "source_file"]).reset_index(drop=True)


def summarize(track: pd.DataFrame) -> pd.DataFrame:
    """MAE / MAPE / WAPE per model over its track-record months."""
    rows = []
    for model_name, group in track.groupby("model_name"):
        abs_error = group["error"].abs()
        rows.append({
            "model": model_name,
            "months_scored": len(group),
            "MAE": round(abs_error.mean()),
            "MAPE": round(group["pct_error"].abs().mean(), 2),
            "WAPE": round(abs_error.sum() / group["actual_count"].sum() * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("WAPE").reset_index(drop=True)
