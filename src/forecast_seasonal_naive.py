"""Freeze the seasonal-naive baseline forecast (same month, one year prior).

This is the same method evaluated as "baseline_seasonal_lag12" in
train_evaluate.py — lag_12, not lag_24. No recursion is needed (unlike
forecast_2026.py): each forecast month just looks up complaint_count
from twelve months earlier.

Uses the same complete-months-only history and horizon as
forecast_2026.py so both frozen runners share a training_end_month, and
likewise skips if that anchor has already been frozen.

Run:
    python src/forecast_seasonal_naive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import FORECAST_DIR
from src.forecast_2026 import HORIZON_MONTHS
from src.forecast_registry import already_frozen
from src.monthly_data import load_monthly_counts, now_nyc

MODEL_NAME = "seasonal_naive"


def main(horizon: int = HORIZON_MONTHS) -> None:
    history = load_monthly_counts()
    series = history.set_index("month")["complaint_count"]
    last_month = history["month"].max()

    existing = already_frozen(MODEL_NAME, last_month)
    if existing:
        print(f"{existing} already forecasts from training_end_month "
              f"{last_month:%Y-%m}; not freezing a duplicate.")
        return

    forecast_dates = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    predictions = []
    for forecast_month in forecast_dates:
        source_month = forecast_month - pd.DateOffset(years=1)
        if source_month not in series.index:
            raise RuntimeError(f"No lag-12 source month for {forecast_month.date()}: "
                               f"need {source_month.date()} in history.")
        predictions.append({
            "month": forecast_month,
            "predicted_count": round(float(series[source_month])),
        })

    created = now_nyc()
    forecast = pd.DataFrame(predictions)
    forecast["forecast_created_at"] = created.isoformat()
    forecast["training_end_month"] = last_month.strftime("%Y-%m")
    forecast["model_name"] = MODEL_NAME

    output = FORECAST_DIR / f"forecast_{MODEL_NAME}_{created:%Y%m%d_%H%M}.csv"
    forecast.to_csv(output, index=False, date_format="%Y-%m-%d")

    print(forecast[["month", "predicted_count"]].to_string(index=False))
    print(f"\nFrozen forecast saved to {output} — do not edit or overwrite this file.")


if __name__ == "__main__":
    main()
