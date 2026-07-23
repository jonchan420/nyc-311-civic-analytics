"""Freezes the seasonal-naive baseline forecast (same month, one year prior).

This is the same method evaluated as "baseline_seasonal_lag12" in
train_evaluate.py — lag_12, not lag_24. No recursion is needed (unlike
forecast_2026.py): each forecast month just looks up complaint_count
from twelve months earlier, so predicting Aug 2026 never depends on
predicting Jul 2026 first.

Applies the same incomplete-month guard as build_forecast_dataset.py
and forecast_2026.py so both frozen runners share a training_end_month
and forecast start.

Run:
    python src/forecast_seasonal_naive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import FORECAST_DIR, PROCESSED_DIR


def main(end_month: str = "2026-12-01") -> None:
    monthly = pd.read_parquet(PROCESSED_DIR / "monthly_citywide_counts.parquet")
    monthly["month"] = pd.to_datetime(monthly["month"])
    history = monthly.sort_values("month")[["month", "complaint_count"]].copy()

    trailing_avg = history["complaint_count"].iloc[-13:-1].mean()
    if history["complaint_count"].iloc[-1] < 0.85 * trailing_avg:
        dropped = history["month"].iloc[-1]
        history = history.iloc[:-1].copy()
        print(f"Dropped {dropped.date()} as a likely-incomplete month.")

    series = history.set_index("month")["complaint_count"]
    last_month = history["month"].max()

    forecast_dates = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        end=pd.Timestamp(end_month),
        freq="MS",
    )
    if len(forecast_dates) == 0:
        print("Nothing to forecast — history already extends past end_month.")
        return

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

    forecast = pd.DataFrame(predictions)
    forecast["forecast_created_at"] = pd.Timestamp.now().isoformat()
    forecast["training_end_month"] = last_month.strftime("%Y-%m")
    forecast["model_name"] = "seasonal_naive"

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    output = FORECAST_DIR / f"forecast_2026_seasonal_naive_{stamp}.csv"
    forecast.to_csv(output, index=False)

    print(forecast[["month", "predicted_count"]].to_string(index=False))
    print(f"\nFrozen forecast saved to {output} — do not edit or overwrite this file.")


if __name__ == "__main__":
    main()
