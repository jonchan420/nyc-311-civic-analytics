"""Freeze a gradient boosting forecast for the next few months.

Recursive forecasting: each predicted month becomes the lag input for
the next. The forecast CSV gets a timestamp and metadata columns so
you can later prove the prediction was made BEFORE the actuals existed.
Never overwrite an old forecast file. Skips (no new file) if an active
forecast with the same training_end_month already exists, so re-running
in the same month is harmless.

(The filename is historical: this originally forecast the rest of 2026;
it now forecasts HORIZON_MONTHS ahead of the latest complete month.)

Run:
    python src/forecast_2026.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import FORECAST_DIR, MODEL_DIR
from src.forecast_registry import already_frozen
from src.monthly_data import load_monthly_counts, now_nyc
from src.train_evaluate import FEATURES

HORIZON_MONTHS = 3


def build_feature_row(history: pd.DataFrame, forecast_month: pd.Timestamp) -> pd.DataFrame:
    values = history["complaint_count"].astype(float)
    row = {
        "year": forecast_month.year,
        "month_number": forecast_month.month,
        "quarter": forecast_month.quarter,
        "month_sin": np.sin(2 * np.pi * forecast_month.month / 12),
        "month_cos": np.cos(2 * np.pi * forecast_month.month / 12),
        "lag_1": values.iloc[-1],
        "lag_2": values.iloc[-2],
        "lag_3": values.iloc[-3],
        "lag_6": values.iloc[-6],
        "lag_12": values.iloc[-12],
        "rolling_mean_3": values.iloc[-3:].mean(),
        "rolling_mean_6": values.iloc[-6:].mean(),
        "rolling_mean_12": values.iloc[-12:].mean(),
    }
    return pd.DataFrame([row])


def main(model_name: str = "gradient_boosting", horizon: int = HORIZON_MONTHS) -> None:
    history = load_monthly_counts()[["month", "complaint_count"]].copy()
    last_month = history["month"].max()

    existing = already_frozen(model_name, last_month)
    if existing:
        print(f"{existing} already forecasts from training_end_month "
              f"{last_month:%Y-%m}; not freezing a duplicate.")
        return

    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    forecast_dates = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )

    predictions: list[dict[str, object]] = []
    for forecast_month in forecast_dates:
        features = build_feature_row(history, forecast_month)
        predicted = max(float(model.predict(features[FEATURES])[0]), 0)
        predictions.append({"month": forecast_month, "predicted_count": round(predicted)})
        history = pd.concat(
            [history, pd.DataFrame({"month": [forecast_month],
                                    "complaint_count": [predicted]})],
            ignore_index=True,
        )

    created = now_nyc()
    forecast = pd.DataFrame(predictions)
    forecast["forecast_created_at"] = created.isoformat()
    forecast["training_end_month"] = last_month.strftime("%Y-%m")
    forecast["model_name"] = model_name

    output = FORECAST_DIR / f"forecast_{model_name}_{created:%Y%m%d_%H%M}.csv"
    forecast.to_csv(output, index=False, date_format="%Y-%m-%d")

    print(forecast[["month", "predicted_count"]].to_string(index=False))
    print(f"\nFrozen forecast saved to {output} — do not edit or overwrite this file.")


if __name__ == "__main__":
    main()
