"""Step 8: forecast the remaining months of 2026 and freeze the prediction.

Recursive forecasting: each predicted month becomes the lag input for
the next. The forecast CSV gets a timestamp and metadata columns so
you can later prove the prediction was made BEFORE the actuals existed.
Never overwrite an old forecast file.

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
from config.settings import FORECAST_DIR, MODEL_DIR, PROCESSED_DIR
from src.train_evaluate import FEATURES


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


def main(model_name: str = "gradient_boosting", end_month: str = "2026-12-01") -> None:
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")

    monthly = pd.read_parquet(PROCESSED_DIR / "monthly_citywide_counts.parquet")
    monthly["month"] = pd.to_datetime(monthly["month"])
    history = monthly.sort_values("month")[["month", "complaint_count"]].copy()

    # same incomplete-month guard as build_forecast_dataset.py: this script
    # reads monthly_citywide_counts.parquet directly, so it needs its own
    # check rather than inheriting the one applied to the training data.
    trailing_avg = history["complaint_count"].iloc[-13:-1].mean()
    if history["complaint_count"].iloc[-1] < 0.85 * trailing_avg:
        dropped = history["month"].iloc[-1]
        history = history.iloc[:-1].copy()
        print(f"Dropped {dropped.date()} as a likely-incomplete month.")

    last_month = history["month"].max()
    forecast_dates = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        end=pd.Timestamp(end_month),
        freq="MS",
    )

    if len(forecast_dates) == 0:
        print("Nothing to forecast — history already extends past end_month.")
        return

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

    forecast = pd.DataFrame(predictions)
    forecast["forecast_created_at"] = pd.Timestamp.now().isoformat()
    forecast["training_end_month"] = last_month.strftime("%Y-%m")
    forecast["model_name"] = model_name

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    output = FORECAST_DIR / f"forecast_2026_{model_name}_{stamp}.csv"
    forecast.to_csv(output, index=False)

    print(forecast[["month", "predicted_count"]].to_string(index=False))
    print(f"\nFrozen forecast saved to {output} — do not edit or overwrite this file.")


if __name__ == "__main__":
    main()
