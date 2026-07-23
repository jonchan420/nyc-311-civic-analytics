"""Step 7: chronological evaluation — baselines first, models second.

The rule that matters most: if your ML model can't beat "same month
last year," the ML model is not adding value and you should say so
honestly in your README. That honesty is a portfolio strength.

Splits (never random for time series):
    train:      2021 through 2024  (2020 rows get consumed by lag_12)
    validation: 2025
    test:       complete months of 2026

Run:
    python src/train_evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import MODEL_DIR, PROCESSED_DIR

FEATURES = [
    "year", "month_number", "quarter", "month_sin", "month_cos",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
]


def evaluate(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual_arr, predicted_arr)
    rmse = float(np.sqrt(mean_squared_error(actual_arr, predicted_arr)))

    nonzero = actual_arr != 0
    mape = float(np.mean(np.abs(
        (actual_arr[nonzero] - predicted_arr[nonzero]) / actual_arr[nonzero]
    )) * 100)
    wape = float(np.abs(actual_arr - predicted_arr).sum() / np.abs(actual_arr).sum() * 100)

    return {"MAE": round(mae), "RMSE": round(rmse),
            "MAPE": round(mape, 2), "WAPE": round(wape, 2)}


def main() -> None:
    data = pd.read_parquet(PROCESSED_DIR / "forecast_dataset.parquet")
    data["month"] = pd.to_datetime(data["month"])

    train = data[data["month"] < "2025-01-01"]
    validation = data[(data["month"] >= "2025-01-01") & (data["month"] < "2026-01-01")]
    test = data[data["month"] >= "2026-01-01"]

    print(f"train: {len(train)} months | validation: {len(validation)} | test: {len(test)}\n")

    if len(validation) == 0:
        raise RuntimeError("No 2025 validation months found — check your data range.")

    results: dict[str, dict[str, float]] = {}

    results["baseline_previous_month"] = evaluate(
        validation["complaint_count"], validation["lag_1"].to_numpy()
    )
    results["baseline_seasonal_lag12"] = evaluate(
        validation["complaint_count"], validation["lag_12"].to_numpy()
    )

    linear = LinearRegression()
    linear.fit(train[FEATURES], train["complaint_count"])
    linear_pred = np.maximum(linear.predict(validation[FEATURES]), 0)
    results["linear_regression"] = evaluate(validation["complaint_count"], linear_pred)

    boosted = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=300, max_depth=3, random_state=42
    )
    boosted.fit(train[FEATURES], train["complaint_count"])
    boosted_pred = np.maximum(boosted.predict(validation[FEATURES]), 0)
    results["gradient_boosting"] = evaluate(validation["complaint_count"], boosted_pred)

    summary = pd.DataFrame(results).T
    print("Validation results (2025):")
    print(summary.to_string())

    best_name = summary["WAPE"].idxmin()
    print(f"\nBest on validation by WAPE: {best_name}")
    if best_name.startswith("baseline"):
        print("NOTE: a baseline is winning. Report this honestly — it is a finding, "
              "not a failure. With ~60 training months, simple beats complex often.")

    if len(test) > 0:
        print("\nTest results (2026 complete months) for the two trained models:")
        retrain = pd.concat([train, validation])
        linear.fit(retrain[FEATURES], retrain["complaint_count"])
        boosted.fit(retrain[FEATURES], retrain["complaint_count"])
        test_results: dict[str, dict[str, float]] = {}
        test_results["baseline_seasonal_lag12"] = evaluate(
            test["complaint_count"], test["lag_12"].to_numpy()
        )
        for name, model in [("linear_regression", linear), ("gradient_boosting", boosted)]:
            pred = np.maximum(model.predict(test[FEATURES]), 0)
            test_results[name] = evaluate(test["complaint_count"], pred)
            print(f"  {name}: {test_results[name]}")
        test_summary = pd.DataFrame(test_results).T
        test_summary.to_csv(PROCESSED_DIR / "model_comparison_test_2026.csv")
        print(f"Test months covered: {test['month'].min().date()} to {test['month'].max().date()}")
        print(f"Saved to model_comparison_test_2026.csv")

    joblib.dump(boosted, MODEL_DIR / "gradient_boosting.joblib")
    joblib.dump(linear, MODEL_DIR / "linear_regression.joblib")
    summary.to_csv(PROCESSED_DIR / "model_comparison.csv")
    print(f"\nModels saved to {MODEL_DIR}, comparison table to model_comparison.csv")


if __name__ == "__main__":
    main()
