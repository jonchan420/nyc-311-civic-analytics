"""Step 6: turn monthly counts into a supervised learning table.

Creates lag features, rolling means, and cyclical month encodings.
Critical detail: rolling means use .shift(1) first so the current
month's target never leaks into its own features.

Also drops the final month if it looks incomplete (a month whose
count is drastically below trend is almost always a partial month
from the data snapshot, not a real drop).

Run:
    python src/build_forecast_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import PROCESSED_DIR


def build() -> pd.DataFrame:
    monthly = pd.read_parquet(PROCESSED_DIR / "monthly_citywide_counts.parquet")
    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly = monthly.sort_values("month").reset_index(drop=True)

    # drop the last month if it's likely incomplete:
    # below 60% of the trailing 12-month average is a red flag.
    trailing_avg = monthly["complaint_count"].iloc[-13:-1].mean()
    if monthly["complaint_count"].iloc[-1] < 0.85 * trailing_avg:
        dropped = monthly["month"].iloc[-1]
        monthly = monthly.iloc[:-1].copy()
        print(f"Dropped {dropped.date()} as a likely-incomplete month.")

    monthly["year"] = monthly["month"].dt.year
    monthly["month_number"] = monthly["month"].dt.month
    monthly["quarter"] = monthly["month"].dt.quarter
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month_number"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month_number"] / 12)

    for lag in [1, 2, 3, 6, 12]:
        monthly[f"lag_{lag}"] = monthly["complaint_count"].shift(lag)

    for window in [3, 6, 12]:
        monthly[f"rolling_mean_{window}"] = (
            monthly["complaint_count"].shift(1).rolling(window=window).mean()
        )

    monthly = monthly.dropna().reset_index(drop=True)

    output = PROCESSED_DIR / "forecast_dataset.parquet"
    monthly.to_parquet(output, index=False)
    print(f"Forecast dataset: {len(monthly)} usable months, saved to {output}")
    print(f"Range: {monthly['month'].min().date()} to {monthly['month'].max().date()}")
    return monthly


if __name__ == "__main__":
    build()
