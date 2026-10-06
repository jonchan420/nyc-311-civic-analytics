"""One loader for monthly citywide counts, returning complete months only.

Prefers data/live/monthly_citywide_counts.csv (written by refresh_live.py,
which only ever includes fully published months). Falls back to the local
snapshot in data/processed/, where the newest month is often partial, so
the original 0.85x-trailing-average guard is applied there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import LIVE_DIR, PROCESSED_DIR

LIVE_MONTHLY = LIVE_DIR / "monthly_citywide_counts.csv"
NYC_TZ = "America/New_York"


def now_nyc() -> pd.Timestamp:
    """Naive NYC wall-clock time, matching the existing forecast timestamps."""
    return pd.Timestamp.now(tz=NYC_TZ).tz_localize(None)


def load_monthly_counts(verbose: bool = True) -> pd.DataFrame:
    if LIVE_MONTHLY.exists():
        df = pd.read_csv(LIVE_MONTHLY, parse_dates=["month"])
        return df.sort_values("month").reset_index(drop=True)

    df = pd.read_parquet(PROCESSED_DIR / "monthly_citywide_counts.parquet")
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)
    trailing_avg = df["complaint_count"].iloc[-13:-1].mean()
    if df["complaint_count"].iloc[-1] < 0.85 * trailing_avg:
        if verbose:
            print(f"Dropped {df['month'].iloc[-1].date()} as a likely-incomplete month.")
        df = df.iloc[:-1].reset_index(drop=True)
    return df
