"""Refresh the committed data/live/ tables straight from the Socrata API.

Asks the API for pre-aggregated counts (monthly citywide; monthly x
borough x complaint type) instead of downloading raw records, so it runs
in a GitHub Action with no local data. Only complete months are kept:
everything before the first day of the current month (NYC time), and
only once the API has published records past that point.

Fails loudly rather than writing suspicious data: if the API looks
stale, months are missing, or already-published history moves by more
than 2%, nothing is written.

Run:
    python src/refresh_live.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import API_BASE, DATA_DIR, LIVE_DIR, START_YEAR, VALID_BOROUGHS
from src.monthly_data import LIVE_MONTHLY, now_nyc

PAGE_SIZE = 50_000
MAX_HISTORY_DRIFT = 0.02


def soql(params: dict[str, object]) -> list[dict]:
    headers = {}
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token
    for attempt in range(1, 5):
        try:
            response = requests.get(API_BASE, params=params, headers=headers, timeout=900)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            if attempt == 4:
                raise
            wait = 30 * attempt
            print(f"  API request failed ({error}); retrying in {wait}s...")
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_paged(params: dict[str, object]) -> pd.DataFrame:
    frames, offset = [], 0
    while True:
        page = soql({**params, "$limit": PAGE_SIZE, "$offset": offset})
        frames.append(pd.DataFrame(page))
        print(f"  fetched {offset + len(page):,} rows")
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    cutoff = now_nyc().normalize().replace(day=1)
    start = pd.Timestamp(f"{START_YEAR}-01-01")
    window = (f"created_date >= '{start:%Y-%m-%dT%H:%M:%S}' "
              f"AND created_date < '{cutoff:%Y-%m-%dT%H:%M:%S}'")

    latest = pd.Timestamp(soql({"$select": "max(created_date) AS latest"})[0]["latest"])
    print(f"API latest record: {latest}; keeping months before {cutoff.date()}")
    if latest < cutoff + pd.Timedelta(days=1):
        raise RuntimeError(
            f"API has only published records through {latest}; the month before "
            f"{cutoff.date()} may still be incomplete. Re-run in a day or two."
        )

    print("Fetching monthly citywide counts...")
    monthly = pd.DataFrame(soql({
        "$select": "date_trunc_ym(created_date) AS month, count(*) AS complaint_count",
        "$where": window,
        "$group": "date_trunc_ym(created_date)",
        "$order": "month",
        "$limit": 5000,
    }))
    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly["complaint_count"] = monthly["complaint_count"].astype(int)

    expected = pd.date_range(start, cutoff - pd.offsets.MonthBegin(1), freq="MS")
    if list(monthly["month"]) != list(expected):
        missing = sorted(set(expected) - set(monthly["month"]))
        raise RuntimeError(f"Monthly series is not contiguous; missing {missing[:5]}")

    if LIVE_MONTHLY.exists():
        previous = pd.read_csv(LIVE_MONTHLY, parse_dates=["month"])
        overlap = previous.merge(monthly, on="month", suffixes=("_old", "_new"))
        drift = ((overlap["complaint_count_new"] - overlap["complaint_count_old"]).abs()
                 / overlap["complaint_count_old"])
        if (drift > MAX_HISTORY_DRIFT).any():
            bad = overlap.loc[drift > MAX_HISTORY_DRIFT, "month"].dt.strftime("%Y-%m").tolist()
            raise RuntimeError(f"Published history moved >{MAX_HISTORY_DRIFT:.0%} in {bad}; "
                               "refusing to overwrite. Investigate before re-running.")

    print("Fetching monthly borough x complaint-type counts (slow, ~5 min)...")
    detail = fetch_paged({
        "$select": ("date_trunc_ym(created_date) AS month, borough, complaint_type, "
                    "count(*) AS complaint_count"),
        "$where": window,
        "$group": "date_trunc_ym(created_date), borough, complaint_type",
        "$order": "month, borough, complaint_type",
    })
    # a late-published record can shift group offsets between pages,
    # which would show up here as a repeated group
    if detail.duplicated(["month", "borough", "complaint_type"]).any():
        raise RuntimeError("Paged API results overlap; the dataset changed mid-fetch. Re-run.")
    detail["month"] = pd.to_datetime(detail["month"])
    detail["complaint_count"] = detail["complaint_count"].astype(int)
    detail["borough"] = detail["borough"].str.strip().str.upper()
    detail["complaint_type"] = detail["complaint_type"].str.strip()
    detail = detail[detail["borough"].isin(VALID_BOROUGHS) & detail["complaint_type"].notna()]
    detail = (detail.groupby(["month", "borough", "complaint_type"], as_index=False)
              ["complaint_count"].sum())

    if detail["complaint_count"].sum() > monthly["complaint_count"].sum():
        raise RuntimeError("Borough detail exceeds the citywide total; the API responses disagree.")

    population = pd.read_csv(DATA_DIR / "borough_population.csv")
    per_capita = (detail.groupby("borough", as_index=False)["complaint_count"].sum()
                  .merge(population[["borough", "population"]], on="borough", validate="one_to_one"))
    per_capita["complaints_per_100k"] = (
        per_capita["complaint_count"] / per_capita["population"] * 100_000).round(0)
    per_capita = per_capita.sort_values("complaints_per_100k", ascending=False)

    monthly.to_csv(LIVE_MONTHLY, index=False, date_format="%Y-%m-%d")
    detail.to_parquet(LIVE_DIR / "monthly_borough_category.parquet", index=False)
    per_capita.to_csv(LIVE_DIR / "borough_per_capita.csv", index=False)
    (LIVE_DIR / "metadata.json").write_text(json.dumps({
        "refreshed_at": now_nyc().isoformat(timespec="seconds"),
        "api_latest_record": latest.isoformat(),
        "complete_through": monthly["month"].max().strftime("%Y-%m"),
        "total_records": int(monthly["complaint_count"].sum()),
    }, indent=2) + "\n")

    print(f"Live data complete through {monthly['month'].max():%Y-%m}: "
          f"{monthly['complaint_count'].sum():,} records")


if __name__ == "__main__":
    main()
