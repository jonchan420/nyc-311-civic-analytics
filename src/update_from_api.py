"""Step 4 (ongoing): pull only NEW records from the Socrata API.

Your 14 GB CSV is a snapshot. As 2026 progresses you need fresh data
to compare against your forecasts. This script finds the last date in
your local Parquet and downloads only records created after it —
never the whole dataset again.

Run monthly:
    python src/update_from_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import API_BASE, RAW_DIR

FIELDS = [
    "unique_key", "created_date", "closed_date", "agency", "agency_name",
    "complaint_type", "descriptor", "location_type", "incident_zip",
    "status", "resolution_action_updated_date", "community_board",
    "borough", "open_data_channel_type", "latitude", "longitude",
]

PAGE_SIZE = 50_000
PARQUET_GLOB = str(RAW_DIR / "parquet_by_year" / "**" / "*.parquet")


def last_local_date() -> pd.Timestamp:
    con = duckdb.connect()
    result = con.execute(f"""
        SELECT max(strptime(created_date, '%m/%d/%Y %I:%M:%S %p'))
        FROM read_parquet('{PARQUET_GLOB}')
    """).fetchone()[0]
    if result is None:
        raise RuntimeError("No local data found. Run convert_csv_to_parquet.py first.")
    return pd.Timestamp(result)


def fetch_new_records(since: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    offset = 0
    since_str = since.strftime("%Y-%m-%dT%H:%M:%S")

    while True:
        params = {
            "$select": ",".join(FIELDS),
            "$where": f"created_date > '{since_str}'",
            "$order": "created_date,unique_key",
            "$limit": PAGE_SIZE,
            "$offset": offset,
        }
        response = requests.get(API_BASE, params=params, timeout=300)
        response.raise_for_status()
        records = response.json()
        if not records:
            break
        frames.append(pd.DataFrame(records))
        print(f"  fetched {offset + len(records):,} new records...")
        if len(records) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    since = last_local_date()
    print(f"Last local record: {since}. Fetching newer records from the API...")

    new_data = fetch_new_records(since)
    if new_data.empty:
        print("No new records. You're up to date.")
        return

    # API returns ISO dates; convert to match the CSV-derived format
    # so all parquet files share one date format.
    new_data["created_date"] = (
        pd.to_datetime(new_data["created_date"]).dt.strftime("%m/%d/%Y %I:%M:%S %p")
    )
    if "closed_date" in new_data.columns:
        new_data["closed_date"] = (
            pd.to_datetime(new_data["closed_date"], errors="coerce")
            .dt.strftime("%m/%d/%Y %I:%M:%S %p")
        )

    stamp = pd.Timestamp.now().strftime("%Y%m%d")
    year = pd.to_datetime(new_data["created_date"]).dt.year.max()
    output = RAW_DIR / "parquet_by_year" / f"yr={year}" / f"update_{stamp}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    new_data["yr"] = pd.to_datetime(new_data["created_date"]).dt.year
    new_data.to_parquet(output, index=False)

    print(f"Saved {len(new_data):,} new records to {output}")
    print("Now re-run build_monthly_counts.py to refresh the aggregates.")


if __name__ == "__main__":
    main()
