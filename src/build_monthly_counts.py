"""Step 3: build small aggregated tables from the yearly Parquet files.

Produces three processed tables:
  1. monthly_citywide_counts.parquet   — one row per month (forecasting target)
  2. monthly_borough_category.parquet  — month x borough x complaint type
  3. borough_totals.parquet            — total per borough (for per-capita analysis)

These are tiny (hundreds to a few thousand rows) and are what every
notebook and the dashboard read. Nothing downstream ever touches the
raw record-level data unless it needs response times or geography.

Run:
    python src/build_monthly_counts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import PROCESSED_DIR, RAW_DIR, VALID_BOROUGHS

PARQUET_GLOB = str(RAW_DIR / "parquet_by_year" / "**" / "*.parquet")
DATE_EXPR = "strptime(created_date, '%m/%d/%Y %I:%M:%S %p')"
BOROUGH_FILTER = "upper(trim(borough)) IN ('" + "', '".join(sorted(VALID_BOROUGHS)) + "')"


def build_all() -> None:
    con = duckdb.connect()

    monthly = con.execute(f"""
        SELECT
            date_trunc('month', {DATE_EXPR}) AS month,
            count(*) AS complaint_count
        FROM read_parquet('{PARQUET_GLOB}')
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()
    monthly.to_parquet(PROCESSED_DIR / "monthly_citywide_counts.parquet", index=False)
    print(f"monthly_citywide_counts: {len(monthly)} rows")

    borough_cat = con.execute(f"""
        SELECT
            date_trunc('month', {DATE_EXPR}) AS month,
            upper(trim(borough)) AS borough,
            trim(complaint_type) AS complaint_type,
            count(*) AS complaint_count
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE {BOROUGH_FILTER}
          AND complaint_type IS NOT NULL
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """).fetchdf()
    borough_cat.to_parquet(PROCESSED_DIR / "monthly_borough_category.parquet", index=False)
    print(f"monthly_borough_category: {len(borough_cat)} rows")

    borough_totals = con.execute(f"""
        SELECT
            upper(trim(borough)) AS borough,
            count(*) AS complaint_count
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE {BOROUGH_FILTER}
        GROUP BY 1
        ORDER BY 2 DESC
    """).fetchdf()
    borough_totals.to_parquet(PROCESSED_DIR / "borough_totals.parquet", index=False)
    print(f"borough_totals: {len(borough_totals)} rows")
    print(borough_totals.to_string(index=False))

    # sanity check: the most recent month is probably incomplete.
    # forecasting code must drop it.
    last = monthly["month"].max()
    print(f"\nMost recent month in data: {last} — verify whether it is complete "
          "before using it for modeling.")


if __name__ == "__main__":
    build_all()
