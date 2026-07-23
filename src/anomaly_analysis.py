"""Reproduces the record-level findings behind the 2026 structural break.

Every number in the README's anomaly discussion traces back to one of
the CSVs this script writes to data/processed/anomaly_findings/. Reads
the raw yearly Parquet partitions directly (record-level fields, not
the small aggregate tables), since borough/channel/ZIP/complaint-type
breakdowns aren't in monthly_citywide_counts.parquet.

Produces:
  baseline_errors_2026.csv        seasonal-naive (same-month-last-year)
                                   error per month, Jan-Jun 2026
  complaint_type_deltas.csv       2025 vs 2026 counts per complaint_type,
                                   for February, March, June
  june_channel_breakdown.csv      open_data_channel_type share, Jun 2025 vs 2026
  june_borough_breakdown.csv      borough counts, Jun 2025 vs 2026
  june_top10_zip.csv              top 10 incident_zip by count, Jun 2026
  illegal_parking_bk_qn_timeline.csv  monthly Illegal Parking count,
                                       Brooklyn + Queens, full history

Run:
    python src/anomaly_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import PROCESSED_DIR, RAW_DIR

PARQUET_GLOB = str(RAW_DIR / "parquet_by_year" / "**" / "*.parquet")
DATE_EXPR = "strptime(created_date, '%m/%d/%Y %I:%M:%S %p')"
OUTPUT_DIR = PROCESSED_DIR / "anomaly_findings"

DELTA_MONTHS = ["02", "03", "06"]


def monthly_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(f"""
        SELECT strftime({DATE_EXPR}, '%Y-%m') AS ym, count(*) AS complaint_count
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE created_date IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()


def build_baseline_errors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    monthly = monthly_counts(con).set_index("ym")["complaint_count"]
    rows = []
    for month in ["01", "02", "03", "04", "05", "06"]:
        actual = monthly.get(f"2026-{month}")
        baseline = monthly.get(f"2025-{month}")
        if actual is None or baseline is None:
            continue
        signed_error = actual - baseline
        rows.append({
            "month": f"2026-{month}",
            "actual": actual,
            "seasonal_naive_baseline": baseline,
            "signed_error": signed_error,
            "pct_error": round(100 * signed_error / baseline, 1),
        })
    return pd.DataFrame(rows)


def build_complaint_type_deltas(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(f"""
        SELECT complaint_type, strftime({DATE_EXPR}, '%Y-%m') AS ym, count(*) AS c
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE created_date IS NOT NULL AND complaint_type IS NOT NULL
        GROUP BY 1, 2
    """).fetchdf()

    frames = []
    for month in DELTA_MONTHS:
        prior = df[df.ym == f"2025-{month}"].set_index("complaint_type")["c"]
        current = df[df.ym == f"2026-{month}"].set_index("complaint_type")["c"]
        merged = pd.DataFrame({"y2025": prior, "y2026": current}).fillna(0)
        merged["delta"] = merged["y2026"] - merged["y2025"]
        merged["month"] = f"2026-{month}"
        merged = merged.reset_index().rename(columns={"index": "complaint_type"})
        frames.append(merged.sort_values("delta", ascending=False))
    return pd.concat(frames, ignore_index=True)[
        ["month", "complaint_type", "y2025", "y2026", "delta"]
    ]


def build_june_channel_breakdown(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(f"""
        SELECT open_data_channel_type AS channel, strftime({DATE_EXPR}, '%Y-%m') AS ym, count(*) AS c
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE created_date IS NOT NULL
          AND strftime({DATE_EXPR}, '%Y-%m') IN ('2025-06', '2026-06')
        GROUP BY 1, 2
    """).fetchdf()
    pivot = df.pivot(index="channel", columns="ym", values="c").fillna(0)
    pivot["delta"] = pivot["2026-06"] - pivot["2025-06"]
    pivot["pct_2025"] = round(100 * pivot["2025-06"] / pivot["2025-06"].sum(), 1)
    pivot["pct_2026"] = round(100 * pivot["2026-06"] / pivot["2026-06"].sum(), 1)
    return pivot.sort_values("delta", ascending=False).reset_index()


def build_june_borough_breakdown(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(f"""
        SELECT borough, strftime({DATE_EXPR}, '%Y-%m') AS ym, count(*) AS c
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE created_date IS NOT NULL
          AND strftime({DATE_EXPR}, '%Y-%m') IN ('2025-06', '2026-06')
        GROUP BY 1, 2
    """).fetchdf()
    pivot = df.pivot(index="borough", columns="ym", values="c").fillna(0)
    pivot["delta"] = pivot["2026-06"] - pivot["2025-06"]
    return pivot.sort_values("delta", ascending=False).reset_index()


def build_june_top10_zip(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(f"""
        SELECT incident_zip, count(*) AS complaint_count
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE created_date IS NOT NULL
          AND strftime({DATE_EXPR}, '%Y-%m') = '2026-06'
          AND incident_zip IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 10
    """).fetchdf()


def build_illegal_parking_timeline(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(f"""
        SELECT strftime({DATE_EXPR}, '%Y-%m') AS ym, count(*) AS complaint_count
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE created_date IS NOT NULL
          AND complaint_type = 'Illegal Parking'
          AND upper(trim(borough)) IN ('BROOKLYN', 'QUEENS')
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    outputs = {
        "baseline_errors_2026.csv": build_baseline_errors(con),
        "complaint_type_deltas.csv": build_complaint_type_deltas(con),
        "june_channel_breakdown.csv": build_june_channel_breakdown(con),
        "june_borough_breakdown.csv": build_june_borough_breakdown(con),
        "june_top10_zip.csv": build_june_top10_zip(con),
        "illegal_parking_bk_qn_timeline.csv": build_illegal_parking_timeline(con),
    }

    for filename, frame in outputs.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False)
        print(f"{filename}: {len(frame)} rows -> {path}")


if __name__ == "__main__":
    main()
