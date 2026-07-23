"""Step 2: convert the 14 GB CSV to yearly Parquet files in ONE pass.

Why DuckDB and not pandas: pandas would need to hold the data in RAM
(or crawl through it in chunks). DuckDB streams the CSV from disk,
filters, renames, and writes compressed Parquet partitioned by year —
all without loading everything into memory. On an M-series MacBook
this typically takes a few minutes for a file this size.

After this runs successfully, you never touch the raw CSV again.

Run:
    python src/convert_csv_to_parquet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import RAW_CSV_PATH, RAW_DIR, START_YEAR

# maps our clean internal name -> possible column names in NYC exports.
# inspect_csv.py tells you which variant your file uses; the code
# below auto-detects among these candidates.
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "unique_key": ["Unique Key"],
    "created_date": ["Created Date"],
    "closed_date": ["Closed Date"],
    "agency": ["Agency"],
    "agency_name": ["Agency Name"],
    "complaint_type": ["Complaint Type", "Problem (formerly Complaint Type)", "Problem"],
    "descriptor": ["Descriptor", "Problem Detail (formerly Descriptor)", "Problem Detail"],
    "location_type": ["Location Type"],
    "incident_zip": ["Incident Zip", "Incident ZIP"],
    "status": ["Status"],
    "resolution_action_updated_date": ["Resolution Action Updated Date"],
    "community_board": ["Community Board"],
    "borough": ["Borough"],
    "open_data_channel_type": ["Open Data Channel Type"],
    "latitude": ["Latitude"],
    "longitude": ["Longitude"],
}


def detect_columns(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Return {internal_name: actual_csv_column} for columns that exist."""
    actual = {
        col[0]
        for col in con.execute(
            "SELECT * FROM read_csv_auto(?, sample_size=2000) LIMIT 0",
            [str(RAW_CSV_PATH)],
        ).description
    }

    mapping: dict[str, str] = {}
    for internal, candidates in COLUMN_CANDIDATES.items():
        for candidate in candidates:
            if candidate in actual:
                mapping[internal] = candidate
                break

    missing = set(COLUMN_CANDIDATES) - set(mapping)
    if missing:
        print(f"Note: these fields were not found and will be skipped: {sorted(missing)}")

    required = {"created_date", "complaint_type", "borough"}
    if not required.issubset(mapping):
        raise RuntimeError(
            f"Required columns missing: {required - set(mapping)}. "
            "Run inspect_csv.py and update COLUMN_CANDIDATES."
        )

    return mapping


def convert() -> None:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {RAW_CSV_PATH}")

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='4GB'")

    mapping = detect_columns(con)

    select_parts = [
        f'"{csv_col}" AS {internal}' for internal, csv_col in mapping.items()
    ]
    select_clause = ", ".join(select_parts)
    date_col = mapping["created_date"]

    output_dir = RAW_DIR / "parquet_by_year"
    output_dir.mkdir(parents=True, exist_ok=True)

    query = f"""
        COPY (
            SELECT
                {select_clause},
                year(strptime("{date_col}", '%m/%d/%Y %I:%M:%S %p')) AS yr
            FROM read_csv_auto(
                '{RAW_CSV_PATH}',
                sample_size=200000,
                ignore_errors=true,
                all_varchar=true
            )
            WHERE year(strptime("{date_col}", '%m/%d/%Y %I:%M:%S %p')) >= {START_YEAR}
        )
        TO '{output_dir}'
        (FORMAT PARQUET, PARTITION_BY (yr), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE true)
    """

    print("Converting CSV to Parquet — this reads the full 14 GB once...")
    con.execute(query)

    total = con.execute(
        f"SELECT yr, count(*) AS rows FROM read_parquet('{output_dir}/**/*.parquet') GROUP BY yr ORDER BY yr"
    ).fetchdf()
    print("\nRows written per year:")
    print(total.to_string(index=False))
    print(f"\nDone. Parquet files are in {output_dir}")
    print("You can now archive or delete the raw CSV — everything downstream reads Parquet.")


if __name__ == "__main__":
    convert()
