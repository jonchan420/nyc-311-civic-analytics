"""Step 1: inspect the raw CSV before doing anything else.

Prints the column names and a small sample so we know exactly what
we're working with. NYC's export column names change over time
("Complaint Type" vs "Problem (formerly Complaint Type)"), so never
assume — always look first.

Run:
    python src/inspect_csv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import RAW_CSV_PATH


def inspect_csv() -> None:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found at {RAW_CSV_PATH}. "
            "Edit RAW_CSV_PATH in config/settings.py to point at your file."
        )

    con = duckdb.connect()

    columns = con.execute(
        "SELECT * FROM read_csv_auto(?, sample_size=2000) LIMIT 0",
        [str(RAW_CSV_PATH)],
    ).description

    print("Columns found in the CSV:")
    for col in columns:
        print(f"  {col[0]}")

    sample = con.execute(
        "SELECT * FROM read_csv_auto(?, sample_size=2000) LIMIT 3",
        [str(RAW_CSV_PATH)],
    ).fetchdf()

    print("\nFirst 3 rows (transposed):")
    print(sample.T.to_string())


if __name__ == "__main__":
    inspect_csv()
