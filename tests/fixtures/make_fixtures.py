"""Write a tiny, hand-checkable copy of data/live/ for the tests.

Same files and columns as data/live/, for Jan 2025 to Mar 2026. QUEENS
carries the interesting cases (monthly counts; "-" = no row at all):

    label                  2025, every month   Jan 2026   Feb 2026
    HEAT/HOT WATER         500                 550        600
    Heat/Hot Water         -                   50         -      spelling variant
    Sewer                  40                  -          -      renamed to ...
    Sewer Maintenance      -                   45         45     ... this
    Illegal Parking        300                 250        250
    Cannabis Retailer      -                   30         30     zero baseline
    Vending                5                   -          -      retired, small
    Noise - Residential    200                 300        -      missing row = 0

The other four boroughs get HEAT/HOT WATER 100 and Illegal Parking 50
every month. The citywide file is the five-borough total plus 10
complaints a month with no borough. metadata.json says the data is
complete through Feb 2026; Mar 2026 is in the files as a partial month.
category_catalog.csv is built from these tables with one reviewed
rename, Sewer -> Sewer Maintenance.

Run from the repo root to inspect the files: python -m tests.fixtures.make_fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analyst.catalog import CATALOG_FILE, build_catalog

MONTHS = [f"2025-{month:02d}" for month in range(1, 13)] + ["2026-01", "2026-02", "2026-03"]
RENAMES = {"Sewer": ("Sewer Maintenance", "fixture: Sewer stops as Sewer Maintenance starts")}
COMPLETE_THROUGH = "2026-02"
OTHER_BOROUGHS = ("BRONX", "BROOKLYN", "MANHATTAN", "STATEN ISLAND")
NO_BOROUGH_PER_MONTH = 10


def queens(month: str) -> dict[str, int]:
    if month.startswith("2025"):
        return {"HEAT/HOT WATER": 500, "Sewer": 40, "Illegal Parking": 300,
                "Vending": 5, "Noise - Residential": 200}
    if month == "2026-01":
        return {"HEAT/HOT WATER": 550, "Heat/Hot Water": 50, "Sewer Maintenance": 45,
                "Illegal Parking": 250, "Cannabis Retailer": 30, "Noise - Residential": 300}
    if month == "2026-02":
        return {"HEAT/HOT WATER": 600, "Sewer Maintenance": 45, "Illegal Parking": 250,
                "Cannabis Retailer": 30}
    return {"HEAT/HOT WATER": 200, "Sewer Maintenance": 15, "Illegal Parking": 80}  # partial month


def other_borough(month: str) -> dict[str, int]:
    return {"HEAT/HOT WATER": 100, "Illegal Parking": 50}


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    for month in MONTHS:
        for borough, counts in [("QUEENS", queens(month))] + [(b, other_borough(month)) for b in OTHER_BOROUGHS]:
            rows += [(f"{month}-01", borough, label, count) for label, count in counts.items()]
    category = pd.DataFrame(rows, columns=["month", "borough", "complaint_type", "complaint_count"])
    category["month"] = pd.to_datetime(category["month"]).astype("datetime64[us]")
    category["complaint_count"] = category["complaint_count"].astype("int64")

    citywide = category.groupby("month", as_index=False)["complaint_count"].sum()
    citywide["complaint_count"] += NO_BOROUGH_PER_MONTH
    citywide["month"] = citywide["month"].dt.strftime("%Y-%m-%d")

    complete = citywide[citywide["month"] <= f"{COMPLETE_THROUGH}-01"]
    metadata = {
        "refreshed_at": "2026-03-05T10:00:00",
        "api_latest_record": "2026-03-04T23:59:00",
        "complete_through": COMPLETE_THROUGH,
        "total_records": int(complete["complaint_count"].sum()),
    }
    return category, citywide, metadata


def write_fixtures(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    category, citywide, metadata = build_tables()
    category.to_parquet(out_dir / "monthly_borough_category.parquet", index=False)
    citywide.to_csv(out_dir / "monthly_citywide_counts.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    build_catalog(category, RENAMES).to_csv(out_dir / CATALOG_FILE, index=False)
    return out_dir


if __name__ == "__main__":
    print(f"Wrote fixtures to {write_fixtures(Path(__file__).parent / 'live')}")
