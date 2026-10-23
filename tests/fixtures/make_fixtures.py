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

forecasts/ holds frozen citywide forecasts: one set made Dec 5, 2025 for
Dec-Feb (scored against the citywide file: 1655, 1835, 1535), a
superseded file from Dec 4, and one set made Mar 5, 2026 for Mar-May
(not yet scorable). backtest_metrics.csv holds made-up backtest WAPEs.

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


FORECASTS = {
    # file: (model, made, trained through, {month: prediction})
    "forecast_gradient_boosting_20251204_0900.csv": (
        "gradient_boosting", "2025-12-04T09:00:00", "2025-11",
        {"2025-12": 9999, "2026-01": 9999, "2026-02": 9999}),
    "forecast_gradient_boosting_20251205_1000.csv": (
        "gradient_boosting", "2025-12-05T10:00:00", "2025-11",
        {"2025-12": 1600, "2026-01": 1700, "2026-02": 1650}),
    "forecast_seasonal_naive_20251205_1000.csv": (
        "seasonal_naive", "2025-12-05T10:00:00", "2025-11",
        {"2025-12": 1655, "2026-01": 1655, "2026-02": 1655}),
    "forecast_gradient_boosting_20260305_1000.csv": (
        "gradient_boosting", "2026-03-05T10:00:00", "2026-02",
        {"2026-03": 1600, "2026-04": 1620, "2026-05": 1640}),
    "forecast_seasonal_naive_20260305_1000.csv": (
        "seasonal_naive", "2026-03-05T10:00:00", "2026-02",
        {"2026-03": 1655, "2026-04": 1655, "2026-05": 1655}),
}
SUPERSEDED = {"forecast_gradient_boosting_20251204_0900.csv": "fixture: anchored on a partial month"}
BACKTEST = [
    # split, model, start, end, months, MAE, RMSE, MAPE, WAPE
    ("validation", "baseline_seasonal_lag12", "2025-01", "2025-06", 6, 40, 50, 2.5, 2.4),
    ("validation", "gradient_boosting", "2025-01", "2025-06", 6, 60, 70, 3.6, 3.6),
    ("test", "baseline_seasonal_lag12", "2025-07", "2026-02", 8, 90, 100, 5.5, 5.4),
    ("test", "gradient_boosting", "2025-07", "2026-02", 8, 70, 80, 4.3, 4.2),
]


def write_fixtures(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    category, citywide, metadata = build_tables()
    category.to_parquet(out_dir / "monthly_borough_category.parquet", index=False)
    citywide.to_csv(out_dir / "monthly_citywide_counts.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    build_catalog(category, RENAMES).to_csv(out_dir / CATALOG_FILE, index=False)

    forecast_dir = out_dir / "forecasts"
    forecast_dir.mkdir(exist_ok=True)
    for name, (model, made, trained, predictions) in FORECASTS.items():
        pd.DataFrame({"month": [f"{m}-01" for m in predictions], "predicted_count": list(predictions.values()),
                      "forecast_created_at": made, "training_end_month": trained, "model_name": model}
                     ).to_csv(forecast_dir / name, index=False)
    pd.DataFrame(list(SUPERSEDED.items()), columns=["source_file", "reason"]).to_csv(
        forecast_dir / "superseded.csv", index=False)
    pd.DataFrame(BACKTEST, columns=["split", "model", "period_start", "period_end", "months",
                                    "MAE", "RMSE", "MAPE", "WAPE"]).to_csv(out_dir / "backtest_metrics.csv", index=False)
    return out_dir


if __name__ == "__main__":
    print(f"Wrote fixtures to {write_fixtures(Path(__file__).parent / 'live')}")
