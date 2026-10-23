"""monthly_trend and month_summary against hand-calculated answers.

Queens monthly totals in the fixture (see tests/fixtures/make_fixtures.py):
  every 2025 month  500 + 40 + 300 + 5 + 200          = 1045
  Jan 2026          550 + 50 + 45 + 250 + 30 + 300    = 1225
  Feb 2026          600 + 45 + 250 + 30               = 925
The other four boroughs add 150 a month each (600); the citywide file adds 10.
"""

import pandas as pd
import pytest

from analyst.tools import DataUnavailableError, compare_periods, month_summary, monthly_trend
from tests.fixtures.make_fixtures import write_fixtures


def test_trend_of_all_complaints(fixture_dir):
    result = monthly_trend({"geography": "QUEENS", "period": {"start": "2025-11", "end": "2026-02"}},
                           data_dir=fixture_dir)
    assert [(r.month, r.category, r.count) for r in result.rows] == [
        ("2025-11", "All complaints", 1045), ("2025-12", "All complaints", 1045),
        ("2026-01", "All complaints", 1225), ("2026-02", "All complaints", 925)]
    assert result.summary == {
        "months": 4, "total": 4240, "first_month": "2025-11", "first_month_count": 1045,
        "last_month": "2026-02", "last_month_count": 925, "change": -120, "pct_change": -11.48,
        "peak_month": "2026-01", "peak_count": 1225, "low_month": "2026-02", "low_count": 925,
        "mean_per_month": 1060.0}


def test_trend_by_category_follows_renames(fixture_dir):
    result = monthly_trend({"geography": "QUEENS", "categories": ["Sewer", "cannabis retailer"],
                            "period": {"start": "2025-11", "end": "2026-02"}}, data_dir=fixture_dir)
    assert [(r.category, r.month, r.count, r.flags) for r in result.rows] == [
        ("Sewer Maintenance", "2025-11", 40, ["renamed"]), ("Sewer Maintenance", "2025-12", 40, ["renamed"]),
        ("Sewer Maintenance", "2026-01", 45, ["renamed"]), ("Sewer Maintenance", "2026-02", 45, ["renamed"]),
        ("Cannabis Retailer", "2025-11", 0, []), ("Cannabis Retailer", "2025-12", 0, []),
        ("Cannabis Retailer", "2026-01", 30, []), ("Cannabis Retailer", "2026-02", 30, []),
    ]
    # combined 40, 40, 75, 75: ties go to the later month for the peak, the earlier for the low
    assert {k: result.summary[k] for k in ("total", "change", "pct_change", "peak_month", "low_month")} == {
        "total": 230, "change": 35, "pct_change": 87.5, "peak_month": "2026-02", "low_month": "2025-11"}
    assert any('"Sewer" was renamed "Sewer Maintenance" in Jan 2026' in w for w in result.warnings)


def test_citywide_trend_reports_complaints_without_a_borough(fixture_dir):
    result = monthly_trend({"geography": "CITYWIDE", "period": {"start": "2026-01", "end": "2026-02"}},
                           data_dir=fixture_dir)
    assert [r.count for r in result.rows] == [1825, 1525]          # 1225 + 600, 925 + 600
    assert result.summary["citywide_total"] == 3370                # 1835 + 1535
    assert any("3,350 of 3,370" in w for w in result.warnings)


def test_trend_refuses_a_partial_month(fixture_dir):
    with pytest.raises(DataUnavailableError, match="complete only through Feb 2026"):
        monthly_trend({"geography": "QUEENS", "period": {"start": "2026-01", "end": "2026-03"}},
                      data_dir=fixture_dir)


@pytest.fixture
def with_spike(tmp_path):
    """Staten Island Drug Activity: 100 a month, except 500 in Jul 2025 and 120 in Aug 2025."""
    data_dir = write_fixtures(tmp_path)
    path = data_dir / "monthly_borough_category.parquet"
    months = pd.date_range("2025-01-01", "2026-02-01", freq="MS")
    counts = [500 if m == pd.Timestamp("2025-07-01") else 120 if m == pd.Timestamp("2025-08-01") else 100
              for m in months]
    extra = pd.DataFrame({"month": months.astype("datetime64[us]"), "borough": "STATEN ISLAND",
                          "complaint_type": "Drug Activity", "complaint_count": counts})
    pd.concat([pd.read_parquet(path), extra]).to_parquet(path, index=False)
    return data_dir


def test_trend_flags_an_unusual_month(with_spike):
    result = monthly_trend({"geography": "STATEN ISLAND", "categories": ["Drug Activity"],
                            "period": {"start": "2025-07", "end": "2025-12"}}, data_dir=with_spike)
    # typical month = median of the 6 months before Jul 2025 in the data = 100; 500 > 3 x 100
    assert [r.month for r in result.rows if "spike" in r.flags] == ["2025-07"]
    assert any("Jul 2025 recorded 500, more than 3 times its typical month (100)" in w for w in result.warnings)


def test_compare_flags_an_unusual_month(with_spike):
    result = compare_periods({"geography": "STATEN ISLAND", "categories": ["Drug Activity"],
                              "period_a": {"start": "2025-07", "end": "2025-08"},
                              "period_b": {"start": "2026-01", "end": "2026-02"}}, data_dir=with_spike)
    # "unreviewed" too: the label was added to the data but not to the fixture catalog
    assert [(r.count_a, r.count_b, r.flags) for r in result.rows] == [(620, 200, ["unreviewed", "spike"])]


def test_month_summary_matches_hand_calculation(fixture_dir):
    result = month_summary({"geography": "QUEENS", "month": "2026-02"}, data_dir=fixture_dir)
    assert result.summary == {
        "month": "2026-02", "month_total": 925,
        "prior_month": "2026-01", "prior_month_total": 1225, "change_vs_prior_month": -300,
        "pct_vs_prior_month": -24.49,                                # -300 / 1225
        "last_year_month": "2025-02", "last_year_total": 1045, "change_vs_last_year": -120,
        "pct_vs_last_year": -11.48,                                  # -120 / 1045
        "average_prior_12_months": 1060.0,                           # (11 x 1045 + 1225) / 12
        "population": 2405464, "per_100k": 38.5,                     # 100,000 x 925 / 2,405,464
    }
    rows = [(r.category, r.count, r.prior_month, r.last_year, r.change_vs_last_year, r.pct_vs_last_year,
             r.share_of_month, r.flags) for r in result.rows]
    assert rows == [
        ("HEAT/HOT WATER", 600, 600, 500, 100, 20.0, 64.86, ["top", "rising"]),
        ("Illegal Parking", 250, 250, 300, -50, -16.67, 27.03, ["top", "falling"]),
        ("Sewer Maintenance", 45, 45, 40, 5, 12.5, 4.86, ["top", "renamed", "rising"]),
        ("Cannabis Retailer", 30, 30, 0, 30, None, 3.24, ["top", "new_vs_last_year", "rising"]),
        ("Noise - Residential", 0, 300, 200, -200, -100.0, 0.0, ["top", "falling"]),
        ("Vending", 0, 0, 5, -5, -100.0, 0.0, ["top", "falling"]),
    ]
    assert any("2020 Census" in w for w in result.warnings)
    assert result.provenance.sources[-1].path.endswith("borough_population.csv")


def test_month_summary_with_no_earlier_months(fixture_dir):
    result = month_summary({"geography": "QUEENS", "month": "2025-01", "top_n": 2}, data_dir=fixture_dir)
    assert (result.summary["prior_month_total"], result.summary["last_year_total"]) == (None, None)
    assert result.summary["average_prior_12_months"] is None
    assert [(r.category, r.flags) for r in result.rows] == [("HEAT/HOT WATER", ["top"]), ("Illegal Parking", ["top"])]
    warnings = " ".join(result.warnings)
    assert "No data for the month before or a year before" in warnings
    assert "no 12-month average" in warnings


def test_citywide_month_summary(fixture_dir):
    result = month_summary({"geography": "CITYWIDE", "month": "2026-02"}, data_dir=fixture_dir)
    assert result.summary["month_total"] == 1525                     # 925 + 600
    assert result.summary["citywide_total"] == 1535
    assert result.summary["population"] == 8804190                   # five 2020 Census boroughs
    assert result.summary["per_100k"] == 17.3                        # 100,000 x 1525 / 8,804,190
    assert any("1,525 of 1,535" in w for w in result.warnings)


def test_month_summary_refuses_a_partial_month(fixture_dir):
    with pytest.raises(DataUnavailableError, match="complete only through Feb 2026"):
        month_summary({"geography": "QUEENS", "month": "2026-03"}, data_dir=fixture_dir)
