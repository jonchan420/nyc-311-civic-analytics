"""The tools agree with an independent pandas implementation on random questions."""

import random

import pytest

from analyst.tools import POPULATION_FILE, compare_periods, month_summary, monthly_trend
from tests.fixtures.oracle import compare, trend
from tests.fixtures.oracle import month_summary as month_summary_oracle

MONTHS = [f"2025-{m:02d}" for m in range(1, 13)] + ["2026-01", "2026-02"]  # complete months
GEOGRAPHIES = ["CITYWIDE", "BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
CATEGORIES = ["Cannabis Retailer", "HEAT/HOT WATER", "Illegal Parking", "Noise - Residential",
              "Sewer Maintenance", "Vending"]


def random_case(rng: random.Random) -> dict:
    length = rng.randint(1, 6)
    a = rng.randint(0, len(MONTHS) - 2 * length)
    b = rng.randint(a + length, len(MONTHS) - length)
    return {
        "geography": rng.choice(GEOGRAPHIES),
        "period_a": {"start": MONTHS[a], "end": MONTHS[a + length - 1]},
        "period_b": {"start": MONTHS[b], "end": MONTHS[b + length - 1]},
        "categories": rng.sample(CATEGORIES, rng.randint(1, 3)) if rng.random() < 0.3 else None,
        "top_n": rng.randint(1, 10),
    }


@pytest.mark.parametrize("seed", range(300))
def test_tool_matches_oracle(fixture_dir, seed):
    case = random_case(random.Random(seed))
    result = compare_periods(case, data_dir=fixture_dir)
    total_a, total_b, rows = compare(
        fixture_dir, case["geography"],
        (case["period_a"]["start"], case["period_a"]["end"]),
        (case["period_b"]["start"], case["period_b"]["end"]),
        case["categories"],
    )
    assert (result.summary["total_a"], result.summary["total_b"]) == (total_a, total_b)
    tool_rows = [(r.category, r.count_a, r.count_b, r.change, r.pct_change, r.share_of_change)
                 for r in result.rows]
    assert tool_rows == rows[: case["top_n"]]


@pytest.mark.parametrize("seed", range(100))
def test_trend_matches_oracle(fixture_dir, seed):
    rng = random.Random(seed)
    start = rng.randint(0, len(MONTHS) - 1)
    end = rng.randint(start, len(MONTHS) - 1)
    categories = rng.sample(CATEGORIES, rng.randint(1, 3)) if rng.random() < 0.5 else None
    geography = rng.choice(GEOGRAPHIES)
    result = monthly_trend({"geography": geography, "period": {"start": MONTHS[start], "end": MONTHS[end]},
                            "categories": categories}, data_dir=fixture_dir)
    expected = trend(fixture_dir, geography, MONTHS[start], MONTHS[end], categories)
    series = expected.pop("series")
    assert {key: result.summary[key] for key in expected} == expected
    for name, counts in series.items():
        assert [row.count for row in result.rows if row.category == name] == counts


@pytest.mark.parametrize("top_n", [1, 3, 8])
@pytest.mark.parametrize("geography", GEOGRAPHIES)
@pytest.mark.parametrize("month", MONTHS)
def test_month_summary_matches_oracle(fixture_dir, month, geography, top_n):
    # every combination: only Jan and Feb 2026 have a year-before month, and only a small
    # top_n leaves room for the rising/falling rows, so random draws miss those cases
    result = month_summary({"geography": geography, "month": month, "top_n": top_n}, data_dir=fixture_dir)
    expected = month_summary_oracle(fixture_dir, geography, month, POPULATION_FILE, top_n)
    rows = expected.pop("rows")
    assert {key: result.summary[key] for key in expected} == expected
    assert [(r.category, r.count, r.last_year, r.change_vs_last_year) for r in result.rows] == rows
