"""compare_periods agrees with an independent pandas implementation on random questions."""

import random

import pytest

from analyst.tools import compare_periods
from tests.fixtures.oracle import compare

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
