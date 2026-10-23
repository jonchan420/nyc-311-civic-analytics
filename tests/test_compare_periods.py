"""compare_periods against hand-calculated answers (see tests/fixtures/make_fixtures.py).

Expected numbers are typed in by hand from the fixture table, never
computed by the code under test.
"""

import hashlib
import json

import pandas as pd
import pytest
from pydantic import ValidationError

from analyst.data import DataSource
from analyst.schemas import CategoryChange, ComparePeriodsParams, MonthRange, ToolResult
from analyst.tools import DataUnavailableError, UnknownCategoryError, compare_periods
from config.settings import PROJECT_ROOT
from tests.fixtures.make_fixtures import write_fixtures

WINTER = {"period_a": {"start": "2025-01", "end": "2025-02"},
          "period_b": {"start": "2026-01", "end": "2026-02"}}

# QUEENS, Jan-Feb 2025 vs Jan-Feb 2026, by hand:
#   HEAT/HOT WATER     A 500+500=1000   B (550+50)+600=1200   +200
#   Illegal Parking    A 300+300=600    B 250+250=500         -100
#   Noise - Residential A 200+200=400   B 300+0 (no row)=300  -100
#   Cannabis Retailer  A 0              B 30+30=60            +60
#   Sewer Maintenance  A 40+40=80 ("Sewer")  B 45+45=90       +10
#   Vending            A 5+5=10         B 0                   -10
#   total              A 2090           B 2150                +60 (2.87%)
# share_of_change = change / 60; ties in |change| sort by name.
QUEENS_ROWS = [
    # category,             count_a, count_b, change, pct,    share,   flags
    ("HEAT/HOT WATER",       1000,    1200,    200,   20.0,   333.33,  []),
    ("Illegal Parking",      600,     500,    -100,  -16.67, -166.67,  []),
    ("Noise - Residential",  400,     300,    -100,  -25.0,  -166.67,  []),
    ("Cannabis Retailer",    0,       60,      60,    None,   100.0,   ["new_in_b"]),
    ("Sewer Maintenance",    80,      90,      10,    12.5,   16.67,   ["renamed"]),
    ("Vending",              10,      0,      -10,   -100.0, -16.67,   ["absent_in_b", "small_count"]),
]


def run(fixture_dir, **params):
    return compare_periods({"geography": "QUEENS", **WINTER, **params}, data_dir=fixture_dir)


def as_tuples(rows):
    return [(r.category, r.count_a, r.count_b, r.change, r.pct_change, r.share_of_change, r.flags)
            for r in rows]


def test_queens_matches_hand_calculation(fixture_dir):
    result = run(fixture_dir)
    assert as_tuples(result.rows) == QUEENS_ROWS
    assert result.summary == {
        "months_per_period": 2, "total_a": 2090, "total_b": 2150, "change": 60,
        "pct_change": 2.87, "categories_compared": 6, "shown_change": 60, "other_change": 0,
    }


def test_top_n_rows_reconcile_with_the_total(fixture_dir):
    result = run(fixture_dir, top_n=3)
    assert [r.category for r in result.rows] == ["HEAT/HOT WATER", "Illegal Parking", "Noise - Residential"]
    # 200 - 100 - 100 = 0 shown; Cannabis +60, Sewer +10, Vending -10 = 60 not shown
    assert (result.summary["shown_change"], result.summary["other_change"]) == (0, 60)


def test_spelling_variants_and_renames_are_combined(fixture_dir):
    rows = {r.category: r for r in run(fixture_dir).rows}
    assert rows["HEAT/HOT WATER"].labels == ["HEAT/HOT WATER", "Heat/Hot Water"]
    assert rows["Sewer Maintenance"].labels == ["Sewer", "Sewer Maintenance"]
    assert "Sewer" not in rows  # the old name never becomes its own row


def test_without_a_catalog_only_spellings_are_combined(tmp_path):
    data_dir = write_fixtures(tmp_path)
    (data_dir / "category_catalog.csv").unlink()
    result = run(data_dir)
    rows = {r.category: r for r in result.rows}
    assert rows["HEAT/HOT WATER"].labels == ["HEAT/HOT WATER", "Heat/Hot Water"]
    assert (rows["Sewer"].count_a, rows["Sewer"].count_b) == (80, 0)
    assert (rows["Sewer Maintenance"].count_a, rows["Sewer Maintenance"].count_b) == (0, 90)
    assert any("No category_catalog.csv was found" in w for w in result.warnings)


def test_a_label_the_catalog_has_not_seen_is_flagged(tmp_path):
    data_dir = write_fixtures(tmp_path)
    path = data_dir / "monthly_borough_category.parquet"
    table = pd.read_parquet(path)
    new = pd.DataFrame({"month": pd.to_datetime(["2026-01-01", "2026-02-01"]).astype("datetime64[us]"),
                        "borough": "QUEENS", "complaint_type": "Street Light Outage", "complaint_count": [25, 35]})
    pd.concat([table, new]).to_parquet(path, index=False)
    result = run(data_dir)
    row = {r.category: r for r in result.rows}["Street Light Outage"]
    assert (row.count_a, row.count_b, row.flags) == (0, 60, ["new_in_b", "unreviewed"])
    assert any("Street Light Outage: not yet in the category catalog" in w for w in result.warnings)


def test_warnings_explain_every_flag(fixture_dir):
    warnings = " ".join(run(fixture_dir).warnings)
    assert '"Sewer" was renamed "Sewer Maintenance" in Jan 2026' in warnings
    assert "No complaints in period A for Cannabis Retailer" in warnings
    assert "No complaints in period B for Vending" in warnings
    assert "Fewer than 20 complaints in both periods for Vending" in warnings
    assert "moved against the overall change" in warnings  # shares of 333% and -167%


def test_citywide_counts_five_boroughs_and_reports_the_rest(fixture_dir):
    result = compare_periods({"geography": "citywide", **WINTER}, data_dir=fixture_dir)
    rows = {r.category: r for r in result.rows}
    # 4 other boroughs add HEAT 100 x 2 months and Illegal Parking 50 x 2 months each
    assert (rows["HEAT/HOT WATER"].count_a, rows["HEAT/HOT WATER"].count_b) == (1800, 2000)
    assert rows["HEAT/HOT WATER"].pct_change == 11.11            # 200 / 1800
    assert (rows["Illegal Parking"].count_a, rows["Illegal Parking"].count_b) == (1000, 900)
    # five boroughs: 2090 + 4 x 300 = 3290 and 2150 + 1200 = 3350
    # citywide file adds 10 a month with no borough: 3310 and 3370
    assert {k: result.summary[k] for k in ("total_a", "total_b", "citywide_total_a", "citywide_total_b")} == {
        "total_a": 3290, "total_b": 3350, "citywide_total_a": 3310, "citywide_total_b": 3370}
    assert any("3,290 of 3,310" in w and "3,350 of 3,370" in w for w in result.warnings)


def test_category_filter_accepts_old_names_and_any_case(fixture_dir):
    result = run(fixture_dir, categories=["heat/hot water", "Sewer"])
    # 1000 + 80 = 1080 -> 1200 + 90 = 1290, +210; shares 200/210 and 10/210
    assert as_tuples(result.rows) == [
        ("HEAT/HOT WATER", 1000, 1200, 200, 20.0, 95.24, []),
        ("Sewer Maintenance", 80, 90, 10, 12.5, 4.76, ["renamed"]),
    ]
    assert result.summary["change"] == 210
    assert "complaint types: HEAT/HOT WATER, Sewer Maintenance" in result.provenance.filters


def test_requested_category_with_no_complaints_still_gets_a_row(fixture_dir):
    result = compare_periods({
        "geography": "QUEENS", "categories": ["Cannabis Retailer"],
        "period_a": {"start": "2025-01", "end": "2025-02"},
        "period_b": {"start": "2025-03", "end": "2025-04"},
    }, data_dir=fixture_dir)
    assert as_tuples(result.rows) == [("Cannabis Retailer", 0, 0, 0, None, None, ["small_count"])]
    assert "The total didn't change" in " ".join(result.warnings)


def test_unknown_category_suggests_close_matches(fixture_dir):
    with pytest.raises(UnknownCategoryError, match="Closest matches: Illegal Parking"):
        run(fixture_dir, categories=["Ilegal Parking"])


def test_category_text_never_reaches_sql(fixture_dir):
    with pytest.raises(UnknownCategoryError):
        run(fixture_dir, categories=["x' OR 1=1 --"])


def test_partial_month_is_refused(fixture_dir):
    with pytest.raises(DataUnavailableError, match="complete only through Feb 2026"):
        compare_periods({"geography": "QUEENS",
                         "period_a": {"start": "2025-02", "end": "2025-03"},
                         "period_b": {"start": "2026-02", "end": "2026-03"}}, data_dir=fixture_dir)


def test_months_before_the_data_are_refused(fixture_dir):
    with pytest.raises(DataUnavailableError, match="the data begins in Jan 2025"):
        compare_periods({"geography": "QUEENS",
                         "period_a": {"start": "2024-12", "end": "2025-01"},
                         "period_b": {"start": "2025-12", "end": "2026-01"}}, data_dir=fixture_dir)


def test_a_missing_month_is_a_gap_not_zero(tmp_path):
    data_dir = write_fixtures(tmp_path)
    path = data_dir / "monthly_borough_category.parquet"
    table = pd.read_parquet(path)
    gap = (table["borough"] == "QUEENS") & (table["month"] == "2025-02-01")
    table[~gap].to_parquet(path, index=False)
    with pytest.raises(DataUnavailableError, match="Queens in Feb 2025"):
        run(data_dir)
    with pytest.raises(DataUnavailableError, match="Queens in Feb 2025"):
        compare_periods({"geography": "CITYWIDE", **WINTER}, data_dir=data_dir)


def test_missing_freshness_record_is_refused(tmp_path):
    data_dir = write_fixtures(tmp_path)
    (data_dir / "metadata.json").unlink()
    with pytest.raises(DataUnavailableError, match="no freshness record"):
        run(data_dir)


def test_changed_columns_are_refused(tmp_path):
    data_dir = write_fixtures(tmp_path)
    path = data_dir / "monthly_borough_category.parquet"
    pd.read_parquet(path).rename(columns={"complaint_count": "count"}).to_parquet(path, index=False)
    with pytest.raises(DataUnavailableError, match="complaint_count"):
        run(data_dir)


def test_provenance_records_inputs_and_file_versions(fixture_dir):
    provenance = run(fixture_dir).provenance
    assert provenance.tool == "compare_periods"
    assert provenance.params == {"geography": "QUEENS", **WINTER, "categories": None, "top_n": 10}
    assert provenance.data_complete_through == "2026-02"
    assert provenance.data_refreshed_at == "2026-03-05T10:00:00"
    assert provenance.unit.startswith("recorded 311 service requests")
    assert provenance.filters[0] == "geography: Queens; complaints with no borough recorded are excluded"
    names = ["monthly_borough_category.parquet", "metadata.json", "category_catalog.csv"]
    assert len(provenance.sources) == len(names)
    for name, source in zip(names, provenance.sources):
        assert source.path.endswith(name)
        assert source.sha256 == hashlib.sha256((fixture_dir / name).read_bytes()).hexdigest()


def test_result_round_trips_as_json(fixture_dir):
    result = run(fixture_dir)
    text = result.model_dump_json()
    assert ToolResult[CategoryChange].model_validate_json(text) == result
    assert json.loads(text)["summary"]["change"] == 60


@pytest.mark.parametrize("bad, message", [
    ({"period_b": {"start": "2026-01", "end": "2026-03"}}, "same length"),
    ({"period_a": {"start": "2026-01", "end": "2026-02"},
      "period_b": {"start": "2025-01", "end": "2025-02"}}, "must end before period_b starts"),
    ({"period_a": {"start": "2025-13", "end": "2025-14"}}, "pattern"),
    ({"period_a": {"start": "2025-02", "end": "2025-01"}}, "after end"),
    ({"geography": "Long Island"}, "geography"),
    ({"top_n": 0}, "greater than or equal to 1"),
    ({"top_n": 26}, "less than or equal to 25"),
    ({"sql": "DROP TABLE x"}, "Extra inputs are not permitted"),
])
def test_invalid_parameters_are_rejected_before_any_query(bad, message):
    with pytest.raises(ValidationError, match=message):
        ComparePeriodsParams.model_validate({"geography": "QUEENS", **WINTER, **bad})


def test_geography_is_case_insensitive():
    assert ComparePeriodsParams.model_validate({"geography": " queens ", **WINTER}).geography == "QUEENS"


LIVE_DIR = PROJECT_ROOT / "data" / "live"


@pytest.mark.skipif(not (LIVE_DIR / "monthly_borough_category.parquet").exists(),
                    reason="committed live data not present")
def test_real_data_totals_match_an_independent_query():
    result = compare_periods({"geography": "QUEENS", "top_n": 25,
                              "period_a": {"start": "2025-06", "end": "2025-08"},
                              "period_b": {"start": "2026-06", "end": "2026-08"}})
    table = pd.read_parquet(LIVE_DIR / "monthly_borough_category.parquet")
    queens = table[table["borough"] == "QUEENS"]
    month = queens["month"].dt.strftime("%Y-%m")
    assert result.summary["total_a"] == queens.loc[month.between("2025-06", "2025-08"), "complaint_count"].sum()
    assert result.summary["total_b"] == queens.loc[month.between("2026-06", "2026-08"), "complaint_count"].sum()
    assert "Sewer" not in {r.category for r in result.rows}


def test_sql_takes_values_only_as_bound_parameters(fixture_dir):
    with DataSource(fixture_dir) as source:     # window names become column names, so they're checked
        with pytest.raises(ValueError, match="bad window name"):
            source.counts_by_label(["QUEENS"], {"a; DROP TABLE borough_category": MonthRange(start="2025-01",
                                                                                           end="2025-01")})
    params = {"geography": "QUEENS", "period_a": {"start": "2025-01", "end": "2025-01"},
              "period_b": {"start": "2026-01", "end": "2026-01"}}
    with pytest.raises(UnknownCategoryError):
        compare_periods({**params, "categories": ["Noise'); DROP TABLE borough_category; --"]}, data_dir=fixture_dir)
    assert compare_periods(params, data_dir=fixture_dir).summary["total_b"] == 1225   # nothing was dropped
