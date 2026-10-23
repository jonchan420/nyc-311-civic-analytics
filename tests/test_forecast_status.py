"""forecast_status against hand-scored fixture forecasts (see tests/fixtures/make_fixtures.py).

Citywide actuals: Dec 2025 1655, Jan 2026 1835, Feb 2026 1535 (total 5025).
  gradient boosting  1600, 1700, 1650 -> errors -55, -135, +115; WAPE 305/5025 = 6.07%
  seasonal naive     1655, 1655, 1655 -> errors   0, -180, +120; WAPE 300/5025 = 5.97%
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from analyst.forecast import forecast_status
from analyst.tools import DataUnavailableError
from config.settings import FORECAST_DIR, LIVE_DIR
from tests.fixtures.make_fixtures import write_fixtures


def run(data_dir):
    return forecast_status(data_dir=data_dir, forecast_dir=data_dir / "forecasts")


def test_track_record_matches_hand_scoring(fixture_dir):
    rows = [(r.model, r.month, r.status, r.predicted, r.actual, r.error, r.pct_error)
            for r in run(fixture_dir).rows]
    assert rows[:6] == [
        ("gradient_boosting", "2025-12", "scored", 1600, 1655, -55, -3.32),
        ("seasonal_naive", "2025-12", "scored", 1655, 1655, 0, 0.0),
        ("gradient_boosting", "2026-01", "scored", 1700, 1835, -135, -7.36),
        ("seasonal_naive", "2026-01", "scored", 1655, 1835, -180, -9.81),
        ("gradient_boosting", "2026-02", "scored", 1650, 1535, 115, 7.49),
        ("seasonal_naive", "2026-02", "scored", 1655, 1535, 120, 7.82),
    ]
    assert rows[6:] == [
        ("gradient_boosting", "2026-03", "upcoming", 1600, None, None, None),
        ("seasonal_naive", "2026-03", "upcoming", 1655, None, None, None),
        ("gradient_boosting", "2026-04", "upcoming", 1620, None, None, None),
        ("seasonal_naive", "2026-04", "upcoming", 1655, None, None, None),
        ("gradient_boosting", "2026-05", "upcoming", 1640, None, None, None),
        ("seasonal_naive", "2026-05", "upcoming", 1655, None, None, None),
    ]


def test_summary_reports_next_month_track_record_and_backtest(fixture_dir):
    assert run(fixture_dir).summary == {
        "data_complete_through": "2026-02",
        "months_scored": 3,
        "next_month": "2026-03",
        "next_month_gradient_boosting": 1600,
        "next_month_seasonal_naive": 1655,
        "wape_gradient_boosting": 6.07, "mae_gradient_boosting": 102,   # 305 / 3 = 101.7
        "wape_seasonal_naive": 5.97, "mae_seasonal_naive": 100,
        "validation_wape_seasonal_naive": 2.4, "validation_wape_gradient_boosting": 3.6,
        "validation_period": "2025-01 to 2025-06",
        "test_wape_seasonal_naive": 5.4, "test_wape_gradient_boosting": 4.2,
        "test_period": "2025-07 to 2026-02",
    }


def test_superseded_forecast_is_left_out_and_explained(fixture_dir):
    result = run(fixture_dir)
    assert 9999 not in {r.predicted for r in result.rows}
    assert "forecast_gradient_boosting_20251204_0900.csv" not in {r.forecast_file for r in result.rows}
    warnings = " ".join(result.warnings)
    assert "1 earlier forecast file is left out as superseded" in warnings
    assert "citywide totals only" in warnings and "early evidence" in warnings


def test_provenance_lists_every_file_used(fixture_dir):
    paths = [s.path for s in run(fixture_dir).provenance.sources]
    assert [p.rsplit("/", 1)[-1] for p in paths] == [
        "metadata.json", "monthly_citywide_counts.csv",
        "forecast_gradient_boosting_20251205_1000.csv", "forecast_gradient_boosting_20260305_1000.csv",
        "forecast_seasonal_naive_20251205_1000.csv", "forecast_seasonal_naive_20260305_1000.csv",
        "superseded.csv", "backtest_metrics.csv"]


def test_a_model_that_is_always_low_is_called_out(tmp_path):
    data_dir = write_fixtures(tmp_path)
    path = data_dir / "forecasts" / "forecast_gradient_boosting_20251205_1000.csv"
    table = pd.read_csv(path)
    table["predicted_count"] = [1500, 1500, 1500]   # below 1655, 1835, 1535
    table.to_csv(path, index=False)
    assert "The gradient boosting forecast was below the actual count in all 3 scored months." in run(data_dir).warnings


def test_missing_forecasts_are_refused(tmp_path):
    data_dir = write_fixtures(tmp_path)
    with pytest.raises(DataUnavailableError, match="No frozen forecasts"):
        forecast_status(data_dir=data_dir, forecast_dir=tmp_path / "empty")


def test_forecast_status_takes_no_parameters():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        forecast_status({"geography": "QUEENS"})


@pytest.mark.skipif(not (LIVE_DIR / "monthly_citywide_counts.csv").exists(), reason="no live data")
def test_live_forecast_status():
    result = forecast_status(data_dir=LIVE_DIR, forecast_dir=FORECAST_DIR)
    assert result.summary["months_scored"] >= 3
    scored = [r for r in result.rows if r.status == "scored"]
    assert all(r.month <= result.summary["data_complete_through"] for r in scored)
    assert all(r.actual is not None for r in scored)
