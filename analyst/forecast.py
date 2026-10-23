"""forecast_status: the latest frozen citywide forecasts and how accurate past ones were.

Reuses the scoring rules in src/forecast_registry.py: each month is scored
with the most recent forecast that existed before the month's data could
be complete, and superseded files are left out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from analyst.data import CITYWIDE_FILE, METADATA_FILE, UNIT, DataUnavailableError, now_utc, read_metadata, source_file
from analyst.schemas import ForecastRow, ForecastStatusParams, Provenance, ToolResult
from config.settings import FORECAST_DIR, LIVE_DIR
from src.forecast_registry import SUPERSEDED_FILE, load_forecasts, score, summarize

BACKTEST_FILE = "backtest_metrics.csv"
EARLY_EVIDENCE_MONTHS = 6
MODEL_NAMES = {
    "gradient_boosting": "gradient boosting",
    "seasonal_naive": "seasonal naive (same month last year)",
    "linear_regression": "linear regression",
    "baseline_previous_month": "previous month",
}
BACKTEST_MODEL_IDS = {"baseline_seasonal_lag12": "seasonal_naive"}  # same method, older name
CALCULATION = (
    "Each month is scored with the most recent forecast made before that month's data could be "
    "complete, leaving out superseded files. error = predicted - actual; pct_error = 100 * error "
    "/ actual; WAPE = 100 * sum(|error|) / sum(actual). Backtest WAPE comes from "
    "src/train_evaluate.py on 2025 (validation) and complete 2026 months (test)."
)


def forecast_status(
    params: ForecastStatusParams | dict[str, Any] | None = None,
    data_dir: Path = LIVE_DIR,
    forecast_dir: Path = FORECAST_DIR,
) -> ToolResult[ForecastRow]:
    """What do the frozen forecasts predict next, and how accurate have they been?"""
    params = ForecastStatusParams.model_validate(params or {})
    data_dir, forecast_dir = Path(data_dir), Path(forecast_dir)
    meta = read_metadata(data_dir)
    complete_through = meta["complete_through"]
    city_path = data_dir / CITYWIDE_FILE
    if not city_path.exists():
        raise DataUnavailableError(f"The data file {city_path} is missing.")
    actuals = pd.read_csv(city_path, parse_dates=["month"])
    actuals = actuals[actuals["month"].dt.strftime("%Y-%m") <= complete_through]
    forecasts = load_forecasts(forecast_dir)
    if forecasts.empty:
        raise DataUnavailableError(f"No frozen forecasts were found in {forecast_dir}.")

    scored = score(forecasts, actuals)
    track = scored[scored["in_track_record"]]
    active = forecasts[~forecasts["superseded"]]
    newest = active[active["training_end_month"] == active.groupby("model_name")["training_end_month"].transform("max")]
    upcoming = newest[newest["month"].dt.strftime("%Y-%m") > complete_through]

    rows = [_row(r, "scored") for r in track.itertuples()] + [_row(r, "upcoming") for r in upcoming.itertuples()]
    rows.sort(key=lambda row: (row.status != "scored", row.month, row.model))
    next_month = min((row.month for row in rows if row.status == "upcoming"), default=None)

    summary: dict[str, Any] = {
        "data_complete_through": complete_through,
        "months_scored": int(track["month"].nunique()),
        "next_month": next_month,
    }
    for row in rows:
        if row.month == next_month:
            summary[f"next_month_{row.model}"] = row.predicted
    if not track.empty:
        for metrics in summarize(track).itertuples():
            summary[f"wape_{metrics.model}"] = float(metrics.WAPE)
            summary[f"mae_{metrics.model}"] = int(metrics.MAE)

    sources = [data_dir / METADATA_FILE, city_path]
    sources += sorted({forecast_dir / row.forecast_file for row in rows})
    if (forecast_dir / SUPERSEDED_FILE.name).exists():
        sources.append(forecast_dir / SUPERSEDED_FILE.name)
    backtest_path = data_dir / BACKTEST_FILE
    if backtest_path.exists():
        sources.append(backtest_path)
        for b in pd.read_csv(backtest_path).itertuples():
            model = BACKTEST_MODEL_IDS.get(b.model, b.model)
            summary[f"{b.split}_wape_{model}"] = float(b.WAPE)
            summary[f"{b.split}_period"] = f"{b.period_start} to {b.period_end}"

    warnings = ["Forecasts cover citywide totals only and come with no prediction intervals."]
    if summary["months_scored"] < EARLY_EVIDENCE_MONTHS:
        warnings.append(f"The track record covers only {summary['months_scored']} months, "
                        "so treat it as early evidence.")
    for model, errors in track.groupby("model_name")["error"]:
        if len(errors) > 1 and ((errors < 0).all() or (errors > 0).all()):
            side = "below" if (errors < 0).all() else "above"
            warnings.append(f"The {MODEL_NAMES.get(model, model)} forecast was {side} the actual count "
                            f"in all {len(errors)} scored months.")
    superseded_files = forecasts.loc[forecasts["superseded"], "source_file"].nunique()
    if superseded_files:
        warnings.append(f"{superseded_files} earlier forecast file{'s are' if superseded_files > 1 else ' is'} "
                        f"left out as superseded; {SUPERSEDED_FILE.name} gives the reasons.")
    if not backtest_path.exists():
        warnings.append(f"No {BACKTEST_FILE} was found, so only the live track record is reported.")
    if next_month is None:
        warnings.append(f"No frozen forecast covers the months after {complete_through} yet.")

    provenance = Provenance(
        tool="forecast_status",
        params=params.model_dump(mode="json"),
        unit=UNIT,
        calculation=CALCULATION,
        filters=["geography: citywide (the forecasts are citywide only)",
                 f"scored months: complete months through {complete_through}"],
        sources=[source_file(path) for path in sources],
        data_complete_through=complete_through,
        data_refreshed_at=meta["refreshed_at"],
        computed_at=now_utc(),
    )
    return ToolResult[ForecastRow](summary=summary, rows=rows, warnings=warnings, provenance=provenance)


def _row(r: Any, status: str) -> ForecastRow:
    scored = status == "scored"
    return ForecastRow(
        model=r.model_name,
        month=f"{r.month:%Y-%m}",
        status=status,
        predicted=int(round(r.predicted_count)),
        actual=int(r.actual_count) if scored else None,
        error=int(round(r.error)) if scored else None,
        pct_error=round(float(r.pct_error), 2) if scored else None,
        trained_through=f"{r.training_end_month:%Y-%m}",
        made_on=f"{r.forecast_created_at:%Y-%m-%d}",
        forecast_file=r.source_file,
    )
