"""Score every frozen forecast against actual counts and write the scoreboard.

Writes data/live/forecast_scoreboard.csv (every forecast-month pair with
an actual, flagged with in_track_record) and prints the per-model track
record summary. See src/forecast_registry.py for the scoring rules.

Run:
    python src/score_forecasts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import LIVE_DIR
from src.forecast_registry import load_forecasts, score, summarize
from src.monthly_data import load_monthly_counts

COLUMNS = [
    "month", "model_name", "source_file", "training_end_month", "forecast_created_at",
    "months_ahead", "predicted_count", "actual_count", "error", "pct_error",
    "in_track_record", "superseded",
]


def main() -> None:
    forecasts = load_forecasts()
    if forecasts.empty:
        print("No frozen forecasts to score.")
        return
    scored = score(forecasts, load_monthly_counts(verbose=False))
    if scored.empty:
        print("No forecast months have actual counts yet.")
        return

    out = scored[COLUMNS].copy()
    out["pct_error"] = out["pct_error"].round(2)
    out["month"] = out["month"].dt.strftime("%Y-%m")
    out["training_end_month"] = out["training_end_month"].dt.strftime("%Y-%m")
    out["forecast_created_at"] = out["forecast_created_at"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out.to_csv(LIVE_DIR / "forecast_scoreboard.csv", index=False)

    track = scored[scored["in_track_record"]]
    print("Track record (best forecast available before each month):")
    print(track[["month", "model_name", "months_ahead", "predicted_count",
                 "actual_count", "pct_error"]].round({"pct_error": 1}).to_string(index=False))
    print()
    print(summarize(track).to_string(index=False))


if __name__ == "__main__":
    main()
