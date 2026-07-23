# NYC 311 Civic Analytics and Complaint Forecasting

End-to-end civic analytics project analyzing NYC 311 service requests
(2020–present): exploratory analysis, population-adjusted borough
comparisons, and monthly complaint-volume forecasting with honest
baseline comparisons, presented through a Streamlit dashboard.

This project began as an EST 389 (Intro to Responsible AI and Data
Science) exploratory analysis of one year of 311 data. It was later
expanded into a full pipeline with API-based incremental updates,
time-series forecasting, and frozen-forecast evaluation against
incoming 2026 data.

Every number below traces to a file in this repo — either
`data/processed/model_comparison*.csv`, a frozen forecast in
`data/forecasts/`, or one of the CSVs in
`data/processed/anomaly_findings/`, produced by
`src/anomaly_analysis.py`. None of it was hand-typed from a terminal
scrollback.

## Key design decisions

- **DuckDB for the 14 GB raw CSV.** The raw export is converted once
  to yearly Parquet partitions in a single streaming pass. No pandas
  chunking, no memory pressure, and the raw CSV is never read again.
- **Incremental updates via the Socrata API.** Only records newer than
  the local maximum date are ever downloaded.
- **Baselines before models — and the result is honestly mixed.**
  On the 2025 validation split, the seasonal-naive baseline
  (same month, one year prior) beats both trained models on every
  metric (see Model performance below). But on the six complete 2026
  test months — which cover the structural break described below —
  both trained models beat the seasonal-naive baseline by a wide
  margin, because the baseline has no way to see a regime change and
  the trained models can partially adapt via recent lags. Neither
  fact is hidden in favor of the other.
- **Frozen forecasts.** Predictions are timestamped and never
  overwritten, so forecast accuracy against incoming 2026 actuals is
  measured without hindsight bias. Two bugs were found and fixed in
  this frozen-forecast machinery — see "Frozen forecasts" below.
- **Record-level findings are scripted, not typed.** Every anomaly
  number (complaint-type deltas, channel shifts, ZIP concentration)
  comes from `src/anomaly_analysis.py`, which reads the raw Parquet
  partitions directly and writes its results to
  `data/processed/anomaly_findings/`. Anyone cloning this repo can
  regenerate every figure in the anomaly section below.
- **Reporting bias is documented, not ignored.** 311 data measures who
  reports, not where conditions are worst.

## Setup (macOS)

```bash
git clone <your-repo-url>
cd nyc-311-civic-analytics

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `config/settings.py` and set `RAW_CSV_PATH` to wherever your
downloaded 311 CSV lives (keep it outside the repo).

## Run order

```bash
python src/inspect_csv.py              # 1. look at the columns first
python src/convert_csv_to_parquet.py   # 2. one-pass CSV -> Parquet (do once)
python src/build_monthly_counts.py     # 3. small aggregate tables
python src/per_capita_analysis.py      # 4. population-adjusted comparison
python src/build_forecast_dataset.py   # 5. lag/rolling feature table
python src/train_evaluate.py           # 6. baselines vs models
python src/forecast_2026.py            # 7. freeze a 2026 GB forecast
python src/forecast_seasonal_naive.py  # 8. freeze a 2026 seasonal-naive forecast
python src/anomaly_analysis.py         # 9. record-level findings behind the 2026 break
streamlit run dashboard/app.py         # 10. dashboard
```

Monthly refresh:

```bash
python src/update_from_api.py          # fetch only new records
python src/build_monthly_counts.py     # rebuild aggregates
```

## Repository structure

```
config/                     paths and constants
data/raw/                   yearly Parquet partitions (gitignored)
data/processed/             small aggregate tables (gitignored)
data/processed/anomaly_findings/  record-level findings, output of anomaly_analysis.py (gitignored)
data/forecasts/             frozen, timestamped forecast files (tracked in git)
data/borough_population.csv 2020 Census borough populations (tracked in git)
src/                        pipeline scripts, numbered by run order above
dashboard/                  Streamlit app
notebooks/                  currently empty — original EST 389 notebook not yet
                             recovered from prior machine
report/                     currently empty — EST 389 final report not yet
                             recovered from prior machine
```

## Data source

NYC Open Data, 311 Service Requests (asset `erm2-nwe9`):
https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9

Borough populations from the 2020 US Census, in
`data/borough_population.csv`:

| Borough | Population |
|---|---|
| Brooklyn | 2,736,074 |
| Queens | 2,405,464 |
| Manhattan | 1,694,251 |
| Bronx | 1,472,654 |
| Staten Island | 495,747 |

## Model performance

Source: `data/processed/model_comparison.csv` (validation, 2025) and
`data/processed/model_comparison_test_2026.csv` (test, six complete
months of 2026: Jan–Jun).

**Validation (2025) — a baseline wins:**

| model | MAE | RMSE | MAPE | WAPE |
|---|---|---|---|---|
| baseline_seasonal_lag12 | 17,203 | 22,959 | 5.44% | 5.65% |
| gradient_boosting | 19,704 | 25,331 | 6.25% | 6.47% |
| linear_regression | 22,294 | 27,713 | 7.19% | 7.32% |
| baseline_previous_month | 25,911 | 34,513 | 8.82% | 8.51% |

**Test (2026 Jan–Jun) — the trained models win instead:**

| model | MAE | RMSE | MAPE | WAPE |
|---|---|---|---|---|
| linear_regression | 24,948 | 26,891 | 7.43% | 7.50% |
| gradient_boosting | 25,630 | 31,660 | 7.55% | 7.71% |
| baseline_seasonal_lag12 | 39,295 | 46,693 | 11.84% | 11.82% |

The reversal is the finding: 2026 is not a normal year (see the
structural break below), and a baseline that can only repeat last
year's number has no mechanism to track a regime change. The trained
models, which see recent lags and rolling means, absorb some of it.

## Frozen forecasts

`data/forecasts/` holds every forecast ever generated, never
overwritten. Two generation bugs were found and fixed here rather
than hidden:

- **Gradient boosting v1/v2 → v3.** `forecast_2026_gradient_boosting_20260722_1857.csv`
  and `..._1916.csv` are byte-identical in their predictions — only
  the timestamp changed between them. Both claimed
  `training_end_month = 2026-07`, but `forecast_2026.py` read
  `monthly_citywide_counts.parquet` directly, bypassing the
  incomplete-month guard that `build_forecast_dataset.py` applies to
  training data. July 2026 had 226,542 rows at generation time — about
  70% of a normal month — so both files' `lag_1` was anchored on a
  partial month. `forecast_2026.py` now applies the same guard.
  `..._2058.csv` is the corrected version: `training_end_month =
  2026-06`, forecasting July onward instead of August onward. v1 and
  v2 are kept, not deleted, per the frozen-forecast policy — they are
  superseded for anchor contamination, not erased.
- **Seasonal-naive baseline was lag-24, not lag-12.**
  `forecast_2026_seasonal_naive_20260722_baseline.csv` has no
  generating script in `src/` and its six predictions turn out to
  match **2024**'s monthly values exactly (verified against
  `monthly_citywide_counts.parquet`) — two years back, not "same
  month last year" as its name and `model_comparison.csv`'s
  `baseline_seasonal_lag12` methodology both imply.
  `src/forecast_seasonal_naive.py` now generates the correct lag-12
  version: `forecast_2026_seasonal_naive_20260722_2106.csv`. The
  original file is kept, not deleted, but should not be used as the
  seasonal-naive baseline going forward.

**Current, correctly-anchored forecasts for July 2026** (both
`training_end_month = 2026-06`):

| model | file | July 2026 prediction |
|---|---|---|
| gradient_boosting | `forecast_2026_gradient_boosting_20260722_2058.csv` | 321,586 |
| seasonal_naive (lag-12) | `forecast_2026_seasonal_naive_20260722_2106.csv` | 315,877 |

The gap is 5,709 (1.8%) — not the ~23,000 gap it would appear to be
against the mislabeled lag-24 file. Given gradient boosting's test-set
win during the current structural break, it should have an edge, but
1.8% is a thin margin the first scoring checkpoint (roughly ten days
after this forecast was frozen) will actually test.

## The 2026 structural break

Every number in this section comes from
`data/processed/anomaly_findings/`, generated by
`src/anomaly_analysis.py`, which reads the raw record-level Parquet
partitions.

**Signed error against the seasonal-naive baseline, Jan–Jun 2026**
(source: `baseline_errors_2026.csv`; baseline = same month, 2025):

| month | actual | baseline | signed error | pct error |
|---|---|---|---|---|
| 2026-01 | 348,511 | 348,180 | +331 | +0.1% |
| 2026-02 | 334,691 | 255,364 | +79,327 | +31.1% |
| 2026-03 | 342,388 | 281,223 | +61,165 | +21.7% |
| 2026-04 | 302,189 | 272,550 | +29,639 | +10.9% |
| 2026-05 | 331,976 | 295,057 | +36,919 | +12.5% |
| 2026-06 | 334,832 | 306,442 | +28,390 | +9.3% |

January tracks the baseline almost exactly; every month from February
on runs well above it. Two events explain most of the gap, by
complaint type (source: `complaint_type_deltas.csv`):

- **February — a blizzard.** `Snow or Ice` complaints went from 2,817
  (Feb 2025) to 30,943 (Feb 2026), +28,126 — nearly 11x, and the
  largest single complaint-type delta of the three months examined.
  `HEAT/HOT WATER` also rose +18,332, consistent with a cold-weather
  event.
- **March — street-condition spillover.** `Street Condition`
  complaints went from 7,035 to 28,690, +21,655 — roughly quadrupled.
  The timing (immediately following the February snow event) is
  consistent with plow/freeze-thaw road damage, though this script
  does not establish causation, only the count.

June shows a different pattern — smaller citywide overshoot (+9.3%)
but a shift in *how* and *where* complaints arrive (source:
`june_channel_breakdown.csv`, `june_borough_breakdown.csv`,
`june_top10_zip.csv`):

- **Reporting channel shifted toward online.** ONLINE share rose from
  42.3% to 46.2% of June complaints; PHONE fell from 27.8% to 23.8%.
- **Brooklyn and Queens drove the citywide increase.** Of June's
  +28,390 citywide complaints (2025 → 2026), Brooklyn contributed
  +14,231 and Queens +11,897 — 92.0% of the citywide increase combined.
  Every other borough also rose, just by far smaller amounts (Staten
  Island +1,015, Manhattan +660, Bronx +454); none decreased.
- **Illegal Parking was June's largest single complaint-type
  increase** (+8,657, 2025 → 2026), and Brooklyn+Queens Illegal
  Parking has been on a multi-year upward trend, not a one-month spike
  (source: `illegal_parking_bk_qn_timeline.csv`): June counts for
  these two boroughs went from 31,195 (2024) to 34,011 (2025, +9.0%)
  to 40,066 (2026, +17.8%) — an accelerating trend, not a reversal.

## Data responsibility

_[To be written by the project author — this is the intellectual
core of the EST 389 analysis and should reflect their own reasoning
about what 311 volume does and doesn't measure, not a generated
summary.]_

## Motivation

_[To be written by the project author — the personal observations
(Chinatown, Brooklyn infrastructure) that motivated this project in
the first place aren't something a script or model can produce.]_

## Limitations

Complaint volume reflects reporting behavior — smartphone access,
language, civic trust — not just underlying conditions. Forecasts in
this project predict recorded 311 complaint volume, not actual
neighborhood problems. See the "Data responsibility" tab in the
dashboard for the full discussion.

The February/March 2026 findings above establish coincident timing
between a snow event and elevated Snow-or-Ice/Street-Condition
complaint volume, and the June findings establish a channel and
borough shift — none of this analysis establishes causation.
