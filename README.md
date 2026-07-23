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

## Key design decisions

- **DuckDB for the 14 GB raw CSV.** The raw export is converted once
  to yearly Parquet partitions in a single streaming pass. No pandas
  chunking, no memory pressure, and the raw CSV is never read again.
- **Incremental updates via the Socrata API.** Only records newer than
  the local maximum date are ever downloaded.
- **Baselines before models.** Every forecast model is compared against
  a previous-month baseline and a same-month-last-year seasonal
  baseline. If a baseline wins, the README says so.
- **Frozen forecasts.** Predictions are timestamped and never
  overwritten, so forecast accuracy against incoming 2026 actuals is
  measured without hindsight bias.
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
python src/forecast_2026.py            # 7. freeze a 2026 forecast
streamlit run dashboard/app.py         # 8. dashboard
```

Monthly refresh:

```bash
python src/update_from_api.py          # fetch only new records
python src/build_monthly_counts.py     # rebuild aggregates
```

## Repository structure

```
config/          paths and constants
data/raw/        yearly Parquet partitions (gitignored)
data/processed/  small aggregate tables (gitignored)
data/forecasts/  frozen, timestamped forecast files
src/             pipeline scripts, numbered by run order above
dashboard/       Streamlit app
notebooks/       original class project + exploratory notebooks
report/          EST 389 final report
```

## Data source

NYC Open Data, 311 Service Requests (asset `erm2-nwe9`):
https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9

Borough populations from the 2020 US Census (see
`data/borough_population.csv`; verify figures before citing).

## Limitations

Complaint volume reflects reporting behavior — smartphone access,
language, civic trust — not just underlying conditions. Forecasts in
this project predict recorded 311 complaint volume, not actual
neighborhood problems. See the "Data responsibility" tab in the
dashboard for the full discussion.
