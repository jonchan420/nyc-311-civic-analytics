-- Recorded complaints per month in each borough: the dashboard's borough comparisons.
-- Table names in curly braces are files in data/live/; src/bigquery_parity.py fills in the
-- BigQuery or DuckDB name, so the same SQL runs on both.
SELECT
  CAST(month AS DATE) AS month,
  borough,
  SUM(complaint_count) AS complaints
FROM {monthly_borough_category}
GROUP BY 1, 2
ORDER BY 1, 2
