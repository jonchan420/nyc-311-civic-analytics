"""The BigQuery sync and parity check, against a stand-in client: no network, no credentials.

The stand-in records every call, so the tests check what would be sent: one load job per
table with WRITE_TRUNCATE, a dry run before every query, and the 1 GB cap on both.
"""

import io
import json

import pandas as pd
import pyarrow.parquet as pq
import pytest
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from src import bigquery_parity as parity
from src.sync_bigquery import LOCATION, Target, live_tables, load_job_config, payload, sync

TARGET = Target("demo-project-123", "nyc_311")


class FakeJob:
    def __init__(self, rows=(), output_rows=0, bytes_processed=0, bytes_billed=0):
        self.rows, self.output_rows = list(rows), output_rows
        self.total_bytes_processed, self.total_bytes_billed = bytes_processed, bytes_billed

    def result(self):
        return [FakeRow(row) for row in self.rows]


class FakeRow:
    def __init__(self, values):
        self._values = tuple(values)

    def values(self):
        return self._values


class FakeClient:
    """Stands in for bigquery.Client; answers queries with whatever `answer` returns."""

    def __init__(self, answer=None, estimate=12_345, existing=()):
        self.answer, self.estimate, self.existing = answer, estimate, set(existing)
        self.datasets, self.loads, self.queries = [], [], []

    def get_dataset(self, ref):
        if ref not in self.existing:
            raise NotFound(f"Dataset {ref} not found")
        return ref

    def create_dataset(self, dataset, exists_ok=False):
        self.datasets.append((dataset.project, dataset.dataset_id, dataset.location, exists_ok))

    def load_table_from_file(self, file_obj, destination, job_config=None, location=None):
        data = file_obj.read()
        self.loads.append((destination, job_config, data, location))
        if job_config.source_format == bigquery.SourceFormat.PARQUET:
            rows = pq.read_table(io.BytesIO(data)).num_rows
        elif job_config.source_format == bigquery.SourceFormat.CSV:
            rows = len(data.decode().splitlines()) - 1
        else:
            rows = len(data.decode().splitlines())
        return FakeJob(output_rows=rows)

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        if job_config.dry_run:
            return FakeJob(bytes_processed=self.estimate)
        return FakeJob(rows=self.answer(sql), bytes_processed=self.estimate, bytes_billed=10_485_760)


def test_sync_replaces_every_table_with_a_load_job(fixture_dir):
    client = FakeClient()
    loaded = sync(client, TARGET, fixture_dir)
    assert client.datasets == [("demo-project-123", "nyc_311", "US", True)]
    tables = {destination.split(".")[-1]: config for destination, config, _, _ in client.loads}
    assert set(tables) == {"monthly_borough_category", "monthly_citywide_counts", "category_catalog",
                           "backtest_metrics", "metadata"}
    assert all(config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE for config in tables.values())
    assert tables["monthly_borough_category"].source_format == bigquery.SourceFormat.PARQUET
    assert tables["monthly_citywide_counts"].source_format == bigquery.SourceFormat.CSV
    assert tables["monthly_citywide_counts"].skip_leading_rows == 1
    assert tables["metadata"].source_format == bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    assert all(destination.startswith("demo-project-123.nyc_311.") and location == LOCATION
               for destination, _, _, location in client.loads)
    # rows as the files hold them: 15 months in the citywide file, one metadata record
    assert loaded["monthly_citywide_counts"] == 15 and loaded["metadata"] == 1
    assert loaded["monthly_borough_category"] == len(pd.read_parquet(fixture_dir / "monthly_borough_category.parquet"))


def test_an_existing_dataset_is_reused_not_created(fixture_dir):
    client = FakeClient(existing={"demo-project-123.nyc_311"})
    sync(client, TARGET, fixture_dir)
    assert client.datasets == [] and len(client.loads) == 5


def test_metadata_becomes_one_json_line(fixture_dir):
    line = payload(fixture_dir / "metadata.json")
    assert line.count(b"\n") == 1 and json.loads(line)["complete_through"] == "2026-02"
    assert load_job_config(fixture_dir / "metadata.json").autodetect


def test_only_table_files_are_loaded(tmp_path):
    (tmp_path / "counts.csv").write_text("a\n1\n")
    (tmp_path / "notes.txt").write_text("not a table")
    assert list(live_tables(tmp_path)) == ["counts"]
    (tmp_path / "bad name.csv").write_text("a\n1\n")
    with pytest.raises(ValueError, match="valid BigQuery table name"):
        live_tables(tmp_path)


@pytest.mark.parametrize("env, message", [
    ({}, "Set GCP_PROJECT_ID"),
    ({"GCP_PROJECT_ID": "Bad_Project"}, "valid Google Cloud project ID"),
    ({"GCP_PROJECT_ID": "demo-project-123", "BQ_DATASET": "nyc-311; DROP"}, "valid BigQuery dataset name"),
])
def test_settings_come_from_the_environment(monkeypatch, env, message):
    for name in ("GCP_PROJECT_ID", "BQ_DATASET"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit, match=message):
        Target.from_env()


def test_dataset_defaults_to_nyc_311(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "demo-project-123")
    monkeypatch.delenv("BQ_DATASET", raising=False)
    assert Target.from_env() == Target("demo-project-123", "nyc_311")


def test_every_query_is_dry_run_first_under_the_cap(fixture_dir):
    sql = (parity.SQL_DIR / "top_complaint_types.sql").read_text()
    client = FakeClient(answer=lambda _: parity.duckdb_rows(sql, fixture_dir))
    rows, used = parity.bigquery_rows(client, TARGET, sql, fixture_dir)
    (dry_sql, dry), (real_sql, real) = client.queries
    assert dry.dry_run and not dry.use_query_cache and not real.dry_run
    assert dry.maximum_bytes_billed == real.maximum_bytes_billed == 1_000_000_000
    assert dry_sql == real_sql and "`demo-project-123.nyc_311.monthly_borough_category`" in real_sql
    assert used == {"estimated": 12_345, "processed": 12_345, "billed": 10_485_760}
    assert rows == parity.duckdb_rows(sql, fixture_dir)


def test_a_query_over_the_cap_never_runs(fixture_dir):
    sql = (parity.SQL_DIR / "top_complaint_types.sql").read_text()
    client = FakeClient(answer=lambda _: [], estimate=2_000_000_000)
    with pytest.raises(SystemExit, match="over the 1,000,000,000-byte cap"):
        parity.bigquery_rows(client, TARGET, sql, fixture_dir)
    assert len(client.queries) == 1 and client.queries[0][1].dry_run


def test_differences_are_reported():
    assert parity.differences([("2025-01-01", 5)], [("2025-01-01", 5)]) == []
    assert parity.differences([("2025-01-01", 5)], [("2025-01-01", 6)]) == [
        "row 0: BigQuery ('2025-01-01', 5) vs DuckDB ('2025-01-01', 6)"]
    assert parity.differences([], [("x", 1)]) == ["0 rows in BigQuery, 1 in DuckDB"]


def test_unknown_tables_in_sql_are_refused():
    with pytest.raises(ValueError, match="isn't a table"):
        parity.render("SELECT * FROM {secrets}", {"monthly_borough_category"}, str)


def test_the_sql_matches_pandas_on_the_fixtures(fixture_dir):
    """The SQL itself, checked against plain pandas: Queens has 1,045 complaints in each 2025
    month, every other borough 150, and the citywide file adds 10 with no borough."""
    table = pd.read_parquet(fixture_dir / "monthly_borough_category.parquet")
    by_borough = parity.duckdb_rows((parity.SQL_DIR / "monthly_borough_totals.sql").read_text(), fixture_dir)
    assert len(by_borough) == 15 * 5 and ("2025-01-01", "QUEENS", 1045) in by_borough
    assert ("2026-02-01", "QUEENS", 925) in by_borough and ("2025-06-01", "BRONX", 150) in by_borough
    top = parity.duckdb_rows((parity.SQL_DIR / "top_complaint_types.sql").read_text(), fixture_dir)
    expected = table.groupby("complaint_type")["complaint_count"].sum().sort_values(ascending=False)
    assert top[0] == (expected.index[0], int(expected.iloc[0]))
    assert [count for _, count in top] == sorted((count for _, count in top), reverse=True)
    citywide = parity.duckdb_rows((parity.SQL_DIR / "citywide_vs_boroughs.sql").read_text(), fixture_dir)
    assert citywide[0] == ("2025-01-01", 1655, 1645, 10) and all(row[3] == 10 for row in citywide)
