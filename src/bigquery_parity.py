"""Run the SQL in sql/ in BigQuery and in DuckDB on the same files, and compare.

    GCP_PROJECT_ID=<project> python src/bigquery_parity.py

Each file in sql/ is one aggregate the dashboard uses, written once in SQL
that both engines accept; table names in curly braces are filled in for each
engine. BigQuery reads the tables src/sync_bigquery.py loaded, DuckDB reads
data/live/ directly, and every row must match.

Every BigQuery query is dry-run first and runs only if the estimate is under
MAX_BYTES_BILLED (1 GB). The real query carries the same cap, so BigQuery
refuses it rather than bill more, even in a project with billing turned on.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any, Callable

import duckdb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import LIVE_DIR, PROJECT_ROOT
from src.sync_bigquery import Target, live_tables, make_client

SQL_DIR = PROJECT_ROOT / "sql"
MAX_BYTES_BILLED = 1_000_000_000
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render(sql: str, tables: set[str], name: Callable[[str], str]) -> str:
    """Fill in each {table} with the engine's name for it; an unknown table is an error."""
    def fill(match: re.Match) -> str:
        if match[1] not in tables:
            raise ValueError(f"{{{match[1]}}} isn't a table in data/live/.")
        return name(match[1])
    return PLACEHOLDER.sub(fill, sql)


def normalize(rows: list[tuple]) -> list[tuple]:
    """Plain Python values, so rows from either engine compare equal."""
    def value(v: Any) -> Any:
        if isinstance(v, (dt.datetime, dt.date)):
            return v.isoformat()[:10]
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v
    return [tuple(value(v) for v in row) for row in rows]


def duckdb_rows(sql: str, data_dir: Path = LIVE_DIR) -> list[tuple]:
    tables = live_tables(data_dir)
    con = duckdb.connect()
    for name, path in tables.items():
        if path.suffix == ".parquet":
            con.read_parquet(str(path)).create_view(name)
        elif path.suffix == ".csv":
            con.read_csv(str(path)).create_view(name)
    return normalize(con.execute(render(sql, set(tables), lambda name: name)).fetchall())


def bigquery_rows(client: Any, target: Target, sql: str, data_dir: Path = LIVE_DIR) -> tuple[list[tuple], dict]:
    """Rows from BigQuery, after a dry run under the byte cap. Returns (rows, bytes reported)."""
    from google.cloud import bigquery

    query = render(sql, set(live_tables(data_dir)), lambda name: f"`{target.table(name)}`")
    dry = client.query(query, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False, maximum_bytes_billed=MAX_BYTES_BILLED))
    estimate = int(dry.total_bytes_processed or 0)
    if estimate > MAX_BYTES_BILLED:
        raise SystemExit(f"The dry run estimates {estimate:,} bytes, over the {MAX_BYTES_BILLED:,}-byte cap; "
                         "nothing was run.")
    job = client.query(query, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES_BILLED))
    rows = normalize([tuple(row.values()) for row in job.result()])
    return rows, {"estimated": estimate, "processed": int(job.total_bytes_processed or 0),
                  "billed": int(job.total_bytes_billed or 0)}


def differences(remote: list[tuple], local: list[tuple], limit: int = 3) -> list[str]:
    out = []
    if len(remote) != len(local):
        out.append(f"{len(remote)} rows in BigQuery, {len(local)} in DuckDB")
    out += [f"row {i}: BigQuery {r} vs DuckDB {l}" for i, (r, l) in enumerate(zip(remote, local)) if r != l]
    return out[:limit]


def sql_files() -> list[Path]:
    return sorted(SQL_DIR.glob("*.sql"))


def main() -> None:
    target = Target.from_env()
    client = make_client(target)
    failed = 0
    for path in sql_files():
        sql = path.read_text()
        remote, used = bigquery_rows(client, target, sql)
        local = duckdb_rows(sql)
        problems = differences(remote, local)
        failed += bool(problems)
        print(f"{path.name:28} {len(remote):>5} rows  dry run {used['estimated']:>9,} bytes  processed "
              f"{used['processed']:>9,}  billed {used['billed']:>10,}  {'MATCH' if not problems else 'DIFFERENT'}")
        for problem in problems:
            print(f"    {problem}")
    print(f"{len(sql_files()) - failed} of {len(sql_files())} queries match DuckDB exactly.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
