"""Load the committed data/live/ tables into BigQuery.

    GCP_PROJECT_ID=<project> python src/sync_bigquery.py

Every file in data/live/ becomes a table in the dataset BQ_DATASET (default
nyc_311, location US): CSV and Parquet files as they are, and metadata.json as
a one-row table. Each table is replaced by a load job with WRITE_TRUNCATE.
Load jobs are free, and they are the only way to write in the BigQuery
sandbox, which allows no DML or streaming inserts. Sandbox tables expire
after 60 days, so the monthly refresh reloads them.

Credentials come from Application Default Credentials: `gcloud auth
application-default login` on a laptop, Workload Identity Federation in
GitHub Actions. This script never reads or writes a key file.

Needs requirements-cloud.txt; the Streamlit app doesn't.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import LIVE_DIR

DEFAULT_DATASET = "nyc_311"
LOCATION = "US"
TABLE_FILES = (".csv", ".parquet", ".json")
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,1023}$")
PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")


@dataclass(frozen=True)
class Target:
    """Where the tables go: a project and a dataset, both checked before use."""

    project: str
    dataset: str = DEFAULT_DATASET

    @classmethod
    def from_env(cls) -> Target:
        project = os.environ.get("GCP_PROJECT_ID", "").strip()
        dataset = os.environ.get("BQ_DATASET", "").strip() or DEFAULT_DATASET
        if not project:
            raise SystemExit("Set GCP_PROJECT_ID to the Google Cloud project to use.")
        if not PROJECT.match(project):
            raise SystemExit(f"{project!r} isn't a valid Google Cloud project ID.")
        if not NAME.match(dataset):
            raise SystemExit(f"{dataset!r} isn't a valid BigQuery dataset name.")
        return cls(project, dataset)

    def table(self, name: str) -> str:
        return f"{self.project}.{self.dataset}.{name}"


def make_client(target: Target) -> Any:
    from google.cloud import bigquery

    return bigquery.Client(project=target.project, location=LOCATION)


def live_tables(data_dir: Path = LIVE_DIR) -> dict[str, Path]:
    """Table name -> file, for every table file in data/live/, named by the file's stem."""
    tables = {}
    for path in sorted(Path(data_dir).iterdir()):
        if path.suffix in TABLE_FILES:
            if not NAME.match(path.stem):
                raise ValueError(f"{path.name} doesn't make a valid BigQuery table name.")
            tables[path.stem] = path
    return tables


def load_job_config(path: Path) -> Any:
    from google.cloud import bigquery

    replace = {"write_disposition": bigquery.WriteDisposition.WRITE_TRUNCATE}
    if path.suffix == ".parquet":  # the schema travels with the file
        return bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET, **replace)
    if path.suffix == ".csv":
        return bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1,
                                      autodetect=True, **replace)
    return bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                                  autodetect=True, **replace)


def payload(path: Path) -> bytes:
    if path.suffix == ".json":  # BigQuery's JSON loader wants one record per line
        return (json.dumps(json.loads(path.read_text())) + "\n").encode()
    return path.read_bytes()


def sync(client: Any, target: Target, data_dir: Path = LIVE_DIR) -> dict[str, int]:
    """Create the dataset if needed and replace every table. Returns rows loaded per table."""
    from google.cloud import bigquery

    dataset = bigquery.Dataset(f"{target.project}.{target.dataset}")
    dataset.location = LOCATION
    client.create_dataset(dataset, exists_ok=True)
    loaded = {}
    for name, path in live_tables(data_dir).items():
        job = client.load_table_from_file(io.BytesIO(payload(path)), target.table(name),
                                          job_config=load_job_config(path), location=LOCATION)
        job.result()  # raises if the load failed
        loaded[name] = int(job.output_rows)
        print(f"  {name:28} {loaded[name]:>9,} rows from {path.name}")
    return loaded


def main() -> None:
    target = Target.from_env()
    client = make_client(target)
    print(f"Loading {LIVE_DIR.relative_to(LIVE_DIR.parents[1])}/ into {target.project}.{target.dataset} ({LOCATION})")
    loaded = sync(client, target)
    expiration = client.get_dataset(f"{target.project}.{target.dataset}").default_table_expiration_ms
    print(f"Loaded {sum(loaded.values()):,} rows into {len(loaded)} tables.")
    if expiration:
        print(f"Tables in this dataset expire after {int(expiration) // 86_400_000} days unless reloaded.")


if __name__ == "__main__":
    main()
