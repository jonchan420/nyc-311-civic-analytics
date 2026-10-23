"""Shared data access for the analysis tools.

One place opens the committed tables, refuses months that are partial or
missing, maps raw labels to categories, and records which file versions
an answer came from. Every query binds its values as parameters.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from analyst.catalog import BOROUGHS, CategoryCatalog
from analyst.schemas import MonthRange, Provenance, SourceFile
from config.settings import PROJECT_ROOT

CATEGORY_FILE = "monthly_borough_category.parquet"
CITYWIDE_FILE = "monthly_citywide_counts.csv"
METADATA_FILE = "metadata.json"
REQUIRED_COLUMNS = {"month", "borough", "complaint_type", "complaint_count"}
UNIT = "recorded 311 service requests (a measure of reporting, not of underlying conditions)"


class ToolError(Exception):
    """The tool can't answer as asked. Messages are written to be shown to the user."""


class DataUnavailableError(ToolError):
    """Data files are missing or malformed, or don't cover the requested months."""


class UnknownCategoryError(ToolError):
    """A requested complaint type matches no label in the data."""


def first_day(year_month: str) -> date:
    year, month = map(int, year_month.split("-"))
    return date(year, month, 1)


def shift_month(year_month: str, months: int) -> str:
    year, month = map(int, year_month.split("-"))
    total = year * 12 + month - 1 + months
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def month_name(year_month: str) -> str:
    return first_day(year_month).strftime("%b %Y")


def pct(part: float, whole: float | None) -> float | None:
    return round(100 * part / whole, 2) if whole else None


def boroughs_for(geography: str) -> list[str]:
    return list(BOROUGHS) if geography == "CITYWIDE" else [geography]


def place_name(geography: str) -> str:
    return "all five boroughs" if geography == "CITYWIDE" else geography.title()


def resolve_categories(requested: list[str], catalog: CategoryCatalog) -> dict[str, str]:
    """category_id -> name for each requested category, in the order asked."""
    found: dict[str, str] = {}
    for text in requested:
        entry = catalog.find(text)
        if entry is None:
            close = catalog.suggest(text)
            hint = f" Closest matches: {', '.join(close)}." if close else ""
            raise UnknownCategoryError(f'No complaint type called "{text}" is in the data.{hint}')
        found.setdefault(entry.category_id, entry.category)
    return found


@dataclass
class Group:
    """One category's counts, combined across the raw labels that belong to it."""

    name: str
    counts: dict[str, int]
    labels: list[str] = field(default_factory=list)


def group_by_category(
    by_label: dict[str, dict[str, int]],
    catalog: CategoryCatalog,
    windows: list[str],
    wanted: dict[str, str] | None = None,
) -> dict[str, Group]:
    groups: dict[str, Group] = {}
    for label, counts in by_label.items():
        entry = catalog.lookup(label)
        if wanted is not None and entry.category_id not in wanted:
            continue
        group = groups.setdefault(entry.category_id, Group(entry.category, dict.fromkeys(windows, 0)))
        for window in windows:
            group.counts[window] += counts.get(window, 0)
        group.labels.append(label)
    for category_id, name in (wanted or {}).items():  # a requested category with no complaints still counts
        groups.setdefault(category_id, Group(name, dict.fromkeys(windows, 0)))
    return groups


class DataSource:
    """The committed tables in one data directory, opened for a single tool call."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.meta = read_metadata(self.data_dir)
        self.complete_through: str = self.meta["complete_through"]
        self._files = [self.data_dir / CATEGORY_FILE, self.data_dir / METADATA_FILE]

        category_path = self.data_dir / CATEGORY_FILE
        if not category_path.exists():
            raise DataUnavailableError(f"The data file {category_path} is missing.")
        self.con = duckdb.connect()
        try:
            self.con.read_parquet(str(category_path)).create_view("borough_category")
            columns = {row[0] for row in self.con.execute("DESCRIBE borough_category").fetchall()}
            if missing := REQUIRED_COLUMNS - columns:
                raise DataUnavailableError(f"{CATEGORY_FILE} is missing the columns {sorted(missing)}.")
            first, last = self.con.execute(
                "SELECT strftime(min(month), '%Y-%m'), strftime(max(month), '%Y-%m') FROM borough_category"
            ).fetchone()
            if first is None:
                raise DataUnavailableError(f"{CATEGORY_FILE} has no rows.")
            self.first_month: str = first
            self.usable_through: str = min(last, self.complete_through)
            label_totals = dict(self.con.execute(
                "SELECT complaint_type, SUM(complaint_count) FROM borough_category GROUP BY complaint_type"
            ).fetchall())
        except Exception:
            self.con.close()
            raise
        self.catalog = CategoryCatalog.load(self.data_dir).including(label_totals)
        if self.catalog.source:
            self._files.append(self.catalog.source)

    def __enter__(self) -> DataSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.con.close()

    def require_months(self, periods: dict[str, MonthRange], boroughs: list[str]) -> None:
        """Refuse months outside the complete data, and whole months with no rows at all.
        periods maps the name used in error messages ("period_a", "month") to its months."""
        for name, period in periods.items():
            if period.start < self.first_month:
                raise DataUnavailableError(
                    f"{name} starts {period.start}, but the data begins {self.first_month}.")
            if period.end > self.usable_through:
                raise DataUnavailableError(
                    f"{name} ends {period.end}, but the data is complete only through "
                    f"{self.usable_through}; later months are missing or still partial.")
        present = set(self.con.execute(
            "SELECT DISTINCT borough, strftime(month, '%Y-%m') FROM borough_category "
            "WHERE list_contains($boroughs, borough)", {"boroughs": boroughs}).fetchall())
        months = sorted({month for period in periods.values() for month in period.each_month()})
        gaps = [f"{borough} {month}" for borough in boroughs for month in months
                if (borough, month) not in present]
        if gaps:
            more = " and others" if len(gaps) > 5 else ""
            raise DataUnavailableError(
                f"There are no rows at all for {', '.join(gaps[:5])}{more}. That is a gap in the "
                "data, not zero complaints, so no answer is given.")

    def counts_by_label(self, boroughs: list[str], windows: dict[str, MonthRange]) -> dict[str, dict[str, int]]:
        """{label: {window: count}} for every label with complaints in any window."""
        for name in windows:
            if not re.fullmatch(r"[a-z_]+", name):  # window names come from code, never from users
                raise ValueError(f"bad window name {name!r}")
        columns = ", ".join(
            f"COALESCE(SUM(complaint_count) FILTER (WHERE month BETWEEN ${w}_start AND ${w}_end), 0) AS {w}"
            for w in windows)
        any_window = " OR ".join(f"month BETWEEN ${w}_start AND ${w}_end" for w in windows)
        params: dict[str, Any] = {"boroughs": boroughs}
        for w, period in windows.items():
            params[f"{w}_start"], params[f"{w}_end"] = first_day(period.start), first_day(period.end)
        rows = self.con.execute(
            f"SELECT complaint_type, {columns} FROM borough_category "
            f"WHERE list_contains($boroughs, borough) AND ({any_window}) GROUP BY complaint_type",
            params).fetchall()
        return {row[0]: {w: int(value) for w, value in zip(windows, row[1:])} for row in rows}

    def monthly_by_label(self, boroughs: list[str], start: str, end: str) -> list[tuple[str, str, int]]:
        """(label, YYYY-MM, count) for every label-month with complaints."""
        rows = self.con.execute(
            "SELECT complaint_type, strftime(month, '%Y-%m'), SUM(complaint_count) FROM borough_category "
            "WHERE list_contains($boroughs, borough) AND month BETWEEN $start AND $end GROUP BY 1, 2",
            {"boroughs": boroughs, "start": first_day(start), "end": first_day(end)}).fetchall()
        return [(label, month, int(count)) for label, month, count in rows]

    def citywide_by_month(self, period: MonthRange) -> dict[str, int]:
        """Citywide totals, including complaints with no borough recorded."""
        path = self.data_dir / CITYWIDE_FILE
        if not path.exists():
            raise DataUnavailableError(f"The data file {path} is missing.")
        if path not in self._files:
            self.con.read_csv(str(path)).create_view("citywide")
            self._files.append(path)
        rows = self.con.execute(
            "SELECT strftime(CAST(month AS DATE), '%Y-%m'), complaint_count FROM citywide "
            "WHERE CAST(month AS DATE) BETWEEN $start AND $end",
            {"start": first_day(period.start), "end": first_day(period.end)}).fetchall()
        totals = {month: int(count) for month, count in rows}
        missing = [month for month in period.each_month() if month not in totals]
        if missing:
            raise DataUnavailableError(f"{CITYWIDE_FILE} has no total for {', '.join(missing)}.")
        return totals

    def add_file(self, path: Path) -> None:
        if path not in self._files:
            self._files.append(path)

    def provenance(self, tool: str, params: dict[str, Any], calculation: str, filters: list[str]) -> Provenance:
        return Provenance(
            tool=tool,
            params=params,
            unit=UNIT,
            calculation=calculation,
            filters=filters,
            sources=[source_file(path) for path in self._files],
            data_complete_through=self.complete_through,
            data_refreshed_at=self.meta["refreshed_at"],
            computed_at=now_utc(),
        )


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / METADATA_FILE
    if not path.exists():
        raise DataUnavailableError(f"There is no freshness record ({path}), so no answer is given.")
    meta = json.loads(path.read_text())
    for name in ("complete_through", "refreshed_at"):
        if name not in meta:
            raise DataUnavailableError(f"{METADATA_FILE} has no {name!r}, so no answer is given.")
    return meta


def source_file(path: Path) -> SourceFile:
    try:
        shown = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        shown = str(path)
    return SourceFile(path=shown, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
