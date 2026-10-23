"""Allowlisted analysis tools over the committed data/live/ tables.

Each tool validates its parameters (schemas.py), runs DuckDB SQL whose
values are always bound parameters, and returns a ToolResult whose
provenance records what was computed, from which file version, with
which filters. A model can only choose a tool and fill its typed
parameters; it never writes SQL.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from analyst.catalog import BOROUGHS, CATALOG_FILE, CategoryCatalog
from analyst.schemas import (
    CategoryChange,
    ComparePeriodsParams,
    MonthRange,
    Provenance,
    SourceFile,
    ToolResult,
)
from config.settings import LIVE_DIR, PROJECT_ROOT

CATEGORY_FILE = "monthly_borough_category.parquet"
CITYWIDE_FILE = "monthly_citywide_counts.csv"
METADATA_FILE = "metadata.json"
REQUIRED_COLUMNS = {"month", "borough", "complaint_type", "complaint_count"}

SMALL_COUNT = 20           # under this in both periods, a change is mostly noise
MIN_SHARE_BASE_PCT = 1.0   # shares of change are skipped when |total change| < 1% of period A
UNIT = "recorded 311 service requests (a measure of reporting, not of underlying conditions)"
CALCULATION = (
    "For each complaint type, count_a and count_b are the sums of its monthly counts in "
    "each period (a month with no row counts as 0); change = count_b - count_a; "
    "pct_change = 100 * change / count_a; share_of_change = 100 * change / total change. "
    "Labels that differ only in case or punctuation, and renamed labels, are combined "
    "under one current name."
)


class ToolError(Exception):
    """The tool can't answer as asked. Messages are written to be shown to the user."""


class DataUnavailableError(ToolError):
    """Data files are missing or malformed, or don't cover the requested months."""


class UnknownCategoryError(ToolError):
    """A requested complaint type matches no label in the data."""


# $names are bound parameters: DuckDB receives the values separately from the
# SQL text, so nothing a user or model types can change what the query does.
COUNTS_SQL = """
SELECT complaint_type,
       COALESCE(SUM(complaint_count) FILTER (WHERE month BETWEEN $a_start AND $a_end), 0) AS count_a,
       COALESCE(SUM(complaint_count) FILTER (WHERE month BETWEEN $b_start AND $b_end), 0) AS count_b
FROM borough_category
WHERE list_contains($boroughs, borough)
  AND (month BETWEEN $a_start AND $a_end OR month BETWEEN $b_start AND $b_end)
GROUP BY complaint_type
"""

MONTHS_PRESENT_SQL = """
SELECT DISTINCT borough, strftime(month, '%Y-%m')
FROM borough_category
WHERE list_contains($boroughs, borough)
  AND (month BETWEEN $a_start AND $a_end OR month BETWEEN $b_start AND $b_end)
"""

LABELS_SQL = "SELECT DISTINCT complaint_type FROM borough_category"

CITYWIDE_SQL = """
SELECT strftime(CAST(month AS DATE), '%Y-%m'), complaint_count
FROM citywide
WHERE CAST(month AS DATE) BETWEEN $start AND $end
"""


def compare_periods(
    params: ComparePeriodsParams | dict[str, Any], data_dir: Path = LIVE_DIR
) -> ToolResult[CategoryChange]:
    """Which complaint types changed most between two equal-length periods?"""
    params = ComparePeriodsParams.model_validate(params)
    data_dir = Path(data_dir)
    citywide = params.geography == "CITYWIDE"
    meta = _read_metadata(data_dir)
    window = {
        "a_start": _first_day(params.period_a.start),
        "a_end": _first_day(params.period_a.end),
        "b_start": _first_day(params.period_b.start),
        "b_end": _first_day(params.period_b.end),
        "boroughs": list(BOROUGHS) if citywide else [params.geography],
    }

    con = _connect(data_dir, citywide)
    try:
        _check_coverage(con, params, window, meta["complete_through"])
        all_labels = [label for (label,) in con.execute(LABELS_SQL).fetchall()]
        counts = con.execute(COUNTS_SQL, window).fetchall()
        city = (_citywide_total(con, params.period_a), _citywide_total(con, params.period_b)) if citywide else None
    finally:
        con.close()
    catalog = CategoryCatalog.load(data_dir).including(all_labels)
    wanted = _resolve_categories(params.categories, catalog) if params.categories else None

    # Combine raw labels into categories (spelling variants, renames).
    groups: dict[str, dict[str, Any]] = {}
    for label, count_a, count_b in counts:
        entry = catalog.lookup(label)
        if wanted is not None and entry.category_id not in wanted:
            continue
        group = groups.setdefault(entry.category_id, {"name": entry.category, "a": 0, "b": 0, "labels": []})
        group["a"] += int(count_a)
        group["b"] += int(count_b)
        group["labels"].append(label)
    for category_id, name in (wanted or {}).items():  # a requested category with no complaints still gets a row
        groups.setdefault(category_id, {"name": name, "a": 0, "b": 0, "labels": []})

    total_a = sum(g["a"] for g in groups.values())
    total_b = sum(g["b"] for g in groups.values())
    total_change = total_b - total_a
    shareable = total_change != 0 and abs(total_change) >= MIN_SHARE_BASE_PCT / 100 * total_a

    rows = [
        _category_row(g["name"], g["a"], g["b"], g["labels"], total_change if shareable else None, catalog)
        for g in groups.values()
    ]
    rows.sort(key=lambda row: (-abs(row.change), row.category))
    shown = rows[: params.top_n]
    shown_change = sum(row.change for row in shown)

    summary: dict[str, int | float | None] = {
        "months_per_period": params.period_a.months,
        "total_a": total_a,
        "total_b": total_b,
        "change": total_change,
        "pct_change": _pct(total_change, total_a),
        "categories_compared": len(rows),
        "shown_change": shown_change,
        "other_change": total_change - shown_change,
    }
    if city:
        summary["citywide_total_a"], summary["citywide_total_b"] = city

    sources = [data_dir / CATEGORY_FILE, data_dir / METADATA_FILE]
    if catalog.source:
        sources.append(catalog.source)
    if citywide:
        sources.append(data_dir / CITYWIDE_FILE)
    provenance = Provenance(
        tool="compare_periods",
        params=params.model_dump(mode="json"),
        unit=UNIT,
        calculation=CALCULATION,
        filters=_filters(params, list(wanted.values()) if wanted else None),
        sources=[_source(path) for path in sources],
        data_complete_through=meta["complete_through"],
        data_refreshed_at=meta["refreshed_at"],
        computed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    warnings = _warnings(rows, shown, total_a, total_b, total_change, shareable, city, catalog)
    return ToolResult[CategoryChange](summary=summary, rows=shown, warnings=warnings, provenance=provenance)


def _category_row(
    name: str, a: int, b: int, labels: list[str], total_change: int | None, catalog: CategoryCatalog
) -> CategoryChange:
    change = b - a
    kinds = {catalog.lookup(label).kind for label in labels}
    flags = []
    if a == 0 and b > 0:
        flags.append("new_in_b")
    if a > 0 and b == 0:
        flags.append("absent_in_b")
    if a < SMALL_COUNT and b < SMALL_COUNT:
        flags.append("small_count")
    if "renamed" in kinds:
        flags.append("renamed")
    if "unreviewed" in kinds:
        flags.append("unreviewed")
    return CategoryChange(
        category=name,
        count_a=a,
        count_b=b,
        change=change,
        pct_change=_pct(change, a),
        share_of_change=_pct(change, total_change),
        flags=flags,
        labels=sorted(labels),
    )


def _warnings(
    rows: list[CategoryChange],
    shown: list[CategoryChange],
    total_a: int,
    total_b: int,
    total_change: int,
    shareable: bool,
    city: tuple[int, int] | None,
    catalog: CategoryCatalog,
) -> list[str]:
    warnings = []
    if catalog.source is None:
        warnings.append(
            f"No {CATALOG_FILE} was found, so renamed categories aren't combined; "
            "only spellings that differ in case or punctuation are."
        )
    for row in rows:
        for label in row.labels:
            entry = catalog.lookup(label)
            if entry.kind == "renamed":
                when = f" in {_month_name(entry.renamed_in)}" if entry.renamed_in else ""
                warnings.append(
                    f'"{label}" was renamed "{entry.category}"{when}; both labels are counted together.'
                )
    unreviewed = [row.category for row in rows if "unreviewed" in row.flags]
    if unreviewed:
        warnings.append(
            f"{', '.join(unreviewed)}: not yet in the category catalog, so a rename of an older "
            "category would not be recognized."
        )

    def named(flag: str) -> str:
        return ", ".join(row.category for row in shown if flag in row.flags)

    if named("new_in_b"):
        warnings.append(
            f"No complaints in period A for {named('new_in_b')}: a new category or an "
            "unrecognized rename, so the increase may not be real growth."
        )
    if named("absent_in_b"):
        warnings.append(
            f"No complaints in period B for {named('absent_in_b')}: a retired category or an "
            "unrecognized rename, so the decrease may not be a real decline."
        )
    if named("small_count"):
        warnings.append(
            f"Fewer than {SMALL_COUNT} complaints in both periods for {named('small_count')}; "
            "too few to read much into."
        )
    if total_change == 0:
        warnings.append("The total didn't change, so shares of the change aren't computed.")
    elif not shareable:
        warnings.append(
            f"The total changed by only {total_change:+,} ({_pct(total_change, total_a)}% of "
            "period A), so shares of the change aren't computed; they would swing wildly."
        )
    elif any(row.share_of_change is not None and not 0 <= row.share_of_change <= 100 for row in shown):
        warnings.append(
            "Some categories moved against the overall change, so shares above 100% or below "
            "0% are expected; across all categories the shares sum to 100%."
        )
    if city:
        warnings.append(
            f"Category counts cover the five boroughs: {total_a:,} of {city[0]:,} complaints in "
            f"period A and {total_b:,} of {city[1]:,} in period B. The rest have no borough recorded."
        )
    return warnings


def _filters(params: ComparePeriodsParams, categories: list[str] | None) -> list[str]:
    place = "all five boroughs" if params.geography == "CITYWIDE" else params.geography.title()
    filters = [
        f"geography: {place}; complaints with no borough recorded are excluded",
        f"period A: {params.period_a.label()} ({params.period_a.months} months)",
        f"period B: {params.period_b.label()} ({params.period_b.months} months)",
    ]
    if categories:
        filters.append("complaint types: " + ", ".join(categories))
    return filters


def _read_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / METADATA_FILE
    if not path.exists():
        raise DataUnavailableError(f"There is no freshness record ({path}), so no answer is given.")
    meta = json.loads(path.read_text())
    for field in ("complete_through", "refreshed_at"):
        if field not in meta:
            raise DataUnavailableError(f"{METADATA_FILE} has no {field!r}, so no answer is given.")
    return meta


def _connect(data_dir: Path, citywide: bool) -> duckdb.DuckDBPyConnection:
    category_path = data_dir / CATEGORY_FILE
    if not category_path.exists():
        raise DataUnavailableError(f"The data file {category_path} is missing.")
    con = duckdb.connect()
    con.read_parquet(str(category_path)).create_view("borough_category")
    columns = {row[0] for row in con.execute("DESCRIBE borough_category").fetchall()}
    if missing := REQUIRED_COLUMNS - columns:
        con.close()
        raise DataUnavailableError(f"{CATEGORY_FILE} is missing the columns {sorted(missing)}.")
    if citywide:
        city_path = data_dir / CITYWIDE_FILE
        if not city_path.exists():
            con.close()
            raise DataUnavailableError(f"The data file {city_path} is missing.")
        con.read_csv(str(city_path)).create_view("citywide")
    return con


def _check_coverage(
    con: duckdb.DuckDBPyConnection,
    params: ComparePeriodsParams,
    window: dict[str, Any],
    complete_through: str,
) -> None:
    first, last = con.execute(
        "SELECT strftime(min(month), '%Y-%m'), strftime(max(month), '%Y-%m') FROM borough_category"
    ).fetchone()
    if first is None:
        raise DataUnavailableError(f"{CATEGORY_FILE} has no rows.")
    usable_through = min(last, complete_through)
    for name, period in (("period_a", params.period_a), ("period_b", params.period_b)):
        if period.start < first:
            raise DataUnavailableError(f"{name} starts {period.start}, but the data begins {first}.")
        if period.end > usable_through:
            raise DataUnavailableError(
                f"{name} ends {period.end}, but the data is complete only through "
                f"{usable_through}; later months are missing or still partial."
            )
    present = set(con.execute(MONTHS_PRESENT_SQL, window).fetchall())
    months = params.period_a.each_month() + params.period_b.each_month()
    gaps = [f"{borough} {month}" for borough in window["boroughs"] for month in months
            if (borough, month) not in present]
    if gaps:
        more = " and others" if len(gaps) > 5 else ""
        raise DataUnavailableError(
            f"There are no rows at all for {', '.join(gaps[:5])}{more}. That is a gap in the "
            "data, not zero complaints, so the comparison isn't run."
        )


def _resolve_categories(requested: list[str], catalog: CategoryCatalog) -> dict[str, str]:
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


def _citywide_total(con: duckdb.DuckDBPyConnection, period: MonthRange) -> int:
    rows = con.execute(
        CITYWIDE_SQL, {"start": _first_day(period.start), "end": _first_day(period.end)}
    ).fetchall()
    missing = sorted(set(period.each_month()) - {month for month, _ in rows})
    if missing:
        raise DataUnavailableError(f"{CITYWIDE_FILE} has no total for {', '.join(missing)}.")
    return sum(int(count) for _, count in rows)


def _source(path: Path) -> SourceFile:
    try:
        shown = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        shown = str(path)
    return SourceFile(path=shown, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _first_day(year_month: str) -> date:
    year, month = map(int, year_month.split("-"))
    return date(year, month, 1)


def _month_name(year_month: str) -> str:
    return _first_day(year_month).strftime("%b %Y")


def _pct(part: int, whole: int | None) -> float | None:
    return round(100 * part / whole, 2) if whole else None
