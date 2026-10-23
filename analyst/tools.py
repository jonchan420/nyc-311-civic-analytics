"""Allowlisted analysis tools over the committed data/live/ tables.

Each tool validates its parameters (schemas.py), reads data only through
DataSource (data.py), and returns a ToolResult whose provenance records
what was computed, from which file version, with which filters. A model
can only choose a tool and fill its typed parameters; it never writes SQL.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from analyst.catalog import CATALOG_FILE, CategoryCatalog
from analyst.data import (  # noqa: F401  (errors re-exported for callers)
    DataSource,
    DataUnavailableError,
    Group,
    ToolError,
    UnknownCategoryError,
    boroughs_for,
    group_by_category,
    month_name,
    pct,
    place_name,
    resolve_categories,
    shift_month,
)
from analyst.schemas import (
    CategoryChange,
    CategoryMonth,
    ComparePeriodsParams,
    MonthCount,
    MonthlyTrendParams,
    MonthRange,
    MonthSummaryParams,
    ToolResult,
)
from config.settings import DATA_DIR, LIVE_DIR

SMALL_COUNT = 20           # under this in both periods, a change is mostly noise
MIN_SHARE_BASE_PCT = 1.0   # shares of change are skipped when |total change| < 1% of period A
SPIKE_RATIO = 3            # a month above 3x the category's typical month is flagged
SPIKE_MIN_TYPICAL = 20     # ...if its typical month has at least this many complaints
POPULATION_FILE = DATA_DIR / "borough_population.csv"

COMPARE_CALCULATION = (
    "For each complaint type, count_a and count_b are the sums of its monthly counts in "
    "each period (a month with no row counts as 0); change = count_b - count_a; "
    "pct_change = 100 * change / count_a; share_of_change = 100 * change / total change. "
    "Labels that differ only in case or punctuation, and renamed labels, are combined "
    "under one current name."
)
TREND_CALCULATION = (
    "Monthly sums of complaint counts (a month with no row counts as 0), combined across "
    "labels that are the same category. change = last month - first month."
)
SUMMARY_CALCULATION = (
    "Complaint counts for the month, the month before, and the same month a year before. "
    "pct values = 100 * change / earlier count; share_of_month = 100 * count / month total; "
    "per_100k = 100,000 * month total / population (2020 Census)."
)


# ---------------------------------------------------------------- compare_periods

def compare_periods(
    params: ComparePeriodsParams | dict[str, Any], data_dir: Path = LIVE_DIR
) -> ToolResult[CategoryChange]:
    """Which complaint types changed most between two equal-length periods?"""
    params = ComparePeriodsParams.model_validate(params)
    boroughs = boroughs_for(params.geography)
    with DataSource(data_dir) as source:
        source.require_months({"period_a": params.period_a, "period_b": params.period_b}, boroughs)
        wanted = resolve_categories(params.categories, source.catalog) if params.categories else None
        by_label = source.counts_by_label(boroughs, {"a": params.period_a, "b": params.period_b})
        groups = group_by_category(by_label, source.catalog, ["a", "b"], wanted)
        city = None
        if params.geography == "CITYWIDE":
            city = (sum(source.citywide_by_month(params.period_a).values()),
                    sum(source.citywide_by_month(params.period_b).values()))

        total_a = sum(g.counts["a"] for g in groups.values())
        total_b = sum(g.counts["b"] for g in groups.values())
        total_change = total_b - total_a
        shareable = total_change != 0 and abs(total_change) >= MIN_SHARE_BASE_PCT / 100 * total_a
        rows = {cid: _category_change(g, total_change if shareable else None, source.catalog)
                for cid, g in groups.items()}
        ranked = sorted(rows, key=lambda cid: (-abs(rows[cid].change), rows[cid].category))
        shown = ranked[: params.top_n]
        spikes = _unusual_months(source, boroughs, {cid: groups[cid] for cid in shown},
                                 params.period_a.each_month() + params.period_b.each_month())
        for cid in spikes:
            rows[cid].flags.append("spike")

        filters = [
            f"geography: {place_name(params.geography)}; complaints with no borough recorded are excluded",
            f"period A: {_span(params.period_a)}",
            f"period B: {_span(params.period_b)}",
        ]
        if wanted:
            filters.append("complaint types: " + ", ".join(wanted.values()))
        provenance = source.provenance("compare_periods", params.model_dump(mode="json"),
                                       COMPARE_CALCULATION, filters)
        catalog = source.catalog

    shown_rows = [rows[cid] for cid in shown]
    shown_change = sum(row.change for row in shown_rows)
    summary: dict[str, Any] = {
        "months_per_period": params.period_a.months,
        "total_a": total_a,
        "total_b": total_b,
        "change": total_change,
        "pct_change": pct(total_change, total_a),
        "categories_compared": len(rows),
        "shown_change": shown_change,
        "other_change": total_change - shown_change,
    }
    if city:
        summary["citywide_total_a"], summary["citywide_total_b"] = city

    warnings = _label_warnings([groups[cid] for cid in shown], catalog)
    warnings += _flag_warnings(shown_rows)
    if total_change == 0:
        warnings.append("The total didn't change, so shares of the change aren't computed.")
    elif not shareable:
        warnings.append(
            f"The total changed by only {total_change:+,} ({pct(total_change, total_a)}% of period A), "
            "so shares of the change aren't computed; they would swing wildly.")
    elif any(r.share_of_change is not None and not 0 <= r.share_of_change <= 100 for r in shown_rows):
        warnings.append(
            "Some categories moved against the overall change, so shares above 100% or below 0% "
            "are expected; across all categories the shares sum to 100%.")
    warnings += _spike_warnings(spikes, groups)
    if city:
        warnings.append(
            f"Category counts cover the five boroughs: {total_a:,} of {city[0]:,} complaints in "
            f"period A and {total_b:,} of {city[1]:,} in period B. The rest have no borough recorded.")
    return ToolResult[CategoryChange](summary=summary, rows=shown_rows, warnings=warnings,
                                      provenance=provenance)


def _category_change(group: Group, total_change: int | None, catalog: CategoryCatalog) -> CategoryChange:
    a, b = group.counts["a"], group.counts["b"]
    change = b - a
    flags = []
    if a == 0 and b > 0:
        flags.append("new_in_b")
    if a > 0 and b == 0:
        flags.append("absent_in_b")
    if a < SMALL_COUNT and b < SMALL_COUNT:
        flags.append("small_count")
    flags += _label_flags(group, catalog)
    return CategoryChange(category=group.name, count_a=a, count_b=b, change=change,
                          pct_change=pct(change, a), share_of_change=pct(change, total_change),
                          flags=flags, labels=sorted(group.labels))


# ---------------------------------------------------------------- monthly_trend

def monthly_trend(params: MonthlyTrendParams | dict[str, Any], data_dir: Path = LIVE_DIR) -> ToolResult[MonthCount]:
    """Complaints per month over a period: all complaints, or one series per complaint type."""
    params = MonthlyTrendParams.model_validate(params)
    boroughs = boroughs_for(params.geography)
    months = params.period.each_month()
    with DataSource(data_dir) as source:
        source.require_months({"period": params.period}, boroughs)
        wanted = resolve_categories(params.categories, source.catalog) if params.categories else None
        series: dict[str, Group] = {}
        for label, month, count in source.monthly_by_label(boroughs, params.period.start, params.period.end):
            entry = source.catalog.lookup(label)
            if wanted is not None and entry.category_id not in wanted:
                continue
            key, name = (entry.category_id, entry.category) if wanted else ("all", "All complaints")
            group = series.setdefault(key, Group(name, dict.fromkeys(months, 0)))
            group.counts[month] += count
            if label not in group.labels:
                group.labels.append(label)
        for category_id, name in (wanted or {}).items():
            series.setdefault(category_id, Group(name, dict.fromkeys(months, 0)))
        spikes = _unusual_months(source, boroughs, {k: g for k, g in series.items() if k != "all"}, months)
        city = sum(source.citywide_by_month(params.period).values()) if params.geography == "CITYWIDE" else None

        filters = [
            f"geography: {place_name(params.geography)}; complaints with no borough recorded are excluded",
            f"period: {_span(params.period)}",
        ]
        if wanted:
            filters.append("complaint types: " + ", ".join(wanted.values()))
        provenance = source.provenance("monthly_trend", params.model_dump(mode="json"),
                                       TREND_CALCULATION, filters)
        catalog = source.catalog

    order = list(wanted) if wanted else ["all"]
    rows = []
    for key in order:
        group = series.get(key) or Group("All complaints", dict.fromkeys(months, 0))
        spike_months = {month for month, _ in spikes.get(key, ([], 0))[0]}
        flags = _label_flags(group, catalog)
        for month in months:
            rows.append(MonthCount(month=month, category=group.name, count=group.counts[month],
                                   flags=flags + (["spike"] if month in spike_months else [])))

    combined = {month: sum(g.counts[month] for g in series.values()) for month in months}
    peak = max(months, key=lambda m: (combined[m], m))
    low = min(months, key=lambda m: (combined[m], m))
    total = sum(combined.values())
    summary: dict[str, Any] = {
        "months": len(months),
        "total": total,
        "first_month": months[0],
        "first_month_count": combined[months[0]],
        "last_month": months[-1],
        "last_month_count": combined[months[-1]],
        "change": combined[months[-1]] - combined[months[0]],
        "pct_change": pct(combined[months[-1]] - combined[months[0]], combined[months[0]]),
        "peak_month": peak,
        "peak_count": combined[peak],
        "low_month": low,
        "low_count": combined[low],
        "mean_per_month": round(total / len(months), 1),
    }
    if city is not None:
        summary["citywide_total"] = city
    warnings = _label_warnings(list(series.values()), catalog) + _spike_warnings(spikes, series)
    if city is not None and not wanted:
        warnings.append(
            f"Monthly counts cover the five boroughs: {total:,} of {city:,} complaints in the period. "
            "The rest have no borough recorded.")
    return ToolResult[MonthCount](summary=summary, rows=rows, warnings=warnings, provenance=provenance)


# ---------------------------------------------------------------- month_summary

def month_summary(
    params: MonthSummaryParams | dict[str, Any], data_dir: Path = LIVE_DIR, population_file: Path = POPULATION_FILE
) -> ToolResult[CategoryMonth]:
    """A briefing on one month: totals against the month before and a year before,
    the largest complaint types, and the types that rose or fell most."""
    params = MonthSummaryParams.model_validate(params)
    boroughs = boroughs_for(params.geography)
    month = MonthRange(start=params.month, end=params.month)
    prior = shift_month(params.month, -1)
    year_ago = shift_month(params.month, -12)
    with DataSource(data_dir) as source:
        source.require_months({"month": month}, boroughs)
        windows = {"month": month}
        for name, ym in (("prior", prior), ("last_year", year_ago)):
            if ym >= source.first_month:
                windows[name] = MonthRange(start=ym, end=ym)
        source.require_months(windows, boroughs)
        by_label = source.counts_by_label(boroughs, windows)
        groups = group_by_category(by_label, source.catalog, list(windows))
        history = [m for m in (shift_month(params.month, -i) for i in range(12, 0, -1)) if m >= source.first_month]
        history_totals = {m: 0 for m in history}
        if history:
            for _, ym, count in source.monthly_by_label(boroughs, history[0], history[-1]):
                history_totals[ym] += count
        city = sum(source.citywide_by_month(month).values()) if params.geography == "CITYWIDE" else None

        totals = {name: sum(g.counts[name] for g in groups.values()) for name in windows}
        rows_by_id = {cid: _category_month(g, totals["month"], windows, source.catalog)
                      for cid, g in groups.items()}
        chosen = _briefing_rows(rows_by_id, params.top_n)
        spikes = _unusual_months(source, boroughs, {cid: groups[cid] for cid in chosen}, [params.month])
        for cid in spikes:
            rows_by_id[cid].flags.append("spike")

        filters = [
            f"geography: {place_name(params.geography)}; complaints with no borough recorded are excluded",
            f"month: {params.month}; compared with {prior} and {year_ago}",
        ]
        source.add_file(Path(population_file))
        provenance = source.provenance("month_summary", params.model_dump(mode="json"),
                                       SUMMARY_CALCULATION, filters)
        catalog = source.catalog

    population = _population(population_file, boroughs)
    month_total = totals["month"]
    prior_total = totals.get("prior")
    year_total = totals.get("last_year")
    summary: dict[str, Any] = {
        "month": params.month,
        "month_total": month_total,
        "prior_month": prior,
        "prior_month_total": prior_total,
        "change_vs_prior_month": None if prior_total is None else month_total - prior_total,
        "pct_vs_prior_month": None if prior_total is None else pct(month_total - prior_total, prior_total),
        "last_year_month": year_ago,
        "last_year_total": year_total,
        "change_vs_last_year": None if year_total is None else month_total - year_total,
        "pct_vs_last_year": None if year_total is None else pct(month_total - year_total, year_total),
        "average_prior_12_months": round(sum(history_totals.values()) / len(history), 1) if history else None,
        "population": population,
        "per_100k": round(100_000 * month_total / population, 1),
    }
    if city is not None:
        summary["citywide_total"] = city

    rows = sorted((rows_by_id[cid] for cid in chosen), key=lambda r: (-r.count, r.category))
    warnings = _label_warnings([groups[cid] for cid in chosen], catalog) + _spike_warnings(spikes, groups)
    if prior_total is None or year_total is None:
        missing = [name for name, total in (("the month before", prior_total), ("a year before", year_total))
                   if total is None]
        warnings.append(f"No data for {' or '.join(missing)}, so that comparison is left out.")
    if not history:
        warnings.append("No earlier months are in the data, so there's no 12-month average.")
    elif len(history) < 12:
        warnings.append(f"The 12-month average uses only the {len(history)} months available.")
    warnings.append("per_100k divides by 2020 Census populations, which don't account for "
                    "commuters, visitors, or population change since 2020.")
    if city is not None:
        warnings.append(
            f"Category counts cover the five boroughs: {month_total:,} of {city:,} complaints. "
            "The rest have no borough recorded.")
    return ToolResult[CategoryMonth](summary=summary, rows=rows, warnings=warnings, provenance=provenance)


def _category_month(group: Group, month_total: int, windows: dict[str, MonthRange],
                    catalog: CategoryCatalog) -> CategoryMonth:
    count = group.counts["month"]
    prior = group.counts.get("prior") if "prior" in windows else None
    year = group.counts.get("last_year") if "last_year" in windows else None
    change = None if year is None else count - year
    flags = _label_flags(group, catalog)
    if year == 0 and count > 0:
        flags.append("new_vs_last_year")
    return CategoryMonth(category=group.name, count=count, prior_month=prior, last_year=year,
                         change_vs_last_year=change, pct_vs_last_year=None if change is None else pct(change, year),
                         share_of_month=pct(count, month_total), flags=flags, labels=sorted(group.labels))


def _briefing_rows(rows: dict[str, CategoryMonth], top_n: int) -> list[str]:
    """The top_n categories by count, plus up to 3 that rose and 3 that fell most vs a year before."""
    by_count = sorted(rows, key=lambda cid: (-rows[cid].count, rows[cid].category))[:top_n]
    for cid in by_count:
        rows[cid].flags.insert(0, "top")
    with_change = [cid for cid in rows if rows[cid].change_vs_last_year is not None]
    rising = sorted((c for c in with_change if rows[c].change_vs_last_year > 0),
                    key=lambda c: (-rows[c].change_vs_last_year, rows[c].category))[:3]
    falling = sorted((c for c in with_change if rows[c].change_vs_last_year < 0),
                     key=lambda c: (rows[c].change_vs_last_year, rows[c].category))[:3]
    for cid in rising:
        rows[cid].flags.append("rising")
    for cid in falling:
        rows[cid].flags.append("falling")
    return list(dict.fromkeys(by_count + rising + falling))


def _population(population_file: Path, boroughs: list[str]) -> int:
    table = pd.read_csv(population_file)
    found = table[table["borough"].isin(boroughs)]
    if len(found) != len(boroughs):
        raise DataUnavailableError(f"{Path(population_file).name} is missing a population for {boroughs}.")
    return int(found["population"].sum())


# ---------------------------------------------------------------- shared checks

def _span(period: MonthRange) -> str:
    return f"{period.label()} ({period.months} month{'s' if period.months != 1 else ''})"


def _label_flags(group: Group, catalog: CategoryCatalog) -> list[str]:
    kinds = {catalog.lookup(label).kind for label in group.labels}
    if catalog.source is None:  # with no catalog every label is unreviewed; one warning says so
        kinds.discard("unreviewed")
    return [kind for kind in ("renamed", "unreviewed") if kind in kinds]


def _label_warnings(groups: list[Group], catalog: CategoryCatalog) -> list[str]:
    if catalog.source is None:
        return [f"No {CATALOG_FILE} was found, so renamed categories aren't combined; "
                "only spellings that differ in case or punctuation are."]
    warnings = []
    for group in groups:
        for label in sorted(group.labels):
            entry = catalog.lookup(label)
            if entry.kind == "renamed":
                when = f" in {month_name(entry.renamed_in)}" if entry.renamed_in else ""
                warnings.append(f'"{label}" was renamed "{entry.category}"{when}; both labels are counted together.')
    unreviewed = [g.name for g in groups if any(catalog.lookup(l).kind == "unreviewed" for l in g.labels)]
    if unreviewed:
        warnings.append(f"{', '.join(unreviewed)}: not yet in the category catalog, so a rename of an "
                        "older category would not be recognized.")
    return warnings


def _flag_warnings(rows: list[CategoryChange]) -> list[str]:
    def named(flag: str) -> str:
        return ", ".join(row.category for row in rows if flag in row.flags)

    warnings = []
    if named("new_in_b"):
        warnings.append(f"No complaints in period A for {named('new_in_b')}: a new category or an "
                        "unrecognized rename, so the increase may not be real growth.")
    if named("absent_in_b"):
        warnings.append(f"No complaints in period B for {named('absent_in_b')}: a retired category or an "
                        "unrecognized rename, so the decrease may not be a real decline.")
    if named("small_count"):
        warnings.append(f"Fewer than {SMALL_COUNT} complaints in both periods for {named('small_count')}; "
                        "too few to read much into.")
    return warnings


def _unusual_months(source: DataSource, boroughs: list[str], groups: dict[str, Group],
                    checked: list[str]) -> dict[str, tuple[list[tuple[str, int]], float]]:
    """Categories with a checked month above SPIKE_RATIO x their typical month.

    Typical = the median month of the 12 months before the first checked month,
    or of the checked months themselves when fewer than 6 of those are in the data.
    Returns {category id: ([(month, count), ...], typical)}.
    """
    if not groups:
        return {}
    first = min(checked)
    history = [m for m in (shift_month(first, -i) for i in range(12, 0, -1)) if m >= source.first_month]
    start = history[0] if history else first
    series: dict[str, dict[str, int]] = {cid: {} for cid in groups}
    for label, month, count in source.monthly_by_label(boroughs, start, max(checked)):
        cid = source.catalog.lookup(label).category_id
        if cid in series:
            series[cid][month] = series[cid].get(month, 0) + count
    base = history if len(history) >= 6 else checked
    found = {}
    for cid, by_month in series.items():
        typical = statistics.median(by_month.get(m, 0) for m in base)
        if typical < SPIKE_MIN_TYPICAL:
            continue
        high = [(m, by_month.get(m, 0)) for m in checked if by_month.get(m, 0) > SPIKE_RATIO * typical]
        if high:
            found[cid] = (high, typical)
    return found


def _spike_warnings(spikes: dict[str, tuple[list[tuple[str, int]], float]], groups: dict[str, Group]) -> list[str]:
    warnings = []
    for cid, (months, typical) in spikes.items():
        month, count = max(months, key=lambda item: item[1])
        warnings.append(
            f"{groups[cid].name} had an unusual month: {month_name(month)} recorded {count:,}, more than "
            f"{SPIKE_RATIO} times its typical month ({typical:,.0f}). Check for a one-off event or a "
            "reporting change before reading it as a trend.")
    return warnings
