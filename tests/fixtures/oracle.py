"""An independent pandas version of compare_periods, used only to check the tool.

It reads the same data files but shares no code with analyst/, so a bug
in the tool's SQL, grouping, or sorting shows up as a disagreement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SHARE_THRESHOLD = 0.01  # the documented rule: no shares when |total change| < 1% of period A


def compare(data_dir: Path, geography: str, a: tuple[str, str], b: tuple[str, str],
            categories: list[str] | None = None) -> tuple[int, int, list[tuple]]:
    """(total_a, total_b, rows) with rows sorted by |change| then name.

    categories are display names from the catalog. Each row is
    (category, count_a, count_b, change, pct_change, share_of_change).
    """
    table = pd.read_parquet(data_dir / "monthly_borough_category.parquet")
    catalog = pd.read_csv(data_dir / "category_catalog.csv", keep_default_na=False)
    table["category"] = table["complaint_type"].map(dict(zip(catalog["label"], catalog["category"])))
    table["ym"] = table["month"].dt.strftime("%Y-%m")
    if geography != "CITYWIDE":
        table = table[table["borough"] == geography]
    if categories:
        table = table[table["category"].isin(categories)]

    count_a = table[table["ym"].between(*a)].groupby("category")["complaint_count"].sum()
    count_b = table[table["ym"].between(*b)].groupby("category")["complaint_count"].sum()
    both = pd.concat({"a": count_a, "b": count_b}, axis=1).fillna(0).astype(int)
    if categories:
        both = both.reindex(categories, fill_value=0)
    both = both[(both["a"] > 0) | (both["b"] > 0) | both.index.isin(categories or [])]

    total_a, total_b = int(both["a"].sum()), int(both["b"].sum())
    total_change = total_b - total_a
    shareable = total_change != 0 and abs(total_change) >= SHARE_THRESHOLD * total_a
    rows = []
    for name, row in both.iterrows():
        change = int(row["b"] - row["a"])
        pct = round(100 * change / row["a"], 2) if row["a"] else None
        share = round(100 * change / total_change, 2) if shareable else None
        rows.append((name, int(row["a"]), int(row["b"]), change, pct, share))
    rows.sort(key=lambda r: (-abs(r[3]), r[0]))
    return total_a, total_b, rows


# ------------------------------------------------------------------ monthly_trend and month_summary
# Same documented rules as the tools: a missing row counts as 0; the peak month is the latest
# of equal months and the low month the earliest; percentages are rounded to 2 places, the mean
# and per-100,000 rate to 1; the briefing lists the top_n types by count plus up to 3 that rose
# and 3 that fell most against the same month a year before.

def _labelled(data_dir: Path) -> pd.DataFrame:
    table = pd.read_parquet(data_dir / "monthly_borough_category.parquet")
    catalog = pd.read_csv(data_dir / "category_catalog.csv", keep_default_na=False)
    table["category"] = table["complaint_type"].map(dict(zip(catalog["label"], catalog["category"])))
    table["ym"] = table["month"].dt.strftime("%Y-%m")
    return table


def _months(start: str, end: str) -> list[str]:
    return list(pd.period_range(start, end, freq="M").strftime("%Y-%m"))


def _shift(month: str, months: int) -> str:
    return (pd.Period(month, freq="M") + months).strftime("%Y-%m")


def _pct(part: int, whole: int | None) -> float | None:
    return round(100 * part / whole, 2) if whole else None


def trend(data_dir: Path, geography: str, start: str, end: str, categories: list[str] | None = None) -> dict:
    """The summary numbers and the monthly series {category or "All complaints": [counts]}."""
    table = _labelled(data_dir)
    if geography != "CITYWIDE":
        table = table[table["borough"] == geography]
    months = _months(start, end)
    table = table[table["ym"].isin(months)]
    names = categories or ["All complaints"]
    if categories:
        table = table[table["category"].isin(categories)]
        table = table.assign(series=table["category"])
    else:
        table = table.assign(series="All complaints")
    grid = table.groupby(["series", "ym"])["complaint_count"].sum().unstack(fill_value=0)
    grid = grid.reindex(index=names, columns=months, fill_value=0)
    combined = grid.sum(axis=0)
    first, last = int(combined[months[0]]), int(combined[months[-1]])
    peak = max(months, key=lambda m: (combined[m], m))
    low = min(months, key=lambda m: (combined[m], m))
    total = int(combined.sum())
    return {
        "months": len(months), "total": total,
        "first_month_count": first, "last_month_count": last, "change": last - first,
        "pct_change": _pct(last - first, first),
        "peak_month": peak, "peak_count": int(combined[peak]),
        "low_month": low, "low_count": int(combined[low]),
        "mean_per_month": round(total / len(months), 1),
        "series": {name: [int(v) for v in grid.loc[name]] for name in names},
    }


def month_summary(data_dir: Path, geography: str, month: str, population_file: Path,
                  top_n: int = 8) -> dict:
    """The summary numbers and the briefing rows (category, count, last_year, change_vs_last_year)."""
    everything = _labelled(data_dir)
    first_month = everything["ym"].min()
    table = everything if geography == "CITYWIDE" else everything[everything["borough"] == geography]
    by_month = table.groupby(["ym", "category"])["complaint_count"].sum()

    def counts(ym: str) -> pd.Series | None:
        if ym < first_month:
            return None
        return by_month[ym] if ym in by_month.index.get_level_values(0) else pd.Series(dtype=int)

    now, prior, year = counts(month), counts(_shift(month, -1)), counts(_shift(month, -12))
    total = int(now.sum())
    prior_total = None if prior is None else int(prior.sum())
    year_total = None if year is None else int(year.sum())
    history = [m for m in (_shift(month, -i) for i in range(12, 0, -1)) if m >= first_month]
    population = pd.read_csv(population_file)
    if geography != "CITYWIDE":
        population = population[population["borough"] == geography]

    categories = sorted(set(now.index) | (set(year.index) if year is not None else set()))
    rows = {}
    for name in categories:
        count = int(now.get(name, 0))
        last_year = None if year is None else int(year.get(name, 0))
        rows[name] = (count, last_year, None if last_year is None else count - last_year)
    top = sorted(rows, key=lambda c: (-rows[c][0], c))[:top_n]
    changed = [c for c in rows if rows[c][2] is not None]
    rising = sorted((c for c in changed if rows[c][2] > 0), key=lambda c: (-rows[c][2], c))[:3]
    falling = sorted((c for c in changed if rows[c][2] < 0), key=lambda c: (rows[c][2], c))[:3]
    chosen = sorted(dict.fromkeys(top + rising + falling), key=lambda c: (-rows[c][0], c))
    return {
        "month_total": total,
        "prior_month_total": prior_total,
        "change_vs_prior_month": None if prior_total is None else total - prior_total,
        "pct_vs_prior_month": None if prior_total is None else _pct(total - prior_total, prior_total),
        "last_year_total": year_total,
        "change_vs_last_year": None if year_total is None else total - year_total,
        "pct_vs_last_year": None if year_total is None else _pct(total - year_total, year_total),
        "average_prior_12_months": (round(sum(int(counts(m).sum()) for m in history) / len(history), 1)
                                    if history else None),
        "per_100k": round(100_000 * total / int(population["population"].sum()), 1),
        "rows": [(c, *rows[c]) for c in chosen],
    }
