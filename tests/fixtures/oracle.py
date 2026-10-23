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
