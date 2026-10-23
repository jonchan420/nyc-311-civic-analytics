"""Complaint categories: which raw 311 labels count as the same thing.

Labels are messy in two ways:
  - one category appears under several spellings ("ELEVATOR", "Elevator");
  - agencies rename categories, so an old label stops and a new one takes
    over ("Sewer" became "Sewer Maintenance" in Aug 2026).
category_catalog.csv maps every raw label to one category. It is built
by src/build_category_catalog.py from the data plus a hand-reviewed list
of renames, and read here at query time. A label that appears after the
catalog was built falls back to spelling rules and is marked "unreviewed".
"""

from __future__ import annotations

import csv
import difflib
import re
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from config.settings import VALID_BOROUGHS

BOROUGHS = tuple(sorted(VALID_BOROUGHS))
CATALOG_FILE = "category_catalog.csv"
CATALOG_COLUMNS = ["label", "category_id", "category", "kind", "renamed_in", "note"]


def spelling_key(label: str) -> str:
    """Uppercase, with each run of non-letters/digits as one space."""
    return re.sub(r"[^A-Z0-9]+", " ", label.upper()).strip()


def slug(name: str) -> str:
    return spelling_key(name).lower().replace(" ", "-")


@dataclass(frozen=True)
class Entry:
    label: str
    category_id: str
    category: str
    kind: str        # current | spelling | renamed | unreviewed
    renamed_in: str  # "YYYY-MM" the new label took over, for renamed labels
    note: str


class CategoryCatalog:
    def __init__(self, entries: list[Entry], source: Path | None = None):
        self.source = source
        self._by_label = {entry.label: entry for entry in entries}
        self._by_key: dict[str, Entry] = {}
        for entry in entries:  # current names first, so a name never resolves to an old label
            if entry.kind == "current":
                self._by_key.setdefault(spelling_key(entry.category), entry)
        for entry in entries:
            self._by_key.setdefault(spelling_key(entry.label), entry)

    @classmethod
    def load(cls, data_dir: Path) -> CategoryCatalog:
        path = Path(data_dir) / CATALOG_FILE
        if not path.exists():
            return cls([], source=None)
        with path.open(newline="") as handle:
            entries = [Entry(**{column: row[column] for column in CATALOG_COLUMNS})
                       for row in csv.DictReader(handle)]
        return cls(entries, source=path)

    def lookup(self, label: str) -> Entry:
        """The category a raw label belongs to, falling back to spelling rules."""
        if label in self._by_label:
            return self._by_label[label]
        known = self._by_key.get(spelling_key(label))
        if known is None:
            return Entry(label, slug(label), label, "unreviewed", "", "not in the category catalog")
        kind = "renamed" if known.kind == "renamed" else "spelling"
        return replace(known, label=label, kind=kind)

    def including(self, labels: list[str]) -> CategoryCatalog:
        """This catalog plus fallback entries for labels it hasn't seen."""
        extra = [self.lookup(label) for label in labels if label not in self._by_label]
        return CategoryCatalog(list(self._by_label.values()) + extra, self.source)

    def find(self, text: str) -> Entry | None:
        """Resolve a category a person typed: any spelling of a name or old label."""
        return self._by_key.get(spelling_key(text))

    def names(self) -> list[str]:
        return sorted({entry.category for entry in self._by_label.values()})

    def suggest(self, text: str, n: int = 3) -> list[str]:
        return difflib.get_close_matches(text, self.names(), n=n, cutoff=0.6)


def build_catalog(
    table: pd.DataFrame,
    renames: dict[str, tuple[str, str]],
    notes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """One row per raw label in table (columns month, complaint_type, complaint_count).

    renames maps an old label to (new label, reviewer's evidence). Every
    other label is grouped by spelling, and each category is named by its
    most-used current spelling.
    """
    notes = notes or {}
    labels = set(table["complaint_type"])
    for old, (new, _) in renames.items():
        missing = {old, new} - labels
        if missing:
            raise ValueError(f"rename {old!r} -> {new!r}: {sorted(missing)} not in the data")
    renamed_key = {spelling_key(old): new for old, (new, _) in renames.items()}

    def group(label: str) -> str:
        return spelling_key(renamed_key.get(spelling_key(label), label))

    totals = table.groupby("complaint_type")["complaint_count"].sum()
    monthly = table.pivot_table(index="month", columns="complaint_type", values="complaint_count",
                                aggfunc="sum", fill_value=0).sort_index()
    last_month = monthly.index.max()

    names: dict[str, str] = {}
    for label in sorted(labels, key=lambda l: (-totals[l], l)):  # most-used first
        if spelling_key(label) == group(label):
            names.setdefault(group(label), label)

    rows = []
    for label in sorted(labels):
        key, name = group(label), names[group(label)]
        current_labels = [l for l in labels if spelling_key(l) == key]
        renamed = spelling_key(label) in renamed_key
        if renamed:
            kind = "renamed"
            took_over = _takeover_month(monthly[current_labels].sum(axis=1), monthly[label])
            note = next(reason for old, (_, reason) in renames.items() if spelling_key(old) == spelling_key(label))
        else:
            kind = "current" if label == name else "spelling"
            took_over = ""
            note = notes.get(label, "")
        last_used = monthly.index[monthly[label] > 0].max()
        if not note and last_used < last_month - pd.DateOffset(months=2):
            note = f"last used {last_used:%Y-%m}"
        rows.append({"label": label, "category_id": slug(name), "category": name, "kind": kind,
                     "renamed_in": took_over, "note": note})
    order = {"current": 0, "spelling": 1, "renamed": 2}
    return (pd.DataFrame(rows, columns=CATALOG_COLUMNS)
            .sort_values(["category", "kind", "label"], key=lambda s: s.map(order) if s.name == "kind" else s.str.lower())
            .reset_index(drop=True))


def _takeover_month(new: pd.Series, old: pd.Series) -> str:
    """The first month from which the new label always has at least as many complaints."""
    behind = new < old
    after_last_behind = behind[behind].index.max() if behind.any() else new.index.min() - pd.DateOffset(months=1)
    later = new[(new.index > after_last_behind) & (new > 0)]
    return f"{later.index.min():%Y-%m}" if len(later) else ""
