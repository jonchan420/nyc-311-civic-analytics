"""Check that every number in an answer came from a tool result.

The writer may round, drop a minus sign ("fell by 11,479" for a change
of -11,479), or restate a month as a year, but it may not compute or
recall a number of its own. An answer with any unsupported number is
rejected and replaced by the template text built from the same result.

A percentage has to match a percentage in a source (a field such as
pct_change or wape_*, or a number written with %), so neither the 900
in a count of 900 nor the 08 in "2025-08" can make "900%" or "8%" grounded.

Limit: signs aren't checked, so "rose 11,479" for a fall of 11,479
passes. Direction words are the writer prompt's job, and the eval checks them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

NUMBER = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?(?:\s?[kKmM]\b)?%?")
PERCENT_FIELD = re.compile(r"pct|percent|share|wape|mape", re.IGNORECASE)
STRUCTURAL = set(range(0, 13)) | {100_000}  # counts of months, "top 3", "5 boroughs"; "per 100,000"


@dataclass
class GroundingReport:
    grounded: bool
    checked: int
    unsupported: list[str] = field(default_factory=list)


def numbers_in(value: Any) -> set[float]:
    """Every number inside a nested value: numeric leaves, and numbers written in strings."""
    found: set[float] = set()
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, (int, float)):
        found.add(float(value))
    elif isinstance(value, str):  # "2026-06" yields 2026 and 6; "1,389" yields 1389
        for token in re.findall(r"\d[\d,]*(?:\.\d+)?", value):
            found.add(float(token.replace(",", "")))
    elif isinstance(value, dict):
        for item in value.values():
            found |= numbers_in(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            found |= numbers_in(item)
    return found


def percents_in(value: Any, key: str = "") -> set[float]:
    """The percentages inside a nested value: numbers in fields named like pct_change,
    share_of_change, or wape_<model>, and numbers written with % in strings."""
    found: set[float] = set()
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, (int, float)):
        if PERCENT_FIELD.search(key):
            found.add(float(value))
    elif isinstance(value, str):
        for token in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s?%", value):
            found.add(float(token.replace(",", "")))
    elif isinstance(value, dict):
        for name, item in value.items():
            found |= percents_in(item, str(name))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            found |= percents_in(item, key)
    return found


def check(text: str, *sources: Any) -> GroundingReport:
    """sources: anything the answer may quote (tool result dicts, question, notes)."""
    percents = set().union(*(percents_in(source) for source in sources))
    anything = set().union(*(numbers_in(source) for source in sources)) | {float(n) for n in STRUCTURAL}
    unsupported, checked = [], 0
    for match in NUMBER.finditer(text):
        token = match.group(0).strip()
        checked += 1
        if re.search(r"[kKmM]%?$", token):  # abbreviations like "16k" hide the real value
            unsupported.append(token)
            continue
        digits = token.rstrip("%").replace(",", "").replace("−", "-").lstrip("+")
        value = float(digits)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        allowed = percents if token.endswith("%") else anything
        if not any(abs(abs(value) - abs(a)) <= tolerance for a in allowed):
            unsupported.append(token)
    return GroundingReport(grounded=not unsupported, checked=checked, unsupported=unsupported)
