"""Fixed vocabularies the analysis tools validate against.

Complaint labels in the 311 data are messy in two ways:
  - one category can appear under several spellings ("ELEVATOR" and
    "Elevator"; "Litter Basket / Request" and "Litter Basket Request");
  - NYC sometimes renames a category, so the old label stops and a new
    one takes over ("Sewer" became "Sewer Maintenance" in 2026).
category_key() gives every label one key per real category, so counts
are compared like for like. Phase 0 replaces RENAMES with a reviewed
data/live/category_catalog.csv.
"""

from __future__ import annotations

import re

from config.settings import VALID_BOROUGHS

BOROUGHS = tuple(sorted(VALID_BOROUGHS))

# old label -> (current label, last month the old label was used).
# Each was checked in data/live/monthly_borough_category.parquet on
# 2026-10-23: the old label stops as the new one takes over at a similar volume.
RENAMES: dict[str, tuple[str, str]] = {
    "Bike Rack Condition": ("Bike Rack", "2024-07"),
    "Bike/Roller/Skate Chronic": ("Bike/Roller/Skate", "2025-11"),
    "Hazardous Materials": ("Hazardous Material", "2026-07"),
    "Sewer": ("Sewer Maintenance", "2026-07"),
    "Water System": ("Water Maintenance", "2026-07"),
}
# Renames were only checked from this month on. Earlier label changes (such
# as the batch of sanitation categories that ended in Sept 2021) aren't mapped.
RENAMES_CHECKED_FROM = "2024-01"


def spelling_key(label: str) -> str:
    """Uppercase, with each run of non-letters/digits as one space."""
    return re.sub(r"[^A-Z0-9]+", " ", label.upper()).strip()


_RENAMES_BY_KEY = {spelling_key(old): (new, last) for old, (new, last) in RENAMES.items()}


def renamed_to(label: str) -> tuple[str, str] | None:
    """(current label, last month the old one was used) if this label was renamed."""
    return _RENAMES_BY_KEY.get(spelling_key(label))


def category_key(label: str) -> str:
    """One key per real category: spelling variants and renamed labels share it."""
    rename = renamed_to(label)
    return spelling_key(rename[0] if rename else label)
