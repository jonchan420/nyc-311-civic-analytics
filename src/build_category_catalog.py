"""Build data/live/category_catalog.csv: which raw 311 labels are the same category.

REVIEWED_RENAMES is the hand-reviewed part. Each pair was checked on
2026-10-23 against the raw records and the monthly counts: the same
agency handles both labels, they describe the same kind of complaint,
and the old label stops as the new one takes over. Agencies usually
renamed their descriptors at the same time, so descriptor text rarely
matches; the meaning and the month-by-month handover are the evidence.
renamed_in in the output is computed from the data: the first month from
which the new label always has at least as many complaints.

Labels that ended without a clear successor stay separate; REVIEWED_NOTES
says why. Every label with at least 100 complaints that stopped appearing
between Jan 2020 and Sep 2026 was reviewed.

    python src/build_category_catalog.py           # rebuild the catalog
    python src/build_category_catalog.py --check   # list labels it hasn't seen yet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analyst.catalog import CATALOG_FILE, CategoryCatalog, build_catalog
from config.settings import LIVE_DIR

# old label: (new label, evidence)
REVIEWED_RENAMES: dict[str, tuple[str, str]] = {
    "Homeless Encampment": ("Encampment", "NYPD; identical descriptors; old label used only Jan-Apr 2020"),
    "Snow": ("Snow or Ice", "DSNY; icy sidewalks and plowing; old label ends Mar 2021, new one starts the next winter"),
    "Snow Removal": ("Snow or Ice", "DSNY; old label ends Feb 2021, new one starts the next winter"),
    "Dirty Conditions": ("Dirty Condition", "DSNY; handover Sep-Oct 2021 at similar volume"),
    "Missed Collection (All Materials)": ("Missed Collection", "DSNY; handover Sep-Oct 2021"),
    "Sweeping/Missed": ("Street Sweeping Complaint", "DSNY; sweeping labels merged into one in Sep 2021"),
    "Sweeping/Inadequate": ("Street Sweeping Complaint", "DSNY; sweeping labels merged into one in Sep 2021"),
    "Sweeping/Missed-Inadequate": ("Street Sweeping Complaint", "DSNY; sweeping labels merged into one in Sep 2021"),
    "Overflowing Litter Baskets": ("Litter Basket Complaint",
                                   "DSNY; handover Sep 2021; the new label also covers basket misuse"),
    "Overflowing Recycling Baskets": ("Recycling Basket Complaint", "DSNY; handover Sep-Oct 2021; very small volume"),
    "Vacant Lot": ("Lot Condition", "DSNY; lot-cleaning requests; handover Sep 2021"),
    "Derelict Bicycle": ("Abandoned Bike", "DSNY; handover Sep 2021"),
    "Smoking": ("Smoking or Vaping", "DOHMH; both labels used Dec 2023-Mar 2024, then the new one dominates"),
    "Bike Rack Condition": ("Bike Rack", "DOT; handover Jul-Aug 2024"),
    "Pet Shop": ("Pet Sale", "DOHMH; handover Jun 2025"),
    "Bike/Roller/Skate Chronic": ("Bike/Roller/Skate", "NYPD; identical descriptors; handover Nov 2025"),
    "Sewer": ("Sewer Maintenance", "DEP; handover Jul-Aug 2026"),
    "Water System": ("Water Maintenance", "DEP; handover Jul-Aug 2026"),
    "Hazardous Materials": ("Hazardous Material", "DEP; handover Jul-Aug 2026"),
    "Plant": ("Waste Water Treatment Plant", "DEP; plant odor and noise; handover Jul-Aug 2026"),
}

REVIEWED_NOTES: dict[str, str] = {
    "Sanitation Condition": "ended Sep 2021 as DSNY relabeled its complaints; its mix (mostly dump-outs, "
                            "also dead animals and weeds) has no single successor label",
    "Other Enforcement": "ended Sep 2021; no clear successor label",
    "Recycling Enforcement": "ended Sep 2021; no clear successor label",
    "Collection Truck Noise": "ended Sep 2021; no clear successor label",
    "Vending": "ended Jan 2021; Vendor Enforcement only starts Mar 2023, so it isn't treated as a rename",
    "Homeless Street Condition": "ended Mar 2021; Encampment already existed alongside it, "
                                 "so it isn't treated as a rename",
    "Request Large Bulky Item Collection": "ended Sep 2022 with no successor label in this data",
    "Electronics Waste Appointment": "ended Oct 2023 with no successor label in this data",
}


def main(check: bool = False, data_dir: Path = LIVE_DIR) -> None:
    table = pd.read_parquet(data_dir / "monthly_borough_category.parquet")
    if check:
        known = CategoryCatalog.load(data_dir)
        unseen = sorted(label for label in set(table["complaint_type"])
                        if known.lookup(label).kind == "unreviewed")
        print("Labels not in the category catalog:", ", ".join(unseen) if unseen else "none")
        return

    catalog = build_catalog(table, REVIEWED_RENAMES, REVIEWED_NOTES)
    catalog.to_csv(data_dir / CATALOG_FILE, index=False)
    counts = catalog["kind"].value_counts()
    print(f"{len(catalog)} labels -> {catalog['category_id'].nunique()} categories "
          f"({counts.get('spelling', 0)} spelling variants, {counts.get('renamed', 0)} renames); "
          f"wrote {data_dir / CATALOG_FILE}")


if __name__ == "__main__":
    main(check="--check" in sys.argv)
