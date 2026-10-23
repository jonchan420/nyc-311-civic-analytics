"""The category catalog: building it, looking labels up, and the live file's integrity."""

import pandas as pd
import pytest

from analyst.catalog import CATALOG_FILE, CategoryCatalog, _takeover_month, build_catalog
from config.settings import LIVE_DIR
from src.build_category_catalog import REVIEWED_NOTES, REVIEWED_RENAMES
from tests.fixtures.make_fixtures import RENAMES, build_tables


def test_fixture_catalog_matches_hand_expectation():
    catalog = build_catalog(build_tables()[0], RENAMES).fillna("")
    expected = [
        # label,               category,              kind,       renamed_in, note
        ("Cannabis Retailer",   "Cannabis Retailer",   "current",  "",         ""),
        ("HEAT/HOT WATER",      "HEAT/HOT WATER",      "current",  "",         ""),
        ("Heat/Hot Water",      "HEAT/HOT WATER",      "spelling", "",         ""),
        ("Illegal Parking",     "Illegal Parking",     "current",  "",         ""),
        ("Noise - Residential", "Noise - Residential", "current",  "",         ""),
        ("Sewer Maintenance",   "Sewer Maintenance",   "current",  "",         ""),
        # Sewer: 40 a month in 2025, 0 after; Sewer Maintenance 0 until Jan 2026
        ("Sewer",               "Sewer Maintenance",   "renamed",  "2026-01",  RENAMES["Sewer"][1]),
        # last row Dec 2025, more than 2 months before the data's last month (Mar 2026)
        ("Vending",             "Vending",             "current",  "",         "last used 2025-12"),
    ]
    assert list(catalog[["label", "category", "kind", "renamed_in", "note"]].itertuples(index=False, name=None)) == expected
    assert catalog.loc[catalog["label"] == "Heat/Hot Water", "category_id"].item() == "heat-hot-water"


def test_a_rename_must_name_labels_that_exist():
    with pytest.raises(ValueError, match="Potholes"):
        build_catalog(build_tables()[0], {"Potholes": ("Street Condition", "made up")})


@pytest.fixture
def catalog(fixture_dir):
    return CategoryCatalog.load(fixture_dir)


def test_lookup_falls_back_to_spelling_rules(catalog):
    assert catalog.lookup("heat/hot water").category == "HEAT/HOT WATER"  # unseen spelling
    assert catalog.lookup("heat/hot water").kind == "spelling"
    assert catalog.lookup("SEWER").category == "Sewer Maintenance"        # unseen spelling of an old name
    assert catalog.lookup("SEWER").kind == "renamed"
    unknown = catalog.lookup("Street Light Outage")
    assert (unknown.category, unknown.kind) == ("Street Light Outage", "unreviewed")


def test_find_accepts_names_old_labels_and_any_spelling(catalog):
    assert catalog.find("Sewer").category == "Sewer Maintenance"
    assert catalog.find("sewer maintenance").category == "Sewer Maintenance"
    assert catalog.find("Heat/Hot Water").category == "HEAT/HOT WATER"
    assert catalog.find("Potholes") is None
    assert catalog.suggest("Ilegal Parking") == ["Illegal Parking"]


def months(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="MS"))


@pytest.mark.parametrize("new, old, expected", [
    ([0, 0, 5, 9, 9], [9, 9, 0, 0, 0], "2024-03"),     # clean handover
    ([0, 3, 5, 0, 8], [9, 4, 6, 2, 1], "2024-05"),     # overlap: the last month behind is Apr
    ([0, 0, 0, 0, 7], [9, 0, 0, 0, 0], "2024-05"),     # seasonal gap before the new label starts
    ([0, 0, 0, 0, 0], [9, 9, 9, 9, 9], ""),            # never takes over
])
def test_takeover_month(new, old, expected):
    assert _takeover_month(months(new), months(old)) == expected


LIVE_CATALOG = LIVE_DIR / CATALOG_FILE
live = pytest.mark.skipif(not LIVE_CATALOG.exists(), reason="no live category catalog")


@live
def test_live_catalog_is_consistent():
    table = pd.read_csv(LIVE_CATALOG, keep_default_na=False)
    assert table["label"].is_unique
    assert (table.groupby("category_id")["category"].nunique() == 1).all()
    renamed = table[table["kind"] == "renamed"]
    assert set(renamed["label"]) == set(REVIEWED_RENAMES)
    assert (renamed["renamed_in"] != "").all()
    current = set(table.loc[table["kind"] == "current", "category"])
    assert set(renamed["category"]) <= current
    when = dict(zip(renamed["label"], renamed["renamed_in"]))
    assert (when["Sewer"], when["Smoking"], when["Snow"], when["Bike Rack Condition"]) == (
        "2026-08", "2024-04", "2021-10", "2024-08")


@live
def test_live_catalog_matches_a_fresh_build_from_the_reviewed_lists():
    data = pd.read_parquet(LIVE_DIR / "monthly_borough_category.parquet")
    fresh = build_catalog(data, REVIEWED_RENAMES, REVIEWED_NOTES).set_index("label")
    saved = pd.read_csv(LIVE_CATALOG, keep_default_na=False).set_index("label")
    columns = ["category_id", "category", "kind", "renamed_in"]
    assert fresh.loc[saved.index, columns].fillna("").equals(saved[columns]), (
        "category_catalog.csv is out of date: run python src/build_category_catalog.py")


@live
def test_every_live_label_is_in_the_catalog():
    labels = set(pd.read_parquet(LIVE_DIR / "monthly_borough_category.parquet")["complaint_type"])
    catalog = CategoryCatalog.load(LIVE_DIR)
    unseen = sorted(label for label in labels if catalog.lookup(label).kind == "unreviewed")
    assert not unseen, (
        f"New complaint labels since the catalog was built: {unseen}. Check them for renames "
        "in src/build_category_catalog.py, then run python src/build_category_catalog.py.")
