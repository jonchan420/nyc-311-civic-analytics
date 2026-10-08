"""Fetch the inputs for the reporting-gap analysis into data/equity/.

Four public sources, no API keys needed:
  - NYC 311 (Socrata API): HPD housing-maintenance complaints filed
    2022-2024, counted by incident ZIP, borough, and reporting channel.
  - Census ACS 2020-2024 5-year estimates, streamed from the Census
    Bureau's bulk table files (its data API now requires a key; these
    files don't) and filtered to NYC ZIP Code Tabulation Areas (ZCTAs).
  - NYC Health's Modified ZCTAs (MODZCTA): the ZIP-to-neighborhood
    crosswalk, neighborhood names, and boundaries for the map.
  - NYCHA (Socrata API): apartments per public-housing development and
    each building's ZIP and tax lot. NYCHA tenants report repairs to
    NYCHA, not HPD, so the analysis leaves them out of the complaint
    rate; HPD complaints filed at NYCHA's own lots are fetched to check
    that.

The 2022-2024 complaint window sits inside the ACS 2020-2024 window and
skips the worst COVID distortion of 2020-2021. Outputs are small and
committed, so the analysis itself runs offline.

Run once:
    python src/fetch_reporting_gap_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import API_BASE, DATA_DIR

EQUITY_DIR = DATA_DIR / "equity"
SOCRATA = "https://data.cityofnewyork.us/resource"
MODZCTA_ID = "pri4-ifjk"
NEIGHBORHOOD_NAMES_ID = "6qs8-44ki"
NYCHA_DATA_BOOK_ID = "evjd-dqpz"
NYCHA_ADDRESSES_ID = "3ub5-4ph8"
ACS_BULK = ("https://www2.census.gov/programs-surveys/acs/summary_file/2024/"
            "table-based-SF/data/5YRData/acsdt5y2024-{table}.dat")
ZCTA_PREFIX = "860Z200US"
COMPLAINT_START, COMPLAINT_END = "2022-01-01", "2025-01-01"

# Summary-file columns (TABLE_Ennn = estimate on line nnn) summed into each
# measure. Line numbers checked against the ACS 2024 table shells and the
# Census API variable metadata (e.g. B16002 line 22 = Chinese, limited
# English speaking household).
ACS_MEASURES: dict[str, list[str]] = {
    "total_population": ["B01003_E001"],
    "households": ["B16002_E001"],
    "lep_spanish": ["B16002_E004"],
    "lep_chinese": ["B16002_E022"],
    # Korean, Vietnamese, Tagalog, other Asian and Pacific Island languages
    "lep_other_asian": ["B16002_E019", "B16002_E025", "B16002_E028", "B16002_E031"],
    # French/Haitian/Cajun, German, Russian/Polish/Slavic, other
    # Indo-European, Arabic, other and unspecified
    "lep_other": ["B16002_E007", "B16002_E010", "B16002_E013",
                  "B16002_E016", "B16002_E034", "B16002_E037"],
    "internet_universe": ["B28002_E001"],
    "no_internet": ["B28002_E013"],
    "renter_households": ["B25003_E003"],
    "renter_units_year_built_universe": ["B25036_E013"],
    "renter_units_pre1950": ["B25036_E022", "B25036_E023"],
    "renter_units_occupancy_universe": ["B25014_E008"],
    "renter_units_overcrowded": ["B25014_E011", "B25014_E012", "B25014_E013"],
    "renter_units_structure_universe": ["B25032_E013"],
    "renter_units_1_to_4": ["B25032_E014", "B25032_E015", "B25032_E016", "B25032_E017"],
    "poverty_universe": ["B17001_E001"],
    "below_poverty": ["B17001_E002"],
    "income_universe": ["B19001_E001"],
    "income_150k_plus": ["B19001_E016", "B19001_E017"],
    "age_universe": ["B01001_E001"],
    "age_65_plus": [f"B01001_E{line:03d}" for line in (*range(20, 26), *range(44, 50))],
}


def fetch_modzcta() -> tuple[pd.DataFrame, dict]:
    geo = requests.get(f"{SOCRATA}/{MODZCTA_ID}.geojson", params={"$limit": 1000}, timeout=300)
    geo.raise_for_status()
    features = [f for f in geo.json()["features"] if f["properties"]["modzcta"] != "99999"]

    names = requests.get(f"{SOCRATA}/{NEIGHBORHOOD_NAMES_ID}.json",
                         params={"$select": "modzcta_first, neighborhood_name", "$limit": 1000},
                         timeout=300)
    names.raise_for_status()
    name_by_modzcta = {r["modzcta_first"]: r["neighborhood_name"] for r in names.json()}

    rows = []
    for feature in features:
        props = feature["properties"]
        for zcta in props["zcta"].split(","):
            rows.append({"modzcta": props["modzcta"], "zcta": zcta.strip(),
                         "neighborhood": name_by_modzcta.get(props["modzcta"], props["label"])})
    crosswalk = pd.DataFrame(rows)
    if crosswalk["zcta"].duplicated().any():
        raise RuntimeError("A ZCTA maps to more than one MODZCTA.")

    def shrink(coords):
        # 5 decimal places is ~1 m; drop points that collapse onto their neighbor
        if coords and isinstance(coords[0], (int, float)):
            return [round(coords[0], 5), round(coords[1], 5)]
        out = [shrink(c) for c in coords]
        if out and isinstance(out[0][0], float):
            out = [p for i, p in enumerate(out) if i == 0 or p != out[i - 1]]
        return out

    boundaries = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature",
                      "properties": {"modzcta": f["properties"]["modzcta"]},
                      "geometry": {"type": f["geometry"]["type"],
                                   "coordinates": shrink(f["geometry"]["coordinates"])}}
                     for f in features],
    }
    print(f"MODZCTA: {crosswalk['modzcta'].nunique()} neighborhoods, {len(crosswalk)} ZCTAs, "
          f"{sum(m in name_by_modzcta for m in crosswalk['modzcta'].unique())} with names")
    return crosswalk, boundaries


def fetch_acs(zctas: set[str]) -> pd.DataFrame:
    tables = sorted({column.split("_")[0].lower() for cols in ACS_MEASURES.values() for column in cols})
    by_table: dict[str, pd.DataFrame] = {}
    for table in tables:
        response = requests.get(ACS_BULK.format(table=table), stream=True, timeout=600)
        response.raise_for_status()
        response.encoding = "utf-8"  # server sends no charset; iter_lines would yield bytes
        lines = response.iter_lines(decode_unicode=True)
        header = next(lines).split("|")
        kept = [line.split("|") for line in lines
                if line.startswith(ZCTA_PREFIX) and line.split("|", 1)[0][len(ZCTA_PREFIX):] in zctas]
        frame = pd.DataFrame(kept, columns=header)
        frame["zcta"] = frame["GEO_ID"].str[len(ZCTA_PREFIX):]
        by_table[table] = frame.set_index("zcta")
        print(f"  {table}: {len(frame)} NYC ZCTAs")

    out = pd.DataFrame(index=sorted(zctas))
    for measure, columns in ACS_MEASURES.items():
        total = 0
        for column in columns:
            values = by_table[column.split("_")[0].lower()][column]
            total = total + pd.to_numeric(values, errors="raise").reindex(out.index)
        out[measure] = total
    missing = out.index[out.isna().any(axis=1)].tolist()
    if missing:
        print(f"  ZCTAs absent from the ACS (no published estimates; counted as 0): {missing}")
    out = out.fillna(0).astype(int)
    if (out < 0).any().any():
        raise RuntimeError("Negative ACS count; a sentinel value slipped through.")
    return out.rename_axis("zcta").reset_index()


def fetch_hpd_complaints() -> pd.DataFrame:
    response = requests.get(API_BASE, params={
        "$select": "incident_zip, borough, open_data_channel_type AS channel, count(*) AS complaints",
        "$where": (f"agency = 'HPD' AND created_date >= '{COMPLAINT_START}T00:00:00' "
                   f"AND created_date < '{COMPLAINT_END}T00:00:00'"),
        "$group": "incident_zip, borough, open_data_channel_type",
        "$limit": 50000,
    }, timeout=900)
    response.raise_for_status()
    complaints = pd.DataFrame(response.json())
    complaints["complaints"] = complaints["complaints"].astype(int)
    return complaints


def fetch_nycha() -> tuple[pd.DataFrame, pd.DataFrame]:
    """NYCHA apartments by development and tax lot, and HPD complaints at those lots.

    A development converted to private management (RAD/PACT) counts as
    NYCHA only for the share of the complaint window before its transfer;
    after that its tenants can file with HPD. Each development's apartments
    are split across its tax lots by number of building addresses.
    Developments converted since and dropped from NYCHA's current address
    file can't be placed and keep a blank ZIP.
    """
    book = requests.get(f"{SOCRATA}/{NYCHA_DATA_BOOK_ID}.json", params={
        "$select": "development, tds_, program, number_of_current_apartments, rad_transferred_date",
        "$limit": 5000}, timeout=300)
    book.raise_for_status()
    book = pd.DataFrame(book.json())
    # a few rows add up others ("DOUGLASS" = "DOUGLASS I" + "II", development
    # numbers "082, 582"; "RED HOOK I" and "II" overlap "RED HOOK EAST" and
    # "WEST") and would double count; every operating row has one plain number
    book = book[book["tds_"].str.fullmatch(r"\d+", na=False)
                & ~book["program"].str.contains("NON-NYCHA", na=False)].copy()
    book["tds"] = book["tds_"].str.lstrip("0")  # zero-padded here, not in the address file
    if book["tds"].duplicated().any():
        raise RuntimeError("A NYCHA development number appears in more than one data book row.")

    addresses = requests.get(f"{SOCRATA}/{NYCHA_ADDRESSES_ID}.json", params={
        "$select": "tds, borough_block_lot, zip_code", "$limit": 50000}, timeout=300)
    addresses.raise_for_status()
    addresses = pd.DataFrame(addresses.json())
    addresses["tds"] = addresses["tds"].str.lstrip("0")
    unknown = set(addresses["tds"]) - set(book["tds"])
    if unknown:
        raise RuntimeError(f"NYCHA addresses with no development in the data book: {sorted(unknown)}")
    lot_share = (addresses.groupby(["tds", "borough_block_lot", "zip_code"]).size()
                 / addresses.groupby("tds").size()).rename("share").reset_index()

    start, end = pd.Timestamp(COMPLAINT_START), pd.Timestamp(COMPLAINT_END)
    transferred = pd.to_datetime(book["rad_transferred_date"])
    book["managed_share"] = ((transferred.clip(start, end).fillna(end) - start) / (end - start)).round(4)
    book["rad_transferred_date"] = transferred.dt.strftime("%Y-%m-%d")
    nycha = book.merge(lot_share, on="tds", how="left").rename(columns={"borough_block_lot": "bbl"})
    nycha["apartments"] = (pd.to_numeric(nycha["number_of_current_apartments"].str.replace(",", ""))
                           * nycha["share"].fillna(1)).round(2)
    nycha = (nycha[["development", "bbl", "zip_code", "apartments", "rad_transferred_date", "managed_share"]]
             .sort_values(["development", "bbl", "zip_code"]).reset_index(drop=True))

    # lots NYCHA managed for the whole window, to check its tenants don't file with HPD
    throughout = nycha.groupby("bbl")["managed_share"].min() == 1
    lots = nycha[nycha["bbl"].isin(throughout[throughout].index)].groupby("bbl")["apartments"].sum()
    counts = []
    for i in range(0, len(lots), 100):
        listed = ", ".join(f"'{bbl}'" for bbl in lots.index[i:i + 100])
        response = requests.get(API_BASE, params={
            "$select": "bbl, count(*) AS complaints",
            "$where": (f"agency = 'HPD' AND created_date >= '{COMPLAINT_START}T00:00:00' "
                       f"AND created_date < '{COMPLAINT_END}T00:00:00' AND bbl IN ({listed})"),
            "$group": "bbl",
            "$limit": 5000,
        }, timeout=900)
        response.raise_for_status()
        counts.extend(response.json())
    counts = pd.DataFrame(counts, columns=["bbl", "complaints"]).astype({"complaints": int})
    nycha_lots = lots.round(2).reset_index().merge(counts, on="bbl", how="left")
    nycha_lots["hpd_complaints"] = nycha_lots.pop("complaints").fillna(0).astype(int)

    managed = nycha["apartments"] * nycha["managed_share"]
    print(f"  {book['development'].nunique()} developments; {managed.sum():,.0f} apartments NYCHA-managed "
          f"on average over the window, {managed[nycha['zip_code'].notna()].sum() / managed.sum():.1%} "
          "placed by ZIP")
    print(f"  {nycha_lots['hpd_complaints'].sum():,} HPD complaints at the {len(nycha_lots)} lots NYCHA "
          f"managed throughout ({nycha_lots['apartments'].sum():,.0f} apartments)")
    return nycha, nycha_lots


def main() -> None:
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching MODZCTA crosswalk, names, and boundaries...")
    crosswalk, boundaries = fetch_modzcta()

    print("Streaming ACS 2020-2024 bulk tables (~720 MB total, filtered as it streams)...")
    acs = fetch_acs(set(crosswalk["zcta"]))
    print(f"  NYC population in these ZCTAs: {acs['total_population'].sum():,}")

    print("Fetching HPD complaints, 2022-2024...")
    complaints = fetch_hpd_complaints()
    mapped = complaints["incident_zip"].isin(crosswalk["zcta"])
    print(f"  {complaints['complaints'].sum():,} complaints; "
          f"{complaints.loc[mapped, 'complaints'].sum() / complaints['complaints'].sum():.2%} "
          "have a ZIP inside a NYC neighborhood")

    print("Fetching NYCHA apartments and HPD complaints at NYCHA lots...")
    nycha, nycha_lots = fetch_nycha()

    crosswalk.to_csv(EQUITY_DIR / "modzcta_crosswalk.csv", index=False)
    (EQUITY_DIR / "modzcta_boundaries.geojson").write_text(json.dumps(boundaries))
    acs.to_csv(EQUITY_DIR / "acs_2024_nyc_zcta.csv", index=False)
    complaints.to_csv(EQUITY_DIR / "hpd_complaints_2022_2024_by_zip.csv", index=False)
    nycha.to_csv(EQUITY_DIR / "nycha_apartments.csv", index=False)
    nycha_lots.to_csv(EQUITY_DIR / "nycha_lots_hpd_complaints.csv", index=False)
    print(f"Saved to {EQUITY_DIR}")


if __name__ == "__main__":
    main()
