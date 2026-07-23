"""Central paths and constants for the NYC 311 project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FORECAST_DIR = DATA_DIR / "forecasts"
MODEL_DIR = PROJECT_ROOT / "models"
VISUAL_DIR = PROJECT_ROOT / "visuals"

for directory in [RAW_DIR, PROCESSED_DIR, FORECAST_DIR, MODEL_DIR, VISUAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# path to the raw 14 GB csv you downloaded.
# EDIT THIS to wherever the file actually lives on your macbook.
# do NOT put it inside the repo folder.
RAW_CSV_PATH = Path("/Users/jonathanchan/Downloads/311_Service_Requests_from_2020_to_Present_20260722.csv")
DATASET_ID = "erm2-nwe9"
API_BASE = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"

VALID_BOROUGHS = {"BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"}

START_YEAR = 2020
