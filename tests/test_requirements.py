"""The Docker image installs requirements-app.txt; Streamlit Cloud installs requirements.txt.
Every package the app needs has to be in both, at the same version."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pins(name: str) -> dict[str, str]:
    lines = (ROOT / name).read_text().splitlines()
    return dict(line.split("==") for line in lines if "==" in line and not line.startswith("#"))


def test_app_requirements_match_the_main_ones():
    app, main = pins("requirements-app.txt"), pins("requirements.txt")
    assert {name: main.get(name) for name in app} == app


def test_the_image_leaves_out_the_training_libraries():
    assert not {"scikit-learn", "scipy", "matplotlib", "joblib"} & set(pins("requirements-app.txt"))
