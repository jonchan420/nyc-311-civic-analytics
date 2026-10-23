from pathlib import Path

import pytest

from tests.fixtures.make_fixtures import write_fixtures


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh copy of the synthetic data/live/ tables, shared by read-only tests."""
    return write_fixtures(tmp_path_factory.mktemp("live"))
