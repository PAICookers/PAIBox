from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--clear-data",
        action="store_true",
        default=False,
        help="Clear the data directory for the current test item.",
    )


@pytest.fixture(scope="module", autouse=True)
def ensure_test_root_dirs(request: pytest.FixtureRequest):
    test_dir = request.path.parent
    (test_dir / "data").mkdir(exist_ok=True)
    (test_dir / "config").mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def ensure_test_item_dirs(request: pytest.FixtureRequest):
    """Create data & config directories for every test module."""
    test_dir = request.path.parent
    func_name: str = request.node.name
    data_subdir: Path = test_dir / "data" / func_name
    config_subdir: Path = test_dir / "config" / func_name

    if not data_subdir.exists():
        data_subdir.mkdir()
    elif request.config.getoption("--clear-data"):
        for f in data_subdir.iterdir():
            f.unlink()
    else:
        pass  # the test will use the existing data

    if not config_subdir.exists():
        config_subdir.mkdir()
    else:
        pass  # will be overwritten every time

    return data_subdir, config_subdir
