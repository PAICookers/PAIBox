import pytest
from pathlib import Path
import tempfile
import os
from paibox.libpaicore import *


@pytest.fixture
def ensure_dump_dir():
    p = Path(__file__).parent / "debug"

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    
    yield p
    # Clean up
    # for f in p.iterdir():
    #     f.unlink()


@pytest.fixture
def cleandir():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        yield
        os.chdir(old_cwd)