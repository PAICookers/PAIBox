from pathlib import Path

import pytest
from paicorelib import CoordXY

from .helpers import make_core_frame, write_pb


@pytest.fixture
def valid_pb(tmp_path: Path) -> Path:
    """Create the smallest valid single-offline-core artifact."""
    return write_pb(tmp_path / "config.pb", make_core_frame(CoordXY(3, 1)))
