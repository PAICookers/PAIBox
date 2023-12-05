import os
import random
import tempfile
import numpy as np
from pathlib import Path

import pytest

from paibox.libpaicore.v2.frame.base import Frame
from paibox.libpaicore import Coord, ReplicationId as RId
from paibox.libpaicore.v2.frame.params import FrameFormat as FF, FrameHeader as FH


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


def gen_random_frame_common():
    chip_coord = Coord.from_addr(random.randint(0, FF.GENERAL_CHIP_ADDR_MASK))
    coord_coord = Coord.from_addr(random.randint(0, FF.GENERAL_CORE_ADDR_MASK))
    rid = RId.from_addr(random.randint(0, FF.GENERAL_CORE_EX_ADDR_MASK))

    return chip_coord, coord_coord, rid


@pytest.fixture
def gen_random_seed():
    random_seed = random.randint(0, FF.GENERAL_MASK)

    payload = np.array(
        [
            (random_seed >> 34) & FF.GENERAL_PAYLOAD_MASK,
            (random_seed >> 4) & FF.GENERAL_PAYLOAD_MASK,
            (random_seed & ((1 << 4) - 1)) << 26,
        ],
        dtype=np.uint64,
    )

    return payload


@pytest.fixture
def mock_gen_random_config1(gen_random_seed):
    f = Frame(FH.CONFIG_TYPE1, *gen_random_frame_common(), payload=gen_random_seed)

    return f
