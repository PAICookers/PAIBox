import os
import random
import tempfile
from pathlib import Path

import pytest

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


@pytest.fixture
def gen_random_params_reg_dict():
    wp = random.choice(list(WeightPrecision))
    lcn_ex = random.choice(list(LCN_EX))
    iwf = random.choice(list(InputWidthFormat))
    swf = random.choice(list(SpikeWidthFormat))
    num_den = random.randint(1, 512)
    mpe = random.choice(list(MaxPoolingEnable))
    tws = random.randint(0, 100)
    twe = random.randint(0, 100)
    sme = random.choice(list(SNNModeEnable))
    target_lcn = random.choice(list(LCN_EX))
    test_chip_addr = Coord(random.randint(0, 31), random.randint(0, 31))

    return dict(
        {
            "weight_width": wp.value,
            "LCN": lcn_ex.value,
            "input_width": iwf.value,
            "spike_width": swf.value,
            "neuron_num": num_den,
            "pool_max": mpe.value,
            "tick_wait_start": tws,
            "tick_wait_end": twe,
            "snn_en": sme.value,
            "target_LCN": target_lcn.value,
            "test_chip_addr": test_chip_addr.address,
        }
    )


@pytest.fixture
def gen_random_neuron_attr_dict():
    pass


@pytest.fixture
def gen_random_dest_info_dict():
    pass
