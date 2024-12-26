import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy as np
import pytest

import paibox as pb
from paibox.base import SynSys
from paibox.components import Neuron
from paibox.naming import clear_name_cache

from .shared_networks import *
from .utils import *

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired


# Import the logging hooks from logging_utils
from ._logging.logging_utils import captured_logs, log_settings_patch


# Add custom markers to eliminate pytest warning
def pytest_configure(config: pytest.Config):
    config.addinivalue_line(
        "markers",
        "make_settings_test(**settings_dict): mark test to set custom settings for logging & perform teardown.",
    )


@pytest.fixture(scope="module")
def ensure_dump_dir():
    p = Path(__file__).parent / "debug"

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    else:
        for f in p.iterdir():
            f.unlink()

    yield p


@pytest.fixture(scope="module")
def ensure_dump_dir_and_clean():
    p = Path(__file__).parent / "debug"

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    else:
        for f in p.iterdir():
            f.unlink()

    yield p

    # Clean up
    for f in p.iterdir():
        f.unlink()


@pytest.fixture
def cleandir():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        yield
        os.chdir(old_cwd)


def _reset_context() -> None:
    clear_name_cache(ignore_warn=True)
    pb.FRONTEND_ENV["t"] = 0
    pb.BACKEND_CONFIG.set_default()
    # To avoid overlapping with multi-chip coordinates
    pb.BACKEND_CONFIG.output_chip_addr = (9, 9)
    SynSys.CFLAG_ENABLE_WP_OPTIMIZATION = True


@pytest.fixture(autouse=True)
def context_reset():
    """Reset the context after each test automatically."""
    _reset_context()
    yield
    _reset_context()


@pytest.fixture
def perf_fixture(request):
    with measure_time(f"{request.node.name}"):
        yield


@pytest.fixture
def fixed_rng() -> np.random.Generator:
    return np.random.default_rng(42)


class ParametrizedTestData(TypedDict):
    """Parametrized test data in dictionary format."""

    args: str
    data: list[Any]
    ids: NotRequired[list[str]]


class Input_to_N1(pb.DynSysGroup):
    """Not nested network
    inp1 -> n1 -> s1 -> n2, n3
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(1,))
        self.n1 = pb.TonicSpiking(1, 3, tick_wait_start=2, delay=1)
        self.s1 = pb.FullConn(
            self.inp1, self.n1, weights=1, conn_type=pb.SynConnType.One2One
        )

        self.probe1 = pb.Probe(self.s1, "output", name="s2_out")
        self.probe2 = pb.Probe(self.n1, "delay_registers", name="n1_reg")
        self.probe3 = pb.Probe(self.n1, "spike", name="n1_spike")
        self.probe4 = pb.Probe(self.n1, "voltage", name="n1_v")


class NotNested_Net_Exp(pb.DynSysGroup):
    """Not nested network
    inp1 -> n1 -> s1 -> n2
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(1,))
        self.n1 = pb.TonicSpiking(1, 2, tick_wait_start=2, delay=3)
        self.n2 = pb.TonicSpiking(1, 2, tick_wait_start=3)

        self.s1 = pb.FullConn(
            self.inp1, self.n1, weights=1, conn_type=pb.SynConnType.One2One
        )
        self.s2 = pb.FullConn(
            self.n1, self.n2, weights=1, conn_type=pb.SynConnType.All2All
        )

        self.probe1 = pb.Probe(self.s2, "output", name="s2_out")
        self.probe2 = pb.Probe(self.n1, "delay_registers", name="n1_reg")
        self.probe3 = pb.Probe(self.n1, "spike", name="n1_spike")
        self.probe4 = pb.Probe(self.n1, "voltage", name="n1_v")
        self.probe5 = pb.Probe(self.n2, "spike", name="n2_spike")
        self.probe6 = pb.Probe(self.n2, "voltage", name="n2_v")


class Network_with_container(pb.DynSysGroup):
    """Network with neurons in list.

    n_list[0] -> s1 -> n_list[1] -> s2 -> n_list[2]
    """

    def __init__(self):
        super().__init__()

        self.inp = pb.InputProj(1, shape_out=(3,))

        n1 = pb.TonicSpiking((3,), 2)
        n2 = pb.TonicSpiking((3,), 3)
        n3 = pb.TonicSpiking((3,), 4)

        n_list: pb.NodeList[Neuron] = pb.NodeList()
        n_list.append(n1)
        n_list.append(n2)
        n_list.append(n3)
        self.n_list = n_list

        self.s1 = pb.FullConn(n_list[0], n_list[1], conn_type=pb.SynConnType.All2All)
        self.s2 = pb.FullConn(n_list[1], n_list[2], conn_type=pb.SynConnType.All2All)

        self.probe1 = pb.Probe(self.n_list[1], "output", name="n2_out")


class Network_with_multi_inodes_onodes(pb.Network):
    """
    INP1 -> S1 -> N1 -> S2 -> N2
    INP2 -> S3 -> N1 -> S4 -> N3
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(40,), name="inp1")
        self.inp2 = pb.InputProj(input=1, shape_out=(50,), name="inp2")
        self.n1 = pb.TonicSpiking(80, 2, name="n1", tick_wait_start=1)
        self.n2 = pb.TonicSpiking(20, 3, name="n2", tick_wait_start=2)
        self.n3 = pb.TonicSpiking(30, 3, name="n3", tick_wait_start=2)

        self.s1 = pb.FullConn(
            self.inp1, self.n1, conn_type=pb.SynConnType.All2All, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.inp2, self.n1, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.FullConn(
            self.n1, self.n3, conn_type=pb.SynConnType.All2All, name="s4"
        )


class Nested_Net_L1(pb.DynSysGroup):
    """Level 1 nested network: pre_n -> syn -> post_n"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

        self.pre_n = pb.LIF((10,), 10)
        self.post_n = pb.LIF((10,), 10)

        self.syn = pb.FullConn(
            self.pre_n,
            self.post_n,
            weights=np.random.randint(-128, 127, (10, 10), dtype=np.int8),
        )

        self.probe1 = pb.Probe(self.post_n, "spike")


class Nested_Net_L2(pb.DynSysGroup):
    """Level 2 nested network: n1 -> s1 -> subnet1 -> s2 -> subnet2"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

        self.n1 = pb.IF((10,), 1)
        self.subnet1 = Nested_Net_L1()
        self.subnet2 = Nested_Net_L1()
        self.s1 = pb.FullConn(self.n1, self.subnet1.pre_n)
        self.s2 = pb.FullConn(self.subnet1.post_n, self.subnet2.pre_n)

        self.probe1 = pb.Probe(self.n1, "spike")
        self.probe2 = pb.Probe(self.subnet1.pre_n, "spike")


class Nested_Net_L3(pb.DynSysGroup):
    """Level 3 nested network: inp1 -> s1 -> subnet_L2_1"""

    def __init__(self):
        self.inp1 = pb.InputProj(1, shape_out=(10,))
        subnet1 = Nested_Net_L2(name="subnet_L2_1")
        self.s1 = pb.FullConn(self.inp1, subnet1.n1)

        super().__init__(subnet1)

        self.probe1 = pb.Probe(self.inp1, "spike", name="pb_L3_1")
        self.probe2 = pb.Probe(subnet1.n1, "spike", name="pb_L3_2")
        self.probe3 = pb.Probe(subnet1.s1, "output", name="pb_L3_3")


@pytest.fixture(scope="class")
def build_Input_to_N1():
    return Input_to_N1()


@pytest.fixture(scope="class")
def build_NotNested_Net():
    return Input_to_N1()


@pytest.fixture(scope="class")
def build_NotNested_Net_Exp():
    return NotNested_Net_Exp()


@pytest.fixture(scope="class")
def build_Network_with_container():
    return Network_with_container()


@pytest.fixture(scope="class")
def build_multi_inodes_onodes():
    return Network_with_multi_inodes_onodes()


@pytest.fixture(scope="class")
def build_Nested_Net_L1():
    return Nested_Net_L1()


@pytest.fixture(scope="class")
def build_Nested_Net_L2():
    return Nested_Net_L2()


@pytest.fixture(scope="class")
def build_Nested_Net_L3():
    return Nested_Net_L3()
