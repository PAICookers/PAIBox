import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional, TypedDict

import numpy as np
import pytest
from typing_extensions import NotRequired

import paibox as pb
from paibox.naming import clear_name_cache


@pytest.fixture(scope="module")
def ensure_dump_dir():
    p = Path(__file__).parent / "debug"

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    else:
        for f in p.iterdir():
            f.unlink()

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


@pytest.fixture(autouse=True)
def clean_name_dict():
    """Clean the global name dictionary after each test automatically."""
    yield
    clear_name_cache(ignore_warn=True)


class ParametrizedTestData(TypedDict):
    """Parametrized test data in dictionary format."""

    args: str
    data: List[Any]
    ids: NotRequired[List[str]]


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

        n_list: pb.NodeList[pb.neuron.Neuron] = pb.NodeList()
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

        w = np.random.randint(-128, 127, (10, 10), dtype=np.int8)
        self.syn = pb.FullConn(
            self.pre_n, self.post_n, conn_type=pb.SynConnType.All2All, weights=w
        )


class Nested_Net_L2(pb.DynSysGroup):
    """Level 2 nested network: inp1 -> s1 -> Nested_Net_L1 -> s2 -> Nested_Net_L1"""

    def __init__(self, name: Optional[str] = None):
        self.inp1 = pb.InputProj(1, shape_out=(10,))
        subnet1 = Nested_Net_L1()
        subnet2 = Nested_Net_L1(name="Named_SubNet_L1_1")
        self.s1 = pb.FullConn(
            self.inp1,
            subnet1.pre_n,
            conn_type=pb.SynConnType.One2One,
        )
        self.s2 = pb.FullConn(
            subnet1.post_n,
            subnet2.pre_n,
            conn_type=pb.SynConnType.One2One,
        )

        super().__init__(subnet1, subnet2, name=name)
        self.probe1 = pb.Probe(self.inp1, "spike")  # won't be discovered in level 3


class Nested_Net_L3(pb.DynSysGroup):
    """Level 3 nested network: inp1 -> s1 -> Named_Nested_Net_L2"""

    def __init__(self):
        self.inp1 = pb.InputProj(1, shape_out=(10,))
        subnet1 = Nested_Net_L2(name="Named_Nested_Net_L2")

        subnet1_of_subnet1 = subnet1[f"{Nested_Net_L1.__name__}_0"]

        self.s1 = pb.FullConn(
            self.inp1,
            subnet1_of_subnet1.pre_n,
            conn_type=pb.SynConnType.One2One,
        )

        super().__init__(subnet1)

        self.probe1 = pb.Probe(self.inp1, "spike")
        self.probe2 = pb.Probe(subnet1_of_subnet1.pre_n, "spike")
        self.probe3 = pb.Probe(subnet1_of_subnet1.pre_n, "voltage")
        self.probe4 = pb.Probe(subnet1.s1, "output")


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
