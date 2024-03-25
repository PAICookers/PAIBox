import pytest
import paibox as pb
import numpy as np

import paibox as pb
from paibox.base import DynamicSys
from paibox.components.modules import FunctionalModule
from paibox.network import DynSysGroup
from paibox.utils import as_shape

from .conftest import TransposeModule_T2d_Net, TransposeModule_T3d_Net


def assert_nodes(network: DynSysGroup, n_node_bef_build: int, n_node_aft_buld: int):
    nodes = (
        network.nodes(include_self=False, find_recursive=True)
        .subset(DynamicSys)
        .unique()
    )
    assert len(nodes) == n_node_bef_build

    # Build the functional modules
    network.build()

    nodes.clear()
    # Must exclude `FunctionalModule`s, because it may be in the probe's `__dict__`.
    nodes = (
        network.nodes(include_self=False, find_recursive=True)
        .subset(DynamicSys)
        .exclude(FunctionalModule)
        .unique()
    )
    assert len(nodes) == n_node_aft_buld


class TestFunctionalModules:
    def test_FModule_ConnWithInput(self, build_FModule_ConnWithInput_Net):
        net = build_FModule_ConnWithInput_Net
        bitwise = 10
        sim = pb.Simulator(net, start_time_zero=False)

        inpa = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)

        # data2 will input to inp2 which is connected with the AND module.
        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim.run(1)

        for i in range(1, 20):
            assert np.array_equal(sim.data[net.probe2][i], inpa[i - 1] & inpb[i])

        assert_nodes(net, 4 + 1 + 2, 4 + 3 + 2)

    def test_FModule_ConnWithModule(self, build_FModule_ConnWithModule_Net):
        net = build_FModule_ConnWithModule_Net
        bitwise = 10
        sim = pb.Simulator(net, start_time_zero=False)

        inpa = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)
        inpc = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)

        for t in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[t], data2=inpb[t], data3=inpc[t])
            sim.run(1)

        # At T >= 1, the AND1 will output the valid result.
        for t in range(1, 20):
            assert np.array_equal(sim.data[net.probe2][t], inpa[t - 1] & inpb[t - 1])

        # At T >= 2, the OR1 will output the valid result.
        for t in range(2, 20):
            assert np.array_equal(
                sim.data[net.probe3][t], (inpa[t - 2] & inpb[t - 2]) | inpc[t - 1]
            )

        assert_nodes(net, 9 + 2 + 2, 9 + 2 * 3 + 2)

    def test_BitwiseAND_function(self, build_BitwiseAND_Net):
        net: pb.Network = build_BitwiseAND_Net
        bitwise = net.bitwise
        sim = pb.Simulator(net, start_time_zero=False)

        inpa = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim.run(1)

        for i in range(1, 20):
            assert np.array_equal(sim.data[net.probe3][i], inpa[i - 1] & inpb[i - 1])

        assert_nodes(net, 6 + 1 + 2, 6 + 3 + 2)

    def test_BitwiseNOT(self, build_BitwiseNOT_Net):
        net: pb.Network = build_BitwiseNOT_Net
        bitwise = net.bitwise
        sim = pb.Simulator(net, start_time_zero=False)

        inpa = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim.run(1)

        for i in range(1, 20):
            assert np.array_equal(sim.data[net.probe2][i], ~inpa[i - 1])

        assert_nodes(net, 3 + 1 + 2, 3 + 2 + 2)

    def test_BitwiseOR(self, build_BitwiseOR_Net):
        net: pb.Network = build_BitwiseOR_Net
        bitwise = net.bitwise
        sim = pb.Simulator(net, start_time_zero=False)

        inpa = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim.run(1)

        for i in range(1, 20):
            assert np.array_equal(sim.data[net.probe3][i], inpa[i - 1] | inpb[i - 1])

        assert_nodes(net, 6 + 1 + 2, 6 + 3 + 2)

    def test_BitwiseXOR(self, build_BitwiseXOR_Net):
        net: pb.Network = build_BitwiseXOR_Net
        bitwise = net.bitwise
        sim = pb.Simulator(net, start_time_zero=False)

        inpa = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim.run(1)

        for i in range(2, 20):
            assert np.array_equal(sim.data[net.probe3][i], inpa[i - 2] ^ inpb[i - 2])

        assert_nodes(net, 6 + 1 + 2, 6 + 5 + 2)

    @pytest.mark.parametrize("shape", [(32, 16), (1, 32), (64,), (128, 1), 48])
    def test_Transpose2d(self, shape):
        net = TransposeModule_T2d_Net(shape)
        shape = net.inp1.shape_out

        sim = pb.Simulator(net, start_time_zero=False)
        inpa = np.random.randint(0, 2, size=(20,) + as_shape(shape), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim.run(1)

        for i in range(2, 20):
            assert np.array_equal(sim.data[net.probe2][i], inpa[i - 2].T.flatten())

    @pytest.mark.parametrize(
        "shape, axes",
        [
            ((32, 16, 24), (1, 2, 0)),
            ((12, 32, 32), None),
            ((28, 28), (2, 0, 1)),
            ((128, 1, 24), (0, 2, 1)),
        ],
    )
    def test_Transpose3d(self, shape, axes):
        net = TransposeModule_T3d_Net(shape, axes)
        sim = pb.Simulator(net, start_time_zero=False)

        if len(shape) == 2:
            shape = (1,) + shape

        inpa = np.random.randint(0, 2, size=(20,) + as_shape(shape), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim.run(1)

        for i in range(2, 20):
            assert np.array_equal(
                sim.data[net.probe2][i], inpa[i - 2].transpose(axes).flatten()
            )
