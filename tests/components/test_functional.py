import pytest
import paibox as pb
import numpy as np

import paibox as pb
from paibox.base import DynamicSys
from paibox.components.modules import FunctionalModule
from paibox.network import DynSysGroup
from paibox.utils import as_shape


def assert_built_nodes(
    network: DynSysGroup, n_node_bef_build: int, n_node_aft_buld: int
):
    nodes = network.components.subset(DynamicSys).unique()
    assert len(nodes) == n_node_bef_build

    # Construct the functional modules
    network.module_construct()

    nodes.clear()
    # Must exclude `FunctionalModule`s, because it may be in the probe's `__dict__`.
    nodes = network.components.subset(DynamicSys).exclude(FunctionalModule).unique()
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

        assert_built_nodes(net, 4 + 1 + 2, 4 + 3 + 2)

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

        assert_built_nodes(net, 9 + 2 + 2, 9 + 2 * 3 + 2)

    def test_BitwiseAND(self, build_BitwiseAND_Net):
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

        assert_built_nodes(net, 6 + 1 + 2, 6 + 3 + 2)

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

        assert_built_nodes(net, 3 + 1 + 2, 3 + 2 + 2)

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

        assert_built_nodes(net, 6 + 1 + 2, 6 + 3 + 2)

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

        for i in range(6, 20):
            assert np.array_equal(sim.data[net.probe4][i], sim.data[net.probe3][i - 4])

        assert_built_nodes(net, 6 + 1 + 2, 6 + 5 + 2)

    def test_DelayChain(self, build_DelayChain_Net):
        net = build_DelayChain_Net
        bitwise = net.bitwise
        delay = net.func_node.external_delay
        sim = pb.Simulator(net, start_time_zero=False)

        inpa = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim.run(1)

        for i in range(2 + delay, 20):
            assert np.array_equal(sim.data[net.probe3][i], inpa[i - 2 - delay])

        assert_built_nodes(net, 3 + 1 + 2, 3 + 2 * net.func_node.inherent_delay + 2)

    def test_SpikingAdd(self, build_SpikingAdd_Net):
        net = build_SpikingAdd_Net
        sim = pb.Simulator(net, start_time_zero=False)

        _base_a = np.array(
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0] + [0] * 8, dtype=np.bool_
        )
        _base_b = np.array(
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1] + [0] * 8, dtype=np.bool_
        )
        _base_expected = np.array(
            [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.bool_
        )

        inpa = np.tile(_base_a, (10, 1)).T  # 20 * 12
        inpb = np.tile(_base_b, (10, 1)).T
        expected = np.tile(_base_expected, (10, 1)).T

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim.run(1)

        for i in range(1, 20):
            assert np.array_equal(sim.data[net.probe3][i], expected[i])

        assert_built_nodes(net, 6 + 1 + 2, 6 + 3 + 2)

    def test_SpikingSub(self, build_SpikingSub_Net):
        net = build_SpikingSub_Net
        sim = pb.Simulator(net, start_time_zero=False)

        _base_a = np.array(
            [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1] + [0] * 8, dtype=np.bool_
        )
        _base_b = np.array(
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0] + [0] * 8, dtype=np.bool_
        )
        _base_expected = np.array(
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool_
        )

        inpa = np.tile(_base_a, (10, 1)).T  # 20 * 12
        inpb = np.tile(_base_b, (10, 1)).T
        expected = np.tile(_base_expected, (10, 1)).T

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim.run(1)

        for i in range(1, 20):
            assert np.array_equal(sim.data[net.probe3][i], expected[i])

        assert_built_nodes(net, 6 + 1 + 2, 6 + 3 + 2)

    @pytest.mark.parametrize(
        "shape, channels, ksize, stride, fm_order, pool_type, p_binomial",
        [
            ((24, 24), 3, (3, 3), (3, 3), "CHW", "avg", 0.7),
            ((12, 12), 1, (2, 3), None, "CHW", "avg", 0.6),
            ((32, 32), 8, (3, 3), None, "CHW", "avg", 0.5),
            ((16, 16), 8, (5, 5), (2, 3), "CHW", "avg", 0.4),
            ((32, 32), 3, (3, 3), (2, 2), "CHW", "max", 0.2),
            ((24, 24), 1, (2, 3), None, "CHW", "max", 0.3),
            ((16, 16), 8, (5, 5), (2, 3), "CHW", "max", 0.4),
            ((32, 32), 8, (3, 3), (3, 4), "CHW", "max", 0.3),
            # ((3, 3), 3, (3, 3), (3, 3), "HWC", "avg", 0.7),
            # ((12, 12), 1, (2, 3), None, "HWC", "avg", 0.6),
            # ((32, 32), 8, (3, 3), None, "HWC", "avg", 0.5),
            # ((16, 16), 8, (5, 5), (2, 3), "HWC", "avg", 0.4),
            # ((32, 32), 3, (3, 3), (2, 2), "HWC", "max", 0.2),
            # ((24, 24), 1, (2, 3), None, "HWC", "max", 0.3),
            # ((16, 16), 8, (5, 5), (2, 3), "HWC", "max", 0.4),
            # ((32, 32), 8, (3, 3), (3, 4), "HWC", "max", 0.3),
        ],
    )
    def test_SpikingPool2d(
        self, shape, channels, ksize, stride, fm_order, pool_type, p_binomial
    ):
        from .conftest import SpikingPool2d_Net
        from .utils import avgpool2d_golden, maxpool2d_golden

        if fm_order == "CHW":
            fm_shape = (channels,) + shape
        else:
            fm_shape = shape + (channels,)

        net1 = SpikingPool2d_Net(
            fm_shape, ksize, stride, (0, 0), True, fm_order, pool_type
        )
        net2 = SpikingPool2d_Net(
            fm_shape, ksize, stride, (0, 0), False, fm_order, pool_type
        )
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        # Use binomial distribution to generate a sparse matrix with more zeros
        inpa = np.random.binomial(1, p_binomial, size=(20,) + fm_shape).astype(np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(1, 20):
            if pool_type == "avg":
                expected = avgpool2d_golden(
                    inpa[i - 1], ksize, stride, (0, 0), fm_order
                ).ravel()
            else:
                expected = maxpool2d_golden(
                    inpa[i - 1], ksize, stride, (0, 0), fm_order
                ).ravel()

            assert np.array_equal(sim1.data[net1.probe2][i], expected)
            assert np.array_equal(sim2.data[net2.probe2][i], expected)

    @pytest.mark.parametrize("shape", [(32, 16), (1, 32), (64,), (128, 1), 48])
    def test_Transpose2d(self, shape):
        from .conftest import TransposeModule_T2d_Net

        net = TransposeModule_T2d_Net(shape)
        sim = pb.Simulator(net, start_time_zero=False)

        inpa = np.random.randint(0, 2, size=(20,) + as_shape(shape), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim.run(1)

        for i in range(2, 20):
            expected = inpa[i - 2].T.ravel()
            assert np.array_equal(sim.data[net.probe2][i], expected)

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
        from .conftest import TransposeModule_T3d_Net

        net = TransposeModule_T3d_Net(shape, axes)
        sim = pb.Simulator(net, start_time_zero=False)

        if len(shape) == 2:
            shape = (1,) + shape

        inpa = np.random.randint(0, 2, size=(20,) + as_shape(shape), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim.run(1)

        for i in range(2, 20):
            expected = inpa[i - 2].transpose(axes).ravel()
            assert np.array_equal(sim.data[net.probe2][i], expected)
