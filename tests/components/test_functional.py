import numpy as np
import pytest

import paibox as pb
from paibox.base import DynamicSys
from paibox.components import NeuModule
from paibox.components.synapses.conv_utils import _pair
from paibox.network import DynSysGroup
from paibox.utils import as_shape, shape2num, typical_round


def _assert_build_fmodule(
    network: DynSysGroup, n_node_bef_build: int, n_node_aft_build: int
):
    nodes = network.components.subset(DynamicSys).unique()
    assert len(nodes) == n_node_bef_build

    # Construct the functional modules
    DynSysGroup.build_fmodule(network)

    # Must exclude `NeuModule`s, because it may be in the probe's `__dict__`.
    nodes = network.components.subset(DynamicSys).exclude(NeuModule).unique()
    assert len(nodes) == n_node_aft_build


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

        _assert_build_fmodule(net, 4 + 1 + 2, 4 + 3 + 2)

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

        _assert_build_fmodule(net, 9 + 2 + 2, 9 + 2 * 3 + 2)

    def test_BitwiseAND(self):
        from tests.shared_networks import FunctionalModule_2to1_Net

        net1 = FunctionalModule_2to1_Net("and")
        net2 = FunctionalModule_2to1_Net("and")
        bitwise = net1.bitwise
        func = net2.func_node
        generated = DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_func = pb.Probe(generated[func][0], "spike")
        sim2.add_probe(probe_func)

        inpa = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(1, 20):
            assert np.array_equal(sim1.data[net1.probe3][i], inpa[i - 1] & inpb[i - 1])
            assert np.array_equal(sim2.data[probe_func][i], inpa[i - 1] & inpb[i - 1])

        for i in range(2, 20):
            assert np.array_equal(sim1.data[net1.probe4][i], inpa[i - 2] & inpb[i - 2])

        _assert_build_fmodule(net1, 6 + 1 + 2, 6 + 3 + 2)

    def test_BitwiseAND_mapping(self, ensure_dump_dir):
        from tests.shared_networks import FunctionalModule_2to1_Net

        net1 = FunctionalModule_2to1_Net("and")

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

    def test_BitwiseNOT(self):
        from tests.shared_networks import FunctionalModule_1to1_Net

        net1 = FunctionalModule_1to1_Net("not")
        net2 = FunctionalModule_1to1_Net("not")
        bitwise = net1.bitwise
        func = net2.func_node
        generated = DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_func = pb.Probe(generated[func][0], "spike")
        sim2.add_probe(probe_func)

        inpa = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(1, 20):
            assert np.array_equal(sim1.data[net1.probe2][i], ~inpa[i - 1])
            assert np.array_equal(sim2.data[probe_func][i], ~inpa[i - 1])

        for i in range(2, 20):
            assert np.array_equal(sim1.data[net1.probe3][i], ~inpa[i - 2])

        _assert_build_fmodule(net1, 3 + 1 + 2, 3 + 2 + 2)

    def test_BitwiseNOT_mapping(self, ensure_dump_dir):
        from tests.shared_networks import FunctionalModule_1to1_Net

        net1 = FunctionalModule_1to1_Net("not")

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

    def test_BitwiseOR(self):
        from tests.shared_networks import FunctionalModule_2to1_Net

        net1 = FunctionalModule_2to1_Net("or")
        net2 = FunctionalModule_2to1_Net("or")
        bitwise = net1.bitwise
        func = net2.func_node
        generated = DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_func = pb.Probe(generated[func][0], "spike")
        sim2.add_probe(probe_func)

        inpa = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(1, 20):
            assert np.array_equal(sim1.data[net1.probe3][i], inpa[i - 1] | inpb[i - 1])
            assert np.array_equal(sim2.data[probe_func][i], inpa[i - 1] | inpb[i - 1])

        for i in range(2, 20):
            assert np.array_equal(sim1.data[net1.probe4][i], inpa[i - 2] | inpb[i - 2])

        _assert_build_fmodule(net1, 6 + 1 + 2, 6 + 3 + 2)

    def test_BitwiseOR_mapping(self, ensure_dump_dir):
        from tests.shared_networks import FunctionalModule_2to1_Net

        net1 = FunctionalModule_2to1_Net("or")

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

    def test_BitwiseXOR(self):
        from tests.shared_networks import FunctionalModule_2to1_Net

        net1 = FunctionalModule_2to1_Net("xor")
        net2 = FunctionalModule_2to1_Net("xor")
        bitwise = net1.bitwise
        func = net2.func_node
        generated = DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_func = pb.Probe(generated[func][1], "spike")
        sim2.add_probe(probe_func)

        inpa = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(2, 20):
            assert np.array_equal(sim1.data[net1.probe3][i], inpa[i - 2] ^ inpb[i - 2])
            assert np.array_equal(sim2.data[probe_func][i], inpa[i - 2] ^ inpb[i - 2])

        for i in range(3, 20):
            assert np.array_equal(sim1.data[net1.probe4][i], inpa[i - 3] ^ inpb[i - 3])

        _assert_build_fmodule(net1, 6 + 1 + 2, 6 + 5 + 2)

    def test_BitwiseXOR_mapping(self, ensure_dump_dir):
        from tests.shared_networks import FunctionalModule_2to1_Net

        net1 = FunctionalModule_2to1_Net("xor")

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

    def test_DelayChain(self):
        from tests.shared_networks import FunctionalModule_1to1_Net

        net1 = FunctionalModule_1to1_Net("delay")
        net2 = FunctionalModule_1to1_Net("delay")
        bitwise = net1.bitwise
        func = net2.func_node
        generated = DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_func = pb.Probe(generated[func][func.chain_level - 1], "spike")
        sim2.add_probe(probe_func)

        inpa = np.random.randint(0, 2, size=(20, bitwise), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        _inh_delay = net1.func_node.inherent_delay
        for i in range(1 + _inh_delay, 20):
            assert np.array_equal(sim1.data[net1.probe2][i], inpa[i - 1 - _inh_delay])
            assert np.array_equal(sim2.data[probe_func][i], inpa[i - 1 - _inh_delay])

        for i in range(2 + _inh_delay, 20):
            assert np.array_equal(sim1.data[net1.probe3][i], inpa[i - 2 - _inh_delay])

        _assert_build_fmodule(net1, 3 + 1 + 2, 3 + 2 * net1.func_node.chain_level + 2)

    def test_DelayChain_mapping(self, ensure_dump_dir):
        from tests.shared_networks import FunctionalModule_1to1_Net

        net1 = FunctionalModule_1to1_Net("delay")

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

    def test_SpikingAdd(self):
        from tests.shared_networks import FunctionalModule_2to1_Net

        net1 = FunctionalModule_2to1_Net("add")
        net2 = FunctionalModule_2to1_Net("add")
        func = net2.func_node
        generated = DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_func = pb.Probe(generated[func][0], "spike")
        sim2.add_probe(probe_func)

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
            sim1.run(1)
            sim2.run(1)

        for i in range(1, 20):
            assert np.array_equal(sim1.data[net1.probe3][i], expected[i])
            assert np.array_equal(sim2.data[probe_func][i], expected[i])

        for i in range(2, 20):
            assert np.array_equal(sim1.data[net1.probe4][i], expected[i - 1])

        _assert_build_fmodule(net1, 6 + 1 + 2, 6 + 3 + 2)

    def test_SpikingAdd_mapping(self, ensure_dump_dir):
        from tests.shared_networks import FunctionalModule_2to1_Net

        net1 = FunctionalModule_2to1_Net("add")

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

    def test_SpikingSub(self):
        from tests.shared_networks import FunctionalModule_2to1_Net

        net1 = FunctionalModule_2to1_Net("sub")
        net2 = FunctionalModule_2to1_Net("sub")
        func = net2.func_node
        generated = DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_func = pb.Probe(generated[func][0], "spike")
        sim2.add_probe(probe_func)

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
            sim1.run(1)
            sim2.run(1)

        for i in range(1, 20):
            assert np.array_equal(sim1.data[net1.probe3][i], expected[i])
            assert np.array_equal(sim2.data[probe_func][i], expected[i])

        for i in range(2, 20):
            assert np.array_equal(sim1.data[net1.probe4][i], expected[i - 1])

        _assert_build_fmodule(net1, 6 + 1 + 2, 6 + 3 + 2)

    def test_SpikingSub_mapping(self, ensure_dump_dir):
        from tests.shared_networks import FunctionalModule_2to1_Net

        net1 = FunctionalModule_2to1_Net("sub")

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

    @pytest.mark.parametrize(
        "shape, channels, ksize, stride, padding, threshold, fm_order, pool_type, p_binomial",
        [
            ((24, 24), 3, (3, 3), 3, 0, None, "CHW", "avg", 0.7),
            ((12, 12), 1, (2, 3), None, 0, None, "CHW", "avg", 0.5),
            ((32, 32), 8, (3, 3), None, 0, 3, "CHW", "avg", 0.6),
            ((16, 16), 8, (5, 5), (2, 3), 0, 16, "CHW", "avg", 0.7),
            ((32, 32), 3, (3, 3), 2, 0, None, "CHW", "max", 0.5),
            ((24, 24), 1, (2, 3), None, 0, None, "CHW", "max", 0.4),
            ((16, 16), 8, (5, 5), (2, 3), 0, None, "CHW", "max", 0.6),
            ((32, 32), 8, (3, 3), (3, 4), 0, None, "CHW", "max", 0.3),
            ((24, 24), 3, (3, 3), 3, 1, 4, "CHW", "avg", 0.6),
            ((12, 12), 1, (2, 3), None, (1, 2), None, "CHW", "avg", 0.5),
            ((32, 32), 8, (3, 3), None, 2, None, "CHW", "avg", 0.5),
            ((16, 16), 8, (5, 5), (2, 3), (2, 3), 12, "CHW", "avg", 0.4),
            ((32, 32), 3, (3, 3), 2, 1, None, "CHW", "max", 0.6),
            ((24, 24), 1, (2, 3), None, 2, None, "CHW", "max", 0.7),
            ((16, 16), 8, (5, 5), (2, 3), (1, 1), None, "CHW", "max", 0.5),
            ((32, 32), 8, (3, 3), (3, 4), (1, 2), None, "CHW", "max", 0.3),
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
        self,
        shape,
        channels,
        ksize,
        stride,
        padding,
        threshold,
        fm_order,
        pool_type,
        p_binomial,
    ):
        from tests.shared_networks import SpikingPool2d_Net

        from .utils import avgpool2d_golden, maxpool2d_golden

        if fm_order == "CHW":
            fm_shape = (channels,) + shape
        else:
            fm_shape = shape + (channels,)

        net1 = SpikingPool2d_Net(fm_shape, ksize, stride, padding, threshold, pool_type)
        net2 = SpikingPool2d_Net(fm_shape, ksize, stride, padding, threshold, pool_type)
        p2d = net2.pool2d
        generated = DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_p2d = pb.Probe(generated[p2d][0], "spike")
        sim2.add_probe(probe_p2d)

        # Use binomial distribution to generate a sparse matrix with more zeros
        inpa = np.random.binomial(1, p_binomial, size=(20,) + fm_shape).astype(np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        _stride = _pair(stride) if stride is not None else ksize
        _padding = _pair(padding)
        if isinstance(threshold, int):
            _threshold = threshold
        else:
            _threshold = typical_round(shape2num(ksize) / 2)

        for i in range(1, 20):
            if pool_type == "avg":
                expected = avgpool2d_golden(
                    inpa[i - 1], ksize, _stride, _padding, fm_order, _threshold
                ).ravel()
            else:
                expected = maxpool2d_golden(
                    inpa[i - 1], ksize, _stride, _padding, fm_order
                ).ravel()

            assert np.array_equal(sim1.data[net1.probe2][i], expected)
            assert np.array_equal(sim2.data[probe_p2d][i], expected)

        for i in range(2, 20):
            if pool_type == "avg":
                expected = avgpool2d_golden(
                    inpa[i - 2], ksize, _stride, _padding, fm_order, _threshold
                ).ravel()
            else:
                expected = maxpool2d_golden(
                    inpa[i - 2], ksize, _stride, _padding, fm_order
                ).ravel()

            assert np.array_equal(sim1.data[net1.probe3][i], expected)

    def test_SpikingPool2d_mapping(self, ensure_dump_dir):
        from tests.shared_networks import SpikingPool2d_Net

        net1 = SpikingPool2d_Net((3, 24, 24), (3, 3), None, 0, None, "avg")

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

    @pytest.mark.parametrize(
        "shape, channels, ksize, stride, padding, threshold, p_binomial",
        [
            ((24, 24), 3, (3, 3), 3, 0, None, 0.5),
            ((12, 12), 1, (3, 3), None, 0, None, 0.7),
            ((32, 32), 8, (3, 3), None, 0, 3, 0.8),
            ((16, 16), 8, (5, 5), 5, 0, 16, 0.5),
            ((24, 24), 3, (3, 3), 3, 1, 4, 0.6),
            ((12, 12), 1, (3, 3), None, (1, 2), None, 0.7),
            ((32, 32), 8, (3, 3), None, 2, None, 0.7),
            ((16, 16), 8, (5, 5), 3, (2, 3), 12, 0.5),
        ],
    )
    def test_SpikingAvgPool2dWithV(
        self,
        shape,
        channels,
        ksize,
        stride,
        padding,
        threshold,
        p_binomial,
    ):
        """NOTE: This function is a native implementation of SNNs and is therefore not  \
            compared to the ANN implementation."""
        from tests.shared_networks import SpikingPool2d_Net

        fm_shape = (channels,) + shape

        net1 = SpikingPool2d_Net(fm_shape, ksize, stride, padding, threshold, "avgv")
        net2 = SpikingPool2d_Net(fm_shape, ksize, stride, padding, threshold, "avgv")
        p2d = net2.pool2d
        generated = DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_p2d = pb.Probe(generated[p2d][0], "spike")
        sim2.add_probe(probe_p2d)

        # Use binomial distribution to generate a sparse matrix with more zeros
        inpa = np.random.binomial(1, p_binomial, size=(20,) + fm_shape).astype(np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(1, 20):
            assert np.array_equal(sim1.data[net1.probe2][i], sim2.data[probe_p2d][i])

    def test_SpikingPool2dWithV_mapping(self, ensure_dump_dir):
        from tests.shared_networks import SpikingPool2d_Net

        net1 = SpikingPool2d_Net((3, 24, 24), (3, 3), None, 0, None, "avgv")

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

    @pytest.mark.parametrize("shape", [(32, 16), (1, 32), (64,), (128, 1), 48])
    def test_Transpose2d(self, shape):
        from tests.shared_networks import TransposeModule_T2d_Net

        net1 = TransposeModule_T2d_Net(shape)
        net2 = TransposeModule_T2d_Net(shape)
        t2d = net2.t2d
        generated = DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_t2d = pb.Probe(generated[t2d][0], "spike")
        sim2.add_probe(probe_t2d)

        inpa = np.random.randint(0, 2, size=(20,) + as_shape(shape), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(2, 20):
            expected = inpa[i - 1].T.ravel()
            assert np.array_equal(sim1.data[net1.probe1][i], expected)
            assert np.array_equal(sim2.data[probe_t2d][i], expected)

        for i in range(3, 20):
            expected = inpa[i - 2].T.ravel()
            assert np.array_equal(sim1.data[net1.probe2][i], expected)

    def test_Transpose2d_mapping(self, ensure_dump_dir):
        from tests.shared_networks import TransposeModule_T2d_Net

        net1 = TransposeModule_T2d_Net((32, 16))

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

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
        from tests.shared_networks import TransposeModule_T3d_Net

        net1 = TransposeModule_T3d_Net(shape, axes)
        net2 = TransposeModule_T3d_Net(shape, axes)
        t3d = net2.t3d
        generated = DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_t3d = pb.Probe(generated[t3d][0], "spike")
        sim2.add_probe(probe_t3d)

        if len(shape) == 2:
            shape = (1,) + shape

        inpa = np.random.randint(0, 2, size=(20,) + as_shape(shape), dtype=np.bool_)

        for i in range(20):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(2, 20):
            expected = inpa[i - 1].transpose(axes).ravel()
            assert np.array_equal(sim1.data[net1.probe1][i], expected)
            assert np.array_equal(sim2.data[probe_t3d][i], expected)

        for i in range(3, 20):
            expected = inpa[i - 2].transpose(axes).ravel()
            assert np.array_equal(sim1.data[net1.probe2][i], expected)

    def test_Transpose3d_mapping(self, ensure_dump_dir):
        from tests.shared_networks import TransposeModule_T3d_Net

        net1 = TransposeModule_T3d_Net((28, 28), (2, 0, 1))

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)
