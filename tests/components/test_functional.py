import numpy as np
import pytest

import paibox as pb
from paibox.base import DynamicSys
from paibox.components import NeuModule
from paibox.components._modules import _SemiFoldedModule
from paibox.components.synapses.conv_utils import _conv2d_faster, _pair, _single
from paibox.network import DynSysGroup
from paibox.types import NEUOUT_U8_DTYPE, VOLTAGE_DTYPE, WEIGHT_DTYPE
from paibox.utils import as_shape, shape2num, typical_round

from .conftest import *  # import test data
from .utils import (
    ann_bit_trunc,
    avgpool1d_golden,
    avgpool2d_golden,
    maxpool1d_golden,
    maxpool2d_golden,
)


def _assert_build_fmodule(
    network: DynSysGroup, n_node_bef_build: int, n_node_aft_build: int
):
    nodes = network.nodes().subset(DynamicSys).unique()
    assert len(nodes) == n_node_bef_build

    # Construct the functional modules
    network.build_modules()

    # Must exclude `NeuModule`, because it may be in the `__dict__` of probe
    nodes = network.nodes().subset(DynamicSys).exclude(NeuModule).unique()
    assert len(nodes) == n_node_aft_build


N_TEST = 20


class TestFunctionalModules:
    def test_FModule_ConnWithInput(self, build_FModule_ConnWithInput_Net):
        net = build_FModule_ConnWithInput_Net
        bitwise = 10
        sim = pb.Simulator(net, start_time_zero=False)

        inpa = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)

        # data2 will input to inp2 which is connected with the AND module.
        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim.run(1)

        for i in range(1, N_TEST):
            assert np.array_equal(sim.data[net.probe2][i], inpa[i - 1] & inpb[i])

        _assert_build_fmodule(net, 4 + 1 + 2, 4 + 3 + 2)

    def test_FModule_ConnWithModule(self, build_FModule_ConnWithModule_Net):
        net = build_FModule_ConnWithModule_Net
        bitwise = 10
        sim = pb.Simulator(net, start_time_zero=False)

        inpa = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)
        inpc = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)

        for t in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[t], data2=inpb[t], data3=inpc[t])
            sim.run(1)

        # At T >= 1, the AND1 will output the valid result.
        for t in range(1, N_TEST):
            assert np.array_equal(sim.data[net.probe2][t], inpa[t - 1] & inpb[t - 1])

        # At T >= 2, the OR1 will output the valid result.
        for t in range(2, N_TEST):
            assert np.array_equal(
                sim.data[net.probe3][t], (inpa[t - 2] & inpb[t - 2]) | inpc[t - 1]
            )

        _assert_build_fmodule(net, 9 + 2 + 2, 9 + 2 * 3 + 2)

    def test_FModule_ConnWithFModule(self, build_FModule_ConnWithFModule_Net):
        net = build_FModule_ConnWithFModule_Net
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph._raw_nodes) == 4

    def test_BitwiseAND(self):
        from tests.shared_networks import FunctionalModule_2to1_Net

        net1 = FunctionalModule_2to1_Net("and")
        net2 = FunctionalModule_2to1_Net("and")
        bitwise = net1.bitwise
        func = net2.func_node
        generated = net2.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_func = pb.Probe(generated[func][0], "spike")
        sim2.add_probe(probe_func)

        inpa = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(1, N_TEST):
            assert np.array_equal(sim1.data[net1.probe3][i], inpa[i - 1] & inpb[i - 1])
            assert np.array_equal(sim2.data[probe_func][i], inpa[i - 1] & inpb[i - 1])

        for i in range(2, N_TEST):
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
        generated = net2.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_func = pb.Probe(generated[func][0], "spike")
        sim2.add_probe(probe_func)

        inpa = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(1, N_TEST):
            assert np.array_equal(sim1.data[net1.probe2][i], ~inpa[i - 1])
            assert np.array_equal(sim2.data[probe_func][i], ~inpa[i - 1])

        for i in range(2, N_TEST):
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
        generated = net2.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_func = pb.Probe(generated[func][0], "spike")
        sim2.add_probe(probe_func)

        inpa = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(1, N_TEST):
            assert np.array_equal(sim1.data[net1.probe3][i], inpa[i - 1] | inpb[i - 1])
            assert np.array_equal(sim2.data[probe_func][i], inpa[i - 1] | inpb[i - 1])

        for i in range(2, N_TEST):
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
        generated = net2.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_func = pb.Probe(generated[func][1], "spike")
        sim2.add_probe(probe_func)

        inpa = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)
        inpb = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(2, N_TEST):
            assert np.array_equal(sim1.data[net1.probe3][i], inpa[i - 2] ^ inpb[i - 2])
            assert np.array_equal(sim2.data[probe_func][i], inpa[i - 2] ^ inpb[i - 2])

        for i in range(3, N_TEST):
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
        generated = net2.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_func = pb.Probe(generated[func][func.chain_level - 1], "spike")
        sim2.add_probe(probe_func)

        inpa = np.random.randint(0, 2, size=(N_TEST, bitwise), dtype=np.bool_)

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        _inh_delay = net1.func_node.inherent_delay
        for i in range(1 + _inh_delay, N_TEST):
            assert np.array_equal(sim1.data[net1.probe2][i], inpa[i - 1 - _inh_delay])
            assert np.array_equal(sim2.data[probe_func][i], inpa[i - 1 - _inh_delay])

        for i in range(2 + _inh_delay, N_TEST):
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
        generated = net2.build_modules()
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

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(1, N_TEST):
            assert np.array_equal(sim1.data[net1.probe3][i], expected[i])
            assert np.array_equal(sim2.data[probe_func][i], expected[i])

        for i in range(2, N_TEST):
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
        generated = net2.build_modules()
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

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i], data2=inpb[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(1, N_TEST):
            assert np.array_equal(sim1.data[net1.probe3][i], expected[i])
            assert np.array_equal(sim2.data[probe_func][i], expected[i])

        for i in range(2, N_TEST):
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
            ((24,), 3, (3,), 3, 0, None, "CL", "avg", 0.7),
            ((12,), 1, (2,), None, 0, None, "CL", "avg", 0.5),
            ((32,), 8, (3,), None, 0, 3, "CL", "avg", 0.6),
            ((16,), 8, (5,), (2,), 0, 16, "CL", "avg", 0.7),
            ((32,), 3, (3,), 2, 0, None, "CL", "max", 0.5),
            ((24,), 1, (2,), None, 0, None, "CL", "max", 0.4),
            ((16,), 8, (5,), (2,), 0, None, "CL", "max", 0.6),
            ((32,), 8, (3,), (3,), 0, None, "CL", "max", 0.3),
            ((24,), 3, (3,), 3, 1, 4, "CL", "avg", 0.6),
            ((12,), 1, (2,), None, (1,), None, "CL", "avg", 0.5),
            ((32,), 8, (3,), None, 2, None, "CL", "avg", 0.5),
            ((16,), 8, (5,), (2,), (2,), 12, "CL", "avg", 0.4),
            ((32,), 3, (3,), 2, 1, None, "CL", "max", 0.6),
            ((24,), 1, (2,), None, 2, None, "CL", "max", 0.7),
            ((16,), 8, (5,), (2,), (1,), None, "CL", "max", 0.5),
            ((32,), 8, (3,), (3,), (1,), None, "CL", "max", 0.3),
        ],
    )
    def test_SpikingPool1d(
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
        from tests.shared_networks import SpikingPool1d_Net

        if fm_order == "CL":
            fm_shape = (channels,) + shape
        else:
            fm_shape = shape + (channels,)

        net1 = SpikingPool1d_Net(fm_shape, ksize, stride, padding, threshold, pool_type)
        net2 = SpikingPool1d_Net(fm_shape, ksize, stride, padding, threshold, pool_type)
        p1d = net2.pool
        generated = net2.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_p1d = pb.Probe(generated[p1d][0], "spike")
        sim2.add_probe(probe_p1d)

        # Use binomial distribution to generate a sparse matrix with more zeros
        inpa = np.random.binomial(1, p_binomial, size=(N_TEST,) + fm_shape).astype(
            np.bool_
        )

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        _stride = _single(stride) if stride is not None else ksize
        _padding = _single(padding)
        if isinstance(threshold, int):
            _threshold = threshold
        else:
            _threshold = typical_round(shape2num(ksize) / 2)

        for i in range(1, N_TEST):
            if pool_type == "avg":
                expected = avgpool1d_golden(
                    inpa[i - 1], ksize, _stride, _padding, _threshold, fm_order
                ).ravel()
            else:
                expected = maxpool1d_golden(
                    inpa[i - 1], ksize, _stride, _padding, fm_order
                ).ravel()

            assert np.array_equal(sim1.data[net1.probe2][i], expected)
            assert np.array_equal(sim2.data[probe_p1d][i], expected)

        for i in range(2, N_TEST):
            if pool_type == "avg":
                expected = avgpool1d_golden(
                    inpa[i - 2], ksize, _stride, _padding, _threshold, fm_order
                ).ravel()
            else:
                expected = maxpool1d_golden(
                    inpa[i - 2], ksize, _stride, _padding, fm_order
                ).ravel()

            assert np.array_equal(sim1.data[net1.probe3][i], expected)

    def test_SpikingPool1d_mapping(self, ensure_dump_dir):
        from tests.shared_networks import SpikingPool1d_Net

        net1 = SpikingPool1d_Net((3, 24), (3,), None, 0, None, "avg")

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

        if fm_order == "CHW":
            fm_shape = (channels,) + shape
        else:
            fm_shape = shape + (channels,)

        net1 = SpikingPool2d_Net(fm_shape, ksize, stride, padding, threshold, pool_type)
        net2 = SpikingPool2d_Net(fm_shape, ksize, stride, padding, threshold, pool_type)
        p2d = net2.pool
        generated = net2.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_p2d = pb.Probe(generated[p2d][0], "spike")
        sim2.add_probe(probe_p2d)

        # Use binomial distribution to generate a sparse matrix with more zeros
        inpa = np.random.binomial(1, p_binomial, size=(N_TEST,) + fm_shape).astype(
            np.bool_
        )

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        _stride = _pair(stride) if stride is not None else ksize
        _padding = _pair(padding)
        if isinstance(threshold, int):
            _threshold = threshold
        else:
            _threshold = typical_round(shape2num(ksize) / 2)

        for i in range(1, N_TEST):
            if pool_type == "avg":
                expected = avgpool2d_golden(
                    inpa[i - 1], ksize, _stride, _padding, _threshold, fm_order
                ).ravel()
            else:
                expected = maxpool2d_golden(
                    inpa[i - 1], ksize, _stride, _padding, fm_order
                ).ravel()

            assert np.array_equal(sim1.data[net1.probe2][i], expected)
            assert np.array_equal(sim2.data[probe_p2d][i], expected)

        for i in range(2, N_TEST):
            if pool_type == "avg":
                expected = avgpool2d_golden(
                    inpa[i - 2], ksize, _stride, _padding, _threshold, fm_order
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
            ((24,), 3, (3,), 3, 0, None, 0.5),
            ((12,), 1, (3,), None, 0, None, 0.7),
            ((32,), 8, (3,), None, 0, 3, 0.8),
            ((16,), 8, (5,), 5, 0, 16, 0.5),
            ((24,), 3, (3,), 3, 1, 4, 0.6),
            ((12,), 1, (3,), None, (1,), None, 0.7),
            ((32,), 8, (3,), None, 2, None, 0.7),
            ((16,), 8, (5,), 3, (2,), 12, 0.5),
        ],
    )
    def test_SpikingAvgPool1dWithV(
        self, shape, channels, ksize, stride, padding, threshold, p_binomial
    ):
        """NOTE: This function is a native implementation of SNNs and is therefore not  \
            compared to the ANN implementation."""
        from tests.shared_networks import SpikingPool1d_Net

        fm_shape = (channels,) + shape

        net1 = SpikingPool1d_Net(fm_shape, ksize, stride, padding, threshold, "avgv")
        net2 = SpikingPool1d_Net(fm_shape, ksize, stride, padding, threshold, "avgv")
        p1d = net2.pool
        generated = net2.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_p1d = pb.Probe(generated[p1d][0], "spike")
        sim2.add_probe(probe_p1d)

        # Use binomial distribution to generate a sparse matrix with more zeros
        inpa = np.random.binomial(1, p_binomial, size=(N_TEST,) + fm_shape).astype(
            np.bool_
        )

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(1, N_TEST):
            assert np.array_equal(sim1.data[net1.probe2][i], sim2.data[probe_p1d][i])

    def test_SpikingAvgPool1dWithV_mapping(self, ensure_dump_dir):
        from tests.shared_networks import SpikingPool1d_Net

        net1 = SpikingPool1d_Net((3, 24), (3,), None, 0, None, "avgv")

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
        p2d = net2.pool
        generated = net2.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_p2d = pb.Probe(generated[p2d][0], "spike")
        sim2.add_probe(probe_p2d)

        # Use binomial distribution to generate a sparse matrix with more zeros
        inpa = np.random.binomial(1, p_binomial, size=(N_TEST,) + fm_shape).astype(
            np.bool_
        )

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(1, N_TEST):
            assert np.array_equal(sim1.data[net1.probe2][i], sim2.data[probe_p2d][i])

    def test_SpikingAvgPool2dWithV_mapping(self, ensure_dump_dir):
        from tests.shared_networks import SpikingPool2d_Net

        net1 = SpikingPool2d_Net((3, 24, 24), (3, 3), None, 0, None, "avgv")

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

    @pytest.mark.skipif(hasattr(pb.Transpose2d, "__deprecated__"), reason="deprecated")
    @pytest.mark.parametrize("shape", [(32, 16), (1, 32), (64,), (128, 1), 48])
    def test_Transpose2d(self, shape):
        from tests.shared_networks import TransposeModule_T2d_Net

        net1 = TransposeModule_T2d_Net(shape)
        net2 = TransposeModule_T2d_Net(shape)
        t2d = net2.t2d
        generated = net2.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_t2d = pb.Probe(generated[t2d][0], "spike")
        sim2.add_probe(probe_t2d)

        inpa = np.random.randint(0, 2, size=(N_TEST,) + as_shape(shape), dtype=np.bool_)

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(2, N_TEST):
            expected = inpa[i - 1].T.ravel()
            assert np.array_equal(sim1.data[net1.probe1][i], expected)
            assert np.array_equal(sim2.data[probe_t2d][i], expected)

        for i in range(3, N_TEST):
            expected = inpa[i - 2].T.ravel()
            assert np.array_equal(sim1.data[net1.probe2][i], expected)

    @pytest.mark.skipif(hasattr(pb.Transpose2d, "__deprecated__"), reason="deprecated")
    def test_Transpose2d_mapping(self, ensure_dump_dir):
        from tests.shared_networks import TransposeModule_T2d_Net

        net1 = TransposeModule_T2d_Net((32, 16))

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

    @pytest.mark.skipif(hasattr(pb.Transpose2d, "__deprecated__"), reason="deprecated")
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
        generated = net2.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_t3d = pb.Probe(generated[t3d][0], "spike")
        sim2.add_probe(probe_t3d)

        if len(shape) == 2:
            shape = (1,) + shape

        inpa = np.random.randint(0, 2, size=(N_TEST,) + as_shape(shape), dtype=np.bool_)

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(2, N_TEST):
            expected = inpa[i - 1].transpose(axes).ravel()
            assert np.array_equal(sim1.data[net1.probe1][i], expected)
            assert np.array_equal(sim2.data[probe_t3d][i], expected)

        for i in range(3, N_TEST):
            expected = inpa[i - 2].transpose(axes).ravel()
            assert np.array_equal(sim1.data[net1.probe2][i], expected)

    @pytest.mark.skipif(hasattr(pb.Transpose2d, "__deprecated__"), reason="deprecated")
    def test_Transpose3d_mapping(self, ensure_dump_dir):
        from tests.shared_networks import TransposeModule_T3d_Net

        net1 = TransposeModule_T3d_Net((28, 28), (2, 0, 1))

        mapper = pb.Mapper()
        mapper.build(net1)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

    @pytest.mark.parametrize(
        conv2d_semifolded_fc_chainnet_data["args"],
        conv2d_semifolded_fc_chainnet_data["data"],
    )
    def test_Conv2dSemiFolded_FC_ChainNet(
        self,
        ishape_chw,
        n_conv,
        kshape_oihw,
        stride,
        padding,
        out_features,
        groups,
        fixed_rng: np.random.Generator,
    ):
        """Test the network with N semi-folded conv2d + 1 semi-folded linear."""
        from tests.shared_networks import Conv2dSemiFolded_FC_ChainNetN

        assert n_conv == len(kshape_oihw) == len(stride)
        assert ishape_chw[0] == groups[0] * kshape_oihw[0][1]
        kernels = []
        strides = []
        paddings = []
        ocs = []
        ohs = []
        ows = []

        for i_conv in range(n_conv):
            kshape, s, p = kshape_oihw[i_conv], stride[i_conv], padding[i_conv]

            k = fixed_rng.integers(-3, 4, size=kshape, dtype=WEIGHT_DTYPE)
            _stride = _pair(s)
            _padding = _pair(p)
            kernels.append(k)
            strides.append(_stride)
            paddings.append(_padding)

            ih = ishape_chw[1] if i_conv == 0 else ohs[-1]
            iw = ishape_chw[2] if i_conv == 0 else ows[-1]
            oc = kshape[0]
            oh = (ih - kshape[2] + 2 * paddings[i_conv][0]) // _stride[0] + 1
            ow = (iw - kshape[3] + 2 * paddings[i_conv][0]) // _stride[1] + 1
            ocs.append(oc)
            ohs.append(oh)
            ows.append(ow)

        fc_weight = fixed_rng.integers(
            -4,
            5,
            size=(ocs[-1] * ohs[-1] * ows[-1], shape2num(out_features)),
            dtype=WEIGHT_DTYPE,
        )

        net1 = Conv2dSemiFolded_FC_ChainNetN(
            ishape_chw[:2], kernels, strides, paddings, out_features, fc_weight, groups
        )
        # `net1.conv_list` will be removed in `build_fmodule`
        conv2d_list = net1.conv_list.copy()
        linear = net1.linear1
        generated = net1.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)

        probe_conv_list = []
        for conv2d in conv2d_list:
            probe = pb.Probe(generated[conv2d][0], "output")
            probe_conv_list.append(probe)
            sim1.add_probe(probe)

        probe_linear = pb.Probe(generated[linear][0], "output")
        sim1.add_probe(probe_linear)

        semi_folded_modules: list[_SemiFoldedModule] = [*conv2d_list, linear]
        # The interval & the time o the first valid data of the external input data stream
        semi_vld_out_intv0 = 1
        t_1st_vld_data0 = 0
        # The interval & the time of the first valid data of the current layers
        semi_vld_out_intv = [m._oflow_format.interval for m in semi_folded_modules]
        t_1st_vld_data = [0] * n_conv
        for i in range(n_conv):
            if i == 0:
                t_1st_vld_data[i] = (
                    t_1st_vld_data0
                    + (kshape_oihw[0][-1] - paddings[0][0]) * semi_vld_out_intv0
                )
            else:
                t_1st_vld_data[i] = (
                    t_1st_vld_data[i - 1]
                    + (kshape_oihw[i][-1] - 1 - paddings[i][0])
                    * semi_vld_out_intv[i - 1]
                )

        n_test = 3  # can be more
        for _ in range(n_test):
            sim1.reset()
            inpa = fixed_rng.integers(0, 4, size=ishape_chw, dtype=NEUOUT_U8_DTYPE)
            if inpa.shape[-1] < 10:
                inp_pad0 = np.concatenate(
                    [
                        inpa,
                        np.zeros((inpa.shape[0], inpa.shape[1], 15), dtype=inpa.dtype),
                    ],
                    axis=2,
                    dtype=inpa.dtype,
                )
            else:
                inp_pad0 = np.concatenate(
                    [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
                )

            for i in range(inp_pad0.shape[-1]):
                pb.FRONTEND_ENV.save(data1=inp_pad0[:, :, i])
                sim1.run(1)

            x = inpa
            for i_conv in range(n_conv):
                x = ann_bit_trunc(
                    _conv2d_faster(
                        x,
                        (ohs[i_conv], ows[i_conv]),
                        kernels[i_conv],
                        strides[i_conv],
                        paddings[i_conv],
                        groups[i_conv],
                    )
                )

                # Check the result of semi-folded convolutions.
                for i in range(ows[i_conv]):
                    assert np.array_equal(
                        x[:, :, i].ravel(),
                        sim1.data[probe_conv_list[i_conv]][
                            conv2d_list[i_conv].tick_wait_start
                            + t_1st_vld_data[i_conv]
                            + i * semi_vld_out_intv[i_conv]
                            - 1
                        ],
                    )

                    assert conv2d_list[i_conv].tick_wait_start + t_1st_vld_data[
                        i_conv
                    ] + i * semi_vld_out_intv[i_conv] - 1 == conv2d_list[
                        i_conv
                    ].tick_wait_start + conv2d_list[
                        i_conv
                    ]._oflow_format.t_at_idx(
                        i
                    )

            # x is the reference result of the last convolution.
            expected_fc_t = ann_bit_trunc(x.ravel() @ fc_weight.astype(VOLTAGE_DTYPE))

            # Check the result of semi-folded linear.
            assert np.array_equal(
                expected_fc_t,
                sim1.data[probe_linear][
                    linear.tick_wait_start + linear._oflow_format.t_last_vld
                ],
            )
            assert (
                linear._oflow_format.get_global_t_1st_vld(linear.tick_wait_start)
                == linear.tick_wait_start + linear._oflow_format.t_last_vld
            )

    @pytest.mark.parametrize(
        pool2d_semifolded_fc_chainnet_data["args"],
        pool2d_semifolded_fc_chainnet_data["data"],
    )
    def test_Pool2dSemiFolded_FC_ChainNet(
        self,
        ishape_chw,
        n_pool,
        kshape_hw,
        stride,
        padding,
        out_features,
        pool_type,
        fixed_rng: np.random.Generator,
    ):
        """Test the network with N semi-folded pool2d + 1 semi-folded linear."""
        from tests.shared_networks import Pool2dSemiFolded_FC_ChainNetN

        if pool_type == "max":
            padding = [(0, 0)] * n_pool

        assert n_pool == len(kshape_hw) == len(stride)
        ksizes = []
        strides = []
        paddings = []
        ocs = []
        ohs = []
        ows = []

        for i_pool in range(n_pool):
            k, s, p = (kshape_hw[i_pool], stride[i_pool], padding[i_pool])

            _ksize = _pair(k)
            _stride = _pair(s if s is not None else _ksize)
            _padding = _pair(p)
            ksizes.append(_ksize)
            strides.append(_stride)
            paddings.append(_padding)

            ih = ishape_chw[1] if i_pool == 0 else ohs[-1]
            iw = ishape_chw[2] if i_pool == 0 else ows[-1]
            oc = ishape_chw[0]
            oh = (ih - _ksize[0] + 2 * paddings[i_pool][0]) // _stride[0] + 1
            ow = (iw - _ksize[1] + 2 * paddings[i_pool][0]) // _stride[1] + 1
            ocs.append(oc)
            ohs.append(oh)
            ows.append(ow)

        fc_weight = fixed_rng.integers(
            -4,
            5,
            size=(ocs[-1] * ohs[-1] * ows[-1], shape2num(out_features)),
            dtype=WEIGHT_DTYPE,
        )

        net1 = Pool2dSemiFolded_FC_ChainNetN(
            ishape_chw[:2],
            ksizes,
            strides,
            paddings,
            out_features,
            fc_weight,
            pool_type,
        )
        # `net1.pool_list` will be removed in `build_fmodule`
        pool2d_list = net1.pool_list.copy()
        linear = net1.linear1
        generated = net1.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)

        probe_pool_list = []
        for poool2d in pool2d_list:
            probe = pb.Probe(generated[poool2d][0], "output")
            probe_pool_list.append(probe)
            sim1.add_probe(probe)

        probe_linear = pb.Probe(generated[linear][0], "output")
        sim1.add_probe(probe_linear)

        semi_folded_modules: list[_SemiFoldedModule] = [*pool2d_list, linear]
        # The interval & the time o the first valid data of the external input data stream
        semi_vld_out_intv0 = 1
        t_1st_vld_data0 = 0
        # The interval & the time of the first valid data of the current layers
        semi_vld_out_intv = [m._oflow_format.interval for m in semi_folded_modules]
        t_1st_vld_data = [0] * n_pool
        for i in range(n_pool):
            if i == 0:
                t_1st_vld_data[i] = (
                    t_1st_vld_data0
                    + (ksizes[i][-1] - paddings[i][0]) * semi_vld_out_intv0
                )
            else:
                t_1st_vld_data[i] = (
                    t_1st_vld_data[i - 1]
                    + (ksizes[i][-1] - 1 - paddings[i][0]) * semi_vld_out_intv[i - 1]
                )

        n_test = 3  # can be more
        _pool_op = {"avg": avgpool2d_golden, "max": maxpool2d_golden}

        for _ in range(n_test):
            sim1.reset()
            inpa = fixed_rng.integers(0, 4, size=ishape_chw, dtype=NEUOUT_U8_DTYPE)
            inp_pad0 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )

            for i in range(inp_pad0.shape[-1]):
                pb.FRONTEND_ENV.save(data1=inp_pad0[:, :, i])
                sim1.run(1)

            x = inpa
            for i_pool in range(n_pool):
                x = ann_bit_trunc(
                    _pool_op[pool_type](
                        x, ksizes[i_pool], strides[i_pool], paddings[i_pool]
                    )
                )

                # Check the result of semi-folded pooling.
                for i in range(ows[i_pool]):
                    assert np.array_equal(
                        x[:, :, i].ravel(),
                        sim1.data[probe_pool_list[i_pool]][
                            pool2d_list[i_pool].tick_wait_start
                            + t_1st_vld_data[i_pool]
                            + i * semi_vld_out_intv[i_pool]
                            - 1
                        ],
                    )

                    assert pool2d_list[i_pool].tick_wait_start + t_1st_vld_data[
                        i_pool
                    ] + i * semi_vld_out_intv[i_pool] - 1 == pool2d_list[
                        i_pool
                    ].tick_wait_start + pool2d_list[
                        i_pool
                    ]._oflow_format.t_at_idx(
                        i
                    )

            # x is the reference result of the last pooling.
            expected_fc_t = ann_bit_trunc(x.ravel() @ fc_weight.astype(VOLTAGE_DTYPE))

            # Check the result of semi-folded linear.
            assert np.array_equal(
                expected_fc_t,
                sim1.data[probe_linear][
                    linear.tick_wait_start + linear._oflow_format.t_last_vld
                ],
            )

            assert (
                linear._oflow_format.get_global_t_1st_vld(linear.tick_wait_start)
                == linear.tick_wait_start + linear._oflow_format.t_last_vld
            )

    @pytest.mark.parametrize(
        "shape, weight",
        [
            (
                (3, 5, 5),
                np.random.randint(0, 5, size=(3 * 5 * 5, 10), dtype=WEIGHT_DTYPE),
            ),
            ((10,), np.random.randint(0, 5, size=(10, 10), dtype=WEIGHT_DTYPE)),
        ],
    )
    def test_Linear(self, shape, weight, fixed_rng: np.random.Generator):
        from tests.shared_networks import Linear_Net

        net1 = Linear_Net(shape, weight)
        net2 = Linear_Net(shape, weight)
        linear = net2.linear1
        generated = net2.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_linear = pb.Probe(generated[linear][0], "output")
        sim2.add_probe(probe_linear)

        inpa = fixed_rng.integers(0, 10, size=(N_TEST,) + shape, dtype=NEUOUT_U8_DTYPE)

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(N_TEST):
            assert np.array_equal(sim1.data[net1.probe1][i], sim2.data[probe_linear][i])

    @pytest.mark.parametrize(ann_pool1d_data["args"], ann_pool1d_data["data"])
    def test_ANNPool1d(
        self,
        ishape_cl,
        n_pool,
        kshape_l,
        stride,
        padding,
        out_features,
        pool_type,
        fixed_rng: np.random.Generator,
    ):
        from tests.shared_networks import Pool1d_FC_ChainNetN

        assert n_pool == len(kshape_l) == len(stride)
        ksizes = []
        strides = []
        paddings = []
        ocs = []
        ols = []

        for i_pool in range(n_pool):
            k, s, p = (kshape_l[i_pool], stride[i_pool], padding[i_pool])

            _ksize = _single(k)
            _stride = _single(s if s is not None else _ksize)
            _padding = _single(p)
            ksizes.append(_ksize)
            strides.append(_stride)
            paddings.append(_padding)

            il = ishape_cl[1] if i_pool == 0 else ols[-1]
            oc = ishape_cl[0]
            ol = (il - _ksize[0] + 2 * paddings[i_pool][0]) // _stride[0] + 1
            ocs.append(oc)
            ols.append(ol)

        fc_weight = fixed_rng.integers(
            -4, 5, size=(ocs[-1] * ols[-1], shape2num(out_features)), dtype=WEIGHT_DTYPE
        )

        net1 = Pool1d_FC_ChainNetN(
            ishape_cl, ksizes, strides, paddings, out_features, fc_weight, pool_type
        )

        # `net1.pool_list` will be removed in `build_fmodule`
        pool1d_list = net1.pool_list.copy()
        linear = net1.linear1
        generated = net1.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)

        probe_pool_list = []
        for pool1d in pool1d_list:
            probe = pb.Probe(generated[pool1d][0], "output")
            probe_pool_list.append(probe)
            sim1.add_probe(probe)

        probe_linear = pb.Probe(generated[linear][0], "output")
        sim1.add_probe(probe_linear)

        _pool_op = {"avg": avgpool1d_golden, "max": maxpool1d_golden}
        n_test = 3  # can be more

        for _ in range(n_test):
            sim1.reset()
            inpa = fixed_rng.integers(0, 4, size=ishape_cl, dtype=NEUOUT_U8_DTYPE)

            for _ in range(3):  # 3 layers
                pb.FRONTEND_ENV.save(data1=inpa)
                sim1.run(1)

            x = inpa
            for i_pool in range(n_pool):
                x = ann_bit_trunc(
                    _pool_op[pool_type](
                        x, ksizes[i_pool], strides[i_pool], paddings[i_pool]
                    )
                )
                assert np.array_equal(
                    x.ravel(), sim1.data[probe_pool_list[i_pool]][2 * i_pool]
                )

    @pytest.mark.parametrize(ann_pool2d_data["args"], ann_pool2d_data["data"])
    def test_ANNPool2d(
        self,
        ishape_chw,
        n_pool,
        kshape_hw,
        stride,
        padding,
        out_features,
        pool_type,
        fixed_rng: np.random.Generator,
    ):
        from tests.shared_networks import Pool2d_FC_ChainNetN

        assert n_pool == len(kshape_hw) == len(stride)
        ksizes = []
        strides = []
        paddings = []
        ocs = []
        ohs = []
        ows = []

        for i_pool in range(n_pool):
            k, s, p = (kshape_hw[i_pool], stride[i_pool], padding[i_pool])

            _ksize = _pair(k)
            _stride = _pair(s if s is not None else _ksize)
            _padding = _pair(p)
            ksizes.append(_ksize)
            strides.append(_stride)
            paddings.append(_padding)

            ih = ishape_chw[1] if i_pool == 0 else ohs[-1]
            iw = ishape_chw[2] if i_pool == 0 else ows[-1]
            oc = ishape_chw[0]
            oh = (ih - _ksize[0] + 2 * paddings[i_pool][0]) // _stride[0] + 1
            ow = (iw - _ksize[1] + 2 * paddings[i_pool][1]) // _stride[1] + 1
            ocs.append(oc)
            ohs.append(oh)
            ows.append(ow)

        fc_weight = fixed_rng.integers(
            -4,
            5,
            size=(ocs[-1] * ohs[-1] * ows[-1], shape2num(out_features)),
            dtype=WEIGHT_DTYPE,
        )

        net1 = Pool2d_FC_ChainNetN(
            ishape_chw, ksizes, strides, paddings, out_features, fc_weight, pool_type
        )
        # `net1.pool_list` will be removed in `build_fmodule`
        pool2d_list = net1.pool_list.copy()
        linear = net1.linear1
        generated = net1.build_modules()
        sim1 = pb.Simulator(net1, start_time_zero=False)

        probe_pool_list = []
        for poool2d in pool2d_list:
            probe = pb.Probe(generated[poool2d][0], "output")
            probe_pool_list.append(probe)
            sim1.add_probe(probe)

        probe_linear = pb.Probe(generated[linear][0], "output")
        sim1.add_probe(probe_linear)

        _pool_op = {"avg": avgpool2d_golden, "max": maxpool2d_golden}
        n_test = 3  # can be more

        for _ in range(n_test):
            sim1.reset()
            inpa = fixed_rng.integers(0, 4, size=ishape_chw, dtype=NEUOUT_U8_DTYPE)

            for _ in range(3):  # 3 layers
                pb.FRONTEND_ENV.save(data1=inpa)
                sim1.run(1)

            x = inpa
            for i_pool in range(n_pool):
                x = ann_bit_trunc(
                    _pool_op[pool_type](
                        x,
                        _pair(ksizes[i_pool]),
                        _pair(strides[i_pool]),
                        paddings[i_pool],
                        0,
                    )
                )
                assert np.array_equal(
                    x.ravel(), sim1.data[probe_pool_list[i_pool]][2 * i_pool]
                )
