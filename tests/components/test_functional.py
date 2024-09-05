import numpy as np
import pytest

import paibox as pb
from paibox.base import DynamicSys
from paibox.components import NeuModule
from paibox.components.synapses.conv_utils import _pair, _single
from paibox.network import DynSysGroup
from paibox.simulator.utils import _conv2d_faster_fp32
from paibox.utils import as_shape, shape2num, typical_round

from .utils import (
    avg_pooling,
    avgpool1d_golden,
    avgpool2d_golden,
    max_pooling,
    maxpool1d_golden,
    maxpool2d_golden,
)


def _assert_build_fmodule(
    network: DynSysGroup, n_node_bef_build: int, n_node_aft_build: int
):
    nodes = network.nodes().subset(DynamicSys).unique()
    assert len(nodes) == n_node_bef_build

    # Construct the functional modules
    DynSysGroup.build_fmodule(network)

    # Must exclude `NeuModule`, because it may be in the `__dict__` of probe
    nodes = network.nodes().subset(DynamicSys).exclude(NeuModule).unique()
    assert len(nodes) == n_node_aft_build


def _ann_bit_trunc(v_array: VoltageType, bit_trunc: int = 8) -> NeuOutType:
    return np.where(v_array <= 0, 0, MetaNeuron._truncate(v_array, bit_trunc)).astype(
        NEUOUT_U8_DTYPE
    )


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
        generated = DynSysGroup.build_fmodule(net2)
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
        generated = DynSysGroup.build_fmodule(net2)
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
        generated = DynSysGroup.build_fmodule(net2)
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
        generated = DynSysGroup.build_fmodule(net2)
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
        generated = DynSysGroup.build_fmodule(net2)
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
        generated = DynSysGroup.build_fmodule(net2)
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
                    inpa[i - 1], ksize, _stride, _padding, fm_order, _threshold
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
                    inpa[i - 2], ksize, _stride, _padding, fm_order, _threshold
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
        generated = DynSysGroup.build_fmodule(net2)
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
                    inpa[i - 1], ksize, _stride, _padding, fm_order, _threshold
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
        generated = DynSysGroup.build_fmodule(net2)
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
        generated = DynSysGroup.build_fmodule(net2)
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
        generated = DynSysGroup.build_fmodule(net2)
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
        generated = DynSysGroup.build_fmodule(net2)
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
        "ishape_chw, n_conv, kshape_oihw, stride, padding, out_features",
        [
            # n_conv = 1
            ((3, 12, 12), 1, [(12, 3, 3, 3)], [(1, 1)], [0], (10,)),
            ((8, 12, 12), 1, [(16, 8, 3, 3)], [(2, 2)], [0], (10,)),
            ((8, 12, 12), 1, [(16, 8, 4, 4)], [2], [0], (10,)),
            ((4, 12, 12), 1, [(8, 4, 3, 3)], [1], [0], (4, 2)),
            ((4, 24, 24), 1, [(8, 4, 3, 3)], [2], [0], 10),
            ((12, 12, 12), 1, [(6, 12, 3, 3)], [1], [0], (3, 3)),
            ((4, 24, 24), 1, [(8, 4, 4, 4)], [2], [0], (10,)),
            ((8, 32, 32), 1, [(4, 8, 3, 3)], [2], [0], 10),
            # n_conv = 2
            (
                (4, 32, 32),
                2,
                [(8, 4, 3, 3), (12, 8, 4, 4)],
                [(2, 2), (2, 2)],
                [0, 0],
                10,
            ),
            (
                (4, 32, 32),
                2,
                [(8, 4, 3, 3), (12, 8, 4, 4)],
                [(2, 2), (1, 1)],
                [0, 0],
                10,
            ),
            ((1, 32, 32), 2, [(1, 1, 3, 3), (1, 1, 3, 3)], [2, 2], [0, 0], 10),
            ((1, 32, 32), 2, [(1, 1, 4, 4), (1, 1, 4, 4)], [1, 2], [0, 0], 10),
            ((1, 32, 32), 2, [(1, 1, 4, 4), (1, 1, 4, 4)], [2, 2], [0, 0], 10),
            ((1, 24, 24), 2, [(1, 1, 3, 3), (1, 1, 4, 4)], [1, 2], [0, 0], 10),
            ((1, 24, 24), 2, [(1, 1, 3, 3), (1, 1, 4, 4)], [2, 2], [0, 0], 10),
            # n_conv = 3
            (
                (4, 32, 32),
                3,
                [(8, 4, 3, 3), (16, 8, 3, 3), (8, 16, 2, 2)],
                [2, 1, 1],
                [0, 0, 0],
                3,
            ),
            (
                (3, 32, 32),
                3,
                [(16, 3, 3, 3), (32, 16, 3, 3), (10, 32, 3, 3)],
                [1, 1, 1],
                [0, 0, 0],
                10,
            ),
        ],
    )
    def test_Conv2dSemiFolded_FC_ChainNet(
        self,
        ishape_chw,
        n_conv,
        kshape_oihw,
        stride,
        padding,
        out_features,
        random_fixture,
    ):
        """Test the network with N semi-folded conv2d + 1 semi-folded linear."""
        from tests.shared_networks import Conv2dSemiFolded_FC_ChainNetN

        assert n_conv == len(kshape_oihw) == len(stride) == len(padding)
        kernels = []
        strides = []
        paddings = []
        ocs = []
        ohs = []
        ows = []

        for i_conv in range(n_conv):
            kshape, s, p = kshape_oihw[i_conv], stride[i_conv], padding[i_conv]

            k = np.random.randint(-3, 4, size=kshape, dtype=WEIGHT_DTYPE)
            _stride = _pair(s)
            _padding = _pair(p)
            kernels.append(k)
            strides.append(_stride)
            paddings.append(_padding)

            ih = ishape_chw[1] if i_conv == 0 else ohs[-1]
            iw = ishape_chw[2] if i_conv == 0 else ows[-1]
            oc = kshape[0]
            oh = (ih + 2 * _padding[0] - kshape[2]) // _stride[0] + 1
            ow = (iw + 2 * _padding[1] - kshape[3]) // _stride[1] + 1
            ocs.append(oc)
            ohs.append(oh)
            ows.append(ow)

        fc_weight = np.random.randint(
            -4,
            5,
            size=(ocs[-1] * ohs[-1] * ows[-1], shape2num(out_features)),
            dtype=WEIGHT_DTYPE,
        )

        net2 = Conv2dSemiFolded_FC_ChainNetN(
            ishape_chw[:2], kernels, strides, paddings, out_features, fc_weight
        )
        # `conv_list` will be removed in `build_fmodule`
        conv2d_list = net2.conv_list.copy()
        linear = net2.linear1
        generated = DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net2, start_time_zero=False)

        probe_conv_list = []
        for conv2d in conv2d_list:
            probe = pb.Probe(generated[conv2d][0], "output")
            probe_conv_list.append(probe)
            sim1.add_probe(probe)

        probe_linear = pb.Probe(generated[linear][0], "output")
        sim1.add_probe(probe_linear)

        semi_folded_modules = [*conv2d_list, linear]
        semi_valid_interval = []
        for m in semi_folded_modules:
            semi_valid_interval.append(m.valid_interval)

        ts_1st_valid = [0] * n_conv
        for i in range(n_conv):
            if i == 0:
                ts_1st_valid[i] = kshape_oihw[0][-1] * semi_valid_interval[0]
            else:
                ts_1st_valid[i] = (
                    ts_1st_valid[i - 1]
                    + (kshape_oihw[i][-1] - 1) * semi_valid_interval[i]
                )

        n_test = 3  # can be more
        for _ in range(n_test):
            sim1.reset()
            inpa = np.random.randint(0, 3, size=ishape_chw, dtype=VOLTAGE_DTYPE)
            inp_pad0 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )

            for i in range(inp_pad0.shape[-1]):
                pb.FRONTEND_ENV.save(data1=inp_pad0[:, :, i])
                sim1.run(1)

            x = inpa
            for i_conv in range(n_conv):
                x = _ann_bit_trunc(
                    _conv2d_faster_fp32(
                        x, kernels[i_conv], strides[i_conv], paddings[i_conv]
                    ).astype(VOLTAGE_DTYPE)
                )

                # Check the result of semi-folded convolutions.
                for i in range(ow):
                    assert np.array_equal(
                        x[:, :, i].ravel(),
                        sim1.data[probe_conv_list[i_conv]][
                            conv2d_list[i_conv].tick_wait_start
                            + ts_1st_valid[i_conv]
                            + i * semi_valid_interval[i_conv + 1]
                            - 1
                        ],
                    )

            # x is the reference result of the last convolution.
            expected_fc_t = _ann_bit_trunc(x.ravel() @ fc_weight.astype(VOLTAGE_DTYPE))

            # Check the result of semi-folded linear.
            assert np.array_equal(
                expected_fc_t,
                sim1.data[probe_linear][
                    linear.tick_wait_start
                    + ts_1st_valid[-1]
                    + (ows[-1] - 1) * semi_valid_interval[-1]
                    - 1
                ],
            )

    @pytest.mark.skip(reason="not implemented yet")
    @pytest.mark.parametrize(
        "shape, kernel, stride, padding, out_features, weight",
        [
            (
                (1, 11),
                np.array([[[[2, 1, 2], [1, -2, 1], [-1, 2, -3]]]], dtype=np.int8),
                [1, 1],
                [0, 0],
                10,
                np.random.randint(-5, 5, size=(7 * 7, 10), dtype=np.int8),
            ),
            (
                (1, 11),
                np.array([[[[2, 1, 2], [1, -2, 1], [-1, 2, -3]]]], dtype=np.int8),
                [1, 2],
                [0, 0],
                10,
                np.random.randint(-5, 5, size=(4 * 4, 10), dtype=np.int8),
            ),
            (
                (1, 11),
                np.array([[[[2, 1, 2], [1, -2, 1], [-1, 2, -3]]]], dtype=np.int8),
                [2, 1],
                [0, 0],
                10,
                np.random.randint(-5, 5, size=(3 * 3, 10), dtype=np.int8),
            ),
            (
                (1, 11),
                np.array([[[[2, 1, 2], [1, -2, 1], [-1, 2, -3]]]], dtype=np.int8),
                [2, 2],
                [0, 0],
                10,
                np.random.randint(-5, 5, size=(2 * 2, 10), dtype=np.int8),
            ),
        ],
    )
    def test_Conv2dSemiFolded_FC_Net2(
        self, shape, kernel, stride, padding, out_features, weight
    ):
        from tests.shared_networks import Conv2dSemiFolded_FC_Net2

        net2 = Conv2dSemiFolded_FC_Net2(
            shape, kernel, stride, padding, out_features, weight
        )
        conv2d = net2.conv2
        linear = net2.linear1
        generated = DynSysGroup.build_fmodule(net2)
        # sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_conv = pb.Probe(generated[conv2d][0], "output")
        probe_linear = pb.Probe(generated[linear][0], "output")
        sim2.add_probe(probe_conv)
        sim2.add_probe(probe_linear)
        inpa = np.random.randint(0, 5, size=(1, 11, 11)).astype(np.int8)
        inpb = np.concatenate([inpa, np.zeros((1, 10, 11))], axis=1)
        for i in range(17):
            pb.FRONTEND_ENV.save(data1=inpb[0][i])
            sim2.run(1)
        expected = _conv2d_faster_fp32(
            np.transpose(inpa, (0, 2, 1)), kernel, _pair(stride[0]), _pair(padding[0])
        )
        expected[expected < 0] = 0

        expected = _conv2d_faster_fp32(
            expected, kernel, _pair(stride[1]), _pair(padding[1])
        )
        expected[expected < 0] = 0

        expected = np.array(expected, dtype=np.int32)
        expected = expected.ravel() @ weight
        expected[expected < 0] = 0
        if (expected >> 8).all() > 0:
            expected = np.full_like(expected, ((1 << 8) - 1))
        else:
            expected = expected & ((1 << 8) - 1)
        # expected = np.clip(expected, 0, 7)
        assert np.array_equal(expected, sim2.data[probe_linear][15])

    @pytest.mark.skip(reason="not implemented yet")
    @pytest.mark.parametrize(
        "shape, kernel_size, stride, weight, pool_type",
        [
            (
                (1, 8),
                (2, 2),
                [1, 1],
                np.random.randint(-5, 5, size=(6 * 6, 2), dtype=np.int8),
                "avg",
            ),
            (
                (1, 8),
                (2, 2),
                [2, 2],
                np.random.randint(-5, 5, size=(2 * 2, 2), dtype=np.int8),
                "avg",
            ),
            (
                (1, 8),
                (2, 2),
                [1, 1],
                np.random.randint(0, 5, size=(6 * 6, 2), dtype=np.int8),
                "max",
            ),
            (
                (1, 8),
                (2, 2),
                [2, 2],
                np.random.randint(0, 5, size=(2 * 2, 2), dtype=np.int8),
                "max",
            ),
        ],
    )
    def test_Pool2dSemiMap(self, shape, kernel_size, stride, weight, pool_type):
        from tests.shared_networks import Pool2dSemiMap_Net

        net1 = Pool2dSemiMap_Net(shape, kernel_size, stride, weight, pool_type)
        pool = net1.pool2
        linear = net1.linear1
        generated = DynSysGroup.build_fmodule(net1)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        probe_linear = pb.Probe(generated[linear][0], "output")
        probe_pool = pb.Probe(generated[pool][0], "output")
        sim1.add_probe(probe_pool)
        sim1.add_probe(probe_linear)
        inpa = np.random.randint(0, 10, size=(1, 8, 8)).astype(np.int8)
        inpb = np.concatenate([inpa, np.zeros((1, 10, 8))], axis=1)
        for i in range(13):
            pb.FRONTEND_ENV.save(data1=inpb[:, i, :])
            sim1.run(1)
        if pool_type == "max":
            expected = max_pooling(np.transpose(inpa, (0, 2, 1)), kernel_size, stride)
            expected = max_pooling(expected, kernel_size, stride)
            expected = np.array(expected, dtype=np.int32)
            expected = expected.ravel() @ weight
            if (expected >> 8).all() > 0:
                expected = np.full_like(expected, ((1 << 8) - 1))
            else:
                expected = expected & ((1 << 8) - 1)
            assert np.array_equal(expected, sim1.data[probe_linear][12])
        else:
            expected = avg_pooling(np.transpose(inpa, (0, 2, 1)), kernel_size, stride)
            expected = avg_pooling(expected, kernel_size, stride)
            expected = np.array(expected, dtype=np.int32)
            expected = expected.ravel() @ weight
            expected[expected < 0] = 0
            if (expected >> 8).all() > 0:
                expected = np.full_like(expected, ((1 << 8) - 1))
            else:
                expected = expected & ((1 << 8) - 1)
            assert np.array_equal(expected, sim1.data[probe_linear][12])

    @pytest.mark.parametrize(
        "shape, weight",
        [
            ((3, 5, 5), np.random.randint(0, 5, size=(3 * 5 * 5, 10), dtype=np.int8)),
            ((10,), np.random.randint(0, 5, size=(10, 10), dtype=np.int8)),
        ],
    )
    def test_Linear(self, shape, weight):
        from tests.shared_networks import Linear_Net

        net1 = Linear_Net(shape, weight)
        net2 = Linear_Net(shape, weight)
        linear = net2.linear1
        generated = pb.DynSysGroup.build_fmodule(net2)
        sim1 = pb.Simulator(net1, start_time_zero=False)
        sim2 = pb.Simulator(net2, start_time_zero=False)

        probe_linear = pb.Probe(generated[linear][0], "output")
        sim2.add_probe(probe_linear)

        inpa = np.random.randint(0, 10, (N_TEST,) + shape, dtype=np.uint8)

        for i in range(N_TEST):
            pb.FRONTEND_ENV.save(data1=inpa[i])
            sim1.run(1)
            sim2.run(1)

        for i in range(N_TEST):
            assert np.array_equal(sim1.data[net1.probe1][i], sim2.data[probe_linear][i])
