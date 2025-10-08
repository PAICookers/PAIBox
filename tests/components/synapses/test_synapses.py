import json
from contextlib import nullcontext
from enum import Enum

import numpy as np
import pytest
from paicorelib import WeightWidth as WW

import paibox as pb
from paibox._logging import set_logs
from paibox.components import FullConnectedSyn
from paibox.components.synapses.lut import LUT_DTYPE
from paibox.exceptions import RegisterError, ShapeError
from paibox.types import NEUOUT_U8_DTYPE, WEIGHT_DTYPE
from paibox.utils import shape2num
from tests.utils import file_not_exist_fail, gen_random_array


class SynCfgJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, Enum):
            return o.value
        return super().default(o)


class TestFullConnectedSyn:
    def test_FullConnectedSyn_properties(self):
        n1 = pb.IF((10, 10), 10)
        n2 = pb.IF((20, 10), 10)
        s1 = pb.FullConn(
            n1, n2, np.random.randint(-128, 128, (100, 200), dtype=np.int8)
        )

        new_source1 = pb.LIF((100,), 3)
        new_source2 = pb.LIF((10,), 5)
        new_target1 = pb.LIF((10, 20), 7)
        new_target2 = pb.LIF(100, 9)

        s1.source = new_source1
        with pytest.raises(RegisterError):
            s1.source = new_source2

        s1.target = new_target1
        with pytest.raises(RegisterError):
            s1.target = new_target2

    def test_FullConn_copy(self):
        n1 = pb.IF((10, 10), 10)
        n2 = pb.IF((20, 10), 10)
        s1 = pb.FullConn(
            n1, n2, np.random.randint(-128, 128, (100, 200), dtype=np.int8)
        )
        s2 = s1.copy()

        assert isinstance(s2, FullConnectedSyn)
        assert id(s1) != id(s2)
        assert s1 != s2
        assert s1.source == s2.source
        assert s1.target == s2.target
        assert s1.weights is not s2.weights
        assert np.array_equal(s1.connectivity, s2.connectivity)

        # Check the size
        s2.source = n1
        s2.target = n2

        new_target1 = pb.LIF((10, 20), 7)
        s2.target = new_target1

        assert s1.target != s2.target

    def test_MatMul2d_copy(self):
        n1 = pb.IF((20, 10), 10)
        n2 = pb.IF((10, 10), 10)
        s1 = pb.MatMul2d(n1, n2, np.random.randint(-128, 128, (20, 10), dtype=np.int8))
        s2 = s1.copy()

        assert isinstance(s2, FullConnectedSyn)
        assert id(s1) != id(s2)
        assert s1 != s2
        assert s1.source == s2.source
        assert s1.target == s2.target
        assert s1.weights is not s2.weights
        assert np.array_equal(s1.connectivity, s2.connectivity)

        s2.source = n1
        s2.target = n2

        new_target1 = pb.LIF((10, 10), 7)
        s2.target = new_target1

        assert s1.target != s2.target

    def test_Conv2d_copy(self):
        n1 = pb.IF((8, 28, 28), 10)
        n2 = pb.IF((16, 14, 14), 10)
        s1 = pb.Conv2d(
            n1,
            n2,
            np.random.randint(-128, 128, (16, 8, 3, 3), dtype=np.int8),
            stride=2,
            padding=1,
        )
        s2 = s1.copy()

        assert isinstance(s2, FullConnectedSyn)
        assert id(s1) != id(s2)
        assert s1 != s2
        assert s1.source == s2.source
        assert s1.target == s2.target
        assert s1.weights is not s2.weights
        assert np.array_equal(s1.connectivity, s2.connectivity)

        s2.source = n1
        s2.target = n2

        new_target1 = pb.IF((16, 14, 14), 7)
        s2.target = new_target1

        assert s1.target != s2.target


class TestFullConn:
    @pytest.mark.parametrize(
        "n1, n2, scalar_weight, expected_wp",
        [
            (pb.IF(10, 3), pb.IF(10, 3), 1, WW.WEIGHT_WIDTH_1BIT),
            (pb.IF((3, 3), 3), pb.IF((3, 3), 3), 4, WW.WEIGHT_WIDTH_4BIT),
            (pb.IF((5,), 3), pb.IF((5,), 3), -1, WW.WEIGHT_WIDTH_2BIT),
            # TODO 3-dimension shape is correct for data flow?
            (pb.IF((10, 2, 3), 3), pb.IF((10, 2, 3), 3), 16, WW.WEIGHT_WIDTH_8BIT),
            (pb.IF((10, 2), 3), pb.IF((4, 5), 3), -100, WW.WEIGHT_WIDTH_8BIT),
            (pb.IF(10, 3), pb.IF((2, 5), 3), 7, WW.WEIGHT_WIDTH_4BIT),
        ],
    )
    def test_FullConn_One2One_scalar(self, n1, n2, scalar_weight, expected_wp):
        s1 = pb.FullConn(n1, n2, scalar_weight, conn_type=pb.SynConnType.One2One)

        assert np.array_equal(s1.weights, scalar_weight)
        assert (s1.num_in, s1.num_out) == (n1.num_out, n2.num_in)
        assert np.array_equal(
            s1.connectivity,
            scalar_weight * np.identity(n1.num_out, dtype=WEIGHT_DTYPE),
        )
        assert s1.connectivity.dtype == WEIGHT_DTYPE
        assert s1.weight_width is expected_wp

    @pytest.mark.parametrize(
        "n1, n2",
        [
            (pb.IF(10, 3), pb.IF(100, 4)),
            (pb.IF((10, 10), 3), pb.IF((5, 10), 4)),
            (pb.IF((10,), 3), pb.IF((5,), 4)),
            (pb.IF(10, 3), pb.IF((5, 10), 4)),
        ],
    )
    def test_FullConn_One2One_scalar_illegal(self, n1, n2):
        with pytest.raises(ShapeError):
            s1 = pb.FullConn(n1, n2, conn_type=pb.SynConnType.One2One)

    def test_FullConn_One2One_matrix(self):
        weight = np.array([2, 3, 4], np.int8)
        s1 = pb.FullConn(
            pb.IF((3,), 3), pb.IF((3,), 3), weight, conn_type=pb.SynConnType.One2One
        )

        assert (s1.num_in, s1.num_out) == (3, 3)
        assert np.array_equal(s1.weights, weight)
        assert np.array_equal(
            s1.connectivity, np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=np.int8)
        )
        assert s1.connectivity.dtype == WEIGHT_DTYPE
        assert s1.weight_width is WW.WEIGHT_WIDTH_4BIT

        weight = np.array([1, 0, 1, 0], np.int8)
        s2 = pb.FullConn(
            pb.IF((2, 2), 3), pb.IF((2, 2), 3), weight, conn_type=pb.SynConnType.One2One
        )

        assert (s2.num_in, s2.num_out) == (4, 4)
        assert np.array_equal(s2.weights, weight)
        assert np.array_equal(
            s2.connectivity,
            np.array(
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.int16
            ),
        )
        assert s2.connectivity.dtype == WEIGHT_DTYPE
        assert s2.weight_width is WW.WEIGHT_WIDTH_1BIT

    @pytest.mark.parametrize(
        "n1, n2",
        [
            (pb.IF(10, 3), pb.IF(10, 3)),
            (pb.IF((3, 3), 3), pb.IF((3, 3), 3)),
            (pb.IF((5,), 3), pb.IF((5,), 3)),
            (pb.IF(10, 3), pb.IF(100, 3)),
            (pb.IF((10, 10), 3), pb.IF((5, 5), 3)),
        ],
    )
    def test_FullConn_All2All(self, n1, n2):
        s1 = pb.FullConn(n1, n2, conn_type=pb.SynConnType.All2All)

        assert (s1.num_in, s1.num_out) == (n1.num_out, n2.num_in)
        assert s1.connectivity.dtype == WEIGHT_DTYPE
        assert np.array_equal(s1.weights, 1)
        assert np.array_equal(s1.connectivity, np.ones((n1.num_out, n2.num_in)))

    def test_FullConn_All2All_with_weights(self):
        n1 = pb.IF(3, 3)
        n2 = pb.IF(3, 3)

        """1. Single weight."""
        weight = 2
        s1 = pb.FullConn(n1, n2, weight, conn_type=pb.SynConnType.All2All)

        assert np.array_equal(s1.weights, weight)
        assert s1.connectivity.dtype == WEIGHT_DTYPE
        assert s1.weight_width is WW.WEIGHT_WIDTH_4BIT

        """2. Weights matrix."""
        weight = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        s2 = pb.FullConn(n1, n2, weight, conn_type=pb.SynConnType.All2All)

        assert s2.connectivity.dtype == WEIGHT_DTYPE
        assert np.array_equal(s2.weights, weight)
        assert np.array_equal(s2.connectivity, weight)

        # Wrong shape
        with pytest.raises(ShapeError):
            s3 = pb.FullConn(
                n1, n2, np.array([1, 2, 3]), conn_type=pb.SynConnType.All2All
            )

        with pytest.raises(ShapeError):
            s3 = pb.FullConn(
                n1,
                n2,
                np.array([[1, 2, 3], [4, 5, 6]]),
                conn_type=pb.SynConnType.All2All,
            )

        with pytest.raises(ShapeError):
            s3 = pb.FullConn(
                n1,
                n2,
                np.array([[1, 2], [4, 5], [6, 7]]),
                conn_type=pb.SynConnType.All2All,
            )

        with pytest.raises(ShapeError):
            s3 = pb.FullConn(
                n1,
                n2,
                np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8], [1, 2, 3]]),
                conn_type=pb.SynConnType.All2All,
            )


class TestMatMul2d:
    @pytest.mark.parametrize(
        "n1, n2, w_shape, expectation",
        [
            (pb.IF(10, 3), pb.IF(10, 3), (10, 10), nullcontext()),
            (pb.IF(10, 3), pb.IF((1, 10), 3), (10, 10), nullcontext()),
            (pb.IF((10, 2), 3), pb.IF((100,), 3), (2, 10), pytest.raises(ShapeError)),
            (pb.IF((2, 4, 6), 3), pb.IF((10,), 3), (12, 10), pytest.raises(ShapeError)),
            (pb.IF((8, 4), 3), pb.IF((4, 2), 3), (8, 2), nullcontext()),
        ],
    )
    def test_MatMul2d_instance(self, n1, n2, w_shape, expectation):
        weights = np.arange(shape2num(w_shape), dtype=np.int8).reshape(w_shape)

        with expectation:
            s = pb.MatMul2d(n1, n2, weights=weights)

            assert (s.num_in, s.num_out) == (n1.num_out, n2.num_in)
            assert s.connectivity.dtype == WEIGHT_DTYPE
            assert np.array_equal(s.weights, weights)


class TestConv:
    def test_Conv1d_instance(self):
        in_shape = (32,)
        ksize = (5,)
        stride = 2
        padding = 1
        groups = 2
        out_shape = ((32 + 2 - 5) // 2 + 1,)
        ci = 8
        co = 16
        ci_in_grp = ci // groups
        korder = "OIL"

        n1 = pb.IF((ci,) + in_shape, 3)  # CL
        n2 = pb.IF((co,) + out_shape, 3)

        weight = gen_random_array((co, ci_in_grp) + ksize, np.int8)
        s1 = pb.Conv1d(
            n1,
            n2,
            weight,
            stride=stride,
            padding=padding,
            kernel_order=korder,
            groups=groups,
        )

        assert s1.num_in == ci * shape2num(in_shape)
        assert s1.connectivity.dtype == WEIGHT_DTYPE
        assert s1.connectivity.shape == (
            ci * shape2num(in_shape),
            co * shape2num(out_shape),
        )

    def test_Conv2d_instance(self):
        in_shape = (32, 32)
        ksize = (5, 5)
        padding = (1, 1)
        stride = 2
        groups = 4
        out_shape = ((32 + 2 - 5) // 2 + 1, (32 + 2 - 5) // 2 + 1)
        ci = 8
        co = 16
        ci_in_grp = ci // groups
        korder = "IOHW"

        n1 = pb.IF((ci,) + in_shape, 3)
        # Strict output shape is no need
        n2 = pb.IF((co * out_shape[0] * out_shape[1],), 3)

        # korder
        weight = gen_random_array((ci_in_grp, co) + ksize, np.int8)
        s1 = pb.Conv2d(
            n1,
            n2,
            weight,
            stride=stride,
            padding=padding,
            kernel_order=korder,
            groups=groups,
        )

        assert s1.num_in == ci * shape2num(in_shape)
        assert s1.connectivity.dtype == WEIGHT_DTYPE
        assert s1.connectivity.shape == (
            ci * shape2num(in_shape),
            co * shape2num(out_shape),
        )

    def test_Conv1d_inchannel_omitted(self):
        in_shape = (32,)
        ksize = (5,)
        stride = 2
        out_shape = ((32 - 5) // 2 + 1,)
        groups = 1
        ci = 1  # omit it
        co = 4
        ci_in_grp = ci // groups
        korder = "IOL"

        n1 = pb.IF(in_shape, 3)  # HW, (ci=1)
        n2 = pb.IF((co,) + out_shape, 3)

        weight = gen_random_array((ci_in_grp, co) + ksize, np.int8)
        s1 = pb.Conv1d(
            n1, n2, weight, stride=stride, kernel_order=korder, groups=groups
        )

        assert s1.num_in == ci * shape2num(in_shape)
        assert s1.connectivity.dtype == WEIGHT_DTYPE
        assert s1.connectivity.shape == (
            ci * shape2num(in_shape),
            co * shape2num(out_shape),
        )

    def test_Conv2d_inchannel_omitted(self):
        in_shape = (32, 32)
        ksize = (5, 5)
        stride = 2
        groups = 1
        out_shape = ((32 - 5) // 2 + 1, (32 - 5) // 2 + 1)
        ci = 1  # omit it
        co = 4
        ci_in_grp = ci // groups
        korder = "IOHW"

        n1 = pb.IF(in_shape, 3)  # HW, (ci=1)
        n2 = pb.IF((co,) + out_shape, 3)

        weight = gen_random_array((ci_in_grp, co) + ksize, np.int8)
        s1 = pb.Conv2d(
            n1, n2, weight, stride=stride, kernel_order=korder, groups=groups
        )

        assert s1.num_in == ci * shape2num(in_shape)
        assert s1.connectivity.shape == (
            ci * shape2num(in_shape),
            co * shape2num(out_shape),
        )


class TestConvTranspose:
    def test_ConvTranspose1d_instance(self):
        in_shape = (14,)
        ksize = (5,)
        stride = 2
        padding = 1
        output_padding = 1
        out_shape = ((14 - 1) * 2 + 5 - 2 * 1 + 1,)
        ci = 16
        co = 8
        korder = "IOL"

        n1 = pb.IF((ci,) + in_shape, 3)  # CL
        n2 = pb.IF((co * out_shape[0],), 3)

        weight = np.random.randint(-128, 128, size=(ci, co) + ksize, dtype=np.int8)
        s1 = pb.ConvTranspose1d(
            n1,
            n2,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            kernel_order=korder,
        )

        assert s1.num_in == ci * shape2num(in_shape)
        assert s1.connectivity.dtype == WEIGHT_DTYPE
        assert s1.connectivity.shape == (
            ci * shape2num(in_shape),
            co * shape2num(out_shape),
        )

    def test_ConvTranspose2d_instance(self):
        in_shape = (14, 14)
        ksize = (5, 5)
        stride = 2
        padding = 1
        output_padding = 1
        out_shape = ((14 - 1) * 2 + 5 - 2 + 1, (14 - 1) * 2 + 5 - 2 + 1)
        ci = 8
        co = 16
        korder = "IOHW"

        n1 = pb.IF((ci,) + in_shape, 3)  # CHW
        n2 = pb.IF((co,) + out_shape, 3)

        weight = np.random.randint(-8, 8, size=(ci, co) + ksize, dtype=np.int32)
        s1 = pb.ConvTranspose2d(
            n1,
            n2,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            kernel_order=korder,
        )

        assert s1.num_in == ci * shape2num(in_shape)
        assert s1.connectivity.dtype == WEIGHT_DTYPE
        assert s1.connectivity.shape == (
            ci * shape2num(in_shape),
            co * shape2num(out_shape),
        )

    def test_ConvTranspose1d_inchannel_omitted(self):
        in_shape = (14,)
        ksize = (5,)
        stride = 2
        padding = 1
        output_padding = 1
        out_shape = ((14 - 1) * 2 + 5 - 2 * 1 + 1,)
        ci = 1  # omit it
        co = 4
        korder = "IOL"

        n1 = pb.IF(in_shape, 3)  # L, (ci=1)
        n2 = pb.IF((co,) + out_shape, 3)

        weight = np.random.randint(-128, 128, size=(ci, co) + ksize, dtype=np.int64)
        s1 = pb.ConvTranspose1d(
            n1,
            n2,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            kernel_order=korder,
        )

        assert s1.num_in == ci * shape2num(in_shape)
        assert s1.connectivity.dtype == WEIGHT_DTYPE
        assert s1.connectivity.shape == (
            ci * shape2num(in_shape),
            co * shape2num(out_shape),
        )

    def test_ConvTranspose2d_inchannel_omitted(self):
        in_shape = (14, 14)
        ksize = (5, 5)
        stride = 2
        padding = 1
        output_padding = 1
        out_shape = ((14 - 1) * 2 + 5 - 2 + 1, (14 - 1) * 2 + 5 - 2 + 1)
        ci = 1  # omit it
        co = 4
        korder = "IOHW"

        n1 = pb.IF(in_shape, 3)  # HW, (ci=1)
        n2 = pb.IF((co,) + out_shape, 3)

        weight = np.random.randint(-128, 128, size=(ci, co) + ksize, dtype=np.int8)
        s1 = pb.ConvTranspose2d(
            n1,
            n2,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            kernel_order=korder,
        )

        assert s1.num_in == ci * shape2num(in_shape)
        assert s1.connectivity.dtype == WEIGHT_DTYPE
        assert s1.connectivity.shape == (
            ci * shape2num(in_shape),
            co * shape2num(out_shape),
        )


class TestSTDPSynapse:
    @pytest.fixture(autouse=True)
    def enable_stdp_logging(self):
        set_logs(stdp=True)

    def test_STDPFullConn_update(self):
        # Use neurons to instantiate synapse but don't update them
        n1 = pb.STDPLIF(
            (3,),
            10,
            reset_v=0,
            leak_v=-1,
            bias=0,
            neg_threshold=-3,
            lateral_inhi_value=-1,
        )
        n2 = pb.STDPLIF(
            (3,),
            10,
            reset_v=0,
            leak_v=-1,
            bias=0,
            neg_threshold=-3,
            lateral_inhi_value=-1,
        )

        shape = (n1.num_out, n2.num_in)
        w = np.zeros(shape, dtype=WEIGHT_DTYPE)
        lut = np.zeros((60,), dtype=LUT_DTYPE)
        lut[:30] = -1
        lut[30:] = 1
        s1 = pb.STDPFullConn(n1, n2, w, weight_decay=-2, lut=lut)
        s1.learn()

        l = 12
        pre_spike = np.zeros((l, n1.num_out), dtype=NEUOUT_U8_DTYPE)
        pre_spike[1] = [1, 0, 0]
        pre_spike[6] = [0, 1, 1]
        pre_spike[9] = [0, 0, 1]
        pre_spike[11] = [1, 1, 0]

        post_spike = np.zeros((l, n2.num_in), dtype=NEUOUT_U8_DTYPE)
        post_spike[2] = [1, 0, 0]
        post_spike[8] = [1, 1, 1]
        post_spike[11] = [0, 1, 1]

        exp_w = np.zeros_like(s1.weights)
        for ts in range(l):
            s1.update_spike_counter(pre_spike[ts], post_spike[ts])
            s1.update_weight(s1.weights)

            # At ts=1, axon #0 LTD, others no learning. No weight decay.
            if ts == 1:
                exp_w[0, :] += -1
                assert np.array_equal(s1.weights, exp_w)

            # At ts=2, neu #0 LTP, others no learning. neu #0 weight decayed(-2).
            if ts == 2:
                exp_w[:, 0] += 1 - 2
                assert np.array_equal(s1.weights, exp_w)

            # At ts=6, axon #1#2 LTD, others no learning. No weight decay.
            if ts == 6:
                exp_w[1:3, :] += -1
                assert np.array_equal(s1.weights, exp_w)

            # At ts=8, all neurons LTP, others no learning. All weights decayed.
            if ts == 8:
                exp_w[:, :] += 1 - 2
                assert np.array_equal(s1.weights, exp_w)

            # At ts=9, axon #2 LTD, others no learning. No weight decay.
            if ts == 9:
                exp_w[2, :] += -1
                assert np.array_equal(s1.weights, exp_w)

            # At ts=11, axon #0#1 neu #0 LTD, neu #1#2 LTP, others no learning. axon #2 neu #1#2 weight decayed.
            if ts == 11:
                exp_w[0:2, 0] += -1  # LTD
                exp_w[:, 1:3] += 1  # LTP
                exp_w[2, 1:3] += -2  # weight decay
                assert np.array_equal(s1.weights, exp_w)

            print(f"ts={ts}, exp_w\n", exp_w)

    def test_attrs_export(self, ensure_dump_dir):
        n1 = pb.STDPLIF(
            (3,),
            10,
            reset_v=0,
            leak_v=-1,
            bias=0,
            neg_threshold=-3,
            lateral_inhi_value=-1,
            tick_wait_start=1,
        )
        n2 = pb.STDPLIF(
            (3,),
            10,
            reset_v=0,
            leak_v=-1,
            bias=0,
            neg_threshold=-3,
            lateral_inhi_value=-1,
            tick_wait_start=2,
        )

        shape = (n1.num_out, n2.num_in)
        w = np.zeros(shape, dtype=WEIGHT_DTYPE)
        lut = np.zeros((60,), dtype=LUT_DTYPE)
        lut[:30] = -1
        lut[30:] = 1
        s1 = pb.STDPFullConn(n1, n2, w, weight_decay=-2, lut=lut)

        attrs = s1.attrs()

        fp = ensure_dump_dir / f"stdp_syn{s1.name}.json"
        file_not_exist_fail(fp)

        with open(fp, "w") as f:
            json.dump({s1.name: attrs}, f, indent=2, cls=SynCfgJsonEncoder)
