from typing import Literal

import numpy as np

import paibox as pb
from paibox.components import Neuron
from paibox.node import NodeList


def _out_bypass1(t, data1, *args, **kwargs):
    return data1


def _out_bypass2(t, data2, *args, **kwargs):
    return data2


def _out_bypass3(t, data3, *args, **kwargs):
    return data3


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

    def __init__(self, name: str | None = None):
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

    def __init__(self, name: str | None = None):
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


class FModule_ConnWithInput_Net(pb.DynSysGroup):
    """A network where an input node is connected to a module.

    Structure:
        inp1 -> s1 -> n1 ->
                    inp2 -> and1 -> s2 -> n2
    """

    def __init__(self):
        super().__init__()

        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=(10,))
        self.inp2 = pb.InputProj(input=_out_bypass2, shape_out=(10,))
        self.n1 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
        self.s1 = pb.FullConn(self.inp1, self.n1, conn_type=pb.SynConnType.One2One)

        self.and1 = pb.BitwiseAND(self.n1, self.inp2, delay=1, tick_wait_start=2)
        self.n2 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=3)
        self.s2 = pb.FullConn(self.and1, self.n2, conn_type=pb.SynConnType.All2All)

        self.probe1 = pb.Probe(self.n1, "spike")
        self.probe2 = pb.Probe(self.and1, "spike")


class FModule_ConnWithModule_Net(pb.DynSysGroup):
    """A network where one module is connected to another module.

    Structure:
        inp1 -> s1 -> n1 ->
        inp2 -> s2 -> n2 -> and1 ->
               inp3-> s3 ->  n3  -> or1 -> s4 -> n4
    """

    def __init__(self):
        super().__init__()

        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=(10,))
        self.inp2 = pb.InputProj(input=_out_bypass2, shape_out=(10,))
        self.inp3 = pb.InputProj(input=_out_bypass3, shape_out=(10,))
        self.n1 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
        self.n2 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
        self.n3 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=2)  # tws = 2!
        self.s1 = pb.FullConn(self.inp1, self.n1, conn_type=pb.SynConnType.One2One)
        self.s2 = pb.FullConn(self.inp2, self.n2, conn_type=pb.SynConnType.One2One)
        self.s3 = pb.FullConn(self.inp3, self.n3, conn_type=pb.SynConnType.One2One)

        self.and1 = pb.BitwiseAND(self.n1, self.n2, delay=1, tick_wait_start=2)
        self.or1 = pb.BitwiseOR(self.and1, self.n3, delay=1, tick_wait_start=3)
        self.n4 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=4)
        self.s4 = pb.FullConn(self.or1, self.n4, conn_type=pb.SynConnType.All2All)

        self.probe1 = pb.Probe(self.n1, "spike")
        self.probe2 = pb.Probe(self.and1, "spike")
        self.probe3 = pb.Probe(self.or1, "spike")


class FModule_ConnWithFModule_Net(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.n1 = pb.IF((8, 16, 16), 10, tick_wait_start=1)
        self.n2 = pb.IF((8, 4, 4), 5, tick_wait_start=2)
        self.mp2d = pb.SpikingMaxPool2d(self.n1, (4, 4), tick_wait_start=2)
        self.sub = pb.SpikingSub(self.n2, self.mp2d, tick_wait_start=3)

        self.s1 = pb.FullConn(self.n1, self.n2)


_2to1_op = {
    "and": pb.BitwiseAND,
    "or": pb.BitwiseOR,
    "xor": pb.BitwiseXOR,
    "add": pb.SpikingAdd,
    "sub": pb.SpikingSub,
}


class FunctionalModule_2to1_Net(pb.DynSysGroup):
    def __init__(self, op: Literal["and", "or", "xor", "add", "sub"]):
        super().__init__()
        self.bitwise = 10

        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=(10,))
        self.inp2 = pb.InputProj(input=_out_bypass2, shape_out=(10,))
        self.n1 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
        self.n2 = pb.IF((10,), 1, 0, delay=1, tick_wait_start=1)
        self.s1 = pb.FullConn(self.inp1, self.n1, conn_type=pb.SynConnType.One2One)
        self.s2 = pb.FullConn(self.inp2, self.n2, conn_type=pb.SynConnType.One2One)

        self.func_node = _2to1_op[op](self.n1, self.n2, delay=1, tick_wait_start=2)

        self.n3 = pb.BypassNeuron(
            (10,),
            delay=1,
            tick_wait_start=self.func_node.tick_wait_start
            + self.func_node.external_delay,
        )
        self.s3 = pb.FullConn(self.func_node, self.n3, conn_type=pb.SynConnType.One2One)

        self.probe1 = pb.Probe(self.n1, "spike")
        self.probe2 = pb.Probe(self.n2, "spike")
        self.probe3 = pb.Probe(self.func_node, "spike")
        self.probe4 = pb.Probe(self.n3, "spike")

        if hasattr(self.func_node, "voltage"):
            self.probe5 = pb.Probe(self.func_node, "voltage")


class FunctionalModule_1to1_Net(pb.DynSysGroup):
    def __init__(self, op: Literal["not", "delay"]):
        super().__init__()
        self.bitwise = 10

        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=(10,))
        self.n1 = pb.IF((10,), 1, 0)
        self.s1 = pb.FullConn(self.inp1, self.n1, conn_type=pb.SynConnType.One2One)

        if op == "not":
            self.func_node = pb.BitwiseNOT(self.n1, tick_wait_start=2)
        elif op == "delay":
            if hasattr(pb, "DelayChain"):
                self.func_node = pb.DelayChain(  # type: ignore
                    self.n1, chain_level=5, tick_wait_start=2
                )
            else:
                from paibox.components._modules import _DelayChainSNN

                self.func_node = _DelayChainSNN(
                    self.n1, chain_level=5, tick_wait_start=2
                )

        self.n2 = pb.BypassNeuron(
            (10,),
            delay=1,
            tick_wait_start=self.func_node.tick_wait_start
            + self.func_node.external_delay,
        )
        self.s3 = pb.FullConn(self.func_node, self.n2, conn_type=pb.SynConnType.One2One)

        self.probe1 = pb.Probe(self.n1, "spike")
        self.probe2 = pb.Probe(self.func_node, "spike")
        self.probe3 = pb.Probe(self.n2, "spike")

        if hasattr(self.func_node, "voltage"):
            self.probe4 = pb.Probe(self.func_node, "voltage")


_pool_op = {
    (1, "avg"): pb.SpikingAvgPool1d,
    (1, "avgv"): pb.SpikingAvgPool1dWithV,
    (2, "avg"): pb.SpikingAvgPool2d,
    (2, "avgv"): pb.SpikingAvgPool2dWithV,
    (1, "max"): pb.SpikingMaxPool1d,
    (2, "max"): pb.SpikingMaxPool2d,
}


class _SpikingPoolNd_Net(pb.DynSysGroup):
    def __init__(
        self, pool_ndim, fm_shape, ksize, stride, padding, threshold, pool_type
    ):
        super().__init__()
        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=fm_shape)
        self.n1 = pb.BypassNeuron(fm_shape, tick_wait_start=1)
        self.s1 = pb.FullConn(self.inp1, self.n1, conn_type=pb.SynConnType.One2One)

        self.pool = _pool_op[(pool_ndim, pool_type)](
            self.n1,
            ksize,
            stride,
            padding,
            threshold=threshold,  # no need for maxpool
            delay=1,
            tick_wait_start=2,
        )

        self.n2 = pb.BypassNeuron(self.pool.shape_out, delay=1, tick_wait_start=3)
        self.s3 = pb.FullConn(self.pool, self.n2, conn_type=pb.SynConnType.One2One)

        self.probe1 = pb.Probe(self.n1, "spike")
        self.probe2 = pb.Probe(self.pool, "spike")
        self.probe3 = pb.Probe(self.n2, "spike")


class SpikingPool1d_Net(_SpikingPoolNd_Net):
    pool_ndim = 1

    def __init__(self, fm_shape, ksize, stride, padding, threshold, pool_type):
        super().__init__(
            self.pool_ndim, fm_shape, ksize, stride, padding, threshold, pool_type
        )


class SpikingPool2d_Net(_SpikingPoolNd_Net):
    pool_ndim = 2

    def __init__(self, fm_shape, ksize, stride, padding, threshold, pool_type):
        super().__init__(
            self.pool_ndim, fm_shape, ksize, stride, padding, threshold, pool_type
        )


class Conv2dSemiFolded_FC_ChainNetN(pb.DynSysGroup):
    def __init__(self, shape, kernels, strides, paddings, out_features, weight, groups):
        super().__init__()

        self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape)
        self.conv_list = NodeList()

        for i, (kernel, stride, padding, g) in enumerate(
            zip(kernels, strides, paddings, groups)
        ):
            self.conv_list.append(
                pb.Conv2dSemiFolded(
                    self.conv_list[-1] if i > 0 else self.i1,
                    kernel,
                    stride,
                    padding,
                    tick_wait_start=1 + 2 * i,
                    groups=g,
                )
            )

        self.linear1 = pb.LinearSemiFolded(
            self.conv_list[-1],
            out_features,
            weight,
            bias=0,
            tick_wait_start=self.conv_list[-1].tick_wait_start + 2,
        )


_pool_semi_op = {
    "avg": pb.AvgPool2dSemiFolded,
    "max": pb.MaxPool2dSemiFolded,
}


class Pool2dSemiFolded_FC_ChainNetN(pb.DynSysGroup):
    def __init__(
        self, shape, kernel_sizes, strides, paddings, out_features, weight, pool_type
    ):
        super().__init__()
        self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape)
        self.pool_list = NodeList()

        for i, (ksize, stride) in enumerate(zip(kernel_sizes, strides)):
            if pool_type == "max":
                pool = _pool_semi_op[pool_type](
                    self.pool_list[-1] if i > 0 else self.i1,
                    ksize,
                    stride,
                    tick_wait_start=1 + 2 * i,
                )
            else:
                pool = _pool_semi_op[pool_type](
                    self.pool_list[-1] if i > 0 else self.i1,
                    ksize,
                    stride,
                    padding=paddings[i],
                    tick_wait_start=1 + 2 * i,
                )
            self.pool_list.append(pool)

        self.linear1 = pb.LinearSemiFolded(
            self.pool_list[-1],
            out_features,
            weights=weight,
            bias=0,
            tick_wait_start=self.pool_list[-1].tick_wait_start + 2,
        )


class Linear_Net(pb.DynSysGroup):
    def __init__(self, shape, weight1):
        super().__init__()
        self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape)
        self.linear1 = pb.Linear(self.i1, 10, weights=weight1, bias=2)
        self.probe1 = pb.Probe(self.linear1, "spike")


_pool_op_1d = {"avg": pb.AvgPool1d, "max": pb.MaxPool1d}
_pool_op_2d = {"avg": pb.AvgPool2d, "max": pb.MaxPool2d}


class Pool1d_FC_ChainNetN(pb.DynSysGroup):
    def __init__(
        self, shape, kernel_sizes, strides, paddings, out_features, weight, pool_type
    ):
        super().__init__()
        self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape)
        self.pool_list = NodeList()

        for i, (ksize, stride) in enumerate(zip(kernel_sizes, strides)):
            pool = _pool_op_1d[pool_type](
                self.pool_list[-1] if i > 0 else self.i1,
                ksize,
                stride,
                paddings[i],
                tick_wait_start=1 + 2 * i,
            )
            self.pool_list.append(pool)

        self.linear1 = pb.Linear(
            self.pool_list[-1],
            out_features,
            weights=weight,
            bias=0,
            tick_wait_start=self.pool_list[-1].tick_wait_start + 2,
        )


class Pool2d_FC_ChainNetN(pb.DynSysGroup):
    def __init__(
        self, shape, kernel_sizes, strides, paddings, out_features, weight, pool_type
    ):
        super().__init__()
        self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape)
        self.pool_list = NodeList()

        for i, (ksize, stride) in enumerate(zip(kernel_sizes, strides)):
            pool = _pool_op_2d[pool_type](
                self.pool_list[-1] if i > 0 else self.i1,
                ksize,
                stride,
                paddings[i],
                tick_wait_start=1 + 2 * i,
            )
            self.pool_list.append(pool)

        self.linear1 = pb.Linear(
            self.pool_list[-1],
            out_features,
            weights=weight,
            bias=0,
            tick_wait_start=self.pool_list[-1].tick_wait_start + 2,
        )


class ANNNetwork(pb.Network):
    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=_out_bypass1, shape_out=(32, 32))

        n1_bias = np.random.randint(-128, 128, size=(4,), dtype=np.int8)
        self.n1 = pb.LIF(
            (4, 30, 30),
            100,
            bias=n1_bias,
            tick_wait_start=1,
            input_width=8,
            spike_width=8,
            snn_en=False,
        )
        n2_bias = np.random.randint(-128, 128, size=(4,), dtype=np.int8)
        self.n2 = pb.LIF(
            (4, 28, 28),
            50,
            bias=n2_bias,
            tick_wait_start=2,
            input_width=8,
            spike_width=8,
            snn_en=False,
        )
        self.n3 = pb.LIF(
            (2, 26, 26),
            20,
            bias=1,
            tick_wait_start=3,
            input_width=8,
            spike_width=8,
            snn_en=False,
        )
        self.n4 = pb.IF(
            (100,), 10, tick_wait_start=4, input_width=8, spike_width=8, snn_en=False
        )

        kernel_1 = np.random.randint(-128, 128, size=(4, 1, 3, 3), dtype=np.int8)
        self.conv2d_1 = pb.Conv2d(self.inp1, self.n1, kernel_1)

        kernel_2 = np.random.randint(-128, 128, size=(4, 4, 3, 3), dtype=np.int8)
        self.conv2d_2 = pb.Conv2d(self.n1, self.n2, kernel_2)

        kernel_3 = np.random.randint(-128, 128, size=(2, 4, 3, 3), dtype=np.int8)
        self.conv2d_3 = pb.Conv2d(self.n2, self.n3, kernel_3)

        w4 = np.random.randint(-128, 128, size=(2 * 26 * 26, 100), dtype=np.int8)
        self.fc1 = pb.FullConn(self.n3, self.n4, w4)


class STDPLinearNet(pb.Network):
    def __init__(
        self, in_feature1: int, in_features2: int, out_features: int, weight1, weight2
    ):
        super().__init__()
        self.input = pb.InputProj(input=None, shape_out=in_feature1)

        lut1 = np.zeros((60,), dtype=np.int8)
        lut1[:30] = -1
        lut1[30:] = 1

        self.n1 = pb.STDPLIF(in_features2, 10, lateral_inhi_value=1, tick_wait_start=1)
        self.n2 = pb.STDPLIF(out_features, 1, -1, tick_wait_start=2)

        self.s1 = pb.STDPFullConn(self.input, self.n1, weight1, lut=lut1)

        lut2 = np.zeros((60,), dtype=np.int8)
        lut2[:30] = -2
        lut2[30:] = 2
        self.s2 = pb.STDPFullConn(self.n1, self.n2, weight2, lut=lut2)
