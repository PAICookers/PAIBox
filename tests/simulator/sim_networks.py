import numpy as np
import paibox as pb


class Net1(pb.DynSysGroup):
    def __init__(self, n_neuron: int):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()

        self.inp = pb.InputProj(pe, shape_out=(n_neuron,), keep_shape=True)
        self.n1 = pb.LIF(n_neuron, threshold=3, reset_v=0, tick_wait_start=1)
        self.n2 = pb.IF(n_neuron, threshold=3, reset_v=1, tick_wait_start=2)
        self.s0 = pb.FullConn(
            self.inp,
            self.n1,
            weights=np.random.randint(-128, 128, size=(n_neuron,), dtype=np.int8),
            conn_type=pb.SynConnType.One2One,
        )
        self.s1 = pb.FullConn(
            self.n1,
            self.n2,
            weights=np.random.randint(
                -128, 128, size=(n_neuron, n_neuron), dtype=np.int8
            ),
            conn_type=pb.SynConnType.All2All,
        )

        # Probes inside
        self.n1_acti = pb.Probe(self.n1, "spike", name="n1_acti")
        self.s1_weight = pb.Probe(self.s1, "weights", name="s1_weight")
        self.n2_acti = pb.Probe(self.n2, "spike", name="n2_acti")


def fake_out_1(t, a, **kwargs):
    return t + a


def fake_out_2(t, b, **kwargs):
    return t + b


class Net2_with_multi_inpproj_func(pb.DynSysGroup):
    def __init__(self, n: int):
        super().__init__()

        self.inp1 = pb.InputProj(fake_out_1, shape_out=(n,), keep_shape=True)
        self.inp2 = pb.InputProj(fake_out_2, shape_out=(n,), keep_shape=True)
        self.n1 = pb.LIF(n, threshold=3, reset_v=0, tick_wait_start=1)
        self.s0 = pb.FullConn(
            self.inp1,
            self.n1,
            weights=np.ones((n,), dtype=np.int8),
            conn_type=pb.SynConnType.One2One,
        )
        self.s1 = pb.FullConn(
            self.inp2,
            self.n1,
            weights=np.ones((n,), dtype=np.int8),
            conn_type=pb.SynConnType.One2One,
        )

        # Probes inside
        self.inp1_output = pb.Probe(self.inp1, "output")
        self.inp2_output = pb.Probe(self.inp2, "output")
        self.n1_output = pb.Probe(self.n1, "spike")


class Net2_with_multi_inpproj_encoder(pb.DynSysGroup):
    def __init__(self, n: int):
        super().__init__()

        pe1 = pb.simulator.PoissonEncoder(seed=21)
        pe2 = pb.simulator.PoissonEncoder(seed=42)

        self.inp1 = pb.InputProj(pe1, shape_out=(n,), keep_shape=True)
        self.inp2 = pb.InputProj(pe2, shape_out=(n,), keep_shape=True)
        self.n1 = pb.LIF(n, threshold=3, reset_v=0, tick_wait_start=1)
        self.s0 = pb.FullConn(
            self.inp1,
            self.n1,
            weights=np.ones((n,), dtype=np.int8),
            conn_type=pb.SynConnType.One2One,
        )
        self.s1 = pb.FullConn(
            self.inp2,
            self.n1,
            weights=np.ones((n,), dtype=np.int8),
            conn_type=pb.SynConnType.One2One,
        )

        # Probes inside
        self.inp1_output = pb.Probe(self.inp1, "output")
        self.inp2_output = pb.Probe(self.inp2, "output")
        self.n1_output = pb.Probe(self.n1, "spike")


class Conv2d_Net(pb.Network):
    def __init__(self):
        super().__init__()

        pe1 = pb.simulator.PoissonEncoder()

        self.inp1 = pb.InputProj(pe1, shape_out=(8, 24, 24))
        self.n1 = pb.IF((16, 22, 22), threshold=10, reset_v=0, keep_shape=True)

        kernel = np.random.randint(-128, 128, size=(8, 16, 3, 3), dtype=np.int8)
        stride = 1

        self.conv1 = pb.Conv2d(
            self.inp1, self.n1, kernel, stride=stride, kernel_order="IOHW"
        )

        self.prob1 = pb.Probe(self.n1, "spike")
        self.prob2 = pb.Probe(self.n1, "feature_map")
