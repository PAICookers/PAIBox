import paibox as pb


class fcnet_2layer_dual_port(pb.Network):
    def __init__(self, weight1, Vthr1, weight2, Vthr2):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(392,))
        self.i2 = pb.InputProj(input=pe, shape_out=(392,))
        self.n1 = pb.neuron.IF(128, threshold=Vthr1, reset_v=0)
        self.s1 = pb.synapses.NoDecay(
            self.i1,
            self.n1,
            weights=weight1[:392],
            conn_type=pb.SynConnType.All2All,
        )
        self.s2 = pb.synapses.NoDecay(
            self.i2,
            self.n1,
            weights=weight1[392:],
            conn_type=pb.SynConnType.All2All,
        )

        # tick_wait_start = 2 for second layer
        self.n2 = pb.neuron.IF(
            5, threshold=Vthr2, reset_v=0, tick_wait_start=2, name="batch_dual_port_o1"
        )
        self.n3 = pb.neuron.IF(
            5, threshold=Vthr2, reset_v=0, tick_wait_start=2, name="batch_dual_port_o2"
        )
        self.s3 = pb.synapses.NoDecay(
            self.n1,
            self.n2,
            weights=weight2[:, :5],
            conn_type=pb.SynConnType.All2All,
        )
        self.s4 = pb.synapses.NoDecay(
            self.n1,
            self.n3,
            weights=weight2[:, 5:],
            conn_type=pb.SynConnType.All2All,
        )

        self.probe1 = pb.simulator.Probe(target=self.n2, attr="spike")
        self.probe2 = pb.simulator.Probe(target=self.n3, attr="spike")
