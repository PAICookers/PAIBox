import paibox as pb
import numpy as np
class NetForTest3(pb.Network):

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=None, shape_out=(400,), name="inp1")
        # self.inp2 = pb.projection.InputProj(input=None, shape_out=(300,), name="inp2")
        self.n0 = pb.TonicSpiking(400, 3, name="n0")
        # self.n0_copy = pb.neuron.TonicSpiking(400, 3, name="n0_copy")
        self.n1 = pb.TonicSpiking(400, 3, name="n1")
        self.n2 = pb.TonicSpiking(800, 3, name="n2")
        self.n3 = pb.TonicSpiking(400, 4, name="n3")
        self.n4 = pb.TonicSpiking(300, 4, name="n4")
        self.s0 = pb.FullConn(
            self.inp1, self.n0, conn_type=pb.SynConnType.One2One, name="s0"
        )
        # self.s0_copy = pb.FullConn(
        #     self.inp1, self.n0_copy, conn_type=pb.SynConnType.One2One, name="s0_copy"
        # )
        self.s1 = pb.FullConn(
            self.n0, self.n1, conn_type=pb.SynConnType.One2One, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.n2, self.n3, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.FullConn(
            self.n0, self.n4, conn_type=pb.SynConnType.All2All, name="s4"
        )
        self.s5 = pb.FullConn(
            self.n4, self.n2, conn_type=pb.SynConnType.All2All, name="s5"
        )
        # self.s6 = pb.FullConn(
        #     self.inp2, self.n4, conn_type=pb.SynConnType.One2One, name="s6"
        # )
        
class NetForTest(pb.Network):

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=None, shape_out=(400,), name="inp1")
        self.inp2 = pb.InputProj(input=None, shape_out=(400,), name="inp2")
        self.n0 = pb.TonicSpiking(400, 3, name="n0")
        self.n1 = pb.TonicSpiking(400, 3, name="n1")
        self.n2 = pb.TonicSpiking(800, 3, name="n2")
        self.n3 = pb.TonicSpiking(400, 4, name="n3")
        self.n4 = pb.TonicSpiking(300, 4, name="n4")
        self.s0 = pb.FullConn(
            self.inp1, self.n0, conn_type=pb.SynConnType.One2One, name="s0"
        )
        self.s1 = pb.FullConn(
            self.inp1, self.n1, conn_type=pb.SynConnType.One2One, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n0, self.n2, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.FullConn(
            self.n2, self.n3, conn_type=pb.SynConnType.All2All, name="s4"
        )
        self.s5 = pb.FullConn(
            self.n2, self.n4, conn_type=pb.SynConnType.All2All, name="s5"
        )
        self.s6 = pb.FullConn(
            self.inp2, self.n4, conn_type=pb.SynConnType.All2All, name="s6"
        )
net = NetForTest()
mapper = pb.Mapper()
mapper.clear()
mapper.build(net)
graph_info = mapper.compile(
    weight_bit_optimization=False, grouping_optim_target="both"
)
print("OK")