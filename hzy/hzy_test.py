

import numpy as np
import paibox as pb
from paibox.components.functional import Conv_HalfRoll, Filter
from paibox.components.synapses import Conv2dHalfRollSyn
from paibox.components.synapses.conv_utils import _conv2d_halfroll
from paibox.simulator.utils import _conv2d_faster_fp32


weight1 = np.random.randint(0, 10, size=(32, 1, 5, 5), dtype=np.int8)
weight2 = np.random.randint(0, 10, size=(32, 32, 2, 2), dtype=np.int8)
weight3 = np.random.randint(0, 10, size=(64, 32, 5, 5), dtype=np.int8)
weight4 = np.random.randint(0, 10, size=(64, 64, 2, 2), dtype=np.int8)

class Conv2d_Net(pb.Network):
    def __init__(self, Vthr1, Vthr2, Vthr3):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(1, 28, 28))
        self.n1 = pb.IF((32, 24, 24), threshold=Vthr1, reset_v=0)
        self.conv2d_1 = pb.Conv2d(self.i1, self.n1, kernel=weight1, stride=1)

        self.n2 = pb.IF((32, 12, 12), threshold=Vthr2, reset_v=0, tick_wait_start=2)
        self.conv2d_2 = pb.Conv2d(self.n1, self.n2, kernel=weight2, stride=2)

        self.n3 = pb.IF((64, 8, 8), threshold=Vthr3, reset_v=0, tick_wait_start=3)
        self.conv2d_3 = pb.Conv2d(self.n2, self.n3, kernel=weight3, stride=1)
        self.n4 = pb.IF((64, 4, 4), threshold=Vthr3, reset_v=0, tick_wait_start=4)
        self.conv2d_4 = pb.Conv2d(self.n3, self.n4, kernel=weight4, stride=2)
        self.n5 = pb.IF((256,), threshold=Vthr3, reset_v=0, tick_wait_start=5)
        self.fc1 = pb.FullConn(
            self.n4, self.n5, weights=np.random.randint(0, 10, size=(1024, 256), dtype=np.int8),
            conn_type=pb.SynConnType.All2All
        )
        self.n6 = pb.IF((64,), threshold=Vthr3, reset_v=0, tick_wait_start=6)
        self.fc2 = pb.FullConn(
            self.n5, self.n6, weights=np.random.randint(0, 10, size=(256, 64), dtype=np.int8),
            conn_type=pb.SynConnType.All2All
        )
        self.n7 = pb.IF((10,), threshold=Vthr3, reset_v=0, tick_wait_start=7)
        self.fc3 = pb.FullConn(
            self.n6, self.n7, weights=np.random.randint(0, 10, size=(64, 10), dtype=np.int8),
            conn_type=pb.SynConnType.All2All
        )

        self.probe1 = pb.Probe(self.n3, "spike")



input_data2 = np.array([1,0,1,0,1], dtype=np.bool_)
class fcnet_4(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(1, 28, 28))
        #self.i1 = pb.InputProj(input=out_bypass1, shape_out=(1, 5))
        self.n1 = pb.IF((1, 28), threshold=4, reset_v=0, name="n_1")
        self.s0 = pb.FullConn(
            self.i1,
            self.n1,
            weights=1,
            conn_type=pb.SynConnType.All2All,
        )
        # self.probe1 = pb.Probe(self.n1, "spike")
        self.n2 = pb.IF((32, 24, 24), threshold=0, reset_v=0, name="n_2")
        #self.conv1 = pb.ConvHalfRoll(self.i1, self.n1, np.array([[[[2,1,2],[1,2,1],[1,2,3]]]], dtype=np.int8), 1, tick_wait_start=1)
        self.conv1 = pb.ConvHalfRoll(self.n1, self.n2, weight1, 1)
        self.n3 = pb.IF((32, 12, 12), threshold=1, reset_v=0, name="n_3")
        self.conv2 = pb.ConvHalfRoll(self.n2, self.n3, weight2, 2)
        self.n4 = pb.IF((64, 8, 8), threshold=1, reset_v=0, name="n_4")
        self.conv3 = pb.ConvHalfRoll(self.n3, self.n4, weight3, 1)
        self.n5 = pb.IF((64, 4, 4), threshold=1, reset_v=0, name="n_5")
        self.conv4 = pb.ConvHalfRoll(self.n4, self.n5, weight4, 2)
        self.n6 = pb.IF((256,), threshold=1, reset_v=0, name="n_6")
        self.linear1 = pb.DelayFullConn(
            self.n5,
            self.n6,
            delay=4,
            weights=np.random.randint(0, 10, size=(1024, 256), dtype=np.int8),
            conn_type=pb.SynConnType.All2All,
        )
        self.n7 = pb.IF((64,), threshold=1, reset_v=0, name="n_7")
        self.linear2 = pb.FullConn(
            self.n6,
            self.n7,
            weights=np.random.randint(0, 10, size=(256, 64), dtype=np.int8),
            conn_type=pb.SynConnType.All2All,
        )
        self.n8 = pb.IF((10,), threshold=1, reset_v=0, name="n_8")
        self.linear2 = pb.FullConn(
            self.n7,
            self.n8,
            weights=np.random.randint(0, 10, size=(64, 10), dtype=np.int8),
            conn_type=pb.SynConnType.All2All,
        )
        self.filter = pb.Filter(self.n8, 28)


def out_bypass1(t, data1, *args, **kwargs):
    return data1

# input_data1 = np.array([[1,2,5,7,5],
#                        [2,0,8,8,2],
#                        [3,8,5,7,5],
#                        [4,9,2,5,4],
#                        [5,10,2,3,8],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0],
#                        [0,0,0,0,0]], dtype=np.int8)
#
# weight1 = np.array([[1,0],
#  [0 ,1],
#  [1 ,0],
#  [0 ,1],
#  [1 ,0],
#  [0 ,1],
#  [0 ,1],
#  [0 ,0],
#  [1 ,1]], dtype=np.int8)
inpa = np.random.randint(0, 2, size=(1, 11, 11)).astype(np.int8)
inpb = np.concatenate([inpa, np.zeros((1, 10, 11))], axis=1)
weight = np.random.randint(0, 2, size=(3*3, 2), dtype=np.int8)
class fcnet_5(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.i1 = pb.InputProj(input=out_bypass1, shape_out=(1, 11))
        self.conv1 = pb.ConvHalfRoll(self.i1, np.array([[[[2,1,2],[1,-2,1],[-1,2,-3]]]], dtype=np.int8), 2, 0, tick_wait_start=1)
        self.conv2 = pb.ConvHalfRoll(self.conv1, np.array([[[[2,1,2],[1,-2,1],[-1,2,-3]]]], dtype=np.int8), 1, 0, tick_wait_start=3)
        self.linear1 = pb.DelayFullConn(
            self.conv2,
            2,
            weights=weight,
            conn_type=pb.SynConnType.All2All,
            tick_wait_start=5
        )

pb_net1 = fcnet_5()
conv = pb_net1.conv2
linear = pb_net1.linear1
generated = pb.DynSysGroup.build_fmodule(pb_net1)

sim1 = pb.Simulator(pb_net1, start_time_zero=False)


probe_conv = pb.Probe(generated[conv][0], "output")
probe_linear = pb.Probe(generated[linear][0], "output")
sim1.add_probe(probe_conv)
sim1.add_probe(probe_linear)
for i in range(20):
    pb.FRONTEND_ENV.save(data1=inpb[0][i])
    sim1.run(1)
    #print(pb_net1.nd_Delay_FullConn_0.output)
    #sim2.run(1)
for i in range(17):
#     print(sim1.data[probe_conv][i])
    print(sim1.data[probe_linear][i])
data = np.array(sim1.data[probe_conv][8:15])
print(data)
#data = np.transpose(data, (1, 0))
print(data)
# output = data.ravel() @ weight
# print(output)
# output =_conv2d_faster_fp32(np.array([[[1,2,3,4,5],[2,0,8,9,10],[5,8,5,2,2],[7,8,7,5,3],[5,2,5,4,8]]]),
#                             np.array([[[[2,1,2],[1,-2,1],[-1,2,-3]]]], dtype=np.int8),
#                             (2,2),
#                             (1,1))
# output[output < 0] = 0
# print(output)
# #output = np.transpose(output, (0, 2, 1))
#
# #print(output.ravel() @ weight1)
# output = _conv2d_faster_fp32(output, np.array([[[[2,1,2],[1,-2,1],[-1,2,-3]]]], dtype=np.int8),(2,2),(0,0))
# output[output < 0] = 0
# print(output)


class deeplabv2(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(3, 256))
        self.n1 = pb.LIF((64, 254), threshold=0, reset_v=0, name="n_1")
        self.conv1 = pb.ConvHalfRoll(self.i1, self.n1, np.random.randint(0, 10, size=(64,3,3,3), dtype=np.int8), 1)
        self.n2 = pb.LIF((64, 252), threshold=0, reset_v=0, name="n_2")
        self.conv2 = pb.ConvHalfRoll(self.n1, self.n2, np.random.randint(0, 10, size=(64,64,3,3), dtype=np.int8), 1)
        self.n3 = pb.LIF((64, 127), threshold=0, reset_v=0, name="n_3")
        self.maxpool2d1 = pb.ConvHalfRoll(self.n2, self.n3, np.random.randint(0, 1, size=(64, 64, 3, 3), dtype=np.bool_), 2)
        self.n4 = pb.LIF((128, 125), threshold=0, reset_v=0, name="n_4")
        self.conv3 = pb.ConvHalfRoll(self.n3, self.n4, np.random.randint(0, 10, size=(128,64,3,3), dtype=np.int8), 1)
        self.n5 = pb.LIF((128, 123), threshold=0, reset_v=0, name="n_5")
        self.conv4 = pb.ConvHalfRoll(self.n4, self.n5, np.random.randint(0, 10, size=(128, 128, 3, 3), dtype=np.int8), 1)
        self.n6 = pb.LIF((128, 62), threshold=0, reset_v=0, name="n_6")
        self.maxpool2d2 = pb.ConvHalfRoll(self.n5, self.n6, np.random.randint(0, 1, size=(128, 128, 3, 3), dtype=np.bool_), 2)
        self.n7 = pb.LIF((128, 60), threshold=0, reset_v=0, name="n_7")
        self.conv5 = pb.ConvHalfRoll(self.n6, self.n7, np.random.randint(0, 10, size=(128, 128, 3, 3), dtype=np.int8), 1)
        self.n8 = pb.LIF((2, 58), threshold=0, reset_v=0, name="n_8")
        self.conv6 = pb.ConvHalfRoll(self.n7, self.n8, np.random.randint(0, 10, size=(2, 128, 3, 3), dtype=np.int8), 1)
        self.n9 = pb.IF((116,), threshold=1, reset_v=0, name="n_9")
        self.linear2 = pb.FullConn(
            self.n8,
            self.n9,
            weights=np.random.randint(0, 1, size=(116, 116), dtype=np.bool_),
            conn_type=pb.SynConnType.All2All,
        )
# w = np.array(   [[
#                 [[2, 2, 2],[5,5,5],[9,9,9]],
#                 [[1, 1, 1],[4,4,4],[7,7,7]],
#                 ]]
# )
w = np.random.randint(1, 10, size=(1,1,3,3), dtype=np.int8)
class deeplabv3(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(1, 10))
        self.n1 = pb.LIF((1, 10), threshold=0, reset_v=0, name="n_1")
        self.conv1 = pb.ConvHalfRoll(self.i1, self.n1, w, 1)


        # self.n2 = pb.LIF((64, 28), threshold=0, reset_v=0, name="n_2")
        # self.conv2 = pb.ConvHalfRoll(self.n1, self.n2, np.random.randint(0, 10, size=(64,64,3,3), dtype=np.int8), 1)
        # self.n3 = pb.LIF((100, 24), threshold=0, reset_v=0, name="n_3")
        # self.maxpool2d1 = pb.ConvHalfRoll(self.n1, self.n3, np.random.randint(0, 1, size=(100, 100, 3, 3), dtype=np.bool_), 2)
        # #
        # self.n4 = pb.LIF((8, 22), threshold=0, reset_v=0, name="n_4")
        # self.conv3 = pb.ConvHalfRoll(self.n3, self.n4, np.random.randint(0, 10, size=(8,100,3,3), dtype=np.int8), 1)
        # # # self.n5 = pb.LIF((128, 10), threshold=0, reset_v=0, name="n_5")
        # # # self.conv4 = pb.ConvHalfRoll(self.n4, self.n5, np.random.randint(0, 10, size=(128, 128, 3, 3), dtype=np.int8), 1)
        # # # self.n6 = pb.LIF((128, 5), threshold=0, reset_v=0, name="n_6")
        # # # self.maxpool2d2 = pb.ConvHalfRoll(self.n5, self.n6, np.random.randint(0, 1, size=(128, 128, 3, 3), dtype=np.bool_), 2)
        # # # self.n7 = pb.LIF((128, 3), threshold=0, reset_v=0, name="n_7")
        # # # self.conv5 = pb.ConvHalfRoll(self.n6, self.n7, np.random.randint(0, 10, size=(128, 128, 3, 3), dtype=np.int8), 1)
        # self.n8 = pb.LIF((2, 251), threshold=0, reset_v=0, name="n_8")
        # self.conv6 = pb.ConvHalfRoll(self.n4, self.n8, np.random.randint(0, 10, size=(2, 8, 3, 3), dtype=np.int8), 1)
        # self.n9 = pb.IF((54,), threshold=1, reset_v=0, name="n_9")
        # self.linear2 = pb.DelayFullConn(
        #     self.n8,
        #     self.n9,
        #     delay=27,
        #     weights=np.random.randint(0, 1, size=(2*27*27, 54), dtype=np.bool_),
        #     conn_type=pb.SynConnType.All2All,
        # )
        # self.linear2 = pb.FullConn(
        #     self.n8,
        #     self.n9,
        #     weights=np.random.randint(0, 1, size=(2 * 9, 10), dtype=np.bool_),
        #     conn_type=pb.SynConnType.All2All,
        # )
class snn3(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(1, 128))
        self.n1 = pb.LIF((64, 128), threshold=0, reset_v=0, name="n_1")
        self.conv1 = pb.ConvHalfRoll(self.i1, self.n1, np.random.randint(0,10, size=(64,1,3,3), dtype=np.int8), 1)
        self.n3 = pb.LIF((64, 64), threshold=0, reset_v=0, name="n_3")
        self.maxpool2d1 = pb.ConvHalfRoll(self.n1, self.n3, np.random.randint(0, 1, size=(64, 64, 2, 2), dtype=np.bool_), 2)
        self.n4 = pb.LIF((64, 64), threshold=0, reset_v=0, name="n_4")
        self.conv2 = pb.ConvHalfRoll(self.n3, self.n4, np.random.randint(0, 10, size=(64, 64, 3, 3), dtype=np.int8), 1)
        self.n5 = pb.LIF((64, 32), threshold=0, reset_v=0, name="n_5")
        self.maxpool2d2 = pb.ConvHalfRoll(self.n4, self.n5,
                                          np.random.randint(0, 1, size=(64, 64, 2, 2), dtype=np.bool_), 2)
        self.n6 = pb.LIF((64, 4, 4), threshold=0, reset_v=0, name="n_6")

        self.linear1 = pb.DelayFullConn(
            self.n5,
            self.n6,
            delay=32,
            weights=np.random.randint(0, 10, size=(64*32*32, 64*4*4), dtype=np.int8),
            conn_type=pb.SynConnType.All2All,
        )
        self.n7 = pb.LIF((10,), threshold=0, reset_v=0, name="n_7")

        self.linear2 = pb.FullConn(
            self.n6,
            self.n7,
            weights=np.random.randint(0, 10, size=(64 * 4 * 4, 10), dtype=np.int8),
            conn_type=pb.SynConnType.All2All,
        )
kernel = np.array([[[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]]], dtype=np.int8)
class paddingnet(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(1, 4))
        self.n1 = pb.IF((1, 4, 4), threshold=0, reset_v=0, name="n_1")
        self.conv = pb.ConvHalfRoll(self.i1, self.n1, kernel, stride=1, padding=1)

#pb_net.conv.build(pb_net, 3)
#
# pb.BACKEND_CONFIG.target_chip_addr = [(0, 0), (0, 1)]
# mapper = pb.Mapper()
# mapper.build(pb_net)
#
# graph_info = mapper.compile()
# print("Core required:", graph_info["n_core_required"])
# print("Core occupied:", graph_info["n_core_occupied"])



# #print(graph_info["members"])
# for k, v in graph_info["members"].items():
#     for c, coreplm in v.items():
#         print(c)
#         for k, v in coreplm.neuron_configs.items():
#             print(k.name,v)
#             for n,s in k.master_nodes.items():
#                 print(s.name)
#                 # print(s.connectivity)

