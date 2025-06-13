import sys
from pathlib import Path

import numpy as np

sys.path.append("/Users/xiahongtu/workspace/PAIBox")

import paibox as pb


class Conv2d_Net(pb.Network):
    def __init__(self, weight1, Vthr1, weight2, Vthr2):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(1, 52, 52))
        self.n1 = pb.IF((8, 48, 48), threshold=Vthr1, reset_v=0)
        self.conv2d_1 = pb.Conv2d(self.i1, self.n1, kernel=weight1, stride=1)

        self.n2 = pb.IF(10, threshold=Vthr2, reset_v=0, tick_wait_start=3)
        self.fc1 = pb.FullConn(
            self.n1, self.n2, weights=weight2, conn_type=pb.SynConnType.All2All
        )


if __name__ == "__main__":
    w1 = np.random.randint(-128, 127, size=(8, 1, 5, 5), dtype=np.int8)
    w2 = np.random.randint(-128, 127, size=(8 * 48 * 48, 10), dtype=np.int8)

    vthr1 = np.random.randint(1, 100)
    vthr2 = np.random.randint(1, 100)

    pb_net = Conv2d_Net(w1, vthr1, w2, vthr2)

    input_data = np.random.randint(0, 255, size=52 * 52, dtype=np.uint8)

    # Input
    pb_net.i1.input = input_data

    out_dir = Path(__file__).parent
    pb.BACKEND_CONFIG.target_chip_addr = (0, 0)

    mapper = pb.Mapper()
    mapper.build(pb_net)
    graph_info = mapper.compile(
        weight_bit_optimization=True, grouping_optim_target="both"
    )

    # #N of cores required
    print("Core required:", graph_info["n_core_required"])

    # mapper.export(
    #     write_to_file=True, fp=out_dir / "debug", format="npy", export_core_params=False
    # )

    # Clear all the results
    mapper.clear()
