import argparse
from pathlib import Path

import numpy as np

import paibox as pb


class Conv2d_Net(pb.Network):
    def __init__(self, weight1, Vthr1, weight2, Vthr2, weight3, Vthr3):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(1, 28, 28))
        self.n1 = pb.IF((2, 26, 26), threshold=Vthr1, reset_v=0)
        self.conv2d_1 = pb.Conv2d(self.i1, self.n1, kernel=weight1, stride=1)

        self.n2 = pb.IF((4, 24, 24), threshold=Vthr2, reset_v=0, tick_wait_start=2)
        self.conv2d_2 = pb.Conv2d(self.n1, self.n2, kernel=weight2, stride=1)

        self.n3 = pb.IF(10, threshold=Vthr3, reset_v=0, tick_wait_start=3)
        self.fc1 = pb.FullConn(
            self.n2, self.n3, weights=weight3, conn_type=pb.SynConnType.All2All
        )

        self.probe1 = pb.Probe(self.n3, "spike")


param_dict = {}


def getNetParam():
    timestep = 8
    layer_num = 3
    delay = layer_num - 1

    weights_dir = Path("./weights")
    w1 = np.load(weights_dir / "weight_conv1.npy").astype(np.int8)
    vthr1 = int(np.load(weights_dir / "Vthr_conv1.npy") / 1.0)
    w2 = np.load(weights_dir / "weight_conv2.npy").astype(np.int8)
    vthr2 = int(np.load(weights_dir / "Vthr_conv2.npy") / 1.0)
    w3 = np.load(weights_dir / "weight_fc1.npy").astype(np.int8).T
    vthr3 = int(np.load(weights_dir / "Vthr_fc1.npy") / 1.0)

    param_dict["timestep"] = timestep
    param_dict["layer_num"] = layer_num
    param_dict["delay"] = delay
    param_dict["w1"] = w1
    param_dict["vthr1"] = vthr1
    param_dict["w2"] = w2
    param_dict["vthr2"] = vthr2
    param_dict["w3"] = w3
    param_dict["vthr3"] = vthr3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", help="visualize the input data", action="store_true"
    )
    args = parser.parse_args()

    getNetParam()
    pb_net = Conv2d_Net(
        param_dict["w1"],
        param_dict["vthr1"],
        param_dict["w2"],
        param_dict["vthr2"],
        param_dict["w3"],
        param_dict["vthr3"],
    )

    # Network simulation
    raw_data = np.load(Path(__file__).parent.parent / "data" / "mnist_input_data7.npy")
    input_data = raw_data.ravel()

    # Visualize
    if args.verbose:
        pe = pb.simulator.PoissonEncoder()
        data_to_see = pe(raw_data).astype(np.int8)
        print(data_to_see)

    # Input
    pb_net.i1.input = input_data

    # Simulation, duration=timestep + delay
    sim = pb.Simulator(pb_net)
    sim.run(param_dict["timestep"] + param_dict["delay"], reset=False)

    # Decode the output
    spike_out = sim.data[pb_net.probe1].astype(np.int8)
    spike_out = spike_out[param_dict["delay"] :]
    spike_sum = spike_out.sum(axis=0)
    pred = np.argmax(spike_sum)

    assert pred == 7, print("failed")  # Correct result is 7

    out_dir = Path(__file__).parent

    mapper = pb.Mapper()
    mapper.build(pb_net)
    graph_info = mapper.compile(
        weight_bit_optimization=True, grouping_optim_target="both"
    )

    # #N of cores required
    print("Core required:", graph_info["n_core_required"])

    mapper.export(
        write_to_file=True,
        fp=out_dir / "debug",
        format="npy",
        split_by_coordinate=True,
        local_chip_addr=(0, 0),
        export_core_params=False,
    )

    # Clear all the results
    mapper.clear()
