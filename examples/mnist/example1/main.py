import argparse
from pathlib import Path

import numpy as np

import paibox as pb


class fcnet_2layer_dual_port(pb.Network):
    def __init__(self, weight1, Vthr1, weight2, Vthr2):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(392,))
        self.i2 = pb.InputProj(input=pe, shape_out=(392,))
        self.n1 = pb.IF(128, threshold=Vthr1, reset_v=0)
        self.s1 = pb.FullConn(
            self.i1,
            self.n1,
            weights=weight1[:392],
            conn_type=pb.SynConnType.All2All,
        )
        self.s2 = pb.FullConn(
            self.i2,
            self.n1,
            weights=weight1[392:],
            conn_type=pb.SynConnType.All2All,
        )

        # tick_wait_start = 2 for second layer
        self.n2 = pb.IF(
            5, threshold=Vthr2, reset_v=0, tick_wait_start=2, name="batch_dual_port_o1"
        )
        self.n3 = pb.IF(
            5, threshold=Vthr2, reset_v=0, tick_wait_start=2, name="batch_dual_port_o2"
        )
        self.s3 = pb.FullConn(
            self.n1,
            self.n2,
            weights=weight2[:, :5],
            conn_type=pb.SynConnType.All2All,
        )
        self.s4 = pb.FullConn(
            self.n1,
            self.n3,
            weights=weight2[:, 5:],
            conn_type=pb.SynConnType.All2All,
        )

        self.probe1 = pb.Probe(target=self.n2, attr="spike")
        self.probe2 = pb.Probe(target=self.n3, attr="spike")


param_dict = {}


def getNetParam():
    timestep = 4
    layer_num = 2
    delay = layer_num - 1

    weights_dir = Path("./weights")
    w1 = np.load(weights_dir / "fc1_weight_np.npy").astype(np.int8).T
    vthr1 = int(np.load(weights_dir / "Vthr1.npy") / 1.0)
    w2 = np.load(weights_dir / "fc2_weight_np.npy").astype(np.int8).T
    vthr2 = int(np.load(weights_dir / "Vthr2.npy") / 1.0)

    param_dict["timestep"] = timestep
    param_dict["layer_num"] = layer_num
    param_dict["delay"] = delay
    param_dict["w1"] = w1
    param_dict["vthr1"] = vthr1
    param_dict["w2"] = w2
    param_dict["vthr2"] = vthr2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", help="visualize the input data", action="store_true"
    )
    args = parser.parse_args()

    getNetParam()
    pb_net = fcnet_2layer_dual_port(
        param_dict["w1"],
        param_dict["vthr1"],
        param_dict["w2"],
        param_dict["vthr2"],
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
    pb_net.i1.input = input_data[:392]
    pb_net.i2.input = input_data[392:]

    # Simulation, duration=timestep + delay
    sim = pb.Simulator(pb_net)
    sim.run(param_dict["timestep"] + param_dict["delay"], reset=False)

    # Decode the output
    spike_out1 = sim.data[pb_net.probe1].astype(np.int8)
    spike_out2 = sim.data[pb_net.probe2].astype(np.int8)
    spike_out = np.concatenate((spike_out1, spike_out2), axis=1)
    spike_sum = spike_out.sum(axis=0)
    pred = np.argmax(spike_sum)

    assert pred == 7, print("failed")  # Correct result is 7

    out_dir = Path(__file__).parent
    pb.BACKEND_CONFIG.target_chip_addr = (0, 0)

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
        split_by_coord=True,
        export_core_params=False,
    )

    # Clear all the results
    mapper.clear()
