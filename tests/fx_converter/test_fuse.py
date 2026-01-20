from spikingjelly.activation_based import neuron
from torch import nn

from paibox._logging import DEFAULT_LOG_SETTINGS, set_logs
from paibox.fx_converter.fuse import apply_fuse_passes, fuse_compute_act
from paibox.fx_converter.trace import trace_spikingjelly_model

set_logs(**DEFAULT_LOG_SETTINGS)


class TestFusionPass:
    def test_fuse_compute_act(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.lif = neuron.LIFNode()
                self.relu = nn.ReLU()
                self.bn = nn.BatchNorm2d(16)  # unused

            def forward(self, x):
                x1 = self.conv(x)
                x1 = self.lif(x1)  # Conv -> LIF
                x2 = self.conv(x)
                x2 = self.relu(x2)  # Conv -> ReLU

                return x1 + x2

        m = M()
        gm = trace_spikingjelly_model(m)
        print(gm.graph.print_tabular())

        gm2 = apply_fuse_passes(gm)
        print(gm2.code)

    def test_fuse_implicit_add(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.conv2 = nn.Conv2d(3, 16, 3)
                self.lif1 = neuron.LIFNode()
                self.lif2 = neuron.LIFNode()

            def forward(self, x, y):
                o1 = self.conv1(x)
                o2 = self.conv2(y)
                return self.lif1(o1 + o2)

        m = M()

        gm = trace_spikingjelly_model(m)
        gm = apply_fuse_passes(gm)
        print(gm.code)

    def test_build_SeqOpNode(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                conv2d_1 = nn.Conv2d(4, 16, 3, bias=False)
                self.seq = nn.Sequential(
                    conv2d_1, neuron.LIFNode(v_threshold=1.0, tau=2.0)
                )
                self.conv2 = nn.Conv2d(16, 8, 3)
                self.if1 = neuron.IFNode(v_threshold=2.0)

            def forward(self, x):
                x1 = self.seq(x)
                x2 = self.if1(self.conv2(x1))
                return x2

        m = M()

        gm = trace_spikingjelly_model(m)
        gm.graph.print_tabular()

        print("Fusing to core op")
        gm = fuse_compute_act(gm)
        gm.graph.print_tabular()
        print(gm.code)
