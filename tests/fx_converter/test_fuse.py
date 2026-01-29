import pprint

import torch
from spikingjelly.activation_based import neuron
from torch import nn

from paibox._logging import DEFAULT_LOG_SETTINGS, set_logs
from paibox.fx_converter.core_op import BaseCoreOp
from paibox.fx_converter.fuse import apply_passes, fuse_compute_act
from paibox.fx_converter.trace import (
    propagate_tensor_shape,
    remove_dropout_identity_and_fuse_conv_bn,
)

set_logs(**DEFAULT_LOG_SETTINGS)


class TestFusionPass:
    def test_fuse_compute_act(self):
        print("\n=== test_fuse_compute_act begin ===")

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.lif = neuron.LIFNode()
                self.relu = nn.ReLU()
                self.bn = nn.BatchNorm2d(16)  # unused
                self.maxpool = nn.MaxPool2d(
                    kernel_size=3, stride=1, padding=1
                )  # unused

            def forward(self, x):
                x1 = self.conv(x)

                x1 = self.lif(x1)  # Conv -> LIF
                x2 = self.conv(x)
                x2 = self.relu(x2)  # Conv -> ReLU

                return x1 + x2

        m = M()
        gm = remove_dropout_identity_and_fuse_conv_bn(m)
        print("Original Graph:")
        print(gm.graph.print_tabular())
        print("doing fuse_compute_act...")
        gm2 = apply_passes(gm)
        print("Fused Code:")
        print(gm2.code)
        propagate_tensor_shape(gm2, torch.randn(1, 3, 32, 32))

        print("\n=== test_fuse_compute_act Exported Attributes Inspection ===")
        for name, module in gm2.named_modules():
            if isinstance(module, BaseCoreOp):
                core_attrs, neu_attrs, comp_attrs = module.get_attrs()
                print(f"\n[Node: {name} ({type(module).__name__})]")
                print(">> Core Attributes:")
                pprint.pprint(core_attrs, indent=2)
                print(">> Neuron Attributes:")
                pprint.pprint(neu_attrs, indent=2)
                print(">> Compute Attributes:")
                pprint.pprint(comp_attrs, indent=2)

    def test_fuse_implicit_add(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.conv2 = nn.Conv2d(3, 16, 3)
                self.lif1 = neuron.LIFNode()
                self.lif2 = neuron.LIFNode()
                self.maxpool = nn.MaxPool2d(2)

            def forward(self, x, y):
                o1 = self.conv1(x)
                o2 = self.conv2(y)
                return self.lif1(o1 + o2)

        m = M()

        gm = remove_dropout_identity_and_fuse_conv_bn(m)

        gm = apply_passes(gm)
        print(gm.code)
        propagate_tensor_shape(gm, torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32))

        print("\n=== test_fuse_implicit_add Exported Attributes Inspection ===")
        for name, module in gm.named_modules():
            if isinstance(module, BaseCoreOp):
                core_attrs, neu_attrs, comp_attrs = module.get_attrs()
                print(f"\n[Node: {name} ({type(module).__name__})]")
                print(">> Core Attributes:")
                pprint.pprint(core_attrs, indent=2)
                print(">> Neuron Attributes:")
                pprint.pprint(neu_attrs, indent=2)
                print(">> Compute Attributes:")
                pprint.pprint(comp_attrs, indent=2)

    def test_fuse_standalone_conv(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

            def forward(self, x):
                return self.conv(x)

        m = M()
        gm = remove_dropout_identity_and_fuse_conv_bn(m)
        gm = apply_passes(gm)

        print("\n=== test_fuse_standalone_conv Exported Attributes Inspection ===")
        for name, module in gm.named_modules():
            if isinstance(module, BaseCoreOp):
                core_attrs, neu_attrs, comp_attrs = module.get_attrs()
                print(f"\n[Node: {name} ({type(module).__name__})]")
                print(">> Core Attributes:")
                pprint.pprint(core_attrs, indent=2)
                print(">> Neuron Attributes:")
                pprint.pprint(neu_attrs, indent=2)
                print(">> Compute Attributes:")
                pprint.pprint(comp_attrs, indent=2)

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

        gm = remove_dropout_identity_and_fuse_conv_bn(m)
        gm.graph.print_tabular()

        print("Fusing to core op")
        gm = fuse_compute_act(gm)
        gm.graph.print_tabular()
        print(gm.code)
        propagate_tensor_shape(gm, torch.randn(1, 4, 32, 32))

        print("\n=== Exported Attributes Inspection ===")
        for name, module in gm.named_modules():
            if isinstance(module, BaseCoreOp):
                core_attrs, neu_attrs, comp_attrs = module.get_attrs()
                print(f"\n[Node: {name} ({type(module).__name__})]")
                print(">> Core Attributes:")
                pprint.pprint(core_attrs, indent=2)
                print(">> Neuron Attributes:")
                pprint.pprint(neu_attrs, indent=2)
                print(">> Compute Attributes:")
                pprint.pprint(comp_attrs, indent=2)

    def test_spiking_inverted_residual(self):
        class RepConv(nn.Module):
            """
            Simulates the structure of 'conv1.body' and 'conv2.body' seen in trace.
            Structure: body.0 -> body.1 -> body.2.0 -> body.2.1 -> body.2.2
            """

            def __init__(
                self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                bias=False,
            ):
                super().__init__()
                # Simplified functional equivalent matching the trace hierarchy
                self.body = nn.Sequential(
                    nn.Identity(),  # 0
                    nn.Identity(),  # 1
                    nn.Sequential(  # 2
                        nn.Identity(),  # 2.0
                        nn.Identity(),  # 2.1
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding,
                            bias=bias,
                        ),  # 2.2
                    ),
                )

            def forward(self, x):
                return self.body(x)

        class SpikingInvertedResidual(nn.Module):
            def __init__(self, in_c, hidden_c, out_c, stride=1):
                super().__init__()
                self.lif1 = neuron.LIFNode()
                self.pwconv1 = nn.Conv2d(in_c, hidden_c, 1, 1, 0, bias=False)
                self.bn1 = nn.BatchNorm2d(hidden_c)

                self.lif2 = neuron.LIFNode()
                self.dwconv2 = nn.Conv2d(
                    hidden_c, hidden_c, 3, stride, 1, groups=hidden_c, bias=False
                )
                self.bn2 = nn.BatchNorm2d(hidden_c)

                self.lif3 = neuron.LIFNode()
                self.pwconv3 = RepConv(hidden_c, out_c, 1, 1, 0)
                self.bn3 = nn.BatchNorm2d(out_c)

                self.use_res_connect = stride == 1 and in_c == out_c

            def forward(self, x):
                identity = x

                x = self.lif1(x)
                x = self.pwconv1(x)
                x = self.bn1(x)

                x = self.lif2(x)
                x = self.dwconv2(x)
                x = self.bn2(x)

                x = self.lif3(x)
                x = self.pwconv3(x)
                x = self.bn3(x)

                if self.use_res_connect:
                    return x + identity
                return x

        class Stage2Block(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.Conv = SpikingInvertedResidual(in_c, in_c // 2, out_c, stride=1)

                # Channels need to be inferred.
                # Assuming standard Bottleneck expansion
                self.lif1 = neuron.LIFNode()
                self.conv1 = RepConv(out_c, out_c * 3, 1, 1, 0)
                self.bn1 = nn.BatchNorm2d(out_c * 3)

                self.lif2 = neuron.LIFNode()
                self.conv2 = RepConv(out_c * 3, out_c, 1, 1, 0)
                self.bn2 = nn.BatchNorm2d(out_c)

            def forward(self, x):
                x = self.Conv(x)

                identity = x
                y = self.lif1(x)
                y = self.conv1(y)
                y = self.bn1(y)

                y = self.lif2(y)
                y = self.conv2(y)
                y = self.bn2(y)

                return y + identity

        m = Stage2Block(in_c=32, out_c=32)

        print("\n=== test_spiking_inverted_residual trace ===")
        gm = remove_dropout_identity_and_fuse_conv_bn(m)
        print("Original Graph:")
        print(gm.graph.print_tabular())

        print("doing fuse_compute_act...")
        gm = apply_passes(gm)
        print("Fused Code:")
        print(gm.code)
        propagate_tensor_shape(gm, torch.randn(1, 32, 32, 32))

        print("\n=== test_spiking_inverted_residual Exported Attributes Inspection ===")
        for name, module in gm.named_modules():
            if isinstance(module, BaseCoreOp):
                core_attrs, neu_attrs, comp_attrs = module.get_attrs()
                print(f"\n[Node: {name} ({type(module).__name__})]")
                print(">> Core Attributes:")
                pprint.pprint(core_attrs, indent=2)
                print(">> Neuron Attributes:")
                pprint.pprint(neu_attrs, indent=2)
                print(">> Compute Attributes:")
                pprint.pprint(comp_attrs, indent=2)

    def test_sppf_snn(self):
        class SPPFSNN(nn.Module):
            def __init__(self, in_c, out_c, k=5):
                super().__init__()
                self.cv1 = nn.Sequential(
                    nn.Conv2d(in_c, in_c // 2, 1, 1, 0, bias=False), neuron.LIFNode()
                )
                self.cv2 = nn.Sequential(
                    nn.Conv2d(in_c * 2, out_c, 1, 1, 0, bias=False), neuron.LIFNode()
                )
                self.m = nn.MaxPool2d(k, 1, k // 2)
                self.lif = neuron.LIFNode()

            def forward(self, x):
                x = self.cv1(x)
                y1 = self.m(x)
                y2 = self.m(y1)
                y3 = self.m(y2)
                return self.cv2(torch.cat((x, y1, y2, y3), 1))

        print("\n=== test_sppf_snn trace ===")
        # in_c needs to be divisible by 2.
        m = SPPFSNN(in_c=32, out_c=64)

        gm = remove_dropout_identity_and_fuse_conv_bn(m)
        print("Original Graph:")
        print(gm.graph.print_tabular())
        print("Original shape:")

        propagate_tensor_shape(gm, torch.randn(1, 32, 64, 64))

        print("doing fuse...")
        gm = apply_passes(gm)
        print("Fused Code:")
        print(gm.code)

        # in_c=32, out_c=64.
        # cv1: 32 -> 16.
        # x becomes 16 channels.
        # m: 16 -> 16.
        # y1, y2, y3 all 16 channels.
        # cat: 16*4 = 64 channels.
        # cv2 input channels: in_c * 2 = 32 * 2 = 64. Matches cat output.

        propagate_tensor_shape(gm, torch.randn(1, 32, 64, 64))

        print("\n=== test_sppf_snn Exported Attributes Inspection ===")
        for name, module in gm.named_modules():
            if isinstance(module, BaseCoreOp):
                core_attrs, neu_attrs, comp_attrs = module.get_attrs()
                print(f"\n[Node: {name} ({type(module).__name__})]")
                print(">> Core Attributes:")
                pprint.pprint(core_attrs, indent=2)
                print(">> Neuron Attributes:")
                pprint.pprint(neu_attrs, indent=2)
                print(">> Compute Attributes:")
                pprint.pprint(comp_attrs, indent=2)
