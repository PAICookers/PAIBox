# import numpy as np
# from paicorelib import FRAME_DTYPE, FrameArrayType


# def export_single_framearray(frame_array: FrameArrayType, file_path: str) -> None:
#     with open(file_path, "a") as f:
#         for frame in frame_array:
#             f.write(f"{frame:016x}\n")


# random_array = np.random.randint(0, 2**64, size=10, dtype=FRAME_DTYPE)
# export_single_framearray(random_array, "output_frames.txt")


import pprint

import torch
from spikingjelly.activation_based import neuron
from torch import nn

from paibox._logging import DEFAULT_LOG_SETTINGS, set_logs
from paibox.backendv2.op_node import CoreOpNode, build_nodes
from paibox.fx_converter.core_op import BaseCoreOp
from paibox.fx_converter.fuse import apply_passes, fuse_compute_act
from paibox.fx_converter.trace import (
    propagate_tensor_shape,
    remove_dropout_identity_and_fuse_conv_bn,
)


class M(nn.Module):
    def __init__(self):
        super().__init__()
        conv2d_1 = nn.Conv2d(4, 16, 3, bias=False)
        self.seq = nn.Sequential(conv2d_1, neuron.LIFNode(v_threshold=1.0, tau=2.0))
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
# print(gm.code)
propagate_tensor_shape(gm, torch.randn(1, 4, 12, 12))

from paibox.backendv2.mapper import Mapper

mapper = Mapper()
mapper.compile(gm)


# print("\n=== Exported Attributes Inspection ===")
# for name, module in gm.named_modules():
#     if isinstance(module, BaseCoreOp):
#         core_attrs, neu_attrs, comp_attrs = module.get_attrs()
#         print(f"\n[Node: {name} ({type(module).__name__})]")
#         print(">> Core Attributes:")
#         pprint.pprint(core_attrs, indent=2)
#         print(">> Neuron Attributes:")
#         pprint.pprint(neu_attrs, indent=2)
#         print(">> Compute Attributes:")
#         pprint.pprint(comp_attrs, indent=2)

# succ_graph: dict[BaseCoreOp, list[BaseCoreOp]] = dict()
# succ_graph_str: dict[str, list[str]] = dict()

# core_op_nodes: dict[str, BaseCoreOp] = dict()
# core_op_node_shapes: dict[str, torch.Size] = dict()
# for name, module in gm.named_modules():
#     print(f"Module: {name}, Type: {type(module).__name__}")
#     if isinstance(module, BaseCoreOp):
#         core_op_nodes[name] = module

# print(f"Core Op Nodes: {list(core_op_nodes.keys())}")

# for node in gm.graph.nodes:
#     node_target = str(node.target)
#     if node_target in core_op_nodes:
#         core_op_node_shapes[node_target] = node.meta["tensor_meta"].shape

#         core_op = core_op_nodes[node_target]
#         succs: list[BaseCoreOp] = []
#         succs_str: list[str] = []
#         print(f"\n[Node: {node.name}]")
#         for succ_node in node.users:
#             succ_node_target = str(succ_node.target)
#             if succ_node_target in core_op_nodes:
#                 succs.append(core_op_nodes[succ_node_target])
#                 succs_str.append(succ_node_target)
#         succ_graph[core_op] = succs
#         succ_graph_str[node_target] = succs_str

# # print("\n=== Core Op Output Shapes ===")
# # for name, shape in core_op_node_shapes.items():
# #     print(f"{name}: {shape}")

# # print("\n=== Succession Graph ===")
# # for name, succs_name in succ_graph_str.items():
# #     print(f"{name} -> {succs_name}")

# nodes: list[CoreOpNode] = []
# for name, raw_node in core_op_nodes.items():
#     shape = core_op_node_shapes[name]
#     node = CoreOpNode(name, raw_node, shape)
#     nodes.append(node)

# for node in nodes:
#     succ_names = succ_graph_str.get(node.name, [])
#     for succ_name in succ_names:
#         for succ_node in nodes:
#             if succ_node.name == succ_name:
#                 node.successors.append(succ_node)
#                 succ_node.predecessors.append(node)

# for node in nodes:
#     print(f"Node {node.name} successors: {[succ.name for succ in node.successors]}")
#     print(f"Node {node.name} predecessors: {[pred.name for pred in node.predecessors]}")
