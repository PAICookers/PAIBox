import torch
from paicorelib import (
    DataSign,
    DataWidth,
    OfflineNeuFullAttrsV2Part2,
    OutputType,
    PoolingMode,
    SNNMode,
    WeightCompressType,
    ZeroOutputMode,
)
from torch.fx import GraphModule, Node

from ..fx_converter.clac_params import NeuV2ClacParams, OfflineCoreV2CalcParams
from ..fx_converter.core_op import BaseCoreOp
from .core_config import Frontend_Core_Config


class CustomIndex:
    def __init__(self, idx: int):
        self.idx = idx

    def __hash__(self) -> int:
        return hash(self.idx)

    def __eq__(self, value: "CustomIndex") -> bool:
        return self.idx == value.idx


def get_frontend_core_conf(
    core_params: OfflineCoreV2CalcParams,
) -> Frontend_Core_Config:
    def to_enum(value, enum_cls):
        return enum_cls(value) if isinstance(value, int) else value

    snn_ann = to_enum(core_params.snn_ann, SNNMode)

    return Frontend_Core_Config(
        snn_ann=to_enum(core_params.snn_ann, SNNMode),
        max_pooling=to_enum(core_params.max_pooling, PoolingMode),
        zero_output=to_enum(core_params.zero_output, ZeroOutputMode),
        input_sign=to_enum(core_params.input_sign, DataSign),
        input_width=to_enum(core_params.input_width, DataWidth),
        output_sign=to_enum(core_params.output_sign, DataSign),
        output_width=to_enum(core_params.output_width, DataWidth),
        weight_sign=to_enum(core_params.weight_sign, DataSign),
        weight_width=to_enum(core_params.weight_width, DataWidth),
        tick_start=core_params.tick_start,
        tick_duration=core_params.tick_duration,
        tick_initial=core_params.tick_initial,
    )


class InputNode:
    def __init__(self, name: str, shape: torch.Size):
        self.name = name
        self.shape = shape
        self.successors: list[CoreOpNode] = []

    def __hash__(self) -> int:
        return hash(id(self))

    def __str__(self) -> str:
        return f"InputNode({self.name})"

    def __repr__(self) -> str:
        return self.__str__()


class CoreOpNode:
    def __init__(self, name: str, raw_node: BaseCoreOp, shape: torch.Size):
        self.name = name
        self.raw_node = raw_node
        self.shape = shape
        self.successors: list[CoreOpNode] = []
        self.predecessors: list[CoreOpNode | InputNode] = []
        self.frontend_core_config: Frontend_Core_Config = get_frontend_core_conf(
            raw_node.core_params
        )

    def __hash__(self) -> int:
        return hash(id(self))

    def __str__(self) -> str:
        return f"CoreOpNode({self.name})"

    def __repr__(self) -> str:
        return self.__str__()

    def attrs_part2(self) -> OfflineNeuFullAttrsV2Part2:
        _, nue_attr, _ = self.raw_node.get_attrs()
        return OfflineNeuFullAttrsV2Part2(
            reset_mode=nue_attr.reset_mode,
            reset_v=int(nue_attr.reset_v),
            threshold_neg_mode=nue_attr.thres_neg_mode,
            threshold_pos_mode=nue_attr.thres_pos_mode,
            threshold_neg=int(nue_attr.thres_neg),
            threshold_pos=int(nue_attr.thres_pos),
            lateral_inhibition=nue_attr.lateral_inhi,
            leak_multi_sequence=nue_attr.leak_multi_sequence,
            leak_multi_input=nue_attr.leak_multi_input,
            leak_multi_mode=nue_attr.leak_multi_mode,
            leak_add_mode=nue_attr.leak_add_mode,
            leak_tau=nue_attr.leak_tau,
            leak_v=int(nue_attr.leak_v),
            weight_compress=WeightCompressType.DENSE,
            vjt_initial=int(nue_attr.init_v),
        )

    def output_type(self) -> OutputType:
        _, neu_attrs, _ = self.raw_node.get_attrs()
        return neu_attrs.output_type

    def core_config(self) -> Frontend_Core_Config:
        return self.frontend_core_config


def build_nodes(gm: GraphModule) -> list[CoreOpNode | InputNode]:
    raw_node_graph: dict[str, list[str]] = dict()
    raw_core_nodes: dict[str, BaseCoreOp] = dict()
    raw_io_nodes: dict[str, Node] = dict()
    raw_node_shapes: dict[str, torch.Size] = dict()
    for name, module in gm.named_modules():
        if isinstance(module, BaseCoreOp):
            raw_core_nodes[name] = module

    for torch_node in gm.graph.nodes:
        raw_node_name = str(torch_node.target)
        if len(torch_node.users) == 0:
            continue  # skip nodes with no users

        if raw_node_name not in raw_core_nodes:
            raw_io_nodes[raw_node_name] = torch_node

        raw_node_shapes[raw_node_name] = torch_node.meta["tensor_meta"].shape
        succ_raw_nodes: list[str] = []
        print(f"\n[Node: {torch_node.name}]")
        for user_node in torch_node.users:
            if len(user_node.users) == 0:
                continue  # skip users with no users

            succ_raw_node = str(user_node.target)
            succ_raw_nodes.append(succ_raw_node)
        raw_node_graph[raw_node_name] = succ_raw_nodes

    print("\n=== Raw Node Graph ===")
    for name, succs_name in raw_node_graph.items():
        print(f"{name} -> {succs_name}")

    nodes: list[CoreOpNode | InputNode] = []
    for name, raw_node in raw_core_nodes.items():
        shape = raw_node_shapes[name]
        node = CoreOpNode(name, raw_node, shape)
        nodes.append(node)

    for name, raw_node in raw_io_nodes.items():
        shape = raw_node_shapes[name]
        node = InputNode(name, shape)
        nodes.append(node)

    for node in nodes:
        succ_names = raw_node_graph.get(node.name, [])
        for succ_name in succ_names:
            for user_node in nodes:
                if user_node.name == succ_name:
                    assert isinstance(user_node, CoreOpNode)
                    node.successors.append(user_node)
                    user_node.predecessors.append(node)

    for node in nodes:
        print(f"\nNode {node.name} ({node.shape}) ({node.shape.numel()}):")
        print(f"\tsuccessors: {[succ.name for succ in node.successors]}")
        if isinstance(node, CoreOpNode):
            print(f"\tpredecessors: {[pred.name for pred in node.predecessors]}")
    return nodes
