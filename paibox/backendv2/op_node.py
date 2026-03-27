from typing import Generic, List, Optional, TypeVar, Union

import torch
from paicorelib import (
    OfflineNeuFullAttrsV2Part2,
    OutputType,
    SNNMode,
    WeightCompressType,
)
from torch import Tensor, nn

from ..paiir import (
    AccumulateOp,
    InputNode,
    LutData,
    OfflineCoreOp,
    OfflineCoreParams,
    OutputNode,
    PAIIRGraph,
    PotentialAddOp,
    ReshapeOp,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from .core_config import Frontend_Core_Config


class CustomIndex:
    def __init__(self, idx: int):
        self.idx = idx

    def __hash__(self) -> int:
        return hash(self.idx)

    def __eq__(self, value: "CustomIndex") -> bool:
        return self.idx == value.idx


def get_frontend_core_conf(
    core_params: OfflineCoreParams, lut_data: Optional[LutData]
) -> Frontend_Core_Config:
    assert core_params.tick_start is not None
    if core_params.snn_mode == SNNMode.ANN:
        assert core_params.tick_initial == 1, "ANN mode requires tick_initial=1"
        # assert lut_data is not None, "lut_data must be provided for ANN mode"
    elif core_params.snn_mode == SNNMode.SNN:
        assert lut_data is None, "lut_data should not be provided for SNN mode"

    return Frontend_Core_Config(
        add_potential=core_params.add_potential,
        snn_ann=core_params.snn_mode,
        max_pooling=core_params.pooling_mode,
        zero_output=core_params.zero_output,
        input_sign=core_params.input_sign,
        input_width=core_params.input_width,
        output_sign=core_params.output_sign,
        output_width=core_params.output_width,
        weight_sign=core_params.weight_sign,
        weight_width=core_params.weight_width,
        tick_start=core_params.tick_start,
        tick_duration=core_params.tick_duration,
        tick_initial=core_params.tick_initial,
        lut_data=lut_data,
    )


# 定义 Raw Node 的类型变量
T_Raw = TypeVar("T_Raw")


class BaseNode(Generic[T_Raw]):
    """所有 Graph 节点的基类"""

    def __init__(self, name: str, shape: tuple[int, ...], raw_node: T_Raw):
        self.name = name
        self.shape = torch.Size(shape)
        self.raw_node = raw_node
        self.successors: List["DestNode"] = []
        self.predecessors: List["SourceNode"] = []

    def __hash__(self) -> int:
        return hash(id(self))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __str__(self) -> str:
        return self.__repr__()


class InNode(BaseNode["InputNode"]):
    def __init__(self, name: str, raw_node: "InputNode", shape: tuple[int, ...]):
        super().__init__(name, shape, raw_node)


class OutNode(BaseNode["OutputNode"]):
    def __init__(self, name: str, raw_node: "OutputNode", shape: tuple[int, ...]):
        super().__init__(name, shape, raw_node)


class ReorderNode(BaseNode["ReshapeOp"]):
    def __init__(self, name: str, raw_node: "ReshapeOp", shape: tuple[int, ...]):
        super().__init__(name, shape, raw_node)

    def get_reorder_info(self) -> dict["SourceElem", "ReorderElem"]:
        if isinstance(self.raw_node, ReshapeOp):
            assert (
                len(self.predecessors) == 1
            ), "ReshapeNode should have exactly one predecessor"
            pred = self.predecessors[0]
            pred_len = pred.shape.numel()
            assert (
                pred_len == self.shape.numel()
            ), "Total number of elements must match for reshape"
            reorder_map: dict["SourceElem", "ReorderElem"] = {}
            for i in range(pred_len):
                pred_elem = get_elem(pred, i)
                reorder_elem = ReorderElem(self, CustomIndex(i))
                reorder_map[pred_elem] = reorder_elem
            return reorder_map
        else:
            raise NotImplementedError(
                f"Unsupported node type for ReorderNode: {type(self.raw_node)}"
            )


class CoreOpNode(BaseNode["OfflineCoreOp"]):
    def __init__(self, name: str, raw_node: "OfflineCoreOp", shape: tuple[int, ...]):
        super().__init__(name, shape, raw_node)
        self.comps: list[Optional[nn.Module]] = []
        self.weights: list[Optional[Tensor]] = []
        # 初始化前端配置
        lut_data: Optional[LutData] = None
        if isinstance(raw_node, SequentialOp):
            lut = raw_node.act.lut
            if lut is not None:
                lut_data = lut.export_lut()

        self.frontend_core_config: "Frontend_Core_Config" = get_frontend_core_conf(
            raw_node.core_params, lut_data
        )
        print(f"lut data for node {self.name}: {lut_data}")
        self.set_comps_and_weights()

    def set_comps_and_weights(self) -> None:
        if isinstance(self.raw_node, SequentialOp):
            self.comps = [self.raw_node.comp]
        elif isinstance(self.raw_node, AccumulateOp):
            self.comps = list(self.raw_node.comps)
        elif isinstance(self.raw_node, StandaloneCompOp):
            self.comps = [self.raw_node.comp]
        elif isinstance(self.raw_node, StandaloneActOp):
            self.comps = [None]
        elif isinstance(self.raw_node, PotentialAddOp):
            self.comps = [None] * len(self.raw_node.signs)
        else:
            raise NotImplementedError(f"Unsupported node type: {type(self.raw_node)}")

        weights = self.raw_node.weights
        if weights is not None:
            self.weights = list(weights)
        else:
            self.weights = [None] * len(self.comps)

    def attrs_part2(self, idx: int = 0) -> "OfflineNeuFullAttrsV2Part2":
        neu_attrs = self.raw_node.neuron_params
        if isinstance(neu_attrs.leak_v, torch.Tensor):
            assert self.shape[0] == 1, "Batch size > 1 not supported for tensor leak_v"
            out_channel = self.shape[1]
            assert (
                neu_attrs.leak_v.numel() == out_channel
            ), "leak_v tensor size mismatch"
            cur_channel = idx // (self.shape.numel() // self.shape[1])
            leak_v = neu_attrs.leak_v[cur_channel].item()
        else:
            leak_v = neu_attrs.leak_v

        return OfflineNeuFullAttrsV2Part2(
            reset_mode=neu_attrs.reset_mode,
            reset_v=round(neu_attrs.reset_v),
            threshold_neg_mode=neu_attrs.thres_neg_mode,
            threshold_pos_mode=neu_attrs.thres_pos_mode,
            threshold_neg=round(neu_attrs.thres_neg),
            threshold_pos=round(neu_attrs.thres_pos),
            lateral_inhibition=neu_attrs.lateral_inhi,
            leak_multi_sequence=neu_attrs.leak_multi_sequence,
            leak_multi_input=neu_attrs.leak_multi_input,
            leak_multi_mode=neu_attrs.leak_multi_mode,
            leak_add_mode=neu_attrs.leak_add_mode,
            leak_tau=neu_attrs.leak_tau,
            leak_v=round(leak_v),
            weight_compress=WeightCompressType.DENSE,
            vjt_initial=round(neu_attrs.init_v),
        )

    def output_type(self) -> "OutputType":
        return self.raw_node.neuron_params.output_type

    def core_config(self) -> "Frontend_Core_Config":
        return self.frontend_core_config


# 类型定义 1：包含三个 Node
SourceNode = Union[InNode, ReorderNode, CoreOpNode]
# 类型定义 2：不包含 Input
DestNode = Union[ReorderNode, CoreOpNode]

AllNode = Union[InNode, ReorderNode, CoreOpNode, OutNode]


T = TypeVar("T")


class BaseElem(Generic[T]):
    def __init__(self, target: T, index: "CustomIndex"):
        self.target = target
        self.index = index

    def __hash__(self) -> int:
        # 统一的哈希逻辑
        return hash((self.index, self.target))

    def __eq__(self, value: object) -> bool:
        # 统一的相等判断逻辑
        if not isinstance(value, type(self)):
            return False
        return self.index == value.index and self.target is value.target

    def __str__(self) -> str:
        # 假设所有 target 都有 .name 属性
        return f"{getattr(self.target, 'name', 'Unknown')}[{self.index.idx}]"

    def __repr__(self) -> str:
        return self.__str__()


# --- 子类实现 ---


class PaddingElem(BaseElem["None"]):
    def __init__(self, index: "CustomIndex"):
        pass


class Neuron(BaseElem["CoreOpNode"]):
    # 仅保留 Neuron 特有的方法
    def attrs_part2(self) -> "OfflineNeuFullAttrsV2Part2":
        return self.target.attrs_part2(self.index.idx)

    def output_type(self) -> "OutputType":
        return self.target.output_type()

    def core_config(self) -> "Frontend_Core_Config":
        return self.target.core_config()


class ReorderElem(BaseElem["ReorderNode"]):
    # 如果没有特有方法，直接 pass 即可
    pass


class InputElem(BaseElem["InNode"]):
    pass


AllElem = Union[Neuron, ReorderElem, InputElem]
SourceElem = AllElem
CoreElem = Union[Neuron, ReorderElem]


def get_elem(Node: BaseNode, idx: int) -> "SourceElem":
    if isinstance(Node, InNode):
        return InputElem(Node, CustomIndex(idx))
    elif isinstance(Node, ReorderNode):
        return ReorderElem(Node, CustomIndex(idx))
    elif isinstance(Node, CoreOpNode):
        return Neuron(Node, CustomIndex(idx))
    else:
        raise NotImplementedError(f"Unsupported node type: {type(Node)}")


def build_nodes(graph: PAIIRGraph) -> list[SourceNode]:
    nodes: list[AllNode] = []
    nodes_map: dict[str, AllNode] = {}
    for raw_node in graph.nodes.values():
        if isinstance(raw_node, OfflineCoreOp):
            node = CoreOpNode(raw_node.name, raw_node, raw_node.output_shape)
        elif isinstance(raw_node, InputNode):
            node = InNode(raw_node.name, raw_node, raw_node.shape)
        elif isinstance(raw_node, OutputNode):
            node = OutNode(raw_node.name, raw_node, raw_node.shape)
        elif isinstance(raw_node, ReshapeOp):
            node = ReorderNode(raw_node.name, raw_node, raw_node.output_shape)
        else:
            raise NotImplementedError(f"Unsupported node type: {type(raw_node)}")
        nodes_map[raw_node.name] = node
        nodes.append(node)

    for node_name, cur_node in nodes_map.items():
        if isinstance(cur_node, OutNode):
            continue
        succ_node_names = graph.successors(node_name)
        for succ_name in succ_node_names:
            succ_node = nodes_map[succ_name]
            if isinstance(succ_node, OutNode):
                continue
            assert isinstance(succ_node, DestNode)
            cur_node.successors.append(succ_node)
        if isinstance(cur_node, CoreOpNode | ReorderNode):
            pred_node_names = graph.predecessors(node_name)
            for pred_name in pred_node_names:
                pred_node = nodes_map[pred_name]
                if isinstance(pred_node, OutNode):
                    raise ValueError(
                        f"CoreOpNode {cur_node.name} has OutNode {pred_node.name} as predecessor"
                    )
                cur_node.predecessors.append(pred_node)

    filtered_nodes = [node for node in nodes if not isinstance(node, OutNode)]

    for node in filtered_nodes:
        print(f"Node {node.name}({node.shape}):")
        print(f"\tPredecessors: {[pred.name for pred in node.predecessors]}")
        print(f"\tSuccessors: {[succ.name for succ in node.successors]}")

    # raise NotImplementedError("Not support output node")

    return filtered_nodes
