from abc import abstractmethod
from typing import Generic, List, Optional, TypeVar, Union

import torch
from paicorelib import (
    AddPotentialMode,
    OfflineNeuFullAttrsV2Part2,
    OutputType,
    SNNMode,
    WeightCompressType,
)
from torch import Tensor, nn

from ..paiir.ir.add_ops import PotentialAddOp
from ..paiir.ir.calc_params import LutData, OfflineCoreParams
from ..paiir.ir.graph import PAIIRGraph
from ..paiir.ir.ir_base import InputNode, OutputNode
from ..paiir.ir.op_node import (
    AccumulateOp,
    ConcatOp,
    OfflineCoreOp,
    ReshapeOp,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from .core_config import Frontend_Core_Config


class CustomIndex:
    def __init__(self, idx: int, copy_id: int = 0):
        self.idx = idx
        self.copy_id = copy_id

    def __hash__(self) -> int:
        return hash((self.idx, self.copy_id))

    def __eq__(self, value: "CustomIndex") -> bool:
        return self.idx == value.idx and self.copy_id == value.copy_id

    def __str__(self) -> str:
        return f"(idx: {self.idx}, copy_id: {self.copy_id})"

    def __repr__(self) -> str:
        return self.__str__()


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
        self.predecessors_set: set["SourceNode"] = (
            set()
        )  # 用于快速判断是否有某个 predecessor
        self.successors_set: set["DestNode"] = set()  # 用于快速判断是否有某个 successor
        self.input_bit_num_: int = -1  # 初始化为 -1，表示未设置
        self.output_bit_num_: int = -1  # 初始化为 -1，表示未设置
        self.io_setted: bool = False  # 标记输入输出位数是否已设置

    @property
    def input_bit_num(self) -> int:
        return self.input_bit_num_

    @property
    def output_bit_num(self) -> int:
        return self.output_bit_num_

    @abstractmethod
    def set_io_bit_num(self, direction: int):
        pass

    def __hash__(self) -> int:
        return hash(id(self))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __str__(self) -> str:
        return self.__repr__()


class InNode(BaseNode["InputNode"]):
    def __init__(self, name: str, raw_node: "InputNode", shape: tuple[int, ...]):
        super().__init__(name, shape, raw_node)

    def set_io_bit_num(self, direction: int):
        assert (
            direction == 1
        ), "InNode should only call set_io_bit_num with direction 1 (from successors)"
        succ_input_bit_nums = set([succ.input_bit_num for succ in self.successors])
        assert (
            len(succ_input_bit_nums) == 1
        ), "All successors must have the same input bit num"

        self.output_bit_num_ = succ_input_bit_nums.pop()


class OutNode(BaseNode["OutputNode"]):
    def __init__(self, name: str, raw_node: "OutputNode", shape: tuple[int, ...]):
        super().__init__(name, shape, raw_node)

    def set_io_bit_num(self, direction: int):
        assert (
            direction == 0
        ), "OutNode should only call set_io_bit_num with direction 0 (from predecessors)"
        pred_output_bit_nums = set([pred.output_bit_num for pred in self.predecessors])
        assert (
            len(pred_output_bit_nums) == 1
        ), "All predecessors must have the same output bit num"

        self.input_bit_num_ = pred_output_bit_nums.pop()


RemapOp = Union[ReshapeOp, ConcatOp]


class ReorderNode(BaseNode[RemapOp]):
    def __init__(self, name: str, raw_node: RemapOp, shape: tuple[int, ...]):
        super().__init__(name, shape, raw_node)

    def get_reorder_info(self) -> dict["SourceElem", "RemapElem"]:
        if isinstance(self.raw_node, ReshapeOp):
            assert (
                len(self.predecessors) == 1
            ), "ReshapeNode should have exactly one predecessor"
            pred = self.predecessors[0]
            pred_len = pred.shape.numel()
            assert (
                pred_len == self.shape.numel()
            ), "Total number of elements must match for reshape"
            reorder_map: dict["SourceElem", "RemapElem"] = {}
            for i in range(pred_len):
                pred_elem = get_elem(pred, i)
                reorder_elem = RemapElem(self, CustomIndex(i))
                reorder_map[pred_elem] = reorder_elem
            return reorder_map
        elif isinstance(self.raw_node, ConcatOp):
            reorder_map: dict["SourceElem", "RemapElem"] = {}
            concat_dim = self.raw_node.dim
            in_shapes = [pred.shape for pred in self.predecessors]
            assert all(
                shape[:concat_dim] == in_shapes[0][:concat_dim]
                and shape[concat_dim + 1 :] == in_shapes[0][concat_dim + 1 :]
                for shape in in_shapes
            ), "All input shapes must match except for the concat dimension"
            dim_offset = 0
            for pred in self.predecessors:
                # concat_dim 后面所有维度的元素数
                inner = 1
                for d in range(concat_dim + 1, len(pred.shape)):
                    inner *= pred.shape[d]
                # concat_dim 及之后所有维度的元素数（前驱）
                pred_dim_stride = pred.shape[concat_dim] * inner
                # 输出在 concat_dim 及之后的 stride
                out_dim_stride = self.shape[concat_dim] * inner
                pred_len = pred.shape.numel()
                for i in range(pred_len):
                    # 把 i 分解为: outer * pred_dim_stride + mid * inner + inner_idx
                    outer = i // pred_dim_stride
                    rem = i % pred_dim_stride
                    mid = rem // inner
                    inner_idx = rem % inner
                    flat_idx = (
                        outer * out_dim_stride + (mid + dim_offset) * inner + inner_idx
                    )
                    pred_elem = get_elem(pred, i)
                    reorder_map[pred_elem] = RemapElem(self, CustomIndex(flat_idx))
                dim_offset += pred.shape[concat_dim]
            return reorder_map
        else:
            raise NotImplementedError(
                f"Unsupported node type for ReorderNode: {type(self.raw_node)}"
            )

    def set_io_bit_num(self, direction: int):
        if direction == 1:
            # Get input bit num from successors
            succ_input_bit_nums = set([succ.input_bit_num for succ in self.successors])
            assert (
                len(succ_input_bit_nums) == 1
            ), "All successors must have the same input bit num"
            self.output_bit_num_ = succ_input_bit_nums.pop()
            self.input_bit_num_ = self.output_bit_num_
        elif direction == 0:
            # Get output bit num from predecessors
            pred_output_bit_nums = set(
                [pred.output_bit_num for pred in self.predecessors]
            )
            assert (
                len(pred_output_bit_nums) == 1
            ), "All predecessors must have the same output bit num"
            self.input_bit_num_ = pred_output_bit_nums.pop()
            self.output_bit_num_ = self.input_bit_num_
        else:
            raise ValueError(
                "Direction must be 0 (from successors) or 1 (from predecessors)"
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

    def set_io_bit_num(self, direction: int):

        assert (
            direction == -1
        ), "CoreOpNode should not call set_io_bit_num with direction 0 or 1, as its input and output bit num are determined by its own configuration rather than predecessors or successors"
        if self.core_config().add_potential == AddPotentialMode.NORMAL:
            input_bit_num = 2 ** self.core_config().input_width
        else:
            input_bit_num = 32
        self.input_bit_num_ = input_bit_num
        if self.output_type() == OutputType.VALUE:
            output_bit_num = 2 ** self.core_config().output_width
        else:
            output_bit_num = 32
        self.output_bit_num_ = output_bit_num


# 类型定义 1：包含三个 Node
SourceNode = Union[InNode, ReorderNode, CoreOpNode]
# 类型定义 2：不包含 Input
DestNode = Union[ReorderNode, CoreOpNode, OutNode]

AllNode = Union[InNode, ReorderNode, CoreOpNode, OutNode]


T = TypeVar("T", CoreOpNode, ReorderNode, InNode)


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
        return f"{getattr(self.target, 'name', 'Unknown')}[{self.index}]"

    def __repr__(self) -> str:
        return self.__str__()

    def get_raw_elem(self) -> "SourceElem":
        return get_elem(self.target, self.index.idx, 0)

    @property
    def input_bit_num(self) -> int:
        return self.target.input_bit_num

    @property
    def output_bit_num(self) -> int:
        return self.target.output_bit_num


# --- 子类实现 ---


class Neuron(BaseElem["CoreOpNode"]):
    # 仅保留 Neuron 特有的方法
    def attrs_part2(self) -> "OfflineNeuFullAttrsV2Part2":
        return self.target.attrs_part2(self.index.idx)

    def output_type(self) -> "OutputType":
        return self.target.output_type()

    def core_config(self) -> "Frontend_Core_Config":
        return self.target.core_config()

    def copy(self, copy_id: int) -> "Neuron":
        return Neuron(self.target, CustomIndex(self.index.idx, copy_id))

    def origin_elem(self) -> "Neuron":
        return self.copy(0)


class RemapElem(BaseElem["ReorderNode"]):
    # 如果没有特有方法，直接 pass 即可
    def copy(self, copy_id: int) -> "RemapElem":
        return RemapElem(self.target, CustomIndex(self.index.idx, copy_id))

    def origin_elem(self) -> "RemapElem":
        return self.copy(0)


class InputElem(BaseElem["InNode"]):
    def copy(self, copy_id: int) -> "InputElem":
        return InputElem(self.target, CustomIndex(self.index.idx, copy_id))

    def origin_elem(self) -> "InputElem":
        return self.copy(0)


AllElem = Union[Neuron, RemapElem, InputElem]
SourceElem = Union[Neuron, RemapElem, InputElem]
CoreElem = Union[Neuron, RemapElem]


def get_elem(Node: BaseNode, idx: int, copy_id: int = 0) -> "SourceElem":
    if isinstance(Node, InNode):
        return InputElem(Node, CustomIndex(idx, copy_id))
    elif isinstance(Node, ReorderNode):
        return RemapElem(Node, CustomIndex(idx, copy_id))
    elif isinstance(Node, CoreOpNode):
        return Neuron(Node, CustomIndex(idx, copy_id))
    else:
        raise NotImplementedError(f"Unsupported node type: {type(Node)}")


def build_nodes(graph: PAIIRGraph) -> list[AllNode]:
    nodes: list[AllNode] = []
    nodes_map: dict[str, AllNode] = {}
    for raw_node in graph.nodes.values():
        if isinstance(raw_node, OfflineCoreOp):
            node = CoreOpNode(raw_node.name, raw_node, raw_node.output_layouts[0].shape)
        elif isinstance(raw_node, InputNode):
            node = InNode(raw_node.name, raw_node, raw_node.shape)
        elif isinstance(raw_node, OutputNode):
            node = OutNode(raw_node.name, raw_node, raw_node.shape)
        elif isinstance(raw_node, RemapOp):
            node = ReorderNode(
                raw_node.name, raw_node, raw_node.output_layouts[0].shape
            )
        else:
            raise NotImplementedError(f"Unsupported node type: {type(raw_node)}")
        nodes_map[raw_node.name] = node
        nodes.append(node)

    for node_name, cur_node in nodes_map.items():
        # if isinstance(cur_node, OutNode):
        #     continue
        succ_node_names = graph.successors(node_name)
        for succ_name in succ_node_names:
            succ_node = nodes_map[succ_name]
            # if isinstance(succ_node, OutNode):
            #     continue
            assert isinstance(succ_node, DestNode)
            cur_node.successors.append(succ_node)
        if not isinstance(cur_node, InNode):
            pred_node_names = graph.predecessors(node_name)
            for pred_name in pred_node_names:
                pred_node = nodes_map[pred_name]
                if isinstance(pred_node, OutNode):
                    raise ValueError(
                        f"CoreOpNode {cur_node.name} has OutNode {pred_node.name} as predecessor"
                    )
                cur_node.predecessors.append(pred_node)

    unset_nodes = set(nodes)
    node_to_process: list[tuple[AllNode, int]] = [
        (node, -1) for node in nodes if isinstance(node, CoreOpNode)
    ]
    unset_nodes -= set(node for node, _ in node_to_process)
    assert (
        len(node_to_process) > 0
    ), "There should be at least one CoreOpNode to dictate the input/output bit num for the whole graph"
    while unset_nodes or len(node_to_process) > 0:
        assert (
            len(node_to_process) > 0
        ), "There is a cycle in the graph or some nodes are not connected to CoreOpNodes"
        node, direction = node_to_process.pop(0)
        node.set_io_bit_num(direction)
        for succ in node.successors:
            # all predecessors of succ have been set_io_bit_num, so we can set_io_bit_num for succ
            if succ in unset_nodes and all(
                pred not in unset_nodes for pred in succ.predecessors
            ):
                unset_nodes.remove(succ)
                node_to_process.append((succ, 0))
        for pred in node.predecessors:
            # all successors of pred have been set_io_bit_num, so we can set_io_bit_num for pred
            if pred in unset_nodes and all(
                succ not in unset_nodes for succ in pred.successors
            ):
                unset_nodes.remove(pred)
                node_to_process.append((pred, 1))

    for node in nodes:
        print(f"Node {node.name}({node.shape}):")
        if isinstance(node, CoreOpNode):
            print(
                f"\tComps: {[type(comp).__name__ if comp is not None else None for comp in node.comps]}"
            )
        print(f"\tPredecessors: {[pred.name for pred in node.predecessors]}")
        print(f"\tSuccessors: {[succ.name for succ in node.successors]}")
        print(
            f"\tInput bit num: {node.input_bit_num}, Output bit num: {node.output_bit_num}"
        )

    # raise NotImplementedError("Not support output node")

    return nodes
