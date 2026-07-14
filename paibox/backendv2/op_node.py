from abc import abstractmethod
from typing import Generic, TypeVar

import torch
from paicorelib import (
    AddPotentialMode,
    LeakMultiMode,
    OfflineNeuFullAttrsV2Part2,
    OutputType,
    SNNMode,
    WeightCompressType,
)
from torch import Tensor, nn
from torch.types import Number

from ..paiir.ir.add_ops import PotentialAddOp
from ..paiir.ir.calc_params import LutData, OfflineCoreParams
from ..paiir.ir.graph import PAIIRGraph
from ..paiir.ir.ir_base import InputNode, OutputNode
from ..paiir.ir.op_node import (
    AccumulateOp,
    ConcatOp,
    OfflineCoreOp,
    PadOp,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
    TransformOp,
)
from .core_config import Frontend_Core_Config


class CustomIndex:
    def __init__(self, idx: int, copy_id: int = 0) -> None:
        self.idx = idx
        self.copy_id = copy_id

    def __hash__(self) -> int:
        return hash((self.idx, self.copy_id))

    def __eq__(self, value: "CustomIndex") -> bool:
        return self.idx == value.idx and self.copy_id == value.copy_id

    def __str__(self) -> str:
        return f"(idx: {self.idx}, copy_id: {self.copy_id})"

    __repr__ = __str__


class PaddingOp:
    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        raw_padding: tuple[int, ...],
        fpad: bool = False,
    ) -> None:
        padding: list[tuple[int, int]] = []
        if not fpad:
            for pad in raw_padding:
                padding.append((pad, pad))
        else:
            for i in reversed(range(0, len(raw_padding), 2)):
                padding.append((raw_padding[i], raw_padding[i + 1]))
        self.name = name
        self.inshape = torch.Size(shape)
        self.outshape = torch.Size(self.compute_out_shape(self.inshape, padding))
        self.shape = self.outshape
        self.num_of_each_dim = []
        for i in range(len(self.outshape)):
            if i == 0:
                self.num_of_each_dim.append(1)
            else:
                shape_index = len(self.outshape) - i
                self.num_of_each_dim.append(
                    self.outshape[shape_index] * self.num_of_each_dim[-1]
                )
        print(
            f"PaddingOp {self.name} inshape: {self.inshape}, outshape: {self.outshape}, num_of_each_dim: {self.num_of_each_dim}"
        )

        self.padding = padding
        self.remap_dict: dict[int, int] = {}
        self.set_remap_dict()

    def compute_out_shape(
        self, inshape: tuple[int, ...], padding: list[tuple[int, int]]
    ) -> tuple[int, ...]:
        shape_list = list(inshape)
        k = len(padding)
        for i in range(k):
            dim = -k + i
            shape_list[dim] += padding[i][0] + padding[i][1]
        outshape = tuple(shape_list)
        return outshape

    def next_in_idx(self, in_idx: list[int]) -> list[int]:
        next_in_idx = in_idx.copy()
        for dim in range(len(self.inshape)):
            shape_index = len(self.inshape) - 1 - dim
            if in_idx[dim] == self.inshape[shape_index] - 1:
                next_in_idx[dim] = 0
            else:
                next_in_idx[dim] += 1
                break
        return next_in_idx

    def set_remap_dict(self) -> None:
        in_idx = [0] * len(self.inshape)
        for i in range(self.inshape.numel()):
            out_idx = in_idx.copy()
            for j in range(len(self.padding)):
                pad = self.padding[j]
                out_idx[j] += pad[0]
            out_idx_flat = 0

            for k, idx in enumerate(out_idx):
                out_idx_flat += idx * self.num_of_each_dim[k]

            self.remap_dict[i] = out_idx_flat
            # print(f"PaddingOp {self.name} remap: {tuple(in_idx)} -> {out_idx}")
            # print(f"PaddingOp {self.name} remap flat: {i} -> {out_idx_flat}")
            in_idx = self.next_in_idx(in_idx)


def conv2d_without_padding(old_conv: nn.Conv2d) -> nn.Conv2d:
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride
    dilation = old_conv.dilation

    assert (
        len(kernel_size) == 2 and len(stride) == 2 and len(dilation) == 2
    ), "Only 2D convolution is supported"
    new_conv = nn.Conv2d(
        in_channels=old_conv.in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=old_conv.groups,
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight)

    return new_conv


def conv1d_without_padding(old_conv: nn.Conv1d) -> nn.Conv1d:
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride
    dilation = old_conv.dilation

    assert (
        len(kernel_size) == 1 and len(stride) == 1 and len(dilation) == 1
    ), "Only 1D convolution is supported"
    new_conv = nn.Conv1d(
        in_channels=old_conv.in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=old_conv.groups,
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight)

    return new_conv


def get_frontend_core_conf(
    core_params: OfflineCoreParams, hw_lut_data: LutData | None
) -> Frontend_Core_Config:
    assert core_params.tick_start is not None
    if core_params.snn_mode == SNNMode.SNN:
        assert hw_lut_data is None, "hw_lut_data should not be provided for SNN mode"

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
        hw_lut_data=hw_lut_data,
    )


# 定义 Raw Node 的类型变量
T_Raw = TypeVar("T_Raw")


class BaseNode(Generic[T_Raw]):
    """所有 Graph 节点的基类"""

    def __init__(self, name: str, shape: tuple[int, ...], raw_node: T_Raw) -> None:
        self.name = name
        self.shape = torch.Size(shape)
        self.raw_node = raw_node
        self.successors: list["DestNode"] = []
        self.predecessors: list["SourceNode"] = []
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
    def set_io_bit_num(self, direction: int) -> None:
        pass

    def __hash__(self) -> int:
        return hash(id(self))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    __repr__ = __str__


class InNode(BaseNode[InputNode]):
    def __init__(self, name: str, raw_node: InputNode, shape: tuple[int, ...]) -> None:
        super().__init__(name, shape, raw_node)

    def set_io_bit_num(self, direction: int) -> None:
        assert (
            direction == 1
        ), "InNode should only call set_io_bit_num with direction 1 (from successors)"
        succ_input_bit_nums = set([succ.input_bit_num for succ in self.successors])
        assert (
            len(succ_input_bit_nums) == 1
        ), "All successors must have the same input bit num"

        self.output_bit_num_ = succ_input_bit_nums.pop()


class OutNode(BaseNode[OutputNode]):
    def __init__(self, name: str, raw_node: OutputNode, shape: tuple[int, ...]) -> None:
        super().__init__(name, shape, raw_node)

    def set_io_bit_num(self, direction: int) -> None:
        assert (
            direction == 0
        ), "OutNode should only call set_io_bit_num with direction 0 (from predecessors)"
        pred_output_bit_nums = set([pred.output_bit_num for pred in self.predecessors])
        assert (
            len(pred_output_bit_nums) == 1
        ), "All predecessors must have the same output bit num"

        self.input_bit_num_ = pred_output_bit_nums.pop()


RemapOp = TransformOp | ConcatOp | PaddingOp


class RemapNode(BaseNode[RemapOp]):
    def __init__(self, name: str, raw_node: RemapOp, shape: tuple[int, ...]) -> None:
        super().__init__(name, shape, raw_node)

    def get_remap_info(self) -> dict["SourceElem", "RemapElem"]:
        if isinstance(self.raw_node, TransformOp):
            assert (
                len(self.predecessors) == 1
            ), "TransformNode should have exactly one predecessor"
            pred = self.predecessors[0]
            pred_len = pred.shape.numel()
            assert (
                pred_len == self.shape.numel()
            ), "Total number of elements must match for transform remap"

            # Drive the routing transform over an index tensor so backend
            # reorder follows the same logical-layout semantics as the IR.
            flat_indices = torch.arange(pred_len, dtype=torch.int64).reshape(pred.shape)
            reordered = self.raw_node(flat_indices).reshape(-1)
            assert (
                reordered.numel() == pred_len
            ), "TransformOp index remap must preserve element count"

            remap_info: dict["SourceElem", "RemapElem"] = {}
            for dst_idx, src_idx in enumerate(reordered.tolist()):
                pred_elem = get_elem(pred, src_idx)
                reorder_elem = RemapElem(self, CustomIndex(dst_idx))
                remap_info[pred_elem] = reorder_elem
            return remap_info
        elif isinstance(self.raw_node, ConcatOp):
            remap_info: dict["SourceElem", "RemapElem"] = {}
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
                    remap_info[pred_elem] = RemapElem(self, CustomIndex(flat_idx))
                dim_offset += pred.shape[concat_dim]
            return remap_info
        elif isinstance(self.raw_node, PaddingOp):
            assert (
                len(self.predecessors) == 1
            ), "PaddingNode should have exactly one predecessor"
            pred = self.predecessors[0]
            pred_len = pred.shape.numel()
            remap_info: dict["SourceElem", "RemapElem"] = {}
            for i in range(pred_len):
                pred_elem = get_elem(pred, i)
                reorder_elem = RemapElem(self, CustomIndex(self.raw_node.remap_dict[i]))
                remap_info[pred_elem] = reorder_elem
            return remap_info
        else:
            raise NotImplementedError(
                f"Unsupported node type for RemapNode: {type(self.raw_node)}"
            )

    def set_io_bit_num(self, direction: int) -> None:
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
    def __init__(
        self, name: str, raw_node: "OfflineCoreOp", shape: tuple[int, ...]
    ) -> None:
        super().__init__(name, shape, raw_node)
        self.comps: list[nn.Module | None] = []
        self.weights: list[Tensor | None] = []
        # 初始化前端配置
        self.frontend_core_config = get_frontend_core_conf(
            raw_node.core_params, raw_node.hw_lut_data
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

    def attrs_part2(self, idx: int = 0) -> OfflineNeuFullAttrsV2Part2:
        neu_attrs = self.raw_node.neu_params

        def _resolve(value: Number | Tensor, idx: int) -> int:
            v = value[idx].item() if torch.is_tensor(value) else value
            if isinstance(v, float):
                return round(v)
            return int(v)

        resolved = {
            name: _resolve(getattr(neu_attrs, name), idx)
            for name in neu_attrs.__vectorized_attrs__
        }

        return OfflineNeuFullAttrsV2Part2(
            reset_mode=neu_attrs.reset_mode,
            reset_v=resolved["reset_v"],
            threshold_neg_mode=neu_attrs.thres_neg_mode,
            threshold_pos_mode=neu_attrs.thres_pos_mode,
            threshold_neg=resolved["thres_neg"],
            threshold_pos=resolved["thres_pos"],
            lateral_inhibition=neu_attrs.lateral_inhi,
            leak_multi_sequence=neu_attrs.leak_multi_sequence,
            leak_multi_input=neu_attrs.leak_multi_input,
            leak_multi_mode=LeakMultiMode(resolved["leak_multi_mode"]),
            leak_add_mode=neu_attrs.leak_add_mode,
            leak_tau=resolved["leak_tau"],
            leak_v=resolved["leak_v"],
            weight_compress=WeightCompressType.DENSE,
            vjt_initial=resolved["init_v"],
        )

    def output_type(self) -> OutputType:
        return self.raw_node.neu_params.output_type

    def core_config(self) -> Frontend_Core_Config:
        return self.frontend_core_config

    def set_io_bit_num(self, direction: int) -> None:
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
SourceNode = InNode | RemapNode | CoreOpNode
# 类型定义 2：不包含 Input
DestNode = RemapNode | CoreOpNode | OutNode

AllNode = InNode | RemapNode | CoreOpNode | OutNode


T = TypeVar("T", CoreOpNode, RemapNode, InNode)


class BaseElem(Generic[T]):
    def __init__(self, target: T, index: "CustomIndex") -> None:
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

    __repr__ = __str__

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
    def attrs_part2(self) -> OfflineNeuFullAttrsV2Part2:
        return self.target.attrs_part2(self.index.idx)

    def output_type(self) -> OutputType:
        return self.target.output_type()

    def core_config(self) -> Frontend_Core_Config:
        return self.target.core_config()

    def copy(self, copy_id: int) -> "Neuron":
        return Neuron(self.target, CustomIndex(self.index.idx, copy_id))

    def origin_elem(self) -> "Neuron":
        return self.copy(0)


class RemapElem(BaseElem["RemapNode"]):
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


AllElem = Neuron | RemapElem | InputElem
SourceElem = Neuron | RemapElem | InputElem
CoreElem = Neuron | RemapElem


def get_elem(Node: BaseNode, idx: int, copy_id: int = 0) -> SourceElem:
    if isinstance(Node, InNode):
        return InputElem(Node, CustomIndex(idx, copy_id))
    elif isinstance(Node, RemapNode):
        return RemapElem(Node, CustomIndex(idx, copy_id))
    elif isinstance(Node, CoreOpNode):
        return Neuron(Node, CustomIndex(idx, copy_id))
    else:
        raise NotImplementedError(f"Unsupported node type: {type(Node)}")


def insert_padding_nodes(nodes: list[AllNode]) -> None:
    new_nodes: list[RemapNode] = []
    for node in nodes:
        if not isinstance(node, CoreOpNode):
            continue

        for i in range(len(node.comps)):
            comp = node.comps[i]
            if isinstance(comp, nn.Conv2d) or isinstance(comp, nn.Conv1d):
                if isinstance(comp.padding, str):
                    raise NotImplementedError(
                        "String padding mode is not supported in this version"
                    )
                if isinstance(comp, nn.Conv2d) and comp.padding == (0, 0):
                    continue
                if isinstance(comp, nn.Conv1d) and comp.padding == (0,):
                    continue

                padding_node: RemapNode | None = None
                for new_node in new_nodes:
                    if new_node.name == f"{node.predecessors[i].name}_Padded":
                        padding_node = new_node
                        if new_node.raw_node.padding == comp.padding:
                            break
                        else:
                            raise ValueError(
                                f"Padding node {new_node.name} already exists with different padding {new_node.raw_node.padding} vs {comp.padding}"
                            )

                if padding_node is None:
                    padding_op = PaddingOp(
                        name=f"{node.predecessors[i].name}_Padded",
                        shape=node.predecessors[i].shape,
                        raw_padding=comp.padding,
                    )

                    padding_node = RemapNode(
                        name=padding_op.name,
                        raw_node=padding_op,
                        shape=padding_op.outshape,
                    )

                    # only set predecessor info when creating new padding node,
                    # if the padding node already exists,
                    # it must have been connected to the same predecessor
                    padding_node.predecessors.append(node.predecessors[i])
                    for j in range(len(node.predecessors[i].successors)):
                        if node.predecessors[i].successors[j] is node:
                            node.predecessors[i].successors[j] = padding_node
                            break
                else:
                    for j in range(len(node.predecessors[i].successors)):
                        if node.predecessors[i].successors[j] is node:
                            # remove the existing connection to node
                            node.predecessors[i].successors.pop(j)
                            break

                if isinstance(comp, nn.Conv2d):
                    no_pad_conv = conv2d_without_padding(comp)
                else:
                    no_pad_conv = conv1d_without_padding(comp)

                padding_node.successors.append(node)
                node.comps[i] = no_pad_conv
                node.predecessors[i] = padding_node
                # weight remains the same, so no need to change node.weights[i]
                new_nodes.append(padding_node)

    nodes.extend(new_nodes)


def set_io_bit_num(nodes: list[AllNode]) -> None:
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
            node = RemapNode(raw_node.name, raw_node, raw_node.output_layouts[0].shape)
        elif isinstance(raw_node, PadOp):
            padding_op = PaddingOp(
                name=raw_node.name,
                shape=raw_node.input_layouts[0].shape,
                raw_padding=raw_node.padding,
                fpad=True,
            )
            node = RemapNode(raw_node.name, padding_op, padding_op.outshape)
        else:
            raise NotImplementedError(f"Unsupported node type: {type(raw_node)}")
        nodes_map[raw_node.name] = node
        nodes.append(node)

    for node_name, cur_node in nodes_map.items():
        succ_node_names = graph.successors(node_name)
        for succ_name in succ_node_names:
            succ_node = nodes_map[succ_name]
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

    for node in nodes:
        print(f"Node {node.name}({node.shape}):")
        if isinstance(node, CoreOpNode):
            print(
                f"\tComps: {[type(comp).__name__ if comp is not None else None for comp in node.comps]}"
            )
        print(f"\tPredecessors: {[pred.name for pred in node.predecessors]}")
        print(f"\tSuccessors: {[succ.name for succ in node.successors]}")

    insert_padding_nodes(nodes)
    set_io_bit_num(nodes)

    print("\nAfter inserting padding nodes and setting IO bit num:")
    for node in nodes:
        print(f"Node {node.name}({node.shape}):")
        print(f"\tPredecessors: {[pred.name for pred in node.predecessors]}")
        print(f"\tSuccessors: {[succ.name for succ in node.successors]}")
        print(
            f"\tInput bit num: {node.input_bit_num}, Output bit num: {node.output_bit_num}"
        )

    return nodes
