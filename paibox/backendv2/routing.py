from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from paicorelib import (
    LCN_EX,
    AERPacketZXYCopy,
    CoordXY,
    DataWidth,
    FoldType,
    NeuronType,
    OfflineCoreRegV2,
    OfflineNeuDestInfoV2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
    OutputType,
    WeightCompressType,
    find_coordxy_shortest_path,
)
from torch import Tensor, nn
from torch.nn import functional as F

from .core_config import Backend_Core_Config, Frontend_Core_Config
from .coreplacement import (
    CorePlacement,
    EmptyOfflineCorePlacementV2,
    OfflineCorePlacementV2,
)
from .neuron import InputElem, Neuron, OfflineNeuronPlacement
from .op_node import CoreOpNode, InNode
from .weight import Weight
from ..paiir import AccumulateOp, AddOp, SequentialOp, StandaloneActOp, StandaloneCompOp

FANIN_BASE = 512


def feature_shape(shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
    # Backend weight extraction works on feature dimensions only.
    # The leading batch dimension is stripped and must stay equal to 1.
    feature_shape = tuple(shape)
    if len(feature_shape) > 1:
        batch = feature_shape[0]
        if batch != 1:
            raise NotImplementedError(
                f"Only batch size 1 is supported, but got shape {feature_shape}."
            )
        feature_shape = feature_shape[1:]

    return feature_shape


def to_nd_tuple(value: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if isinstance(value, tuple):
        if len(value) != ndim:
            raise ValueError(f"Expected a tuple of length {ndim}, but got {value}.")
        return value

    return (value,) * ndim


def ensure_target_components(target: CoreOpNode) -> None:
    # Lazily reconstruct per-predecessor comps/weights so routing can
    # query them even if node initialization did not populate them yet.
    if not target.predecessors:
        return

    if len(target.comps) == len(target.predecessors) and len(target.weights) == len(
        target.predecessors
    ):
        return

    raw_node = target.raw_node
    if isinstance(raw_node, SequentialOp):
        target.comps = [raw_node.comp]
    elif isinstance(raw_node, AccumulateOp):
        target.comps = list(raw_node.comps)
    elif isinstance(raw_node, StandaloneCompOp):
        target.comps = [raw_node.comp]
    elif isinstance(raw_node, StandaloneActOp):
        target.comps = [None]
    elif isinstance(raw_node, AddOp):
        target.comps = [None] * len(raw_node.signs)
    else:
        raise NotImplementedError(f"Unsupported node type: {type(raw_node)}")

    raw_weights = raw_node.weights
    if raw_weights is None:
        target.weights = [None] * len(target.comps)
    else:
        target.weights = list(raw_weights)


def path_signs(target: CoreOpNode) -> list[int]:
    # Accumulate/Add nodes may encode subtraction through per-path signs.
    raw_node = target.raw_node
    if isinstance(raw_node, (AccumulateOp, AddOp)):
        return list(raw_node.signs)

    return [1] * len(target.predecessors)


def direct_weight_matrix(
    weight: Tensor, input_shape: tuple[int, ...], output_shape: tuple[int, ...], sign: int
) -> np.ndarray:
    # Linear/identity-like paths already expose a dense [out, in] matrix.
    matrix = np.asarray(weight.detach().cpu(), dtype=np.int32)
    n_input = int(np.prod(input_shape))
    n_output = int(np.prod(output_shape))

    if matrix.shape != (n_output, n_input):
        raise ValueError(
            f"Direct weight shape mismatch: expected {(n_output, n_input)}, got {matrix.shape}."
        )

    if sign != 1:
        matrix = sign * matrix

    return matrix


def unfold_input_indices_2d(
    channels: int,
    spatial_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> torch.Tensor:
    # Build a lookup table from each sliding-window position back to the
    # flattened input indices. Zero means "came from padding".
    height, width = spatial_shape
    n_input = channels * height * width
    input_ids = torch.arange(1, n_input + 1, dtype=torch.float64).reshape(
        1, channels, height, width
    )
    patches = F.unfold(
        input_ids,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    return patches.to(torch.int64).squeeze(0)


def conv2d_weight_matrix(
    weight: Tensor,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    sign: int,
) -> np.ndarray:
    # Expand a Conv2d kernel into a full dense matrix with shape [out, in],
    # where rows are flattened output neurons and columns are flattened inputs.
    in_channels, height, width = input_shape
    out_channels, out_height, out_width = output_shape
    kernel = np.asarray(weight.detach().cpu(), dtype=np.int32)
    _, in_channels_per_group, kernel_height, kernel_width = kernel.shape

    if in_channels != in_channels_per_group * groups:
        raise ValueError(
            f"Conv groups mismatch: input channels {in_channels}, kernel channels {in_channels_per_group}, groups {groups}."
        )

    out_channels_per_group = out_channels // groups
    out_size = out_height * out_width
    n_input = in_channels * height * width
    matrix = np.zeros((out_channels * out_size, n_input), dtype=np.int32)

    patches = unfold_input_indices_2d(
        in_channels,
        (height, width),
        (kernel_height, kernel_width),
        stride,
        padding,
        dilation,
    ).cpu().numpy()

    if patches.shape[1] != out_size:
        raise ValueError(
            f"Conv unfold mismatch: expected {out_size} output positions, got {patches.shape[1]}."
        )

    row_offsets = np.tile(
        np.arange(out_size, dtype=np.int64),
        in_channels_per_group * kernel_height * kernel_width,
    )

    for out_channel in range(out_channels):
        group_idx = out_channel // out_channels_per_group
        patch_start = group_idx * in_channels_per_group * kernel_height * kernel_width
        patch_end = patch_start + in_channels_per_group * kernel_height * kernel_width

        # Scatter each kernel coefficient to the input positions that feed the
        # corresponding flattened output locations.
        flat_cols = patches[patch_start:patch_end].reshape(-1)
        valid = flat_cols > 0
        flat_rows = row_offsets[valid]
        flat_values = np.repeat(kernel[out_channel].reshape(-1), out_size)[valid]
        matrix[out_channel * out_size + flat_rows, flat_cols[valid] - 1] = flat_values

    if sign != 1:
        matrix *= sign

    return matrix


def pool2d_weight_matrix(
    channels: int,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    sign: int,
) -> np.ndarray:
    # Pooling has no explicit weight tensor, but connectivity still forms a
    # dense [out, in] matrix. Every valid input in the pooling window gets 1.
    in_channels, height, width = input_shape
    out_channels, out_height, out_width = output_shape
    if in_channels != channels or out_channels != channels:
        raise ValueError(
            f"Pooling channel mismatch: input={in_channels}, output={out_channels}, channels={channels}."
        )

    out_size = out_height * out_width
    n_input = in_channels * height * width
    matrix = np.zeros((out_channels * out_size, n_input), dtype=np.int32)

    patches = unfold_input_indices_2d(
        in_channels,
        (height, width),
        kernel_size,
        stride,
        padding,
        dilation,
    ).cpu().numpy()

    if patches.shape[1] != out_size:
        raise ValueError(
            f"Pooling unfold mismatch: expected {out_size} output positions, got {patches.shape[1]}."
        )

    kernel_elems = int(np.prod(kernel_size))
    row_offsets = np.tile(np.arange(out_size, dtype=np.int64), kernel_elems)

    for channel in range(channels):
        patch_start = channel * kernel_elems
        patch_end = patch_start + kernel_elems

        flat_cols = patches[patch_start:patch_end].reshape(-1)
        valid = flat_cols > 0
        matrix[channel * out_size + row_offsets[valid], flat_cols[valid] - 1] = sign

    return matrix


def conv1d_weight_matrix(
    weight: Tensor,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    groups: int,
    sign: int,
) -> np.ndarray:
    # Reuse the 2D expansion path by treating 1D signals as H=1 feature maps.
    in_channels, in_length = input_shape
    out_channels, out_length = output_shape
    kernel = weight.reshape(weight.shape[0], weight.shape[1], 1, weight.shape[2])
    return conv2d_weight_matrix(
        kernel,
        (in_channels, 1, in_length),
        (out_channels, 1, out_length),
        (1, stride[0]),
        (0, padding[0]),
        (1, dilation[0]),
        groups,
        sign,
    )


def pool1d_weight_matrix(
    channels: int,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    kernel_size: tuple[int],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    sign: int,
) -> np.ndarray:
    # Reuse the 2D pooling expansion path by treating 1D signals as H=1 maps.
    in_channels, in_length = input_shape
    out_channels, out_length = output_shape
    return pool2d_weight_matrix(
        channels,
        (in_channels, 1, in_length),
        (out_channels, 1, out_length),
        (1, kernel_size[0]),
        (1, stride[0]),
        (0, padding[0]),
        (1, dilation[0]),
        sign,
    )


def expanded_path_weight_matrix(
    predecessor: CoreOpNode | InNode,
    target: CoreOpNode,
    comp: Optional[nn.Module],
    weight: Optional[Tensor],
    sign: int,
) -> np.ndarray:
    # Convert one predecessor path into a dense [out, in] matrix, choosing
    # the correct expansion strategy from the comp type.
    input_shape = feature_shape(predecessor.shape)
    output_shape = feature_shape(target.shape)

    if weight is not None and (comp is None or isinstance(comp, nn.Linear) or weight.ndim == 2):
        return direct_weight_matrix(weight, input_shape, output_shape, sign)

    if isinstance(comp, nn.Conv1d):
        return conv1d_weight_matrix(
            weight,
            input_shape,
            output_shape,
            to_nd_tuple(comp.stride, 1),
            to_nd_tuple(comp.padding, 1),
            to_nd_tuple(comp.dilation, 1),
            comp.groups,
            sign,
        )

    if isinstance(comp, nn.Conv2d):
        return conv2d_weight_matrix(
            weight,
            input_shape,
            output_shape,
            to_nd_tuple(comp.stride, 2),
            to_nd_tuple(comp.padding, 2),
            to_nd_tuple(comp.dilation, 2),
            comp.groups,
            sign,
        )

    if isinstance(comp, nn.MaxPool1d):
        if comp.ceil_mode:
            raise NotImplementedError("MaxPool1d with ceil_mode=True is not supported.")

        return pool1d_weight_matrix(
            input_shape[0],
            input_shape,
            output_shape,
            to_nd_tuple(comp.kernel_size, 1),
            to_nd_tuple(comp.stride or comp.kernel_size, 1),
            to_nd_tuple(comp.padding, 1),
            to_nd_tuple(comp.dilation, 1),
            sign,
        )

    if isinstance(comp, nn.AvgPool1d):
        if comp.ceil_mode:
            raise NotImplementedError("AvgPool1d with ceil_mode=True is not supported.")

        return pool1d_weight_matrix(
            input_shape[0],
            input_shape,
            output_shape,
            to_nd_tuple(comp.kernel_size, 1),
            to_nd_tuple(comp.stride or comp.kernel_size, 1),
            to_nd_tuple(comp.padding, 1),
            (1,),
            sign,
        )

    if isinstance(comp, nn.MaxPool2d):
        if comp.ceil_mode:
            raise NotImplementedError("MaxPool2d with ceil_mode=True is not supported.")

        return pool2d_weight_matrix(
            input_shape[0],
            input_shape,
            output_shape,
            to_nd_tuple(comp.kernel_size, 2),
            to_nd_tuple(comp.stride or comp.kernel_size, 2),
            to_nd_tuple(comp.padding, 2),
            to_nd_tuple(comp.dilation, 2),
            sign,
        )

    if isinstance(comp, nn.AvgPool2d):
        if comp.ceil_mode:
            raise NotImplementedError("AvgPool2d with ceil_mode=True is not supported.")

        return pool2d_weight_matrix(
            input_shape[0],
            input_shape,
            output_shape,
            to_nd_tuple(comp.kernel_size, 2),
            to_nd_tuple(comp.stride or comp.kernel_size, 2),
            to_nd_tuple(comp.padding, 2),
            (1, 1),
            sign,
        )

    raise NotImplementedError(
        f"Unsupported weight expansion for comp {type(comp)} with weight {type(weight)}."
    )


def get_raw_weights(
    raw_neus: list[Neuron], input_neus: list[Neuron | InputElem]
) -> np.ndarray:
    # Return a dense routing-group weight matrix with shape
    # [number of output neurons, number of input elements].
    n_output = len(raw_neus)
    n_input = len(input_neus)
    weights = np.zeros((n_output, n_input), dtype=np.int32)

    # Cache expanded matrices per target node and predecessor node because
    # many raw neurons share the same source/target pair.
    target_cache: dict[CoreOpNode, dict[CoreOpNode | InNode, np.ndarray]] = {}

    for neu in raw_neus:
        target = neu.target
        if target in target_cache:
            continue

        ensure_target_components(target)
        signs = path_signs(target)
        path_matrices: dict[CoreOpNode | InNode, np.ndarray] = {}

        for predecessor, comp, weight, sign in zip(
            target.predecessors,
            target.comps,
            target.weights,
            signs,
        ):
            # Each predecessor contributes one dense [target_out, pred_out] block.
            matrix = expanded_path_weight_matrix(
                predecessor,
                target,
                comp,
                weight,
                sign,
            )
            if predecessor in path_matrices:
                path_matrices[predecessor] = path_matrices[predecessor] + matrix
            else:
                path_matrices[predecessor] = matrix

        target_cache[target] = path_matrices

    for i, neu in enumerate(raw_neus):
        path_matrices = target_cache[neu.target]
        for j, elem in enumerate(input_neus):
            matrix = path_matrices.get(elem.target)
            if matrix is None:
                # No edge from this input node to the current output neuron.
                continue

            # Both neuron and input indices are already flattened indices.
            weights[i, j] = matrix[neu.index.idx, elem.index.idx]

    return weights


class RoutingGroup:
    _counter = 0

    # core_blocks in the same routing group share the same following properties
    # lcn, input_sign, input_width
    def __init__(
        self,
        raw_neus: list[Neuron],
        input_list: list[Neuron | InputElem],
        nodes: Optional[set[CoreOpNode]] = None,
        input_nodes: Optional[set[CoreOpNode | InNode]] = None,
    ):
        self.id: int = type(self)._counter
        type(self)._counter += 1
        self.name: str = f"RG_{self.id}"

        # for better optimization, if nodes and input_nodes are provided,
        # it means the raw_neus and input_list are all neu and input_elem generated from these nodes,
        # so we can directly get the weight matrix for this routing group without checking the connection between each raw_neu and input_elem
        self.nodes = nodes
        self.input_nodes = input_nodes

        # set by generate_routing_groups
        self.raw_neus: list[Neuron] = raw_neus
        self.input_list: list[Neuron | InputElem] = (
            input_list  # input_list can be reordered later
        )
        self.input_set: set[Neuron | InputElem] = set(input_list)

        # set by mapper.set_rough_dest()
        self.dests: dict[Neuron, "RoutingGroup"] = {}
        self.lcn: LCN_EX = LCN_EX.LCN_1X

        # self.core_blocks: list[CoreBlock] = []
        self.core_placements: list[CorePlacement] = []
        self.n_core_required: int = -1

        # set by mapper.routing() call self.assign_coord()
        self.assigned_cores: dict[CoordXY, CorePlacement] = {}
        self._multicast_config: Optional[AERPacketZXYCopy] = None
        self._base_coord: Optional[CoordXY] = None

    def set_lcn(self):
        input_widths: set[DataWidth] = set(
            [neu.core_config().input_width for neu in self.raw_neus]
        )
        assert (
            len(input_widths) == 1
        ), "All neurons in the routing group must have the same input width."
        # if input width, input sign weight are different, need to split routing group before calling this function
        input_width = input_widths.pop()
        max_axon_addr = len(self.input_list) * (2**input_width)
        lcn = ((max_axon_addr - 1) // FANIN_BASE).bit_length()
        self.lcn = LCN_EX(lcn)

    def allocate_neurons(self):
        """core placement generation"""
        # you can get neu_attrs_part2, inherited_core_config, target_lcn, lcn for each neuron in self.raw_neus, like:
        # all the attrs in neu_attrs_part2 are valid except weight compress, you should set weight compress according to your weight storage strategy
        # inherited core_config are valid except weight_width, you can set weight_width larger than or equal to the original value for optimization

        weights = get_raw_weights(self.raw_neus, self.input_list)

        print(
            f"Allocating neurons for Routing Group {self.name} with {len(self.raw_neus)} neurons."
        )

        # 1. Grouping Phase
        core_groups: dict[
            tuple[Frontend_Core_Config, Backend_Core_Config],
            list[tuple[Neuron, np.ndarray]],
        ] = {}
        for i, neu in enumerate(self.raw_neus):
            frontend_core_conf = neu.core_config()
            backend_core_conf = Backend_Core_Config(
                lcn=self.lcn,
                target_lcn=self.dests[neu].lcn,
            )
            weight_of_neu = weights[i]
            key = (frontend_core_conf, backend_core_conf)
            if key not in core_groups:
                core_groups[key] = []
            core_groups[key].append((neu, weight_of_neu))

        self.core_placements: list[CorePlacement] = []

        # 2. Allocation Phase
        for key, group_items in core_groups.items():
            frontend_core_conf, backend_core_conf = key
            # Initialize the first core for the current group
            current_core = OfflineCorePlacementV2(frontend_core_conf, backend_core_conf)

            last_full_attrs = None

            for neu, weight_of_neu in group_items:
                attrs_part2 = neu.attrs_part2()

                # Weight Strategy
                current_weight_width = frontend_core_conf.weight_width
                w_dense = Weight(
                    weight_of_neu, WeightCompressType.DENSE, current_weight_width
                )
                w_sparse = Weight(
                    weight_of_neu, WeightCompressType.SPARSE, current_weight_width
                )

                if w_sparse.n_sram_required() < w_dense.n_sram_required():
                    selected_weight = w_sparse
                    attrs_part2.weight_compress = WeightCompressType.SPARSE
                else:
                    selected_weight = w_dense
                    attrs_part2.weight_compress = WeightCompressType.DENSE

                neuron_type = (
                    NeuronType.HALF
                    if attrs_part2 == last_full_attrs
                    else NeuronType.FULL
                )
                output_type = neu.output_type()
                attrs_part1 = OfflineNeuFullAttrsV2Part1(
                    weight_skew=0,
                    weight_address_start=0,
                    weight_address_end=0,
                    fold_type=FoldType.UNFOLDED,
                    neuron_type=neuron_type,
                    output_type=output_type,
                )
                neu_placement = OfflineNeuronPlacement([neu], attrs_part1, attrs_part2)

                # SRAM Check
                neu_sram_req = neu_placement.n_sram_required()
                weight_sram_req = selected_weight.n_sram_required()
                total_req = neu_sram_req + weight_sram_req

                if total_req > 4096:
                    raise NotImplementedError(
                        f"Neuron {neu} with its weight requires {total_req} SRAM lines."
                    )

                if current_core.n_sram_required() + total_req > 4096:
                    self.core_placements.append(current_core)

                    # Create new core with inheritance
                    current_core = OfflineCorePlacementV2(
                        frontend_core_conf, backend_core_conf
                    )
                    neuron_type = NeuronType.FULL
                    neu_placement.neu_attrs_part1.neuron_type = NeuronType.FULL
                    last_full_attrs = attrs_part2

                elif neuron_type == NeuronType.FULL:
                    last_full_attrs = attrs_part2

                # print(f"Neuron {neu} assigned type {neu_placement.neuron_type}.")
                current_core.neus.append(neu_placement)
                current_core.weights.append(selected_weight)

                idx = len(current_core.neus) - 1
                current_core.neu_weight_map[idx] = idx

            if len(current_core.neus) > 0:
                self.core_placements.append(current_core)

        self.n_core_required = len(self.core_placements)

        for core_placement in self.core_placements:
            core_placement.set_weight_address()

    def assign_coord(self, coords: list[CoordXY], copy_config: AERPacketZXYCopy):
        assert len(coords) >= len(
            self.core_placements
        ), "Not enough coordinates provided to assign."
        for i, coord in enumerate(coords):
            if i < len(self.core_placements):
                self.assigned_cores[coord] = self.core_placements[i]
            else:
                self.assigned_cores[coord] = EmptyOfflineCorePlacementV2()
            self.assigned_cores[coord]._coord = coord
        self._base_coord = coords[0]
        self._multicast_config = copy_config

    def set_detail_dest(self):
        for core_placement in self.core_placements:
            for neu_placement in core_placement.neus:
                # mostly each neu_placement contains only one raw_neu
                # for folded neuron placement, it may contain multiple raw_neus
                # but we only need to set dest_info for the first raw_neu
                main_neu = neu_placement.raw_neus[0]
                dest_routing_group = self.dests[main_neu]

                dest_coord = dest_routing_group.base_coord
                coord_copy = dest_routing_group.multicast_config
                axon_addr_logic = dest_routing_group.input_list.index(main_neu)

                # the input width of all cores in dest routing group should be the same,
                # so we can use the output width of this core as the input width of dest routing group
                axon_addr_count = axon_addr_logic * (2**core_placement.output_width)

                addr_axon = axon_addr_count % FANIN_BASE
                tick_relative = axon_addr_count // FANIN_BASE

                coord_offset, _ = find_coordxy_shortest_path(
                    target=dest_coord, start=core_placement.coord
                )

                neu_placement.dest_info = OfflineNeuDestInfoV2(
                    tick_relative=tick_relative,
                    addr_axon=addr_axon,  # to be set during core allocation
                    addr_core_xy=coord_offset.z,
                    addr_core_x=coord_offset.x,
                    addr_core_y=coord_offset.y,
                    addr_copy_xy=coord_copy.z,
                    addr_copy_x=coord_copy.x,
                    addr_copy_y=coord_copy.y,
                )

    def set_auto_core_config(self):
        for core_placement in self.core_placements:
            core_placement.set_auto_core_config()

    @property
    def multicast_config(self) -> AERPacketZXYCopy:
        if self._multicast_config is None:
            raise ValueError("multicast_config has not been set yet.")
        return self._multicast_config

    @property
    def base_coord(self) -> CoordXY:
        if self._base_coord is None:
            raise ValueError("base_coord has not been set yet.")
        return self._base_coord

    def __hash__(self):
        # use hash of each raw_neu to identify routing group
        return hash(tuple(sorted([hash(neu) for neu in self.raw_neus])))

    def __str__(self) -> str:
        neu_strs = [str(neu) for neu in self.raw_neus]
        # if too many neurons, only show first 3 and last 3
        if len(neu_strs) > 6:
            neu_strs = neu_strs[:3] + ["..."] + neu_strs[-3:]
        # if too many inputs, only show first 3 and last 3
        input_strs = [str(neu) for neu in self.input_list]
        if len(input_strs) > 6:
            input_strs = input_strs[:3] + ["..."] + input_strs[-3:]
        return f"\n{self.name}(\nraw_neus=[{', '.join(neu_strs)}], \ninput_list=[{', '.join(input_strs)}])\n"

    def __repr__(self) -> str:
        return self.__str__()

    def info(self) -> str:
        info_str = f"{self.name}:\n"
        # if too many dests, only show first 3 and last 3
        dest_strs = []
        for neu, dest_rg in self.dests.items():
            dest_strs.append(f"{str(neu)} -> {dest_rg.name}")
        if len(dest_strs) > 6:
            dest_strs = dest_strs[:3] + ["..."] + dest_strs[-3:]
        info_str += f"  Number of Neurons: {len(self.raw_neus)}\n"
        info_str += f"  Number of Inputs: {len(self.input_list)}\n"
        info_str += f"  Dests:\n    " + "\n    ".join(dest_strs) + "\n"
        info_str += f"  LCN: {self.lcn.name}\n"
        info_str += f"  Number of Core Placements: {len(self.core_placements)}\n"
        if self._base_coord is not None:
            info_str += f"  Base Coord: ({self._base_coord.x}, {self._base_coord.y})\n"
        if self._multicast_config is not None:
            info_str += f"  Multicast Config: (Z: {self._multicast_config.z}, X: {self._multicast_config.x}, Y: {self._multicast_config.y})\n"
        return info_str

    def routing_summary(self) -> str:
        summary_str = f"{self.name} Routing Summary:\n"
        for i, core_placement in enumerate(self.core_placements):
            summary_str += f"  Core Placement {i} at {core_placement.coord}:\n"
            summary_str += f"    Number of Neurons: {len(core_placement.neus)}\n"
            summary_str += f"    Number of Weights: {len(core_placement.weights)}\n"
        return summary_str


def toposort_for_rg(
    routing_groups: list[RoutingGroup],
) -> tuple[list[RoutingGroup], dict[int, list[int]]]:
    """topological sort for routing groups based on their dests"""
    from collections import defaultdict, deque

    indegree = {rg: 0 for rg in routing_groups}
    graph = defaultdict(list)

    for rg in routing_groups:
        for dest_rg in rg.dests.values():
            if dest_rg not in routing_groups:
                continue
            graph[rg].append(dest_rg)
            indegree[dest_rg] += 1

    queue = deque([rg for rg in routing_groups if indegree[rg] == 0])
    sorted_rgs = []

    while queue:
        rg = queue.popleft()
        sorted_rgs.append(rg)
        for neighbor in graph[rg]:
            if neighbor not in routing_groups:
                continue
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    next_rg_id = {}
    for i, rg in enumerate(sorted_rgs):
        next_rg_id[i] = [sorted_rgs.index(dest_rg) for dest_rg in graph[rg]]

    if len(sorted_rgs) != len(routing_groups):
        raise ValueError("Cycle detected in routing groups.")

    return sorted_rgs, next_rg_id
