from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from google.protobuf.json_format import MessageToJson
from paicorelib import (
    CoordZXYOffset,
    DataSign,
    DataWidth,
    OnlineDataWidth,
    OnlineFrameGenV2,
    find_coordxy_shortest_path,
)

from paibox.paiir import PAIIRGraph
from paibox.paiir.ir.calc_params import OnlineCoreSemanticMode
from paibox.paiir.ir.ir_base import InputNode, OutputNode
from paibox.paiir.ir.op_node import OnlineCoreOp, TransformOp
from paibox.paiir.ir.signal_domain import SignalDomain

from ..coreplacement import CorePlacement, Frontend_Core_Config, OnlineCorePlacementV2
from ..op_node import Neuron, RemapElem
from ..proto import get_schema_version
from ..proto.compile_artifacts_pb2 import (
    CompileArtifacts,
    ConfigFrames,
    DataType,
    InputEntry,
    InputTensorMapping,
    OutputEntry,
    OutputTensorMapping,
    OutputTensorMappings,
    RuntimeParams,
    TickParams,
)
from .utils import (
    FrameRecords,
    TargetPlatform,
    WordOrder,
    export_framearray_to_int32,
    iter_frame_arrays_core_major,
    resolve_platform_exports,
)

if TYPE_CHECKING:
    from ..op_node import SourceElem
    from ..routing import InputGroup, OutputGroup, RemapGroup, RoutingGroup

TickTriple = tuple[int, int, int]
DataFormat = tuple[DataSign, DataWidth]

_DATA_TYPE_BY_FORMAT: Mapping[DataFormat, DataType.Code] = {
    (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT): DataType.UINT1,
    (DataSign.SIGNED, DataWidth.WIDTH_1BIT): DataType.INT1,
    (DataSign.UNSIGNED, DataWidth.WIDTH_2BIT): DataType.UINT2,
    (DataSign.SIGNED, DataWidth.WIDTH_2BIT): DataType.INT2,
    (DataSign.UNSIGNED, DataWidth.WIDTH_4BIT): DataType.UINT4,
    (DataSign.SIGNED, DataWidth.WIDTH_4BIT): DataType.INT4,
    (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT): DataType.UINT8,
    (DataSign.SIGNED, DataWidth.WIDTH_8BIT): DataType.INT8,
}

_ONLINE_DATA_TYPE_BY_WIDTH: Mapping[OnlineDataWidth, tuple[int, int]] = {
    OnlineDataWidth.TYPE_1BIT: (DataType.UINT1, 1),
    OnlineDataWidth.TYPE_FP16: (DataType.FLOAT16, 16),
    OnlineDataWidth.TYPE_UINT8: (DataType.UINT8, 8),
    OnlineDataWidth.TYPE_INT8: (DataType.INT8, 8),
}


def _bit_width_from_data_width(width: DataWidth) -> int:
    return int(1 << width)


def _dtype_from_format(fmt: DataFormat, context: str) -> DataType.Code:
    dtype = _DATA_TYPE_BY_FORMAT.get(fmt)
    if dtype is None:
        sign, width = fmt
        raise ValueError(
            f"{context} has unsupported data format "
            f"{sign.name}/{width.name}; expected signed/unsigned 1/2/4/8-bit DATA."
        )
    return dtype


def _set_entry_dtype(
    entry: InputEntry | OutputEntry, fmt: DataFormat, bit_width: int, context: str
) -> None:
    """Set entry dtype and verify it matches the mapping-level bit width."""
    expected_bit_width = _bit_width_from_data_width(fmt[1])
    if bit_width != expected_bit_width:
        sign, width = fmt
        raise ValueError(
            f"{context} bit width mismatch: mapping has {bit_width}, "
            f"format {sign.name}/{width.name} implies {expected_bit_width}."
        )
    entry.dtype = _dtype_from_format(fmt, context)


def _set_mapping_bit_width(
    mapping: InputTensorMapping | OutputTensorMapping,
    mapping_kind: str,
    mapping_name: str,
    bit_width: int,
) -> None:
    """Set tensor-level bit width and reject mixed-width entries."""
    if mapping.HasField("bit_width"):
        if mapping.bit_width != bit_width:
            raise ValueError(
                f"{mapping_kind} mapping '{mapping_name}' contains mixed bit widths: "
                f"{mapping.bit_width} and {bit_width}."
            )
    else:
        mapping.bit_width = bit_width


def _set_thread_output_lcn(
    output_mappings: OutputTensorMappings, output_group: OutputGroup, thread_id: int
) -> None:
    """Set thread-level output LCN and reject conflicting output groups."""
    target_lcn = output_group.lcn.value
    if output_mappings.HasField("target_lcn"):
        if output_mappings.target_lcn != target_lcn:
            raise ValueError(
                f"thread {thread_id} contains output groups with mixed target_lcn: "
                f"{output_mappings.target_lcn} and {target_lcn}."
            )
    else:
        output_mappings.target_lcn = target_lcn


def _online_dtype_and_bit_width(
    width: OnlineDataWidth, context: str
) -> tuple[int, int]:
    dtype_and_width = _ONLINE_DATA_TYPE_BY_WIDTH.get(width)
    if dtype_and_width is None:
        raise ValueError(f"{context} has unsupported online data width {width.name}.")
    return dtype_and_width


def _output_kind_from_elem(elem: SourceElem) -> OutputTensorMapping.OutputKind:
    domain = elem.target.raw_node.signal_semantics.output_domain
    if domain is SignalDomain.VALUE:
        kind = OutputTensorMapping.DATA
    else:
        kind = OutputTensorMapping.VOLTAGE

    if kind == OutputTensorMapping.DATA:
        if elem.output_bit_num > 8:
            raise ValueError(
                f"DATA output {elem} has unsupported bit width "
                f"{elem.output_bit_num}; expected <= 8."
            )
    else:
        if elem.output_bit_num != 32:
            raise ValueError(
                f"VOLTAGE output {elem} has bit width "
                f"{elem.output_bit_num}; expected 32."
            )

    return kind


def _set_output_mapping_kind(
    output_mapping: OutputTensorMapping, output_name: str, elem: SourceElem
) -> OutputTensorMapping.OutputKind:
    kind = _output_kind_from_elem(elem)
    if output_mapping.HasField("kind"):
        if output_mapping.kind != kind:
            raise ValueError(
                f"output mapping '{output_name}' contains mixed output kinds: "
                f"{OutputTensorMapping.OutputKind.Name(output_mapping.kind)} and "
                f"{OutputTensorMapping.OutputKind.Name(kind)}."
            )
    else:
        output_mapping.kind = kind

    return kind


def _tick_tuple_from_core_conf(core_conf: Frontend_Core_Config) -> TickTriple:
    return core_conf.tick_start, core_conf.tick_duration, core_conf.tick_initial


def _tick_tuple_from_online_placement(
    core_placement: OnlineCorePlacementV2,
) -> TickTriple:
    params = core_placement.core_params
    assert params.tick_start is not None
    return params.tick_start, params.tick_duration, params.tick_initial


def _core_tick_export_parts(
    core_placement: CorePlacement,
) -> tuple[tuple[str, ...], int, TickTriple]:
    if isinstance(core_placement, OnlineCorePlacementV2):
        return (
            core_placement.node_names,
            core_placement.core_params.thread_number,
            _tick_tuple_from_online_placement(core_placement),
        )

    if not core_placement.neus:
        return (
            (),
            core_placement.default_core_config.thread_number,
            _tick_tuple_from_core_conf(core_placement.frontend_core_config),
        )

    nodes = {
        raw_neu.target.raw_node.name
        for neu_placement in core_placement.neus
        for raw_neu in neu_placement.raw_neus
    }
    return (
        tuple(sorted(nodes)),
        core_placement.default_core_config.thread_number,
        _tick_tuple_from_core_conf(core_placement.frontend_core_config),
    )


def _producer_core_conf_from_source_elem(
    elem: SourceElem, groups: Sequence[RoutingGroup | RemapGroup]
) -> Frontend_Core_Config:
    from ..routing import RemapGroup

    current = elem
    visited: set[SourceElem] = set()
    while True:
        if isinstance(current, Neuron):
            return current.core_config()
        if current in visited:
            raise ValueError(
                f"Detected remap cycle while tracing compute source for {elem}."
            )
        visited.add(current)
        if not isinstance(current, RemapElem):
            raise TypeError(
                f"Expected output source to trace to a Neuron, got {type(current)}."
            )

        for group in groups:
            if isinstance(group, RemapGroup) and current in group.source_dict:
                current = group.source_dict[current]
                break
        else:
            raise ValueError(f"Cannot trace remap output {current} to a compute core.")


def _tick_tuple_from_source_elem(
    elem: SourceElem, groups: Sequence[RoutingGroup | RemapGroup]
) -> TickTriple:
    return _tick_tuple_from_core_conf(
        _producer_core_conf_from_source_elem(elem, groups)
    )


def _output_format_from_source_elem(
    elem: SourceElem, groups: Sequence[RoutingGroup | RemapGroup]
) -> DataFormat:
    core_conf = _producer_core_conf_from_source_elem(elem, groups)
    return core_conf.output_sign, core_conf.output_width


def _routing_group_input_formats(routing_group: RoutingGroup) -> set[DataFormat]:
    return {
        (
            core_placement.frontend_core_config.input_sign,
            core_placement.frontend_core_config.input_width,
        )
        for core_placement in routing_group.core_placements
        if core_placement.neus
    }


def _require_single_data_format(
    formats: set[DataFormat], mapping_kind: str, mapping_name: str
) -> DataFormat:
    if not formats:
        raise ValueError(
            f"{mapping_kind} mapping '{mapping_name}' has no data type source."
        )
    if len(formats) != 1:
        ordered = sorted((sign.name, width.name) for sign, width in formats)
        raise ValueError(
            f"{mapping_kind} mapping '{mapping_name}' maps to multiple data "
            f"formats: {ordered}"
        )
    return next(iter(formats))


def _set_tick_params(tick_params: TickParams, tick: TickTriple) -> None:
    tick_params.tick_start = tick[0]
    tick_params.tick_duration = tick[1]
    tick_params.tick_initial = tick[2]


def _require_single_tick(
    ticks: set[TickTriple], mapping_kind: str, mapping_name: str
) -> TickTriple:
    if not ticks:
        raise ValueError(f"{mapping_kind} mapping '{mapping_name}' has no tick source.")
    if len(ticks) != 1:
        ordered = sorted(ticks)
        raise ValueError(
            f"{mapping_kind} mapping '{mapping_name}' maps to multiple compute "
            f"ticks: {ordered}"
        )
    return next(iter(ticks))


def _routing_group_ticks(routing_group: RoutingGroup) -> set[TickTriple]:
    return {
        _tick_tuple_from_core_conf(core_placement.frontend_core_config)
        for core_placement in routing_group.core_placements
        if core_placement.neus
    }


def _thread_tick_depth(
    output_groups: Sequence[OutputGroup], groups: Sequence[RoutingGroup | RemapGroup]
) -> int | None:
    """Return the latest producer start tick for one exported thread."""
    if not output_groups:
        return None

    tick_starts = [
        int(_producer_core_conf_from_source_elem(elem, groups).tick_start)
        for out_grp in output_groups
        for elem in out_grp.input_list
    ]
    if not tick_starts:
        raise ValueError("Output groups contain no output elements.")

    return max(tick_starts)


def _decode_mode_from_target_lcn(
    target_lcn: int, timesteps: int
) -> RuntimeParams.DecodeMode:
    ts_width, _ = OnlineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[target_lcn]
    max_stream_timesteps = 1 << ts_width
    return (
        RuntimeParams.STREAM
        if max_stream_timesteps >= timesteps
        else RuntimeParams.STEP
    )


def _thread_decode_mode(
    output_groups: Sequence[OutputGroup], timesteps: int
) -> RuntimeParams.DecodeMode:
    """Derive STREAM/STEP from the final output LCN timestep capacity."""
    if not output_groups:
        return RuntimeParams.STREAM
    if len(output_groups) != 1:
        raise NotImplementedError(
            "RuntimeParams export currently supports one OutputGroup per thread."
        )

    out_grp = output_groups[0]
    return _decode_mode_from_target_lcn(int(out_grp.lcn.value), timesteps)


def _online_placements_by_node(
    coreplacements: Sequence[CorePlacement],
) -> dict[str, OnlineCorePlacementV2]:
    placements: dict[str, OnlineCorePlacementV2] = {}
    for core_placement in coreplacements:
        if not isinstance(core_placement, OnlineCorePlacementV2):
            continue
        for node_name in core_placement.node_names:
            placements[node_name] = core_placement
    return placements


def _trace_online_input_path(
    graph: PAIIRGraph, first_forward_name: str
) -> tuple[InputNode, tuple[TransformOp, ...]]:
    transforms: list[TransformOp] = []
    current_names = graph.predecessors(first_forward_name)
    if len(current_names) != 1:
        raise ValueError(
            f"online input path expects '{first_forward_name}' to have exactly one predecessor."
        )
    current_name = current_names[0]

    while True:
        node = graph.nodes[current_name]
        if isinstance(node, InputNode):
            return node, tuple(reversed(transforms))
        if not isinstance(node, TransformOp):
            raise ValueError(
                f"online input path to '{first_forward_name}' must contain only "
                f"TransformOp nodes before the first forward core, got {type(node).__name__}."
            )

        transforms.append(node)
        current_names = graph.predecessors(current_name)
        if len(current_names) != 1:
            raise ValueError(
                f"online transform path expects '{current_name}' to have exactly one predecessor."
            )
        current_name = current_names[0]


def _input_source_to_target_positions(
    input_node: InputNode,
    transforms: Sequence[TransformOp],
    expected_shape: torch.Size,
) -> list[int]:
    numel = prod(input_node.shape) if input_node.shape else 1
    index_tensor = torch.arange(numel, dtype=torch.int64).reshape(
        tuple(input_node.shape)
    )
    transformed = index_tensor
    for transform in transforms:
        transformed = transform(transformed)

    if expected_shape and transformed.shape != expected_shape:
        raise ValueError(
            f"online input transform result shape {tuple(transformed.shape)} does not match "
            f"the first forward input shape {tuple(expected_shape)}."
        )

    flat = transformed.reshape(-1).tolist()
    if sorted(flat) != list(range(numel)):
        raise ValueError("online input transform path must remain a 1:1 element remap.")

    source_to_target = [0] * numel
    for target_idx, source_idx in enumerate(flat):
        source_to_target[int(source_idx)] = target_idx

    return source_to_target


def _online_tick_relative_and_addr_axon(
    flat_idx: int, target_lcn: int
) -> tuple[int, int]:
    _, axon_width = OnlineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[target_lcn]
    axon_mask = (1 << axon_width) - 1
    return flat_idx >> axon_width, flat_idx & axon_mask


def _export_online_input_mapping(
    thread_mapping,
    graph: PAIIRGraph,
    first_forward: OnlineCoreOp,
    first_forward_name: str,
    placements_by_node: Mapping[str, OnlineCorePlacementV2],
) -> None:
    input_node, transforms = _trace_online_input_path(graph, first_forward_name)
    first_forward_placement = placements_by_node[first_forward_name]
    input_mapping = thread_mapping.input_mappings.items.add()
    input_mapping.name = input_node.name
    input_mapping.shape.size.extend(list(input_node.shape))
    _set_tick_params(
        input_mapping.tick, _tick_tuple_from_online_placement(first_forward_placement)
    )

    dtype, bit_width = _online_dtype_and_bit_width(
        first_forward.core_params.input_width,
        f"online input mapping '{input_node.name}'",
    )
    input_mapping.bit_width = bit_width

    target_lcn = int(first_forward.core_params.lcn_at)
    core_offset, _ = find_coordxy_shortest_path(first_forward_placement.coord)
    expected_shape = (
        first_forward.input_layouts[0].shape
        if first_forward.input_layouts
        else torch.Size()
    )
    source_to_target = _input_source_to_target_positions(
        input_node, transforms, expected_shape
    )

    for elem_idx, target_idx in enumerate(source_to_target):
        tick_relative, addr_axon = _online_tick_relative_and_addr_axon(
            target_idx, target_lcn
        )
        entry = input_mapping.entries.add()
        entry.elem_idx = elem_idx
        entry.copy_id = 0
        entry.dtype = dtype
        entry.tick_relative = tick_relative
        entry.addr_axon = addr_axon
        entry.target_lcn = target_lcn
        entry.core_offset.xy = core_offset.z
        entry.core_offset.x = core_offset.x
        entry.core_offset.y = core_offset.y
        entry.copy_count.xy = 0
        entry.copy_count.x = 0
        entry.copy_count.y = 0


def _export_online_output_mappings(
    thread_mapping,
    graph: PAIIRGraph,
    thread_id: int,
    placements_by_node: Mapping[str, OnlineCorePlacementV2],
) -> None:
    target_lcn: int | None = None

    for output_node in graph.output_nodes():
        assert isinstance(output_node, OutputNode)
        pred_names = graph.predecessors(output_node.name)
        if len(pred_names) != 1:
            raise ValueError(
                f"online output node '{output_node.name}' must have exactly one predecessor."
            )

        pred_name = pred_names[0]
        pred = graph.nodes[pred_name]
        if not isinstance(pred, OnlineCoreOp):
            continue
        if pred.core_params.thread_number != thread_id:
            continue
        if pred.core_params.semantic_mode is not OnlineCoreSemanticMode.FORWARD:
            continue

        output_width = pred.core_params.output_width
        if not isinstance(output_width, OnlineDataWidth):
            raise ValueError(
                f"online output mapping '{pred_name}' requires data output_width, "
                f"got {type(output_width).__name__}."
            )

        dtype, bit_width = _online_dtype_and_bit_width(
            output_width, f"online output mapping '{pred_name}'"
        )
        placement = placements_by_node[pred_name]
        output_mapping = thread_mapping.output_mappings.items.add()
        output_mapping.name = pred_name
        output_mapping.kind = OutputTensorMapping.DATA
        output_mapping.bit_width = bit_width
        output_mapping.shape.size.extend(list(output_node.shape))
        _set_tick_params(
            output_mapping.tick, _tick_tuple_from_online_placement(placement)
        )

        current_target_lcn = int(pred.core_params.target_lcn_at)
        if target_lcn is None:
            target_lcn = current_target_lcn
            thread_mapping.output_mappings.target_lcn = current_target_lcn
        elif target_lcn != current_target_lcn:
            raise ValueError(
                f"online thread {thread_id} output mappings use multiple target_lcn values: "
                f"{target_lcn} and {current_target_lcn}."
            )

        for elem_idx in range(prod(output_node.shape) if output_node.shape else 1):
            entry = output_mapping.entries.add()
            entry.elem_idx = elem_idx
            entry.copy_id = 0
            entry.dtype = dtype
            entry.axon_bit_idx = elem_idx


def _export_online_thread_io_mapping(
    thread_mapping,
    graph: PAIIRGraph,
    thread_id: int,
    placements_by_node: Mapping[str, OnlineCorePlacementV2],
) -> None:
    forward_names = [
        name
        for name in graph.topo_sort()
        if isinstance(graph.nodes[name], OnlineCoreOp)
        and graph.nodes[name].core_params.thread_number == thread_id
        and graph.nodes[name].core_params.semantic_mode
        is OnlineCoreSemanticMode.FORWARD
    ]
    if not forward_names:
        return

    first_forward_name = forward_names[0]
    first_forward = graph.nodes[first_forward_name]
    assert isinstance(first_forward, OnlineCoreOp)

    _export_online_input_mapping(
        thread_mapping,
        graph,
        first_forward,
        first_forward_name,
        placements_by_node,
    )
    _export_online_output_mappings(
        thread_mapping,
        graph,
        thread_id,
        placements_by_node,
    )


def _export_online_thread_runtime(thread_mapping, timesteps: int) -> None:
    if not thread_mapping.output_mappings.items:
        return

    if not thread_mapping.output_mappings.HasField("target_lcn"):
        raise ValueError("online thread runtime export requires output target_lcn.")

    tick_depth = max(
        int(output_mapping.tick.tick_start)
        for output_mapping in thread_mapping.output_mappings.items
    )
    thread_mapping.runtime.timesteps = timesteps
    thread_mapping.runtime.tick_depth = tick_depth
    thread_mapping.runtime.sync_steps = tick_depth + timesteps - 1
    thread_mapping.runtime.decode_mode = _decode_mode_from_target_lcn(
        int(thread_mapping.output_mappings.target_lcn), timesteps
    )


def _export_offline_thread_runtime(
    thread_mapping,
    thread_output_groups: Sequence[OutputGroup],
    groups: Sequence[RoutingGroup | RemapGroup],
    timesteps: int,
) -> None:
    tick_depth = _thread_tick_depth(thread_output_groups, groups)
    if tick_depth is None:
        return

    thread_mapping.runtime.timesteps = timesteps
    thread_mapping.runtime.tick_depth = tick_depth
    thread_mapping.runtime.sync_steps = tick_depth + timesteps - 1
    thread_mapping.runtime.decode_mode = _thread_decode_mode(
        thread_output_groups, timesteps
    )


def _export_offline_thread_io_mapping(
    thread_mapping,
    thread_id: int,
    groups: Sequence[RoutingGroup | RemapGroup],
    input_groups: Sequence[InputGroup],
    thread_output_groups: Sequence[OutputGroup],
) -> None:
    input_mappings_by_name: dict[str, InputTensorMapping] = {}
    input_ticks_by_name: dict[str, set[TickTriple]] = {}
    from ..routing import RoutingGroup

    for in_grp in input_groups:
        if in_grp.thread_id != thread_id:
            continue
        for elem, dest in in_grp.dest_infos.items():
            input_name = elem.target.raw_node.name
            if input_name not in input_mappings_by_name:
                input_mapping = thread_mapping.input_mappings.items.add()
                input_mapping.name = input_name
                input_mapping.shape.size.extend(list(elem.target.shape))
                input_mappings_by_name[input_name] = input_mapping
                input_ticks_by_name[input_name] = set()

            input_mapping = input_mappings_by_name[input_name]
            dest_group = in_grp.get_dest(elem)
            if not isinstance(dest_group, RoutingGroup):
                raise TypeError(
                    f"Input entry {input_name}[{elem.index.idx}] maps to "
                    f"{type(dest_group).__name__}, expected RoutingGroup."
                )
            input_ticks_by_name[input_name].update(_routing_group_ticks(dest_group))
            _set_mapping_bit_width(
                input_mapping, "input", input_name, elem.output_bit_num
            )
            input_entry = input_mapping.entries.add()
            input_entry.elem_idx = elem.index.idx
            input_entry.copy_id = elem.index.copy_id
            _set_entry_dtype(
                input_entry,
                _require_single_data_format(
                    _routing_group_input_formats(dest_group),
                    "input entry",
                    f"{input_name}[{elem.index.idx}]",
                ),
                elem.output_bit_num,
                f"input entry {input_name}[{elem.index.idx}]",
            )
            input_entry.tick_relative = dest.tick_relative
            input_entry.addr_axon = dest.addr_axon
            input_entry.core_offset.xy = dest.addr_core_xy
            input_entry.core_offset.x = dest.addr_core_x
            input_entry.core_offset.y = dest.addr_core_y
            input_entry.copy_count.xy = dest.addr_copy_xy
            input_entry.copy_count.x = dest.addr_copy_x
            input_entry.copy_count.y = dest.addr_copy_y
            input_entry.target_lcn = in_grp.dest_lcn[elem]

    for input_name, input_mapping in input_mappings_by_name.items():
        _set_tick_params(
            input_mapping.tick,
            _require_single_tick(input_ticks_by_name[input_name], "input", input_name),
        )

    output_mappings_by_name: dict[str, OutputTensorMapping] = {}
    output_ticks_by_name: dict[str, set[TickTriple]] = {}
    for out_grp in thread_output_groups:
        _set_thread_output_lcn(thread_mapping.output_mappings, out_grp, thread_id)
        for axon_bit_idx, elem in sorted(
            out_grp.axon_bit_allocator.axon_infos, key=lambda item: item[0]
        ):
            if elem not in out_grp.input_set:
                continue
            output_name = elem.target.raw_node.name
            if output_name not in output_mappings_by_name:
                output_mapping = thread_mapping.output_mappings.items.add()
                output_mapping.name = output_name
                output_mapping.shape.size.extend(list(elem.target.shape))
                output_mappings_by_name[output_name] = output_mapping
                output_ticks_by_name[output_name] = set()

            output_mapping = output_mappings_by_name[output_name]
            output_ticks_by_name[output_name].add(
                _tick_tuple_from_source_elem(elem, groups)
            )
            _set_mapping_bit_width(
                output_mapping, "output", output_name, elem.output_bit_num
            )
            output_entry = output_mapping.entries.add()
            output_entry.elem_idx = elem.index.idx
            output_entry.copy_id = elem.index.copy_id
            output_entry.axon_bit_idx = axon_bit_idx
            output_kind = _set_output_mapping_kind(output_mapping, output_name, elem)
            if output_kind == OutputTensorMapping.DATA:
                _set_entry_dtype(
                    output_entry,
                    _output_format_from_source_elem(elem, groups),
                    elem.output_bit_num,
                    f"output entry {output_name}[{elem.index.idx}]",
                )

    for output_name, output_mapping in output_mappings_by_name.items():
        _set_tick_params(
            output_mapping.tick,
            _require_single_tick(
                output_ticks_by_name[output_name], "output", output_name
            ),
        )


def export_compile_artifacts(
    output_path: str | Path,
    target_platform: TargetPlatform,
    word_order: WordOrder,
    export_python: bool,
    debug: bool,
    timesteps: int,
    groups: Sequence[RoutingGroup | RemapGroup],
    input_groups: Sequence[InputGroup],
    output_groups: Sequence[OutputGroup],
    coreplacements: Sequence[CorePlacement],
    global_starts: Mapping[int, CoordZXYOffset],
    frame_records: FrameRecords,
    online_graph: PAIIRGraph | None = None,
) -> Path:
    """Export protobuf artifacts for config frames and I/O mappings."""
    export_x86, _ = resolve_platform_exports(target_platform, debug)

    proto_dir = Path(__file__).parent.parent / "proto"
    proto_out_dir = Path(output_path) / "proto"
    proto_out_dir.mkdir(parents=True, exist_ok=True)

    pb_path = proto_out_dir / "config.pb"
    pb_text_path = proto_out_dir / "config.json"

    proto_files = ["compile_artifacts.proto"]
    if export_python and export_x86:
        proto_files.extend(["compile_artifacts_pb2.py", "compile_artifacts_pb2.pyi"])

    for file_name in proto_files:
        src_file = proto_dir / file_name
        if not src_file.exists():
            raise FileNotFoundError(src_file)
        shutil.copy2(src_file, proto_out_dir / file_name)

    artifacts = CompileArtifacts()
    artifacts.schema_version = get_schema_version()
    io_mapping = artifacts.io_mapping
    online_placements = (
        _online_placements_by_node(coreplacements) if online_graph is not None else {}
    )

    for thread_id, global_start in global_starts.items():
        thread_mapping = io_mapping.threads.add()
        thread_mapping.thread_id = thread_id
        thread_mapping.root_core_offset.xy = global_start.z
        thread_mapping.root_core_offset.x = global_start.x
        thread_mapping.root_core_offset.y = global_start.y

        if online_graph is not None:
            _export_online_thread_io_mapping(
                thread_mapping,
                online_graph,
                thread_id,
                online_placements,
            )
            _export_online_thread_runtime(thread_mapping, timesteps)
        else:
            thread_output_groups = [
                out_grp for out_grp in output_groups if out_grp.thread_id == thread_id
            ]
            _export_offline_thread_runtime(
                thread_mapping, thread_output_groups, groups, timesteps
            )
            _export_offline_thread_io_mapping(
                thread_mapping,
                thread_id,
                groups,
                input_groups,
                thread_output_groups,
            )

        for core_placement in coreplacements:
            node_names, placement_thread_id, tick_params = _core_tick_export_parts(
                core_placement
            )
            if not node_names:
                continue
            if placement_thread_id != thread_id:
                continue
            core_tick = thread_mapping.core_ticks.add()
            core_offset, _ = find_coordxy_shortest_path(core_placement.coord)
            core_tick.core_offset.xy = core_offset.z
            core_tick.core_offset.x = core_offset.x
            core_tick.core_offset.y = core_offset.y
            core_tick.nodes.extend(node_names)
            _set_tick_params(core_tick.tick, tick_params)

    config_words: list[int] = []
    for frame_array in iter_frame_arrays_core_major(frame_records):
        config_words.extend(export_framearray_to_int32(frame_array, word_order))
    artifacts.config_frames.words.extend(config_words)

    artifacts.config_frames.word_order = (
        ConfigFrames.HIGH_FIRST
        if word_order == "high_first"
        else ConfigFrames.LOW_FIRST
    )

    pb_path.write_bytes(artifacts.SerializeToString())
    if debug:
        pb_text_path.write_text(
            MessageToJson(artifacts, always_print_fields_with_no_presence=True)
        )

    return pb_path
