from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from paicorelib import CoordZXYOffset, DataSign, DataWidth, find_coordxy_shortest_path

from paibox.paiir.ir.signal_domain import SignalDomain

from ..coreplacement import CorePlacement, Frontend_Core_Config
from ..op_node import Neuron, RemapElem
from ..routing import InputGroup, OutputGroup, RemapGroup, RoutingGroup, SourceElem
from .utils import (
    FrameRecords,
    WordOrder,
    export_framearray_to_int32,
    iter_frame_arrays_core_major,
)

TickTriple = tuple[int, int, int]
DataFormat = tuple[DataSign, DataWidth]

DATA_TYPE_NOT_SET = 0
DATA_TYPE_UINT1 = 1
DATA_TYPE_INT1 = 2
DATA_TYPE_UINT2 = 3
DATA_TYPE_INT2 = 4
DATA_TYPE_UINT4 = 5
DATA_TYPE_INT4 = 6
DATA_TYPE_UINT8 = 7
DATA_TYPE_INT8 = 8

OUTPUT_KIND_DATA = 0
OUTPUT_KIND_VOLTAGE = 1

DECODE_MODE_STREAM = 0
DECODE_MODE_STEP = 1

WORD_ORDER_HIGH_FIRST = 0
WORD_ORDER_LOW_FIRST = 1

SCHEMA_VERSION = 1


@dataclass
class CoreOffsetData:
    xy: int = 0
    x: int = 0
    y: int = 0


@dataclass
class CopyCountData:
    xy: int = 0
    x: int = 0
    y: int = 0


@dataclass
class TickParamsData:
    tick_start: int = 0
    tick_duration: int = 0
    tick_initial: int = 0


@dataclass
class ShapeData:
    size: list[int] = field(default_factory=list)


@dataclass
class InputEntryData:
    elem_idx: int = 0
    core_offset: CoreOffsetData = field(default_factory=CoreOffsetData)
    copy_count: CopyCountData = field(default_factory=CopyCountData)
    tick_relative: int = 0
    addr_axon: int = 0
    target_lcn: int = 0
    copy_id: int = 0
    dtype: int = DATA_TYPE_NOT_SET


@dataclass
class OutputEntryData:
    elem_idx: int = 0
    copy_id: int = 0
    axon_bit_idx: int = 0
    dtype: int = DATA_TYPE_NOT_SET


@dataclass
class InputTensorMappingData:
    name: str
    shape: ShapeData = field(default_factory=ShapeData)
    bit_width: int = 0
    tick: TickParamsData = field(default_factory=TickParamsData)
    entries: list[InputEntryData] = field(default_factory=list)


@dataclass
class OutputTensorMappingData:
    name: str
    shape: ShapeData = field(default_factory=ShapeData)
    kind: int | None = None
    bit_width: int = 0
    tick: TickParamsData = field(default_factory=TickParamsData)
    entries: list[OutputEntryData] = field(default_factory=list)


@dataclass
class InputTensorMappingsData:
    items: list[InputTensorMappingData] = field(default_factory=list)


@dataclass
class OutputTensorMappingsData:
    target_lcn: int | None = None
    items: list[OutputTensorMappingData] = field(default_factory=list)


@dataclass
class RuntimeParamsData:
    timesteps: int = 0
    tick_depth: int = 0
    sync_steps: int = 0
    decode_mode: int = DECODE_MODE_STREAM


@dataclass
class CoreTickData:
    core_offset: CoreOffsetData = field(default_factory=CoreOffsetData)
    tick: TickParamsData = field(default_factory=TickParamsData)
    nodes: list[str] = field(default_factory=list)


@dataclass
class ThreadIOMappingData:
    thread_id: int = 0
    root_core_offset: CoreOffsetData = field(default_factory=CoreOffsetData)
    runtime: RuntimeParamsData = field(default_factory=RuntimeParamsData)
    input_mappings: InputTensorMappingsData = field(
        default_factory=InputTensorMappingsData
    )
    output_mappings: OutputTensorMappingsData = field(
        default_factory=OutputTensorMappingsData
    )
    core_ticks: list[CoreTickData] = field(default_factory=list)


@dataclass
class IOMappingData:
    threads: list[ThreadIOMappingData] = field(default_factory=list)


@dataclass
class ConfigFramesData:
    words: list[int] = field(default_factory=list)
    word_order: int = WORD_ORDER_HIGH_FIRST


@dataclass
class CompileArtifactsData:
    schema_version: int = 0
    io_mapping: IOMappingData = field(default_factory=IOMappingData)
    config_frames: ConfigFramesData = field(default_factory=ConfigFramesData)


_DATA_TYPE_BY_FORMAT: Mapping[DataFormat, int] = {
    (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT): DATA_TYPE_UINT1,
    (DataSign.SIGNED, DataWidth.WIDTH_1BIT): DATA_TYPE_INT1,
    (DataSign.UNSIGNED, DataWidth.WIDTH_2BIT): DATA_TYPE_UINT2,
    (DataSign.SIGNED, DataWidth.WIDTH_2BIT): DATA_TYPE_INT2,
    (DataSign.UNSIGNED, DataWidth.WIDTH_4BIT): DATA_TYPE_UINT4,
    (DataSign.SIGNED, DataWidth.WIDTH_4BIT): DATA_TYPE_INT4,
    (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT): DATA_TYPE_UINT8,
    (DataSign.SIGNED, DataWidth.WIDTH_8BIT): DATA_TYPE_INT8,
}


def _bit_width_from_data_width(width: DataWidth) -> int:
    return int(1 << width)


def _dtype_from_format(fmt: DataFormat, context: str) -> int:
    dtype = _DATA_TYPE_BY_FORMAT.get(fmt)
    if dtype is None:
        sign, width = fmt
        raise ValueError(
            f"{context} has unsupported data format "
            f"{sign.name}/{width.name}; expected signed/unsigned 1/2/4/8-bit DATA."
        )
    return dtype


def _set_entry_dtype(
    entry: InputEntryData | OutputEntryData,
    fmt: DataFormat,
    bit_width: int,
    context: str,
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
    mapping: InputTensorMappingData | OutputTensorMappingData,
    mapping_kind: str,
    mapping_name: str,
    bit_width: int,
) -> None:
    """Set tensor-level bit width and reject mixed-width entries."""
    if mapping.bit_width:
        if mapping.bit_width != bit_width:
            raise ValueError(
                f"{mapping_kind} mapping '{mapping_name}' contains mixed bit widths: "
                f"{mapping.bit_width} and {bit_width}."
            )
    else:
        mapping.bit_width = bit_width


def _set_thread_output_lcn(
    output_mappings: OutputTensorMappingsData, output_group: OutputGroup, thread_id: int
) -> None:
    """Set thread-level output LCN and reject conflicting output groups."""
    target_lcn = output_group.lcn.value
    if output_mappings.target_lcn is not None:
        if output_mappings.target_lcn != target_lcn:
            raise ValueError(
                f"thread {thread_id} contains output groups with mixed target_lcn: "
                f"{output_mappings.target_lcn} and {target_lcn}."
            )
    else:
        output_mappings.target_lcn = target_lcn


def _output_kind_from_elem(elem: SourceElem) -> int:
    domain = elem.target.raw_node.signal_semantics.output_domain
    if domain is SignalDomain.VALUE:
        kind = OUTPUT_KIND_DATA
    else:
        kind = OUTPUT_KIND_VOLTAGE

    if kind == OUTPUT_KIND_DATA:
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


def _output_kind_name(kind: int) -> str:
    return "DATA" if kind == OUTPUT_KIND_DATA else "VOLTAGE"


def _set_output_mapping_kind(
    output_mapping: OutputTensorMappingData, output_name: str, elem: SourceElem
) -> int:
    kind = _output_kind_from_elem(elem)
    if output_mapping.kind is not None:
        if output_mapping.kind != kind:
            raise ValueError(
                f"output mapping '{output_name}' contains mixed output kinds: "
                f"{_output_kind_name(output_mapping.kind)} and "
                f"{_output_kind_name(kind)}."
            )
    else:
        output_mapping.kind = kind

    return kind


def _tick_tuple_from_core_conf(core_conf: Frontend_Core_Config) -> TickTriple:
    return core_conf.tick_start, core_conf.tick_duration, core_conf.tick_initial


def _producer_core_conf_from_source_elem(
    elem: SourceElem, groups: Sequence[RoutingGroup | RemapGroup]
) -> Frontend_Core_Config:
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


def _set_tick_params(tick_params: TickParamsData, tick: TickTriple) -> None:
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


def _thread_decode_mode(output_groups: Sequence[OutputGroup], timesteps: int) -> int:
    """Derive STREAM/STEP from the final output LCN timestep capacity."""
    if not output_groups:
        return DECODE_MODE_STREAM
    max_stream_timesteps = min(
        1 << (8 - out_grp.lcn.value) for out_grp in output_groups
    )
    return DECODE_MODE_STREAM if max_stream_timesteps >= timesteps else DECODE_MODE_STEP


def build_compile_artifacts(
    word_order: WordOrder,
    timesteps: int,
    groups: Sequence[RoutingGroup | RemapGroup],
    input_groups: Sequence[InputGroup],
    output_groups: Sequence[OutputGroup],
    coreplacements: Sequence[CorePlacement],
    global_starts: Mapping[int, CoordZXYOffset],
    frame_records: FrameRecords,
) -> CompileArtifactsData:
    """Build backendv2 compile metadata shared by protobuf and FlatBuffers."""
    artifacts = CompileArtifactsData(schema_version=SCHEMA_VERSION)
    io_mapping = artifacts.io_mapping

    for thread_id, global_start in global_starts.items():
        thread_mapping = ThreadIOMappingData(
            thread_id=thread_id,
            root_core_offset=CoreOffsetData(
                xy=global_start.z, x=global_start.x, y=global_start.y
            ),
        )
        io_mapping.threads.append(thread_mapping)

        thread_output_groups = [
            out_grp for out_grp in output_groups if out_grp.thread_id == thread_id
        ]
        tick_depth = _thread_tick_depth(thread_output_groups, groups)
        if tick_depth is not None:
            thread_mapping.runtime.timesteps = timesteps
            thread_mapping.runtime.tick_depth = tick_depth
            thread_mapping.runtime.sync_steps = tick_depth + timesteps - 1
            thread_mapping.runtime.decode_mode = _thread_decode_mode(
                thread_output_groups, timesteps
            )

        input_mappings_by_name: dict[str, InputTensorMappingData] = {}
        input_ticks_by_name: dict[str, set[TickTriple]] = {}

        for in_grp in input_groups:
            if in_grp.thread_id != thread_id:
                continue
            for elem, dest in in_grp.dest_infos.items():
                input_name = elem.target.raw_node.name
                if input_name not in input_mappings_by_name:
                    input_mapping = InputTensorMappingData(name=input_name)
                    input_mapping.shape.size.extend(list(elem.target.shape))
                    thread_mapping.input_mappings.items.append(input_mapping)
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
                input_entry = InputEntryData(
                    elem_idx=elem.index.idx, copy_id=elem.index.copy_id
                )
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
                input_mapping.entries.append(input_entry)
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
                _require_single_tick(
                    input_ticks_by_name[input_name], "input", input_name
                ),
            )

        output_mappings_by_name: dict[str, OutputTensorMappingData] = {}
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
                    output_mapping = OutputTensorMappingData(name=output_name)
                    output_mapping.shape.size.extend(list(elem.target.shape))
                    thread_mapping.output_mappings.items.append(output_mapping)
                    output_mappings_by_name[output_name] = output_mapping
                    output_ticks_by_name[output_name] = set()

                output_mapping = output_mappings_by_name[output_name]
                output_ticks_by_name[output_name].add(
                    _tick_tuple_from_source_elem(elem, groups)
                )
                _set_mapping_bit_width(
                    output_mapping, "output", output_name, elem.output_bit_num
                )
                output_entry = OutputEntryData(
                    elem_idx=elem.index.idx,
                    copy_id=elem.index.copy_id,
                    axon_bit_idx=axon_bit_idx,
                )
                output_kind = _set_output_mapping_kind(
                    output_mapping, output_name, elem
                )
                if output_kind == OUTPUT_KIND_DATA:
                    _set_entry_dtype(
                        output_entry,
                        _output_format_from_source_elem(elem, groups),
                        elem.output_bit_num,
                        f"output entry {output_name}[{elem.index.idx}]",
                    )
                output_mapping.entries.append(output_entry)

        for output_name, output_mapping in output_mappings_by_name.items():
            _set_tick_params(
                output_mapping.tick,
                _require_single_tick(
                    output_ticks_by_name[output_name], "output", output_name
                ),
            )

        for core_placement in coreplacements:
            if not core_placement.neus:
                continue
            if core_placement.default_core_config.thread_number != thread_id:
                continue
            core_offset, _ = find_coordxy_shortest_path(core_placement.coord)
            core_tick = CoreTickData(
                core_offset=CoreOffsetData(
                    xy=core_offset.z, x=core_offset.x, y=core_offset.y
                )
            )
            thread_mapping.core_ticks.append(core_tick)
            nodes = {
                raw_neu.target.raw_node.name
                for neu_placement in core_placement.neus
                for raw_neu in neu_placement.raw_neus
            }
            core_tick.nodes.extend(sorted(nodes))
            _set_tick_params(
                core_tick.tick,
                _tick_tuple_from_core_conf(core_placement.frontend_core_config),
            )

    config_words: list[int] = []
    for frame_array in iter_frame_arrays_core_major(frame_records):
        config_words.extend(export_framearray_to_int32(frame_array, word_order))
    artifacts.config_frames.words.extend(config_words)

    artifacts.config_frames.word_order = (
        WORD_ORDER_HIGH_FIRST if word_order == "high_first" else WORD_ORDER_LOW_FIRST
    )

    return artifacts
