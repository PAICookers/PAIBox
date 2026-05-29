import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from google.protobuf.json_format import MessageToJson
from paicorelib import CoordZXYOffset, DataSign, DataWidth, find_coordxy_shortest_path

from paibox.paiir.ir.signal_domain import SignalDomain

from ..coreplacement import CorePlacement, Frontend_Core_Config
from ..op_node import Neuron, RemapElem
from ..proto import PROTO_SCHEMA_VERSION
from ..proto.compile_artifacts_pb2 import (
    CompileArtifacts,
    ConfigFrames,
    DataType,
    InputTensorMapping,
    OutputTensorMapping,
    TickParams,
)
from ..routing import InputGroup, OutputGroup, RemapGroup, RoutingGroup, SourceElem
from .utils import (
    FrameRecords,
    TargetPlatform,
    WordOrder,
    export_framearray_to_int32,
    iter_frame_arrays_core_major,
    resolve_platform_exports,
)

TickTriple = tuple[int, int, int]
DataFormat = tuple[DataSign, DataWidth]

_DATA_TYPE_BY_FORMAT: Mapping[DataFormat, int] = {
    (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT): DataType.UINT1,
    (DataSign.SIGNED, DataWidth.WIDTH_1BIT): DataType.INT1,
    (DataSign.UNSIGNED, DataWidth.WIDTH_2BIT): DataType.UINT2,
    (DataSign.SIGNED, DataWidth.WIDTH_2BIT): DataType.INT2,
    (DataSign.UNSIGNED, DataWidth.WIDTH_4BIT): DataType.UINT4,
    (DataSign.SIGNED, DataWidth.WIDTH_4BIT): DataType.INT4,
    (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT): DataType.UINT8,
    (DataSign.SIGNED, DataWidth.WIDTH_8BIT): DataType.INT8,
}


def _bit_width_from_data_width(width: DataWidth) -> int:
    return int(2**width)


def _dtype_from_format(fmt: DataFormat, context: str) -> int:
    dtype = _DATA_TYPE_BY_FORMAT.get(fmt)
    if dtype is None:
        sign, width = fmt
        raise ValueError(
            f"{context} has unsupported data format "
            f"{sign.name}/{width.name}; expected signed/unsigned 1/2/4/8-bit DATA."
        )
    return dtype


def _set_entry_dtype(entry, fmt: DataFormat, bit_width: int, context: str) -> None:
    expected_bit_width = _bit_width_from_data_width(fmt[1])
    if bit_width != expected_bit_width:
        sign, width = fmt
        raise ValueError(
            f"{context} bit width mismatch: entry has {bit_width}, "
            f"format {sign.name}/{width.name} implies {expected_bit_width}."
        )
    entry.dtype = _dtype_from_format(fmt, context)


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


def export_compile_artifacts(
    output_path: str | Path,
    target_platform: TargetPlatform,
    word_order: WordOrder,
    export_python: bool,
    debug: bool,
    groups: Sequence[RoutingGroup | RemapGroup],
    input_groups: Sequence[InputGroup],
    output_groups: Sequence[OutputGroup],
    coreplacements: Sequence[CorePlacement],
    global_starts: Mapping[int, CoordZXYOffset],
    frame_records: FrameRecords,
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
    artifacts.schema_version = PROTO_SCHEMA_VERSION
    io_mapping = artifacts.io_mapping

    for thread_id, global_start in global_starts.items():
        thread_mapping = io_mapping.threads.add()
        thread_mapping.thread_id = thread_id
        thread_mapping.root_core_offset.xy = global_start.z
        thread_mapping.root_core_offset.x = global_start.x
        thread_mapping.root_core_offset.y = global_start.y

        input_mappings_by_name: dict[str, InputTensorMapping] = {}
        input_ticks_by_name: dict[str, set[TickTriple]] = {}

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
                input_entry = input_mapping.entries.add()
                input_entry.elem_idx = elem.index.idx
                input_entry.copy_id = elem.index.copy_id
                input_entry.bit_width = elem.output_bit_num
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
                _require_single_tick(
                    input_ticks_by_name[input_name], "input", input_name
                ),
            )

        output_mappings_by_name: dict[str, OutputTensorMapping] = {}
        output_ticks_by_name: dict[str, set[TickTriple]] = {}
        for out_grp in output_groups:
            if out_grp.thread_id != thread_id:
                continue
            thread_mapping.output_mappings.target_lcn = out_grp.lcn
            for axon_bit_idx, elem in sorted(
                out_grp.axon_bit_allocator.axon_infos, key=lambda item: item[0]
            ):
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
                output_entry = output_mapping.entries.add()
                output_entry.elem_idx = elem.index.idx
                output_entry.copy_id = elem.index.copy_id
                output_entry.bit_width = elem.output_bit_num
                output_entry.axon_bit_idx = axon_bit_idx
                output_kind = _set_output_mapping_kind(
                    output_mapping, output_name, elem
                )
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

        for core_placement in coreplacements:
            if not core_placement.neus:
                continue
            if core_placement.default_core_config.thread_number != thread_id:
                continue
            core_tick = thread_mapping.core_ticks.add()
            core_offset, _ = find_coordxy_shortest_path(core_placement.coord)
            core_tick.core_offset.xy = core_offset.z
            core_tick.core_offset.x = core_offset.x
            core_tick.core_offset.y = core_offset.y
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
