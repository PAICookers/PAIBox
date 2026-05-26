import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from google.protobuf.json_format import MessageToJson
from paicorelib import CoordZXYOffset

from paibox.paiir.ir.signal_domain import SignalDomain

from ..coreplacement import CorePlacement
from ..proto import PROTO_SCHEMA_VERSION
from ..proto.compile_artifacts_pb2 import (
    CompileArtifacts,
    ConfigFrames,
    InputTensorMapping,
    OutputEntry,
    OutputTensorMapping,
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


def _set_output_entry_kind(output_entry: OutputEntry, elem: SourceElem) -> None:
    domain = elem.target.raw_node.signal_semantics.output_domain
    if domain is SignalDomain.VALUE:
        kind = OutputEntry.DATA
    else:
        kind = OutputEntry.VOLTAGE

    if kind == OutputEntry.DATA:
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

    output_entry.kind = kind


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

                input_mapping = input_mappings_by_name[input_name]
                input_entry = input_mapping.entries.add()
                input_entry.elem_idx = elem.index.idx
                input_entry.copy_id = elem.index.copy_id
                input_entry.bit_width = elem.output_bit_num
                input_entry.tick_relative = dest.tick_relative
                input_entry.addr_axon = dest.addr_axon
                input_entry.core_offset.xy = dest.addr_core_xy
                input_entry.core_offset.x = dest.addr_core_x
                input_entry.core_offset.y = dest.addr_core_y
                input_entry.copy_count.xy = dest.addr_copy_xy
                input_entry.copy_count.x = dest.addr_copy_x
                input_entry.copy_count.y = dest.addr_copy_y
                input_entry.target_lcn = in_grp.dest_lcn[elem]

        output_mappings_by_name: dict[str, OutputTensorMapping] = {}
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

                output_mapping = output_mappings_by_name[output_name]
                output_entry = output_mapping.entries.add()
                output_entry.elem_idx = elem.index.idx
                output_entry.copy_id = elem.index.copy_id
                output_entry.bit_width = elem.output_bit_num
                output_entry.axon_bit_idx = axon_bit_idx
                _set_output_entry_kind(output_entry, elem)

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
