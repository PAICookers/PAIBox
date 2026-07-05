import shutil
from pathlib import Path

from google.protobuf.json_format import MessageToJson

from ..generated.proto import compile_artifacts_pb2 as pb
from .compile_artifacts import (
    CompileArtifactsData,
    ConfigFramesData,
    CopyCountData,
    CoreOffsetData,
    CoreTickData,
    InputEntryData,
    InputTensorMappingData,
    InputTensorMappingsData,
    IOMappingData,
    OutputEntryData,
    OutputTensorMappingData,
    OutputTensorMappingsData,
    RuntimeParamsData,
    ShapeData,
    ThreadIOMappingData,
    TickParamsData,
)


def _copy_core_offset(src: CoreOffsetData, dst: pb.CoreOffset) -> None:
    dst.xy = src.xy
    dst.x = src.x
    dst.y = src.y


def _copy_copy_count(src: CopyCountData, dst: pb.CopyCount) -> None:
    dst.xy = src.xy
    dst.x = src.x
    dst.y = src.y


def _copy_tick_params(src: TickParamsData, dst: pb.TickParams) -> None:
    dst.tick_start = src.tick_start
    dst.tick_duration = src.tick_duration
    dst.tick_initial = src.tick_initial


def _copy_shape(src: ShapeData, dst: pb.Shape) -> None:
    dst.size.extend(src.size)


def _copy_input_entry(src: InputEntryData, dst: pb.InputEntry) -> None:
    dst.elem_idx = src.elem_idx
    _copy_core_offset(src.core_offset, dst.core_offset)
    _copy_copy_count(src.copy_count, dst.copy_count)
    dst.tick_relative = src.tick_relative
    dst.addr_axon = src.addr_axon
    dst.target_lcn = src.target_lcn
    dst.copy_id = src.copy_id
    if src.dtype:
        dst.dtype = src.dtype


def _copy_output_entry(src: OutputEntryData, dst: pb.OutputEntry) -> None:
    dst.elem_idx = src.elem_idx
    dst.copy_id = src.copy_id
    dst.axon_bit_idx = src.axon_bit_idx
    if src.dtype:
        dst.dtype = src.dtype


def _copy_input_mapping(
    src: InputTensorMappingData, dst: pb.InputTensorMapping
) -> None:
    dst.name = src.name
    _copy_shape(src.shape, dst.shape)
    dst.bit_width = src.bit_width
    _copy_tick_params(src.tick, dst.tick)
    for entry in src.entries:
        _copy_input_entry(entry, dst.entries.add())


def _copy_output_mapping(
    src: OutputTensorMappingData, dst: pb.OutputTensorMapping
) -> None:
    dst.name = src.name
    _copy_shape(src.shape, dst.shape)
    if src.kind is not None:
        dst.kind = src.kind
    dst.bit_width = src.bit_width
    _copy_tick_params(src.tick, dst.tick)
    for entry in src.entries:
        _copy_output_entry(entry, dst.entries.add())


def _copy_input_mappings(
    src: InputTensorMappingsData, dst: pb.InputTensorMappings
) -> None:
    for item in src.items:
        _copy_input_mapping(item, dst.items.add())


def _copy_output_mappings(
    src: OutputTensorMappingsData, dst: pb.OutputTensorMappings
) -> None:
    if src.target_lcn is not None:
        dst.target_lcn = src.target_lcn
    for item in src.items:
        _copy_output_mapping(item, dst.items.add())


def _copy_runtime_params(src: RuntimeParamsData, dst: pb.RuntimeParams) -> None:
    dst.timesteps = src.timesteps
    dst.tick_depth = src.tick_depth
    dst.sync_steps = src.sync_steps
    dst.decode_mode = src.decode_mode


def _copy_core_tick(src: CoreTickData, dst: pb.CoreTick) -> None:
    _copy_core_offset(src.core_offset, dst.core_offset)
    _copy_tick_params(src.tick, dst.tick)
    dst.nodes.extend(src.nodes)


def _copy_thread_mapping(
    src: ThreadIOMappingData, dst: pb.ThreadIOMapping
) -> None:
    dst.thread_id = src.thread_id
    _copy_core_offset(src.root_core_offset, dst.root_core_offset)
    _copy_runtime_params(src.runtime, dst.runtime)
    _copy_input_mappings(src.input_mappings, dst.input_mappings)
    _copy_output_mappings(src.output_mappings, dst.output_mappings)
    for item in src.core_ticks:
        _copy_core_tick(item, dst.core_ticks.add())


def _copy_io_mapping(src: IOMappingData, dst: pb.IOMapping) -> None:
    for thread in src.threads:
        _copy_thread_mapping(thread, dst.threads.add())


def _copy_config_frames(src: ConfigFramesData, dst: pb.ConfigFrames) -> None:
    dst.words.extend(src.words)
    dst.word_order = src.word_order


def compile_artifacts_to_proto(data: CompileArtifactsData) -> pb.CompileArtifacts:
    artifacts = pb.CompileArtifacts()
    artifacts.schema_version = data.schema_version
    _copy_io_mapping(data.io_mapping, artifacts.io_mapping)
    _copy_config_frames(data.config_frames, artifacts.config_frames)
    return artifacts


def export_compile_artifacts(
    output_path: str | Path,
    export_python: bool,
    debug: bool,
    artifacts: CompileArtifactsData,
) -> Path:
    """Export protobuf artifacts for config frames and I/O mappings."""
    backendv2_dir = Path(__file__).parent.parent
    schema_dir = backendv2_dir / "schemas"
    generated_dir = backendv2_dir / "generated" / "proto"
    proto_out_dir = Path(output_path) / "proto"
    proto_out_dir.mkdir(parents=True, exist_ok=True)

    pb_path = proto_out_dir / "config.pb"
    pb_text_path = proto_out_dir / "config.json"

    shutil.copy2(schema_dir / "compile_artifacts.proto", proto_out_dir)
    if export_python:
        for file_name in ("compile_artifacts_pb2.py", "compile_artifacts_pb2.pyi"):
            shutil.copy2(generated_dir / file_name, proto_out_dir / file_name)

    proto_artifacts = compile_artifacts_to_proto(artifacts)
    pb_path.write_bytes(proto_artifacts.SerializeToString())
    if debug:
        pb_text_path.write_text(
            MessageToJson(proto_artifacts, always_print_fields_with_no_presence=True)
        )

    return pb_path
