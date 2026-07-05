import shutil
from pathlib import Path

import flatbuffers

from ..generated.fbs import CompileArtifacts as FbsCompileArtifacts
from ..generated.fbs import ConfigFrames as FbsConfigFrames
from ..generated.fbs import CopyCount as FbsCopyCount
from ..generated.fbs import CoreOffset as FbsCoreOffset
from ..generated.fbs import CoreTick as FbsCoreTick
from ..generated.fbs import InputEntry as FbsInputEntry
from ..generated.fbs import InputTensorMapping as FbsInputTensorMapping
from ..generated.fbs import InputTensorMappings as FbsInputTensorMappings
from ..generated.fbs import IOMapping as FbsIOMapping
from ..generated.fbs import OutputEntry as FbsOutputEntry
from ..generated.fbs import OutputTensorMapping as FbsOutputTensorMapping
from ..generated.fbs import OutputTensorMappings as FbsOutputTensorMappings
from ..generated.fbs import RuntimeParams as FbsRuntimeParams
from ..generated.fbs import Shape as FbsShape
from ..generated.fbs import ThreadIOMapping as FbsThreadIOMapping
from ..generated.fbs import TickParams as FbsTickParams
from .compile_artifacts import OUTPUT_KIND_DATA, CompileArtifactsData

COMPILE_ARTIFACTS_FILE = "compile_artifacts.bin"
COMPILE_ARTIFACTS_SCHEMA_FILE = "compile_artifacts.fbs"
COMPILE_ARTIFACTS_FILE_IDENTIFIER = b"PBCA"


def _int32_vector(builder: flatbuffers.Builder, values) -> int:
    data = list(values)
    builder.StartVector(4, len(data), 4)
    for value in reversed(data):
        builder.PrependInt32(int(value))
    return builder.EndVector()


def _uint32_vector(builder: flatbuffers.Builder, values) -> int:
    data = list(values)
    builder.StartVector(4, len(data), 4)
    for value in reversed(data):
        builder.PrependUint32(int(value))
    return builder.EndVector()


def _table_vector(builder: flatbuffers.Builder, offsets: list[int]) -> int:
    builder.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def _string_vector(builder: flatbuffers.Builder, values) -> int:
    offsets = [builder.CreateString(str(value)) for value in values]
    return _table_vector(builder, offsets)


def _core_offset(builder: flatbuffers.Builder, value) -> int:
    FbsCoreOffset.Start(builder)
    FbsCoreOffset.AddXy(builder, int(value.xy))
    FbsCoreOffset.AddX(builder, int(value.x))
    FbsCoreOffset.AddY(builder, int(value.y))
    return FbsCoreOffset.End(builder)


def _copy_count(builder: flatbuffers.Builder, value) -> int:
    FbsCopyCount.Start(builder)
    FbsCopyCount.AddXy(builder, int(value.xy))
    FbsCopyCount.AddX(builder, int(value.x))
    FbsCopyCount.AddY(builder, int(value.y))
    return FbsCopyCount.End(builder)


def _tick_params(builder: flatbuffers.Builder, value) -> int:
    FbsTickParams.Start(builder)
    FbsTickParams.AddTickStart(builder, int(value.tick_start))
    FbsTickParams.AddTickDuration(builder, int(value.tick_duration))
    FbsTickParams.AddTickInitial(builder, int(value.tick_initial))
    return FbsTickParams.End(builder)


def _shape(builder: flatbuffers.Builder, value) -> int:
    size = _int32_vector(builder, value.size)
    FbsShape.Start(builder)
    FbsShape.AddSize(builder, size)
    return FbsShape.End(builder)


def _input_entry(builder: flatbuffers.Builder, value) -> int:
    core_offset = _core_offset(builder, value.core_offset)
    copy_count = _copy_count(builder, value.copy_count)
    FbsInputEntry.Start(builder)
    FbsInputEntry.AddElemIdx(builder, int(value.elem_idx))
    FbsInputEntry.AddCoreOffset(builder, core_offset)
    FbsInputEntry.AddCopyCount(builder, copy_count)
    FbsInputEntry.AddTickRelative(builder, int(value.tick_relative))
    FbsInputEntry.AddAddrAxon(builder, int(value.addr_axon))
    FbsInputEntry.AddTargetLcn(builder, int(value.target_lcn))
    FbsInputEntry.AddCopyId(builder, int(value.copy_id))
    FbsInputEntry.AddDtype(builder, int(value.dtype))
    return FbsInputEntry.End(builder)


def _output_entry(builder: flatbuffers.Builder, value) -> int:
    FbsOutputEntry.Start(builder)
    FbsOutputEntry.AddElemIdx(builder, int(value.elem_idx))
    FbsOutputEntry.AddCopyId(builder, int(value.copy_id))
    FbsOutputEntry.AddAxonBitIdx(builder, int(value.axon_bit_idx))
    FbsOutputEntry.AddDtype(builder, int(value.dtype))
    return FbsOutputEntry.End(builder)


def _input_mapping(builder: flatbuffers.Builder, value) -> int:
    name = builder.CreateString(value.name)
    shape = _shape(builder, value.shape)
    tick = _tick_params(builder, value.tick)
    entries = _table_vector(
        builder, [_input_entry(builder, item) for item in value.entries]
    )
    FbsInputTensorMapping.Start(builder)
    FbsInputTensorMapping.AddName(builder, name)
    FbsInputTensorMapping.AddShape(builder, shape)
    FbsInputTensorMapping.AddBitWidth(builder, int(value.bit_width))
    FbsInputTensorMapping.AddTick(builder, tick)
    FbsInputTensorMapping.AddEntries(builder, entries)
    return FbsInputTensorMapping.End(builder)


def _output_mapping(builder: flatbuffers.Builder, value) -> int:
    name = builder.CreateString(value.name)
    shape = _shape(builder, value.shape)
    tick = _tick_params(builder, value.tick)
    entries = _table_vector(
        builder, [_output_entry(builder, item) for item in value.entries]
    )
    FbsOutputTensorMapping.Start(builder)
    FbsOutputTensorMapping.AddName(builder, name)
    FbsOutputTensorMapping.AddShape(builder, shape)
    kind = OUTPUT_KIND_DATA if value.kind is None else value.kind
    FbsOutputTensorMapping.AddKind(builder, int(kind))
    FbsOutputTensorMapping.AddBitWidth(builder, int(value.bit_width))
    FbsOutputTensorMapping.AddTick(builder, tick)
    FbsOutputTensorMapping.AddEntries(builder, entries)
    return FbsOutputTensorMapping.End(builder)


def _input_mappings(builder: flatbuffers.Builder, value) -> int:
    items = _table_vector(
        builder, [_input_mapping(builder, item) for item in value.items]
    )
    FbsInputTensorMappings.Start(builder)
    FbsInputTensorMappings.AddItems(builder, items)
    return FbsInputTensorMappings.End(builder)


def _output_mappings(builder: flatbuffers.Builder, value) -> int:
    items = _table_vector(
        builder, [_output_mapping(builder, item) for item in value.items]
    )
    FbsOutputTensorMappings.Start(builder)
    target_lcn = 0 if value.target_lcn is None else value.target_lcn
    FbsOutputTensorMappings.AddTargetLcn(builder, int(target_lcn))
    FbsOutputTensorMappings.AddItems(builder, items)
    return FbsOutputTensorMappings.End(builder)


def _runtime_params(builder: flatbuffers.Builder, value) -> int:
    FbsRuntimeParams.Start(builder)
    FbsRuntimeParams.AddTimesteps(builder, int(value.timesteps))
    FbsRuntimeParams.AddTickDepth(builder, int(value.tick_depth))
    FbsRuntimeParams.AddSyncSteps(builder, int(value.sync_steps))
    FbsRuntimeParams.AddDecodeMode(builder, int(value.decode_mode))
    return FbsRuntimeParams.End(builder)


def _core_tick(builder: flatbuffers.Builder, value) -> int:
    core_offset = _core_offset(builder, value.core_offset)
    tick = _tick_params(builder, value.tick)
    nodes = _string_vector(builder, value.nodes)
    FbsCoreTick.Start(builder)
    FbsCoreTick.AddCoreOffset(builder, core_offset)
    FbsCoreTick.AddTick(builder, tick)
    FbsCoreTick.AddNodes(builder, nodes)
    return FbsCoreTick.End(builder)


def _thread_mapping(builder: flatbuffers.Builder, value) -> int:
    root_core_offset = _core_offset(builder, value.root_core_offset)
    runtime = _runtime_params(builder, value.runtime)
    input_mappings = _input_mappings(builder, value.input_mappings)
    output_mappings = _output_mappings(builder, value.output_mappings)
    core_ticks = _table_vector(
        builder, [_core_tick(builder, item) for item in value.core_ticks]
    )
    FbsThreadIOMapping.Start(builder)
    FbsThreadIOMapping.AddThreadId(builder, int(value.thread_id))
    FbsThreadIOMapping.AddRootCoreOffset(builder, root_core_offset)
    FbsThreadIOMapping.AddRuntime(builder, runtime)
    FbsThreadIOMapping.AddInputMappings(builder, input_mappings)
    FbsThreadIOMapping.AddOutputMappings(builder, output_mappings)
    FbsThreadIOMapping.AddCoreTicks(builder, core_ticks)
    return FbsThreadIOMapping.End(builder)


def _io_mapping(builder: flatbuffers.Builder, value) -> int:
    threads = _table_vector(
        builder, [_thread_mapping(builder, item) for item in value.threads]
    )
    FbsIOMapping.Start(builder)
    FbsIOMapping.AddThreads(builder, threads)
    return FbsIOMapping.End(builder)


def _config_frames(builder: flatbuffers.Builder, value) -> int:
    words = _uint32_vector(builder, value.words)
    FbsConfigFrames.Start(builder)
    FbsConfigFrames.AddWords(builder, words)
    FbsConfigFrames.AddWordOrder(builder, int(value.word_order))
    return FbsConfigFrames.End(builder)


def compile_artifacts_to_flatbuffer(artifacts: CompileArtifactsData) -> bytes:
    builder = flatbuffers.Builder(1024)
    io_mapping = _io_mapping(builder, artifacts.io_mapping)
    config_frames = _config_frames(builder, artifacts.config_frames)
    FbsCompileArtifacts.Start(builder)
    FbsCompileArtifacts.AddSchemaVersion(builder, int(artifacts.schema_version))
    FbsCompileArtifacts.AddIoMapping(builder, io_mapping)
    FbsCompileArtifacts.AddConfigFrames(builder, config_frames)
    root = FbsCompileArtifacts.End(builder)
    builder.Finish(root, file_identifier=COMPILE_ARTIFACTS_FILE_IDENTIFIER)
    return bytes(builder.Output())


def export_compile_artifacts_flatbuffer(
    output_path: str | Path, artifacts: CompileArtifactsData
) -> Path:
    """Export FlatBuffers artifacts for RISC-V runtime consumption."""
    runtime_dir = Path(output_path) / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    schema_path = (
        Path(__file__).parent.parent / "schemas" / COMPILE_ARTIFACTS_SCHEMA_FILE
    )
    shutil.copy2(schema_path, runtime_dir / COMPILE_ARTIFACTS_SCHEMA_FILE)

    artifacts_path = runtime_dir / COMPILE_ARTIFACTS_FILE
    artifacts_path.write_bytes(compile_artifacts_to_flatbuffer(artifacts))
    return artifacts_path
