import json
from pathlib import Path

import numpy as np
import pytest
import torch
from paicorelib import LCN_EX, DataSign, DataWidth, find_coordxy_shortest_path
from torch import nn

from paibox.backendv2.artifacts.compile_artifacts import SCHEMA_VERSION
from paibox.backendv2.artifacts.utils import export_framearray_to_int32
from paibox.backendv2.generated.fbs.CompileArtifacts import (
    CompileArtifacts as FbsCompileArtifacts,
)
from paibox.backendv2.generated.proto.compile_artifacts_pb2 import (
    CompileArtifacts,
    ConfigFrames,
    DataType,
    OutputTensorMapping,
    RuntimeParams,
)
from paibox.backendv2.mapper import Mapper
from paibox.backendv2.routing import FANIN_BASE
from paibox.paiir import compile_to_paiir
from tests.paiir.conftest import ANNClassifier, SimpleCNN, SNNTwoLayer, make_img_3ch_8x8

FRAME_STEMS = ("cfg_frame1", "cfg_frame2", "cfg_frame3")
NPY_FILES = {f"{stem}.npy" for stem in FRAME_STEMS} | {"cfg_frames.npy"}
HEADER_FILES = {f"{stem}.h" for stem in FRAME_STEMS} | {"cfg_frames.h"}
TEXT_FILES = {f"{stem}.txt" for stem in FRAME_STEMS} | {"cfg_frames.txt"}
PROTO_FILES = {
    "proto/config.pb",
    "proto/compile_artifacts.proto",
    "proto/compile_artifacts_pb2.py",
    "proto/compile_artifacts_pb2.pyi",
}
PROTO_JSON_FILES = {"proto/config.json"}
RUNTIME_FILES = {
    "runtime/compile_artifacts.bin",
    "runtime/compile_artifacts.fbs",
}

CoreTickKey = tuple[int, int, int, tuple[str, ...], tuple[int, int, int]]


class ConvPotential(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture(scope="module")
def export_root(tmp_path_factory):
    return tmp_path_factory.mktemp("backendv2-artifacts")


def _relative_files(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def _assert_files(root: Path, *, present: set[str], absent: set[str]) -> None:
    actual = _relative_files(root)
    assert present <= actual
    assert absent.isdisjoint(actual)


def _export_simple_cnn(
    export_root: Path,
    case_name: str,
    target_platform: str,
    *,
    debug: bool,
    word_order: str = "high_first",
    export_merged_frames: bool = True,
) -> tuple[Path, Mapper]:
    export_dir = export_root / case_name
    export_dir.mkdir(parents=True, exist_ok=False)

    graph = compile_to_paiir(SimpleCNN().eval(), make_img_3ch_8x8(), strict=True)
    mapper = Mapper()
    mapper.compile(
        graph,
        export_dir,
        target_platform=target_platform,  # type: ignore[arg-type]
        word_order=word_order,  # type: ignore[arg-type]
        debug=debug,
        export_merged_frames=export_merged_frames,
    )

    return export_dir, mapper


def _export_graph_proto_with_context(
    export_root: Path,
    case_name: str,
    model,
    sample,
    mapper_timesteps: int | None = None,
    **compile_kwargs,
) -> tuple[Path, set[str], Mapper]:
    export_dir = export_root / case_name
    export_dir.mkdir(parents=True, exist_ok=False)

    graph = compile_to_paiir(model.eval(), sample, strict=True, **compile_kwargs)
    output_source_names = {
        pred_name
        for output_node in graph.output_nodes()
        for pred_name in graph.predecessors(output_node.name)
    }
    mapper = Mapper()
    mapper.compile(
        graph, export_dir, target_platform="x86", debug=True, timesteps=mapper_timesteps
    )

    return export_dir / "proto" / "config.pb", output_source_names, mapper


def _export_graph_proto(
    export_root: Path, case_name: str, model, sample, **compile_kwargs
) -> Path:
    pb_path, _, _ = _export_graph_proto_with_context(
        export_root, case_name, model, sample, **compile_kwargs
    )
    return pb_path


def _load_compile_artifacts_proto(pb_path: Path) -> CompileArtifacts:
    artifacts = CompileArtifacts()
    artifacts.ParseFromString(pb_path.read_bytes())
    return artifacts


def _load_compile_artifacts_flatbuffer(bundle_path: Path) -> FbsCompileArtifacts:
    data = bytearray(bundle_path.read_bytes())
    assert FbsCompileArtifacts.CompileArtifactsBufferHasIdentifier(data, 0)
    return FbsCompileArtifacts.GetRootAs(data, 0)


def _assert_compile_artifacts_matches_proto(
    bundle: FbsCompileArtifacts, artifacts: CompileArtifacts
) -> None:
    assert bundle.SchemaVersion() == artifacts.schema_version
    config_frames = bundle.ConfigFrames()
    assert config_frames is not None
    assert config_frames.WordOrder() == artifacts.config_frames.word_order
    assert config_frames.WordsLength() == len(artifacts.config_frames.words)
    assert config_frames.WordsLength() > 0

    io_mapping = bundle.IoMapping()
    assert io_mapping is not None
    assert io_mapping.ThreadsLength() == len(artifacts.io_mapping.threads)
    assert io_mapping.ThreadsLength() > 0
    thread = io_mapping.Threads(0)
    assert thread is not None
    proto_thread = artifacts.io_mapping.threads[0]
    assert thread.ThreadId() == proto_thread.thread_id
    assert thread.InputMappings().ItemsLength() == len(
        proto_thread.input_mappings.items
    )
    assert thread.OutputMappings().ItemsLength() == len(
        proto_thread.output_mappings.items
    )


def _assert_debug_frame_text(path: Path) -> None:
    text = path.read_text()
    assert "# Core at coord (X,Y)=" in text
    assert "0x" in text


def _tick_tuple(tick) -> tuple[int, int, int]:
    return (tick.tick_start, tick.tick_duration, tick.tick_initial)


def _assert_entries_have_dtype(entries, dtype: int) -> None:
    assert entries
    assert all(entry.HasField("dtype") for entry in entries)
    assert {entry.dtype for entry in entries} == {dtype}


def _assert_mapping_bit_width(mapping, expected: int) -> None:
    assert mapping.HasField("bit_width")
    assert mapping.bit_width == expected
    assert all(
        "bit_width" not in entry.DESCRIPTOR.fields_by_name for entry in mapping.entries
    )


def _assert_json_field_order(mapping_json: dict[str, object]) -> None:
    keys = list(mapping_json)
    assert keys.index("bitWidth") < keys.index("tick")
    assert keys.index("bitWidth") < keys.index("entries")


def _expected_core_major_frames(mapper: Mapper) -> np.ndarray:
    parts = []
    for core_placement in mapper.coreplacements:
        for frame_array in core_placement.to_frame():
            if frame_array is not None:
                parts.append(np.asarray(frame_array, dtype="<u8"))
    return np.concatenate(parts) if parts else np.array([], dtype="<u8")


def _expected_core_major_words(mapper: Mapper, word_order: str) -> list[int]:
    words = []
    for core_placement in mapper.coreplacements:
        for frame_array in core_placement.to_frame():
            if frame_array is not None:
                words.extend(export_framearray_to_int32(frame_array, word_order))  # type: ignore
    return words


def _expected_data_lcn(num_outputs: int) -> LCN_EX:
    for lcn_value in range(8):
        if num_outputs <= FANIN_BASE * (1 << lcn_value):
            return LCN_EX(lcn_value)
    raise AssertionError(f"{num_outputs} outputs exceed LCN_128X capacity")


def _expected_lcn_from_max_axon(max_axon_addr: int) -> LCN_EX:
    min_tick_relative_bit = (max_axon_addr // FANIN_BASE).bit_length()
    return LCN_EX(min_tick_relative_bit)


def _core_tick_key(core_tick) -> CoreTickKey:
    return (
        core_tick.core_offset.xy,
        core_tick.core_offset.x,
        core_tick.core_offset.y,
        tuple(core_tick.nodes),
        _tick_tuple(core_tick.tick),
    )


def _expected_core_tick_keys(mapper: Mapper, thread_id: int) -> set[CoreTickKey]:
    keys = set()
    for core_placement in mapper.coreplacements:
        if not core_placement.neus:
            continue
        if core_placement.default_core_config.thread_number != thread_id:
            continue
        core_offset, _ = find_coordxy_shortest_path(core_placement.coord)
        nodes = tuple(
            sorted(
                {
                    raw_neu.target.raw_node.name
                    for neu_placement in core_placement.neus
                    for raw_neu in neu_placement.raw_neus
                }
            )
        )
        tick = (
            int(core_placement.frontend_core_config.tick_start),
            int(core_placement.frontend_core_config.tick_duration),
            int(core_placement.frontend_core_config.tick_initial),
        )
        keys.add((core_offset.z, core_offset.x, core_offset.y, nodes, tick))
    return keys


def _assert_core_ticks_match_mapper(thread, mapper: Mapper) -> None:
    core_ticks = list(thread.core_ticks)
    assert core_ticks
    assert all(core_tick.nodes for core_tick in core_ticks)
    assert {_core_tick_key(core_tick) for core_tick in core_ticks} == (
        _expected_core_tick_keys(mapper, thread.thread_id)
    )


def _assert_thread_tick_duration_initial(thread, expected: tuple[int, int]) -> None:
    assert thread.input_mappings.items
    assert thread.output_mappings.items
    assert all(
        _tick_tuple(mapping.tick)[1:] == expected
        for mapping in thread.input_mappings.items
    )
    assert all(
        _tick_tuple(mapping.tick)[1:] == expected
        for mapping in thread.output_mappings.items
    )
    assert all(
        _tick_tuple(core_tick.tick)[1:] == expected for core_tick in thread.core_ticks
    )


@pytest.mark.parametrize(
    ("target_platform", "debug", "present", "absent"),
    [
        (
            "x86",
            False,
            NPY_FILES | PROTO_FILES,
            HEADER_FILES | TEXT_FILES | PROTO_JSON_FILES | RUNTIME_FILES,
        ),
        (
            "riscv",
            False,
            HEADER_FILES | RUNTIME_FILES,
            NPY_FILES | TEXT_FILES | PROTO_FILES | PROTO_JSON_FILES,
        ),
        (
            "all",
            False,
            NPY_FILES | HEADER_FILES | PROTO_FILES | RUNTIME_FILES,
            TEXT_FILES | PROTO_JSON_FILES,
        ),
        (
            "riscv",
            True,
            NPY_FILES
            | HEADER_FILES
            | TEXT_FILES
            | PROTO_FILES
            | PROTO_JSON_FILES
            | RUNTIME_FILES,
            set(),
        ),
    ],
    ids=["x86", "riscv", "all", "debug-forces-all"],
)
def test_export_artifacts_by_platform(
    export_root, target_platform, debug, present, absent
):
    export_dir, _ = _export_simple_cnn(
        export_root,
        f"platform_{target_platform}_{debug}",
        target_platform=target_platform,
        debug=debug,
    )

    _assert_files(export_dir, present=present, absent=absent)
    if "cfg_frame1.npy" in present:
        assert np.load(export_dir / "cfg_frame1.npy").dtype == np.dtype("<u8")
    if "cfg_frame1.txt" in present:
        _assert_debug_frame_text(export_dir / "cfg_frame1.txt")


def test_export_artifacts_can_skip_merged_frames(export_root):
    export_dir, _ = _export_simple_cnn(
        export_root,
        "skip_merged_frames",
        target_platform="all",
        debug=True,
        export_merged_frames=False,
    )

    per_type_files = {
        *(f"{stem}.txt" for stem in FRAME_STEMS),
        *(f"{stem}.npy" for stem in FRAME_STEMS),
        *(f"{stem}.h" for stem in FRAME_STEMS),
    }
    _assert_files(
        export_dir,
        present=per_type_files | PROTO_FILES | PROTO_JSON_FILES | RUNTIME_FILES,
        absent={"cfg_frames.txt", "cfg_frames.npy", "cfg_frames.h"},
    )


def test_export_compile_artifacts_flatbuffer_matches_proto(export_root):
    export_dir, _ = _export_simple_cnn(
        export_root,
        "compile_artifacts_matches_proto",
        target_platform="all",
        debug=False,
    )
    artifacts = _load_compile_artifacts_proto(export_dir / "proto" / "config.pb")
    bundle = _load_compile_artifacts_flatbuffer(
        export_dir / "runtime" / "compile_artifacts.bin"
    )

    _assert_compile_artifacts_matches_proto(bundle, artifacts)


@pytest.mark.parametrize(
    ("word_order", "expected_enum", "expected_json_value"),
    [
        ("high_first", ConfigFrames.HIGH_FIRST, "HIGH_FIRST"),
        ("low_first", ConfigFrames.LOW_FIRST, "LOW_FIRST"),
    ],
)
def test_export_proto_real_workflow_keeps_pb_and_json(
    export_root, word_order, expected_enum, expected_json_value
):
    export_dir, _ = _export_simple_cnn(
        export_root,
        f"proto_{word_order}",
        target_platform="x86",
        word_order=word_order,
        debug=True,
    )
    proto_dir = export_dir / "proto"
    artifacts = _load_compile_artifacts_proto(proto_dir / "config.pb")

    assert artifacts.schema_version == SCHEMA_VERSION
    assert len(artifacts.io_mapping.threads) == 1
    assert len(artifacts.config_frames.words) > 0
    assert artifacts.config_frames.word_order == expected_enum

    payload = json.loads((proto_dir / "config.json").read_text())
    assert payload["schemaVersion"] == SCHEMA_VERSION
    assert payload["configFrames"]["wordOrder"] == expected_json_value
    assert len(payload["configFrames"]["words"]) > 0
    threads = payload["ioMapping"]["threads"]
    assert len(threads) == 1
    for mapping in threads[0]["inputMappings"]["items"]:
        _assert_json_field_order(mapping)
    for mapping in threads[0]["outputMappings"]["items"]:
        _assert_json_field_order(mapping)


def test_export_merged_frames_use_core_major_order(export_root):
    export_dir, mapper = _export_simple_cnn(
        export_root,
        "core_major_order",
        target_platform="x86",
        word_order="high_first",
        debug=False,
    )

    np.testing.assert_array_equal(
        np.load(export_dir / "cfg_frames.npy"), _expected_core_major_frames(mapper)
    )
    artifacts = _load_compile_artifacts_proto(export_dir / "proto" / "config.pb")
    assert list(artifacts.config_frames.words) == _expected_core_major_words(
        mapper, "high_first"
    )


def test_export_proto_marks_data_outputs_and_target_lcn(export_root):
    pb_path, output_source_names, _ = _export_graph_proto_with_context(
        export_root,
        "data_output_kind",
        ANNClassifier(),
        make_img_3ch_8x8(),
    )

    artifacts = _load_compile_artifacts_proto(pb_path)
    thread = artifacts.io_mapping.threads[0]
    output_mappings = thread.output_mappings

    assert thread.runtime.timesteps == 1
    assert thread.runtime.tick_depth >= 1
    assert thread.runtime.sync_steps == thread.runtime.tick_depth
    assert thread.runtime.decode_mode == RuntimeParams.STREAM
    output_count = sum(len(mapping.entries) for mapping in output_mappings.items)
    assert output_mappings.target_lcn == _expected_data_lcn(output_count).value
    assert output_mappings.HasField("target_lcn")
    assert len(output_mappings.items) == 1
    assert {mapping.name for mapping in output_mappings.items} == output_source_names

    output_mapping = output_mappings.items[0]
    assert output_mapping.kind == OutputTensorMapping.DATA
    assert output_mapping.HasField("kind")

    entries = list(output_mapping.entries)
    assert entries
    assert output_mapping.HasField("bit_width")
    assert output_mapping.bit_width <= 8
    assert all("bit_width" not in entry.DESCRIPTOR.fields_by_name for entry in entries)
    assert all(entry.HasField("dtype") for entry in entries)
    assert {entry.dtype for entry in entries}.issubset({DataType.UINT8, DataType.INT8})


def test_export_proto_marks_voltage_outputs_and_base_addresses(export_root):
    model = ConvPotential()
    with torch.no_grad():
        model.conv.weight.fill_(1)

    pb_path = _export_graph_proto(
        export_root,
        "voltage_output_kind",
        model,
        torch.ones(1, 1, 3, 3),
    )

    artifacts = _load_compile_artifacts_proto(pb_path)
    thread = artifacts.io_mapping.threads[0]
    output_mappings = thread.output_mappings

    assert thread.runtime.timesteps == 1
    assert thread.runtime.sync_steps == 1
    assert thread.runtime.decode_mode == RuntimeParams.STREAM
    assert len(output_mappings.items) == 1

    output_mapping = output_mappings.items[0]
    assert output_mapping.kind == OutputTensorMapping.VOLTAGE
    assert output_mapping.HasField("kind")

    entries = list(output_mapping.entries)
    assert entries
    _assert_mapping_bit_width(output_mapping, 32)
    assert all(not entry.HasField("dtype") for entry in entries)

    bases = [entry.axon_bit_idx for entry in entries[:10]]
    assert bases == [0, 1, 2, 3, 4, 5, 6, 7, 32]
    assert output_mappings.target_lcn == _expected_lcn_from_max_axon(max(bases)).value


def test_output_lcn_uses_timesteps_not_external_sync_steps(export_root):
    pb_path, _, mapper = _export_graph_proto_with_context(
        export_root,
        "output_lcn_uses_timesteps",
        ANNClassifier(),
        make_img_3ch_8x8(),
        timesteps=1,
        auto_reset=False,
    )

    artifacts = _load_compile_artifacts_proto(pb_path)
    thread = artifacts.io_mapping.threads[0]

    assert thread.runtime.timesteps == 1
    assert thread.runtime.tick_depth > 1
    assert thread.runtime.sync_steps == thread.runtime.tick_depth
    output_count = sum(len(mapping.entries) for mapping in thread.output_mappings.items)
    expected_lcn = _expected_data_lcn(output_count)
    assert thread.output_mappings.target_lcn == expected_lcn.value
    assert mapper.output_groups[0].lcn == expected_lcn
    assert not hasattr(mapper.output_groups[0], "runtime_timesteps")


def test_export_proto_exports_input_dtype_from_consumer_format(export_root):
    pb_path = _export_graph_proto(
        export_root,
        "input_dtype_uint1",
        SNNTwoLayer(),
        make_img_3ch_8x8(),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)},
    )

    artifacts = _load_compile_artifacts_proto(pb_path)
    input_mappings = artifacts.io_mapping.threads[0].input_mappings.items

    assert input_mappings
    for mapping in input_mappings:
        _assert_mapping_bit_width(mapping, 1)
        _assert_entries_have_dtype(list(mapping.entries), DataType.UINT1)


@pytest.mark.parametrize(
    ("case_name", "model_factory", "compile_kwargs", "expected_tick"),
    [
        ("ann_default", ANNClassifier, {}, (0, 1)),
        (
            "ann_manual_reset",
            ANNClassifier,
            {"timesteps": 6, "auto_reset": False},
            (6, 0),
        ),
        ("snn_auto_reset", SNNTwoLayer, {"timesteps": 7, "auto_reset": True}, (0, 7)),
        (
            "snn_manual_reset",
            SNNTwoLayer,
            {"timesteps": 7, "auto_reset": False},
            (7, 0),
        ),
    ],
    ids=["ann-default", "ann-manual-reset", "snn-auto-reset", "snn-manual-reset"],
)
def test_export_proto_exports_io_tick_policy_and_core_ticks(
    export_root, case_name, model_factory, compile_kwargs, expected_tick
):
    pb_path, output_source_names, mapper = _export_graph_proto_with_context(
        export_root,
        f"tick_metadata_{case_name}",
        model_factory(),
        make_img_3ch_8x8(),
        **compile_kwargs,
    )

    artifacts = _load_compile_artifacts_proto(pb_path)
    thread = artifacts.io_mapping.threads[0]
    timesteps = compile_kwargs.get("timesteps", 1)
    expected_tick_depth = max(
        mapping.tick.tick_start for mapping in thread.output_mappings.items
    )

    assert {mapping.name for mapping in thread.output_mappings.items} == (
        output_source_names
    )
    assert thread.runtime.timesteps == timesteps
    assert thread.runtime.tick_depth == expected_tick_depth
    assert thread.runtime.sync_steps == expected_tick_depth + timesteps - 1
    assert thread.runtime.decode_mode == RuntimeParams.STREAM
    _assert_core_ticks_match_mapper(thread, mapper)
    _assert_thread_tick_duration_initial(thread, expected_tick)
