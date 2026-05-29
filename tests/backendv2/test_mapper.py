import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from paicorelib import LCN_EX, DataSign, DataWidth, find_coordxy_shortest_path
from torch import nn

from paibox.backendv2.export.utils import export_framearray_to_int32
from paibox.backendv2.mapper import Mapper
from paibox.backendv2.proto import PROTO_SCHEMA_VERSION
from paibox.backendv2.proto.compile_artifacts_pb2 import (
    CompileArtifacts,
    ConfigFrames,
    DataType,
    OutputTensorMapping,
)
from paibox.paiir import compile_to_paiir
from tests.paiir.conftest import (
    ANNClassifier,
    SimpleCNN,
    SNNTwoLayer,
    make_img_3ch_8x8,
)
from tests.utils import is_ci_env

DEBUG_EXPORT_ROOT = Path(__file__).with_name("debug") / "mapper_proto_export"


def _assert_debug_frame_text(path: Path) -> None:
    text = path.read_text()
    assert "# Core at coord (X,Y)=" in text
    assert "0x" in text


def _export_simple_cnn_proto(export_root: Path, word_order: str) -> Path:
    export_dir = export_root / word_order
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(SimpleCNN().eval(), make_img_3ch_8x8(), strict=True)
    mapper = Mapper()
    mapper.compile(
        graph,
        export_dir,
        target_platform="x86",
        word_order=word_order,  # type: ignore[arg-type]
        debug=True,
    )

    return export_dir / "proto"


def _export_simple_cnn(
    export_root: Path,
    case_name: str,
    *,
    target_platform: str,
    debug: bool,
    export_merged_frames: bool = True,
) -> Path:
    export_dir = export_root / case_name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(SimpleCNN().eval(), make_img_3ch_8x8(), strict=True)
    mapper = Mapper()
    mapper.compile(
        graph,
        export_dir,
        target_platform=target_platform,  # type: ignore[arg-type]
        debug=debug,
        export_merged_frames=export_merged_frames,
    )

    return export_dir


class ConvPotential(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


def _load_compile_artifacts(pb_path: Path) -> CompileArtifacts:
    artifacts = CompileArtifacts()
    artifacts.ParseFromString(pb_path.read_bytes())
    return artifacts


def _export_graph_proto_with_context(
    export_root: Path, case_name: str, model, sample, **compile_kwargs
) -> tuple[Path, set[str], Mapper]:
    export_dir = export_root / case_name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(model.eval(), sample, strict=True, **compile_kwargs)
    output_source_names = {
        pred_name
        for output_node in graph.output_nodes()
        for pred_name in graph.predecessors(output_node.name)
    }
    mapper = Mapper()
    mapper.compile(graph, export_dir, target_platform="x86", debug=True)

    return export_dir / "proto" / "config.pb", output_source_names, mapper


def _export_graph_proto(
    export_root: Path, case_name: str, model, sample, **compile_kwargs
) -> Path:
    pb_path, _, _ = _export_graph_proto_with_context(
        export_root, case_name, model, sample, **compile_kwargs
    )
    return pb_path


def _tick_tuple(tick) -> tuple[int, int, int]:
    return (tick.tick_start, tick.tick_duration, tick.tick_initial)


def _assert_entries_have_dtype(entries, dtype: int, bit_width: int) -> None:
    assert entries
    assert all(entry.HasField("dtype") for entry in entries)
    assert {entry.dtype for entry in entries} == {dtype}
    assert {entry.bit_width for entry in entries} == {bit_width}


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
                words.extend(export_framearray_to_int32(frame_array, word_order))
    return words


CoreTickKey = tuple[int, int, int, tuple[str, ...], tuple[int, int, int]]


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


@pytest.fixture(scope="module")
def ensure_backendv2_debug_dir(tmp_path_factory):
    if is_ci_env():
        export_root = tmp_path_factory.mktemp("backendv2-debug") / "mapper_proto_export"
    else:
        export_root = DEBUG_EXPORT_ROOT

    if export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    return export_root


@pytest.mark.parametrize(
    ("word_order", "expected_enum", "expected_json_value"),
    [
        ("high_first", ConfigFrames.HIGH_FIRST, "HIGH_FIRST"),
        ("low_first", ConfigFrames.LOW_FIRST, "LOW_FIRST"),
    ],
)
def test_export_proto_real_workflow_keeps_pb_and_json(
    ensure_backendv2_debug_dir, word_order, expected_enum, expected_json_value
):
    proto_dir = _export_simple_cnn_proto(ensure_backendv2_debug_dir, word_order)

    pb_path = proto_dir / "config.pb"
    json_path = proto_dir / "config.json"

    assert pb_path.exists()
    assert json_path.exists()
    assert (proto_dir / "compile_artifacts.proto").exists()
    assert (proto_dir / "compile_artifacts_pb2.py").exists()
    assert (proto_dir / "compile_artifacts_pb2.pyi").exists()

    artifacts = _load_compile_artifacts(pb_path)

    assert artifacts.schema_version == PROTO_SCHEMA_VERSION
    assert len(artifacts.io_mapping.threads) == 1
    assert len(artifacts.config_frames.words) > 0
    assert artifacts.config_frames.word_order == expected_enum

    payload = json.loads(json_path.read_text())
    assert payload["schemaVersion"] == PROTO_SCHEMA_VERSION
    assert payload["configFrames"]["wordOrder"] == expected_json_value
    assert len(payload["configFrames"]["words"]) > 0
    assert len(payload["ioMapping"]["threads"]) == 1


def test_export_merged_frames_use_core_major_order(ensure_backendv2_debug_dir):
    export_dir = ensure_backendv2_debug_dir / "core_major_order"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(SimpleCNN().eval(), make_img_3ch_8x8(), strict=True)
    mapper = Mapper()
    mapper.compile(
        graph,
        export_dir,
        target_platform="x86",
        word_order="high_first",
        debug=False,
    )

    merged_frames = np.load(export_dir / "cfg_frames.npy")
    np.testing.assert_array_equal(merged_frames, _expected_core_major_frames(mapper))

    artifacts = _load_compile_artifacts(export_dir / "proto" / "config.pb")
    assert list(artifacts.config_frames.words) == _expected_core_major_words(
        mapper, "high_first"
    )


def test_export_proto_marks_data_outputs_and_target_lcn(
    ensure_backendv2_debug_dir,
):
    pb_path, output_source_names, _ = _export_graph_proto_with_context(
        ensure_backendv2_debug_dir,
        "data_output_kind",
        ANNClassifier(),
        make_img_3ch_8x8(),
    )

    artifacts = _load_compile_artifacts(pb_path)
    output_mappings = artifacts.io_mapping.threads[0].output_mappings

    assert output_mappings.target_lcn == LCN_EX.LCN_128X.value
    assert output_mappings.HasField("target_lcn")
    assert len(output_mappings.items) == 1
    assert {mapping.name for mapping in output_mappings.items} == output_source_names

    output_mapping = output_mappings.items[0]
    assert output_mapping.kind == OutputTensorMapping.DATA
    assert output_mapping.HasField("kind")

    entries = list(output_mapping.entries)
    assert entries
    assert all(entry.bit_width <= 8 for entry in entries)
    assert all(entry.HasField("dtype") for entry in entries)
    assert {entry.dtype for entry in entries}.issubset({DataType.UINT8, DataType.INT8})


def test_export_proto_marks_voltage_outputs_and_base_addresses(
    ensure_backendv2_debug_dir,
):
    model = ConvPotential()
    with torch.no_grad():
        model.conv.weight.fill_(1)

    pb_path = _export_graph_proto(
        ensure_backendv2_debug_dir,
        "voltage_output_kind",
        model,
        torch.ones(1, 1, 3, 3),
    )

    artifacts = _load_compile_artifacts(pb_path)
    output_mappings = artifacts.io_mapping.threads[0].output_mappings

    assert output_mappings.target_lcn == LCN_EX.LCN_128X.value
    assert len(output_mappings.items) == 1

    output_mapping = output_mappings.items[0]
    assert output_mapping.kind == OutputTensorMapping.VOLTAGE
    assert output_mapping.HasField("kind")

    entries = list(output_mapping.entries)
    assert entries
    assert all(entry.bit_width == 32 for entry in entries)
    assert all(not entry.HasField("dtype") for entry in entries)

    bases = [entry.axon_bit_idx for entry in entries[:10]]
    assert bases == [0, 1, 2, 3, 4, 5, 6, 7, 32]


def test_export_proto_exports_ann_io_ticks_and_core_ticks(
    ensure_backendv2_debug_dir,
):
    pb_path, output_source_names, mapper = _export_graph_proto_with_context(
        ensure_backendv2_debug_dir,
        "ann_tick_metadata",
        ANNClassifier(),
        make_img_3ch_8x8(),
    )

    artifacts = _load_compile_artifacts(pb_path)
    thread = artifacts.io_mapping.threads[0]

    assert {mapping.name for mapping in thread.output_mappings.items} == (
        output_source_names
    )
    _assert_core_ticks_match_mapper(thread, mapper)
    _assert_thread_tick_duration_initial(thread, (1, 1))


def test_export_proto_exports_explicit_ann_tick_policy(
    ensure_backendv2_debug_dir,
):
    pb_path, _, mapper = _export_graph_proto_with_context(
        ensure_backendv2_debug_dir,
        "ann_explicit_tick_metadata",
        ANNClassifier(),
        make_img_3ch_8x8(),
        tick_duration=6,
        auto_reset=False,
    )

    artifacts = _load_compile_artifacts(pb_path)
    thread = artifacts.io_mapping.threads[0]

    _assert_core_ticks_match_mapper(thread, mapper)
    _assert_thread_tick_duration_initial(thread, (6, 0))


def test_export_proto_exports_input_dtype_from_consumer_format(
    ensure_backendv2_debug_dir,
):
    pb_path = _export_graph_proto(
        ensure_backendv2_debug_dir,
        "input_dtype_uint1",
        SNNTwoLayer(),
        make_img_3ch_8x8(),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)},
    )

    artifacts = _load_compile_artifacts(pb_path)
    input_mappings = artifacts.io_mapping.threads[0].input_mappings.items

    assert input_mappings
    for mapping in input_mappings:
        _assert_entries_have_dtype(list(mapping.entries), DataType.UINT1, 1)


@pytest.mark.parametrize(
    ("tick_duration", "auto_reset", "expected_initial"),
    [(7, True, 7), (7, False, 0), (0, True, 0)],
    ids=["finite_reset", "finite_no_reset", "always_on"],
)
def test_export_proto_exports_snn_tick_policy(
    ensure_backendv2_debug_dir, tick_duration, auto_reset, expected_initial
):
    pb_path, output_source_names, mapper = _export_graph_proto_with_context(
        ensure_backendv2_debug_dir,
        f"snn_tick_metadata_{tick_duration}_{auto_reset}",
        SNNTwoLayer(),
        make_img_3ch_8x8(),
        tick_duration=tick_duration,
        auto_reset=auto_reset,
    )

    artifacts = _load_compile_artifacts(pb_path)
    thread = artifacts.io_mapping.threads[0]
    expected = (tick_duration, expected_initial)

    assert {mapping.name for mapping in thread.output_mappings.items} == (
        output_source_names
    )
    _assert_core_ticks_match_mapper(thread, mapper)
    _assert_thread_tick_duration_initial(thread, expected)


def test_export_artifacts_all_platforms_when_requested(ensure_backendv2_debug_dir):
    export_dir = _export_simple_cnn(
        ensure_backendv2_debug_dir,
        "all_platforms",
        target_platform="all",
        debug=False,
    )

    assert (export_dir / "cfg_frame1.npy").exists()
    assert (export_dir / "cfg_frame2.npy").exists()
    assert (export_dir / "cfg_frame3.npy").exists()
    assert (export_dir / "cfg_frames.npy").exists()

    assert (export_dir / "cfg_frame1.h").exists()
    assert (export_dir / "cfg_frame2.h").exists()
    assert (export_dir / "cfg_frame3.h").exists()
    assert (export_dir / "cfg_frames.h").exists()

    assert not (export_dir / "cfg_frame1.txt").exists()
    assert not (export_dir / "cfg_frame2.txt").exists()
    assert not (export_dir / "cfg_frame3.txt").exists()
    assert not (export_dir / "cfg_frames.txt").exists()
    assert not (export_dir / "proto" / "config.json").exists()

    cfg_frame1 = np.load(export_dir / "cfg_frame1.npy")
    assert cfg_frame1.dtype == np.dtype("<u8")


def test_export_artifacts_can_skip_merged_frames(ensure_backendv2_debug_dir):
    export_dir = _export_simple_cnn(
        ensure_backendv2_debug_dir,
        "skip_merged_frames",
        target_platform="all",
        debug=True,
        export_merged_frames=False,
    )

    assert not (export_dir / "cfg_frames.txt").exists()
    assert not (export_dir / "cfg_frames.npy").exists()
    assert not (export_dir / "cfg_frames.h").exists()

    assert (export_dir / "cfg_frame1.txt").exists()
    assert (export_dir / "cfg_frame1.npy").exists()
    assert (export_dir / "cfg_frame1.h").exists()
    assert (export_dir / "proto" / "config.pb").exists()
    assert (export_dir / "proto" / "config.json").exists()


def test_export_artifacts_debug_forces_all_platform_outputs(
    ensure_backendv2_debug_dir,
):
    export_dir = _export_simple_cnn(
        ensure_backendv2_debug_dir,
        "debug_forces_all",
        target_platform="riscv",
        debug=True,
    )

    assert (export_dir / "cfg_frame1.txt").exists()
    assert (export_dir / "cfg_frame2.txt").exists()
    assert (export_dir / "cfg_frame3.txt").exists()
    assert (export_dir / "cfg_frames.txt").exists()

    assert (export_dir / "cfg_frame1.npy").exists()
    assert (export_dir / "cfg_frame2.npy").exists()
    assert (export_dir / "cfg_frame3.npy").exists()
    assert (export_dir / "cfg_frames.npy").exists()

    assert (export_dir / "cfg_frame1.h").exists()
    assert (export_dir / "cfg_frame2.h").exists()
    assert (export_dir / "cfg_frame3.h").exists()
    assert (export_dir / "cfg_frames.h").exists()

    assert (export_dir / "proto" / "config.pb").exists()
    assert (export_dir / "proto" / "config.json").exists()
    assert (export_dir / "proto" / "compile_artifacts_pb2.py").exists()
    assert (export_dir / "proto" / "compile_artifacts_pb2.pyi").exists()

    _assert_debug_frame_text(export_dir / "cfg_frame1.txt")
    _assert_debug_frame_text(export_dir / "cfg_frame2.txt")
    _assert_debug_frame_text(export_dir / "cfg_frame3.txt")
    _assert_debug_frame_text(export_dir / "cfg_frames.txt")
