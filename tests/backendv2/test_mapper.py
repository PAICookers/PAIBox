import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from paicorelib import LCN_EX
from torch import nn

from paibox.backendv2.export.utils import export_framearray_to_int32
from paibox.backendv2.mapper import Mapper
from paibox.backendv2.proto import PROTO_SCHEMA_VERSION
from paibox.backendv2.proto.compile_artifacts_pb2 import (
    CompileArtifacts,
    ConfigFrames,
    OutputEntry,
)
from paibox.paiir import compile_to_paiir
from tests.paiir.conftest import (
    ANNClassifier,
    SimpleCNN,
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


@pytest.fixture(scope="module")
def ensure_backendv2_debug_dir(request, tmp_path_factory):
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

    entries = list(output_mappings.items[0].entries)
    assert entries
    assert {entry.kind for entry in entries} == {OutputEntry.DATA}
    assert all(entry.HasField("kind") for entry in entries)
    assert all(entry.bit_width <= 8 for entry in entries)


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

    entries = list(output_mappings.items[0].entries)
    assert entries
    assert {entry.kind for entry in entries} == {OutputEntry.VOLTAGE}
    assert all(entry.bit_width == 32 for entry in entries)

    bases = [entry.axon_bit_idx for entry in entries[:10]]
    assert bases == [0, 1, 2, 3, 4, 5, 6, 7, 32]
    for entry in entries:
        lanes = {entry.axon_bit_idx + 8 * i for i in range(4)}
        assert len(lanes) == 4


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
