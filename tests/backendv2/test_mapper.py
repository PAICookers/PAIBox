import json
import shutil
from pathlib import Path

import pytest

from paibox.backendv2.mapper import Mapper
from paibox.backendv2.proto import PROTO_SCHEMA_VERSION
from paibox.backendv2.proto.config_pb2 import CompileArtifacts, ConfigFrames
from paibox.paiir import compile_to_paiir
from tests.paiir.conftest import SimpleCNN, make_img_3ch_8x8
from tests.utils import is_ci_env

DEBUG_EXPORT_ROOT = Path(__file__).with_name("debug") / "mapper_proto_export"


def _export_simple_cnn_proto(export_root: Path, word_order: str) -> Path:
    export_dir = export_root / word_order
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(SimpleCNN().eval(), make_img_3ch_8x8(), strict=True)
    mapper = Mapper()
    mapper.compile(graph, export_dir, target_platform="x86", word_order=word_order)  # type: ignore[arg-type]

    return export_dir / "proto"


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
    assert (proto_dir / "config.proto").exists()
    assert (proto_dir / "config_pb2.py").exists()
    assert (proto_dir / "config_pb2.pyi").exists()

    artifacts = CompileArtifacts()
    artifacts.ParseFromString(pb_path.read_bytes())

    assert artifacts.schema_version == PROTO_SCHEMA_VERSION
    assert len(artifacts.io_mapping.threads) == 1
    assert len(artifacts.config_frames.words) > 0
    assert artifacts.config_frames.word_order == expected_enum

    payload = json.loads(json_path.read_text())
    assert payload["schemaVersion"] == PROTO_SCHEMA_VERSION
    assert payload["configFrames"]["wordOrder"] == expected_json_value
    assert len(payload["configFrames"]["words"]) > 0
    assert len(payload["ioMapping"]["threads"]) == 1
