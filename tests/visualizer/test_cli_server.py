import builtins
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from paicorelib import CoordXY
from paicorelib.framelib.frame_defs import FFV2, FrameHeader

from paibox.visualizer.artifact import load_artifact, load_artifact_session
from paibox.visualizer.backends.v2.errors import FrameDecodeError
from paibox.visualizer.cli import cli as viz_cli

from .helpers import endpoint_map, make_core_frame, write_pb


def test_backendv2_generated_proto_import_is_lightweight() -> None:
    """Generated artifact bindings must not initialize the compiler backend."""
    source = (
        "import sys; "
        "import paibox.backendv2.generated.proto.compile_artifacts_pb2; "
        "assert 'paibox.backendv2.mapper' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'numba' not in sys.modules"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_visualizer_cli_import_does_not_initialize_server_dependencies() -> None:
    """Validation-only CLI imports must not initialize FastAPI or uvicorn."""
    source = (
        "import sys; import paibox.visualizer.cli; "
        "assert 'paibox.visualizer.server' not in sys.modules; "
        "assert 'fastapi' not in sys.modules; assert 'uvicorn' not in sys.modules"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_load_pb_frame_first_and_metadata_cross_check(tmp_path: Path) -> None:
    frames = make_core_frame(CoordXY(3, 1))
    pb_path = write_pb(tmp_path / "config.pb", frames, root_core_offset=(0, 3, 1))

    model = load_artifact(pb_path)

    assert model.summary.frame_count == 4
    assert model.summary.used_core_count == 1
    assert model.summary.validation_error_count == 0
    assert model.runtime[0].thread_id == 3
    core = next(core for core in model.chips[0].cores if core.x == 3 and core.y == 1)
    assert core.source == "frame+metadata"
    assert core.nodes == ["SequentialOp_0"]
    assert core.core_config["neuron_number"] == 4
    assert core.global_signal.send_dirs == []
    assert core.global_signal.receive_dirs == []
    assert core.global_signal.is_source
    assert core.global_signal.source_thread_ids == [3]
    assert len(core.global_signal.control_paths) == 1
    assert [(p.x, p.y) for p in core.global_signal.control_paths[0].points] == [
        (3, 1),
        (2, 0),
        (1, 0),
        (0, 0),
    ]
    assert core.global_signal.control_paths[0].target_x == 0
    assert core.global_signal.control_paths[0].target_y == 0
    assert core.global_signal.control_paths[0].offset_xy == -1
    assert core.global_signal.control_paths[0].offset_x == -2
    assert core.global_signal.control_paths[0].offset_y == 0


@pytest.mark.parametrize(
    "word_order",
    ["high", "low"],
    ids=["high-first", "low-first"],
)
def test_load_pb_supports_both_word_orders(tmp_path: Path, word_order: str) -> None:
    from paibox.backendv2.generated.proto.compile_artifacts_pb2 import ConfigFrames

    frames = make_core_frame(CoordXY(3, 1))
    pb_path = write_pb(
        tmp_path / f"{word_order}.pb",
        frames,
        word_order=(
            ConfigFrames.HIGH_FIRST if word_order == "high" else ConfigFrames.LOW_FIRST
        ),
    )

    model = load_artifact(pb_path)

    assert model.summary.frame_count == len(frames)
    used = [core for core in model.chips[0].cores if core.used]
    assert [(core.x, core.y) for core in used] == [(3, 1)]


def test_global_signal_links_include_send_and_receive(tmp_path: Path) -> None:
    frames = np.concatenate(
        [
            make_core_frame(CoordXY(3, 1), global_send=1 << 3),
            make_core_frame(CoordXY(4, 1), global_receive=1 << 2),
        ]
    )
    pb_path = write_pb(tmp_path / "config.pb", frames, root_core_offset=(0, 3, 1))

    model = load_artifact(pb_path)

    link_keys = {
        (
            link.kind,
            link.direction,
            link.source["x"],
            link.source["y"],
            link.target["x"],
            link.target["y"],
        )
        for link in model.links
    }
    assert ("global_send", "+x", 3, 1, 4, 1) in link_keys
    assert ("global_receive", "-x", 3, 1, 4, 1) in link_keys
    assert model.summary.validation_error_count == 0


def test_global_signal_source_missing_root_metadata_is_hidden(valid_pb: Path) -> None:
    pb_path = valid_pb
    model = load_artifact(pb_path)

    core = next(core for core in model.chips[0].cores if core.x == 3 and core.y == 1)
    assert not core.global_signal.is_source
    assert core.global_signal.control_paths == []


def test_global_signal_control_path_out_of_grid_fails(tmp_path: Path) -> None:
    frames = make_core_frame(
        CoordXY(0, 8),
        test_core_xy=-1,
        test_core_x=1,
        test_core_y=0,
    )
    pb_path = write_pb(tmp_path / "config.pb", frames, root_core_offset=(0, 0, 8))

    with pytest.raises(
        FrameDecodeError, match="global signal source control path leaves chip grid"
    ) as exc_info:
        load_artifact(pb_path)

    assert exc_info.value.frame_index is not None
    assert exc_info.value.raw_frame is not None
    assert exc_info.value.context["route_axis"] == "z"
    assert exc_info.value.context["illegal_x"] == -1
    assert exc_info.value.context["illegal_y"] == 7


@pytest.mark.parametrize(
    ("coord", "signal_kwargs", "message", "context_key", "context_value"),
    [
        (
            CoordXY(8, 2),
            {"global_send": 1 << 3},
            "global signal send target leaves chip grid",
            "target_x",
            9,
        ),
        (
            CoordXY(0, 2),
            {"global_receive": 1 << 2},
            "global signal receive source leaves chip grid",
            "source_x",
            -1,
        ),
    ],
    ids=["send-target-out-of-grid", "receive-source-out-of-grid"],
)
def test_global_signal_route_out_of_grid_fails(
    tmp_path: Path,
    coord: CoordXY,
    signal_kwargs: dict[str, int],
    message: str,
    context_key: str,
    context_value: int,
) -> None:
    pb_path = write_pb(tmp_path / "config.pb", make_core_frame(coord, **signal_kwargs))

    with pytest.raises(FrameDecodeError, match=message) as exc_info:
        load_artifact(pb_path)

    assert exc_info.value.frame_index is not None
    assert exc_info.value.raw_frame is not None
    assert exc_info.value.context[context_key] == context_value


@pytest.mark.parametrize(
    ("coord", "signal_kwargs", "message", "signal_kind", "direction"),
    [
        (
            CoordXY(3, 1),
            {"global_send": 1 << 3},
            "global signal send missing matching receive",
            "global_send",
            "-x",
        ),
        (
            CoordXY(4, 1),
            {"global_receive": 1 << 2},
            "global signal receive missing matching send",
            "global_receive",
            "+x",
        ),
    ],
    ids=["send-without-receive", "receive-without-send"],
)
def test_global_signal_route_missing_peer_fails(
    tmp_path: Path,
    coord: CoordXY,
    signal_kwargs: dict[str, int],
    message: str,
    signal_kind: str,
    direction: str,
) -> None:
    pb_path = write_pb(tmp_path / "config.pb", make_core_frame(coord, **signal_kwargs))

    with pytest.raises(FrameDecodeError, match=message) as exc_info:
        load_artifact(pb_path)

    assert exc_info.value.context["signal_kind"] == signal_kind
    assert exc_info.value.context["expected_direction"] == direction


def test_merged_npy_content_cross_check(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    proto_dir = artifact_dir / "proto"
    proto_dir.mkdir(parents=True)
    frames = make_core_frame(CoordXY(3, 1))
    write_pb(proto_dir / "config.pb", frames)
    np.save(artifact_dir / "cfg_frames.npy", frames)

    model = load_artifact(artifact_dir)

    assert model.summary.frame_count == len(frames)
    assert not [
        item for item in model.validation if item.code.endswith("content_mismatch")
    ]


def test_typed_npy_frames_are_loaded_without_protobuf(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "typed"
    artifact_dir.mkdir()
    frames = make_core_frame(CoordXY(3, 1))
    np.save(artifact_dir / "cfg_frame1.npy", frames)

    model = load_artifact(artifact_dir)

    assert model.artifact.has_pb is False
    assert model.artifact.has_typed_npy is True
    assert model.summary.frame_count == len(frames)


def test_cli_help_lists_current_commands() -> None:
    result = CliRunner().invoke(viz_cli, ["--help"])

    assert result.exit_code == 0
    assert "serve" in result.output
    assert "validate" in result.output
    assert "inspect" not in result.output


def test_cli_validate_text_and_json(valid_pb: Path) -> None:
    pb_path = valid_pb
    runner = CliRunner()

    result = runner.invoke(
        viz_cli, ["validate", "--artifact", str(pb_path), "--backend", "v2"]
    )
    assert result.exit_code == 0
    assert "errors=0" in result.output

    json_result = runner.invoke(
        viz_cli, ["validate", "--artifact", str(pb_path), "--json"]
    )
    assert json_result.exit_code == 0
    assert json_result.output.strip() == "[]"


def test_cli_validate_reports_decode_error_without_traceback(tmp_path: Path) -> None:
    bad = np.array(
        [int(FrameHeader.WORK_TYPE1) << FFV2.GENERAL_HEADER_OFFSET],
        dtype=np.uint64,
    )
    pb_path = write_pb(tmp_path / "config.pb", bad)

    result = CliRunner().invoke(viz_cli, ["validate", "--artifact", str(pb_path)])

    assert result.exit_code == 1
    assert f"artifact={pb_path.resolve()}" in result.output
    assert "frame decode error:" in result.output
    assert "unsupported frame header" in result.output
    assert "Traceback" not in result.output


def test_cli_rejects_removed_inspect_command(valid_pb: Path) -> None:
    pb_path = valid_pb

    result = CliRunner().invoke(viz_cli, ["inspect", "--artifact", str(pb_path)])

    assert result.exit_code == 2
    assert "No such command 'inspect'" in result.output


def test_cli_rejects_old_backend_name(valid_pb: Path) -> None:
    pb_path = valid_pb

    result = CliRunner().invoke(
        viz_cli, ["validate", "--artifact", str(pb_path), "--backend", "backendv2"]
    )

    assert result.exit_code == 2
    assert "Invalid value for '--backend'" in result.output
    assert "v2" in result.output


@pytest.mark.parametrize("loader", [load_artifact, load_artifact_session])
def test_public_loaders_reject_unknown_backend(
    valid_pb: Path, loader: Callable[..., object]
) -> None:
    with pytest.raises(ValueError, match="unknown visualizer backend"):
        loader(valid_pb, backend="legacy")


@pytest.mark.parametrize(
    ("command", "options", "expected"),
    [
        (
            [],
            [
                "--host",
                "0.0.0.0",
                "--port",
                "1234",
                "--no-browser",
                "--backend",
                "v2",
            ],
            {
                "host": "0.0.0.0",
                "port": 1234,
                "open_browser": False,
                "backend": "v2",
            },
        ),
        (
            ["serve"],
            ["--port", "4321"],
            {
                "host": "127.0.0.1",
                "port": 4321,
                "open_browser": True,
                "backend": "v2",
            },
        ),
    ],
    ids=["default-command", "explicit-serve-command"],
)
def test_cli_serve_invokes_server(
    valid_pb: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
    options: list[str],
    expected: dict[str, object],
) -> None:
    pytest.importorskip("fastapi")
    from paibox.visualizer import server as server_module

    calls = []

    def fake_serve(*args, **kwargs) -> int:
        calls.append((args, kwargs))
        return 0

    monkeypatch.setattr(server_module, "serve", fake_serve)

    result = CliRunner().invoke(
        viz_cli,
        [*command, "--artifact", str(valid_pb), *options],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            (valid_pb,),
            expected,
        )
    ]


def test_cli_serve_reports_missing_optional_dependencies(
    valid_pb: Path, monkeypatch
) -> None:
    pb_path = valid_pb
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fastapi":
            raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "paibox.visualizer.server", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = CliRunner().invoke(viz_cli, ["serve", "--artifact", str(pb_path)])

    assert result.exit_code == 2
    assert "optional visualizer dependencies" in result.output


def test_fastapi_app_summary_and_core(valid_pb: Path) -> None:
    pytest.importorskip("fastapi")

    from fastapi.responses import FileResponse

    from paibox.visualizer.server import _find_ui_dist, create_app

    pb_path = valid_pb
    app = create_app(pb_path)
    assert app.state.viewer_model.io.input_entries == []
    assert app.state.viewer_model.io.output_entries == []
    endpoints = endpoint_map(app)

    summary = endpoints["/api/summary"]()
    assert summary["used_core_count"] == 1

    chips = endpoints["/api/chips"]()
    overview_core = next(item for item in chips[0]["cores"] if item["used"])
    core = endpoints["/api/cores/{chip_id}/{x}/{y}"](0, 3, 1)
    assert overview_core["core_config"]["neuron_number"] == 4
    assert overview_core["neurons"]["summary"] == core["neurons"]["summary"]
    assert "packages" not in overview_core
    assert "decoded_core_config" not in overview_core
    assert "lut" not in overview_core
    assert "weights" not in overview_core
    assert "raw_frames" not in overview_core
    assert "records" not in overview_core["neurons"]

    assert core["core_config"]["neuron_number"] == 4
    assert "packages" in core
    assert "records" in core["neurons"]
    assert "records" in core["weights"]
    assert "raw_frames" in core

    if _find_ui_dist() is not None:
        index = endpoints["/"]()
        assert isinstance(index, FileResponse)


def test_fastapi_auxiliary_endpoints_and_missing_ui(
    valid_pb: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")

    import paibox.visualizer.server as server_module

    monkeypatch.setattr(server_module, "_find_ui_dist", lambda: None)
    endpoints = endpoint_map(server_module.create_app(valid_pb))

    validation = endpoints["/api/validation"]()
    favicon = endpoints["/favicon.ico"]()
    missing_ui = endpoints["/"]()

    assert validation == []
    assert favicon.status_code == 204
    assert missing_ui.status_code == 503
    assert missing_ui.body == b"PAIBox visualizer UI assets are not built."


def test_server_serve_picks_port_and_opens_browser(
    valid_pb: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")

    import paibox.visualizer.server as server_module

    calls: dict[str, object] = {}

    class FakeServer:
        def __init__(self, config: object) -> None:
            calls["server_config"] = config

        def run(self) -> None:
            calls["ran"] = True

    def fake_config(app: object, **kwargs: object) -> object:
        calls["app"] = app
        calls["config"] = kwargs
        return object()

    monkeypatch.setattr(server_module, "create_app", lambda *args, **kwargs: object())
    monkeypatch.setattr(server_module, "_pick_free_port", lambda host: 4567)
    monkeypatch.setattr(server_module.uvicorn, "Config", fake_config)
    monkeypatch.setattr(server_module.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(
        server_module.webbrowser, "open", lambda url: calls.setdefault("url", url)
    )

    assert server_module.serve(valid_pb, port=0, open_browser=True) == 0
    assert calls["config"] == {"host": "127.0.0.1", "port": 4567, "log_level": "info"}
    assert calls["url"] == "http://127.0.0.1:4567"
    assert calls["ran"] is True


def test_fastapi_defers_offline_core_decode_until_selection(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("fastapi")

    import paibox.visualizer.backends.v2.artifact as artifact_module
    from paibox.visualizer.server import create_app

    frames = make_core_frame(CoordXY(3, 2))
    pb_path = write_pb(tmp_path / "config.pb", frames)
    calls = 0
    original = artifact_module.decode_offline_core

    def counted_decode(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(artifact_module, "decode_offline_core", counted_decode)
    app = create_app(pb_path)
    assert calls == 0

    endpoints = endpoint_map(app)
    core_endpoint = endpoints["/api/cores/{chip_id}/{x}/{y}"]
    first = core_endpoint(0, 3, 2)
    second = core_endpoint(0, 3, 2)
    assert calls == 1
    assert second == first


def test_fastapi_lazy_core_matches_full_loader(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")

    from dataclasses import asdict

    from paibox.visualizer.server import create_app

    frames = make_core_frame(CoordXY(3, 2))
    pb_path = write_pb(tmp_path / "config.pb", frames)
    full_model = load_artifact(pb_path)
    expected = next(
        core for core in full_model.chips[0].cores if (core.x, core.y) == (3, 2)
    )

    app = create_app(pb_path)
    endpoints = endpoint_map(app)
    actual = endpoints["/api/cores/{chip_id}/{x}/{y}"](0, 3, 2)

    assert actual == asdict(expected)


def test_fastapi_io_endpoints(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")

    from paibox.backendv2.generated.proto.compile_artifacts_pb2 import (
        CompileArtifacts,
        DataType,
    )
    from paibox.visualizer.server import create_app

    frames = make_core_frame(CoordXY(3, 1))
    pb_path = write_pb(tmp_path / "config.pb", frames)

    artifacts = CompileArtifacts()
    artifacts.ParseFromString(pb_path.read_bytes())
    mapping = artifacts.io_mapping.threads[0].input_mappings.items.add()
    mapping.name = "input0"
    mapping.shape.size.extend([1, 2, 4, 5])
    mapping.bit_width = 8
    for elem_idx, addr_axon in ((5, 10), (6, 11)):
        entry = mapping.entries.add()
        entry.elem_idx = elem_idx
        entry.core_offset.x = 3
        entry.core_offset.y = 1
        entry.tick_relative = 7
        entry.addr_axon = addr_axon
        entry.target_lcn = 0
        entry.dtype = DataType.UINT8
    pb_path.write_bytes(artifacts.SerializeToString())

    from paibox.visualizer.backends.v2.io_mapping import build_io_view

    full_io = build_io_view(artifacts, [])
    summary_io = build_io_view(artifacts, [], include_entries=False)
    assert summary_io.tensors == full_io.tensors
    assert summary_io.core_summaries == full_io.core_summaries
    assert len(load_artifact(pb_path).io.input_entries) == 2

    app = create_app(pb_path)
    assert app.state.viewer_model.io.input_entries == []
    assert app.state.viewer_model.io.output_entries == []
    endpoints = endpoint_map(app)

    summary = endpoints["/api/io/summary"]()
    assert summary["available"]
    assert summary["tensors"][0]["shape"] == [1, 2, 4, 5]
    assert summary["core_summaries"][0]["input_count"] == 2

    core_io = endpoints["/api/io/cores/{chip_id}/{x}/{y}"](0, 3, 1)
    assert core_io["summary"]["input_count"] == 2
    assert core_io["input_regions"][0]["bbox_width"] == 2
    assert core_io["input_buffer_spans"][0]["bit_start"] == 10

    regions = endpoints["/api/io/regions"](
        direction="input",
        thread_id=None,
        tensor_name="input0",
        slice_key=None,
        chip_id=0,
        x=3,
        y=1,
        offset=0,
        limit=10,
    )
    assert regions["total"] == 1
    assert regions["items"][0]["bbox_width"] == 2

    entries = endpoints["/api/io/entries"](
        direction="input",
        thread_id=None,
        tensor_name="input0",
        slice_key=None,
        chip_id=0,
        x=3,
        y=1,
        offset=0,
        limit=1,
    )
    assert entries["total"] == 2
    assert len(entries["items"]) == 1
    assert entries["items"][0]["elem_idx"] == 5
