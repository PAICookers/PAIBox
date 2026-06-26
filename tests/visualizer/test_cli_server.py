import builtins
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from paicorelib import CoordXY
from paicorelib.framelib.frame_defs import FFV2, FrameHeader

from paibox.visualizer.artifact import load_artifact
from paibox.visualizer.backends.v2.errors import FrameDecodeError
from paibox.visualizer.cli import cli as viz_cli

from .helpers import make_core_frame, write_pb


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


def test_global_signal_source_missing_root_metadata_is_hidden(tmp_path: Path) -> None:
    frames = make_core_frame(CoordXY(3, 1))
    pb_path = write_pb(tmp_path / "config.pb", frames)

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


def test_global_signal_send_out_of_grid_fails(tmp_path: Path) -> None:
    frames = make_core_frame(CoordXY(8, 2), global_send=1 << 3)
    pb_path = write_pb(tmp_path / "config.pb", frames)

    with pytest.raises(
        FrameDecodeError, match="global signal send target leaves chip grid"
    ) as exc_info:
        load_artifact(pb_path)

    assert exc_info.value.frame_index is not None
    assert exc_info.value.raw_frame is not None
    assert exc_info.value.context["signal_kind"] == "global_send"
    assert exc_info.value.context["target_x"] == 9


def test_global_signal_receive_out_of_grid_fails(tmp_path: Path) -> None:
    frames = make_core_frame(CoordXY(0, 2), global_receive=1 << 2)
    pb_path = write_pb(tmp_path / "config.pb", frames)

    with pytest.raises(
        FrameDecodeError, match="global signal receive source leaves chip grid"
    ) as exc_info:
        load_artifact(pb_path)

    assert exc_info.value.frame_index is not None
    assert exc_info.value.raw_frame is not None
    assert exc_info.value.context["signal_kind"] == "global_receive"
    assert exc_info.value.context["source_x"] == -1


def test_global_signal_send_missing_matching_receive_fails(
    tmp_path: Path,
) -> None:
    frames = make_core_frame(CoordXY(3, 1), global_send=1 << 3)
    pb_path = write_pb(tmp_path / "config.pb", frames)

    with pytest.raises(
        FrameDecodeError, match="global signal send missing matching receive"
    ) as exc_info:
        load_artifact(pb_path)

    assert exc_info.value.context["signal_kind"] == "global_send"
    assert exc_info.value.context["expected_direction"] == "-x"


def test_global_signal_receive_missing_matching_send_fails(
    tmp_path: Path,
) -> None:
    frames = make_core_frame(CoordXY(4, 1), global_receive=1 << 2)
    pb_path = write_pb(tmp_path / "config.pb", frames)

    with pytest.raises(
        FrameDecodeError, match="global signal receive missing matching send"
    ) as exc_info:
        load_artifact(pb_path)

    assert exc_info.value.context["signal_kind"] == "global_receive"
    assert exc_info.value.context["expected_direction"] == "+x"


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


def test_cli_help_lists_current_commands() -> None:
    result = CliRunner().invoke(viz_cli, ["--help"])

    assert result.exit_code == 0
    assert "serve" in result.output
    assert "validate" in result.output
    assert "inspect" not in result.output


def test_cli_validate_text_and_json(tmp_path: Path) -> None:
    frames = make_core_frame(CoordXY(3, 1))
    pb_path = write_pb(tmp_path / "config.pb", frames)
    runner = CliRunner()

    result = runner.invoke(
        viz_cli, ["validate", "--artifact", str(pb_path), "--backend", "v2"]
    )
    assert result.exit_code == 0
    assert "errors=0" in result.output

    json_result = runner.invoke(viz_cli, ["validate", "--artifact", str(pb_path), "--json"])
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


def test_cli_rejects_removed_inspect_command(tmp_path: Path) -> None:
    frames = make_core_frame(CoordXY(3, 1))
    pb_path = write_pb(tmp_path / "config.pb", frames)

    result = CliRunner().invoke(viz_cli, ["inspect", "--artifact", str(pb_path)])

    assert result.exit_code == 2
    assert "No such command 'inspect'" in result.output


def test_cli_rejects_old_backend_name(tmp_path: Path) -> None:
    frames = make_core_frame(CoordXY(3, 1))
    pb_path = write_pb(tmp_path / "config.pb", frames)

    result = CliRunner().invoke(
        viz_cli, ["validate", "--artifact", str(pb_path), "--backend", "backendv2"]
    )

    assert result.exit_code == 2
    assert "Invalid value for '--backend'" in result.output
    assert "v2" in result.output


def test_cli_default_serve_invokes_server(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from paibox.visualizer import server as server_module

    frames = make_core_frame(CoordXY(3, 1))
    pb_path = write_pb(tmp_path / "config.pb", frames)
    calls = []

    def fake_serve(*args, **kwargs) -> int:
        calls.append((args, kwargs))
        return 0

    monkeypatch.setattr(server_module, "serve", fake_serve)

    result = CliRunner().invoke(
        viz_cli,
        [
            "--artifact",
            str(pb_path),
            "--host",
            "0.0.0.0",
            "--port",
            "1234",
            "--no-browser",
            "--backend",
            "v2",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            (pb_path,),
            {
                "host": "0.0.0.0",
                "port": 1234,
                "open_browser": False,
                "backend": "v2",
            },
        )
    ]


def test_cli_explicit_serve_invokes_server(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from paibox.visualizer import server as server_module

    frames = make_core_frame(CoordXY(3, 1))
    pb_path = write_pb(tmp_path / "config.pb", frames)
    calls = []

    def fake_serve(*args, **kwargs) -> int:
        calls.append((args, kwargs))
        return 0

    monkeypatch.setattr(server_module, "serve", fake_serve)

    result = CliRunner().invoke(
        viz_cli,
        ["serve", "--artifact", str(pb_path), "--port", "4321"],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            (pb_path,),
            {
                "host": "127.0.0.1",
                "port": 4321,
                "open_browser": True,
                "backend": "v2",
            },
        )
    ]


def test_cli_serve_reports_missing_optional_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    frames = make_core_frame(CoordXY(3, 1))
    pb_path = write_pb(tmp_path / "config.pb", frames)
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


def test_fastapi_app_summary_and_core(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")

    from fastapi.responses import FileResponse

    from paibox.visualizer.server import _find_ui_dist, create_app

    frames = make_core_frame(CoordXY(3, 1))
    pb_path = write_pb(tmp_path / "config.pb", frames)
    app = create_app(pb_path)
    endpoints = {
        route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")
    }

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


def test_fastapi_io_endpoints(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")

    from paibox.backendv2.proto.compile_artifacts_pb2 import DataType
    from paibox.visualizer.server import create_app

    frames = make_core_frame(CoordXY(3, 1))
    pb_path = write_pb(tmp_path / "config.pb", frames)

    from paibox.backendv2.proto.compile_artifacts_pb2 import CompileArtifacts

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

    app = create_app(pb_path)
    endpoints = {
        route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")
    }

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
