from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from google.protobuf.json_format import Parse
from paicorelib.framelib.parser_v2 import (
    FramePackageInfo,
    FrameParseError,
    ParsedCoreFrames,
    decode_core_config,
    parse_frame_stream,
    sign_magnitude_to_int,
)

from paibox.backendv2.proto.compile_artifacts_pb2 import CompileArtifacts, ConfigFrames

from ...model import (
    ArtifactInfo,
    ChipView,
    ControlPathView,
    CoreConfigView,
    CoreFrameSummary,
    CoreView,
    FramePackageSummary,
    GlobalSignal,
    LinkView,
    LutView,
    NeuronView,
    RoutePointView,
    RuntimeInfo,
    ValidationEntry,
    ViewerModel,
    ViewerSummary,
    WeightView,
)
from .errors import FrameDecodeError
from .frame_parser import (
    CHIP_ID,
    GLOBAL_SIGNAL_DIRECTIONS,
    GRID_HEIGHT,
    GRID_WIDTH,
    chip_core_role,
    global_signal_dirs,
    reverse_global_signal_dir,
)
from .io_mapping import (
    attribute_output_entries_to_cores,
    build_io_view,
    io_core_summary_map,
)
from .offline_v2 import decode_offline_core


class ArtifactLoadError(ValueError):
    """Raised when an artifact path cannot be loaded as a visualizer input."""


@dataclass(frozen=True)
class _CoreConfigSource:
    config: dict[str, int]
    frame_index: int | None
    raw_frame: int | None
    config_word1_index: int | None
    config_word1_raw: int | None
    config_word2_index: int | None
    config_word2_raw: int | None


CONTROL_PATH_TARGET = (0, 0)


def load_artifact(path: str | Path) -> ViewerModel:
    """Load a backendv2 artifact and build the viewer JSON model.

    Frame-derived data is the source of truth for core configuration, LUT,
    neuron, weight, and raw-frame views. Protobuf metadata is then layered on for
    node/runtime/IO context and cross-check validation.
    """
    artifact_path = Path(path).resolve()
    resolved = _resolve_artifact_files(artifact_path)
    artifacts = _load_compile_artifacts(resolved)
    frames = _load_frames(resolved, artifacts)
    try:
        frame_stream = parse_frame_stream(frames)
    except FrameParseError as exc:
        raise FrameDecodeError(
            exc.reason,
            frame_index=exc.frame_index,
            raw_frame=exc.raw_frame,
            context=exc.context,
        ) from exc
    validation: list[ValidationEntry] = []

    metadata_by_coord, runtime_infos = _metadata_from_artifacts(artifacts, validation)
    io_view = build_io_view(artifacts, validation)
    io_summaries = io_core_summary_map(io_view)
    core_config_sources = _decode_core_configs(frame_stream.cores)
    links = _build_global_signal_links(core_config_sources)
    source_control_paths = _build_global_source_control_paths(
        core_config_sources, runtime_infos
    )
    cores: list[CoreView] = []

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            frame_core = frame_stream.cores.get((x, y))
            metadata = metadata_by_coord.get((x, y), {})
            config_source = core_config_sources.get((x, y))
            core_config = config_source.config if config_source else {}
            core_validation = _validate_core(x, y, core_config, metadata)
            validation.extend(core_validation)

            global_send = core_config.get("global_send", 0)
            global_receive = core_config.get("global_receive", 0)
            control_paths = source_control_paths.get((x, y), [])

            packages = frame_core.packages if frame_core else []
            package_summaries = _package_summaries(packages)
            try:
                decoded = (
                    decode_offline_core(
                        core_config,
                        packages,
                        core_coord=(x, y),
                        grid_width=GRID_WIDTH,
                        grid_height=GRID_HEIGHT,
                    )
                    if frame_core and chip_core_role(x, y) == "offline"
                    else None
                )
            except FrameDecodeError as exc:
                raise exc.with_context(chip_id=CHIP_ID, core_x=x, core_y=y) from exc
            frames_summary = CoreFrameSummary(
                frame_type1_count=(
                    len(frame_core.frame_type1_payloads) if frame_core else 0
                ),
                frame_type2_count=(
                    len(frame_core.frame_type2_payloads) if frame_core else 0
                ),
                frame_type3_count=(
                    len(frame_core.frame_type3_payloads) if frame_core else 0
                ),
                package_count=len(package_summaries),
            )
            used = bool(frame_core or metadata)
            cores.append(
                CoreView(
                    chip_id=CHIP_ID,
                    x=x,
                    y=y,
                    role=chip_core_role(x, y),
                    used=used,
                    source=_source_label(bool(frame_core), bool(metadata)),
                    nodes=list(metadata.get("nodes", [])),
                    thread_id=_thread_id(core_config, metadata),
                    core_config=core_config,
                    io_summary=io_summaries.get((CHIP_ID, x, y)),
                    global_signal=GlobalSignal(
                        send_bits=global_send,
                        receive_bits=global_receive,
                        send_dirs=global_signal_dirs(global_send, include_local=True),
                        receive_dirs=global_signal_dirs(global_receive),
                        sends_local=bool(global_send & (1 << 6)),
                        is_source=bool(control_paths),
                        source_thread_ids=[path.thread_id for path in control_paths],
                        control_paths=control_paths,
                    ),
                    frames=frames_summary,
                    packages=package_summaries,
                    decoded_core_config=(
                        decoded.core_config if decoded is not None else CoreConfigView()
                    ),
                    lut=decoded.lut if decoded is not None else LutView(),
                    neurons=decoded.neurons if decoded is not None else NeuronView(),
                    weights=decoded.weights if decoded is not None else WeightView(),
                    raw_frames=decoded.raw_frames if decoded is not None else [],
                    validation=core_validation,
                )
            )

    io_view = attribute_output_entries_to_cores(io_view, cores)
    io_summaries = io_core_summary_map(io_view)
    cores = [
        replace(core, io_summary=io_summaries.get((core.chip_id, core.x, core.y)))
        for core in cores
    ]

    _validate_frame_content(resolved, frames, validation)
    warning_count = sum(1 for item in validation if item.severity == "warning")
    error_count = sum(1 for item in validation if item.severity == "error")
    return ViewerModel(
        schema_version=1,
        artifact=ArtifactInfo(
            path=str(artifact_path),
            kind=resolved["kind"],
            has_pb=resolved.get("pb") is not None,
            has_json=resolved.get("json") is not None,
            has_merged_npy=resolved.get("merged_npy") is not None,
            has_typed_npy=any(resolved.get(f"frame{idx}_npy") for idx in (1, 2, 3)),
        ),
        chips=[
            ChipView(
                chip_id=CHIP_ID,
                grid_width=GRID_WIDTH,
                grid_height=GRID_HEIGHT,
                cores=cores,
            )
        ],
        links=links,
        io=io_view,
        validation=validation,
        runtime=runtime_infos,
        summary=ViewerSummary(
            chip_count=1,
            core_count=len(cores),
            used_core_count=sum(1 for core in cores if core.used),
            frame_count=frame_stream.frame_count,
            package_count=len(frame_stream.packages),
            validation_error_count=error_count,
            validation_warning_count=warning_count,
        ),
    )


def _package_summaries(packages: list[FramePackageInfo]) -> list[FramePackageSummary]:
    return [
        FramePackageSummary(
            frame_type=package.frame_type,
            start_addr=package.start_addr,
            package_type=package.package_type,
            package_count=package.package_count,
            frame_start=package.frame_start,
            frame_end=package.frame_end,
            header=package.header,
            header_hex=package.header_hex,
            payload_hex=list(package.payload_hex),
        )
        for package in packages
    ]


def _resolve_artifact_files(path: Path) -> dict[str, Any]:
    """Normalize supported inputs to a small artifact-file manifest.

    Users may point the CLI at `config.pb`, `config.json`, a frame `.npy`, or an
    artifact directory. Downstream loaders use the returned manifest instead of
    repeating path-guessing rules.
    """
    result: dict[str, Any] = {"kind": "file" if path.is_file() else "directory"}
    if path.is_dir():
        result["root"] = path
        result["pb"] = _first_existing(path / "proto" / "config.pb", path / "config.pb")
        result["json"] = _first_existing(
            path / "proto" / "config.json", path / "config.json"
        )
        result["merged_npy"] = _first_existing(path / "cfg_frames.npy")
        for idx in (1, 2, 3):
            result[f"frame{idx}_npy"] = _first_existing(path / f"cfg_frame{idx}.npy")
        return result

    if not path.exists():
        raise ArtifactLoadError(f"artifact path does not exist: {path}")

    if path.name == "config.pb" or path.suffix == ".pb":
        root = path.parent.parent if path.parent.name == "proto" else path.parent
        result.update(_resolve_artifact_files(root))
        result["kind"] = "pb"
        result["pb"] = path
        return result
    if path.name == "config.json" or path.suffix == ".json":
        root = path.parent.parent if path.parent.name == "proto" else path.parent
        result.update(_resolve_artifact_files(root))
        result["kind"] = "json"
        result["json"] = path
        return result
    if path.suffix == ".npy":
        result["root"] = path.parent
        if path.name == "cfg_frames.npy":
            result["kind"] = "merged_npy"
            result["merged_npy"] = path
        else:
            result["kind"] = "typed_npy"
            for idx in (1, 2, 3):
                if path.name == f"cfg_frame{idx}.npy":
                    result[f"frame{idx}_npy"] = path
        return result

    raise ArtifactLoadError(f"unsupported artifact input: {path}")


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _load_compile_artifacts(resolved: dict[str, Any]) -> CompileArtifacts | None:
    pb_path = resolved.get("pb")
    if pb_path is not None:
        artifacts = CompileArtifacts()
        artifacts.ParseFromString(Path(pb_path).read_bytes())
        return artifacts

    json_path = resolved.get("json")
    if json_path is not None:
        artifacts = CompileArtifacts()
        Parse(Path(json_path).read_text(encoding="utf-8"), artifacts)
        return artifacts

    return None


def _load_frames(
    resolved: dict[str, Any], artifacts: CompileArtifacts | None
) -> np.ndarray:
    """Load final config frames from PB first, then numpy fallbacks.

    PB `config_frames.words` is preferred because it is the packaged artifact
    users actually deploy; numpy frame dumps are kept as developer fallbacks.
    """
    if artifacts is not None and artifacts.config_frames.words:
        return _frames_from_words(artifacts.config_frames)

    merged_npy = resolved.get("merged_npy")
    if merged_npy is not None:
        return np.load(merged_npy).astype(np.uint64, copy=False)

    parts = []
    for idx in (1, 2, 3):
        path = resolved.get(f"frame{idx}_npy")
        if path is not None:
            parts.append(np.load(path).astype(np.uint64, copy=False))
    if parts:
        return np.concatenate(parts).astype(np.uint64, copy=False)

    raise ArtifactLoadError("artifact does not contain config frames")


def _frames_from_words(config_frames: ConfigFrames) -> np.ndarray:
    words = list(config_frames.words)
    if len(words) % 2 != 0:
        raise ArtifactLoadError("config frame word count is not even")

    frames = np.zeros(len(words) // 2, dtype=np.uint64)
    for idx in range(0, len(words), 2):
        a = int(words[idx])
        b = int(words[idx + 1])
        if config_frames.word_order == ConfigFrames.HIGH_FIRST:
            hi, lo = a, b
        else:
            lo, hi = a, b
        frames[idx // 2] = (np.uint64(hi) << np.uint64(32)) | np.uint64(lo)
    return frames


def _metadata_from_artifacts(
    artifacts: CompileArtifacts | None, validation: list[ValidationEntry]
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[RuntimeInfo]]:
    if artifacts is None:
        return {}, []

    metadata: dict[tuple[int, int], dict[str, Any]] = {}
    runtime_infos: list[RuntimeInfo] = []
    for thread in artifacts.io_mapping.threads:
        runtime_infos.append(
            RuntimeInfo(
                thread_id=thread.thread_id,
                root_core_offset=_core_offset_to_dict(thread.root_core_offset),
                runtime=_runtime_to_dict(thread.runtime),
                input_count=len(thread.input_mappings.items),
                output_count=len(thread.output_mappings.items),
                core_tick_count=len(thread.core_ticks),
            )
        )
        for core_tick in thread.core_ticks:
            x = core_tick.core_offset.xy + core_tick.core_offset.x
            y = core_tick.core_offset.xy + core_tick.core_offset.y
            if not _coord_in_grid(x, y):
                validation.append(
                    ValidationEntry(
                        severity="error",
                        code="metadata_core_out_of_grid",
                        message=f"metadata core ({x}, {y}) is outside the 9x9 grid",
                        chip_id=CHIP_ID,
                        x=x,
                        y=y,
                    )
                )
            metadata[(x, y)] = {
                "thread_id": thread.thread_id,
                "nodes": list(core_tick.nodes),
                "tick": {
                    "tick_start": core_tick.tick.tick_start,
                    "tick_duration": core_tick.tick.tick_duration,
                    "tick_initial": core_tick.tick.tick_initial,
                },
            }

    return metadata, runtime_infos


def _core_offset_to_dict(offset: Any) -> dict[str, int]:
    if not (
        _has_optional_field(offset, "xy")
        and _has_optional_field(offset, "x")
        and _has_optional_field(offset, "y")
    ):
        return {}
    return {"xy": offset.xy, "x": offset.x, "y": offset.y}


def _has_optional_field(message: Any, field_name: str) -> bool:
    try:
        return bool(message.HasField(field_name))
    except ValueError:
        return True


def _runtime_to_dict(runtime: Any) -> dict[str, int | str]:
    decode_mode = "STREAM" if runtime.decode_mode == 0 else "STEP"
    return {
        "timesteps": runtime.timesteps,
        "tick_depth": runtime.tick_depth,
        "sync_steps": runtime.sync_steps,
        "decode_mode": decode_mode,
    }


def _validate_core(
    x: int, y: int, core_config: dict[str, int], metadata: dict[str, Any]
) -> list[ValidationEntry]:
    validation: list[ValidationEntry] = []
    if not core_config and metadata:
        validation.append(
            ValidationEntry(
                severity="warning",
                code="metadata_without_frame",
                message="core exists in protobuf metadata but no config frame was parsed",
                chip_id=CHIP_ID,
                x=x,
                y=y,
            )
        )
    if core_config and not metadata:
        validation.append(
            ValidationEntry(
                severity="info",
                code="frame_without_metadata",
                message="core has config frames but no protobuf core_tick metadata",
                chip_id=CHIP_ID,
                x=x,
                y=y,
            )
        )

    tick = metadata.get("tick")
    if core_config and tick:
        for key in ("tick_start", "tick_duration", "tick_initial"):
            if core_config.get(key, -1) != tick[key]:
                validation.append(
                    ValidationEntry(
                        severity="error",
                        code=f"{key}_mismatch",
                        message=(
                            f"frame {key}={core_config.get(key)} does not match "
                            f"metadata {tick[key]}"
                        ),
                        chip_id=CHIP_ID,
                        x=x,
                        y=y,
                    )
                )

    if core_config and not _coord_in_grid(x, y):
        validation.append(
            ValidationEntry(
                severity="error",
                code="frame_core_out_of_grid",
                message=f"frame core ({x}, {y}) is outside the 9x9 grid",
                chip_id=CHIP_ID,
                x=x,
                y=y,
            )
        )
    return validation


def _decode_core_configs(
    frame_cores: dict[tuple[int, int], ParsedCoreFrames],
) -> dict[tuple[int, int], _CoreConfigSource]:
    configs: dict[tuple[int, int], _CoreConfigSource] = {}
    for coord, frame_core in frame_cores.items():
        config = decode_core_config(frame_core.frame_type1_payloads)
        if not config:
            continue
        frame_index, raw_frame = _global_signal_source_frame(frame_core)
        word1_index, word1_raw = _config_payload_frame(frame_core, 0)
        word2_index, word2_raw = _config_payload_frame(frame_core, 1)
        configs[coord] = _CoreConfigSource(
            config=config,
            frame_index=frame_index,
            raw_frame=raw_frame,
            config_word1_index=word1_index,
            config_word1_raw=word1_raw,
            config_word2_index=word2_index,
            config_word2_raw=word2_raw,
        )
    return configs


def _config_payload_frame(
    frame_core: ParsedCoreFrames, payload_index: int
) -> tuple[int | None, int | None]:
    for package in frame_core.packages:
        if (
            package.frame_type == 1
            and len(frame_core.frame_type1_payloads) > payload_index
        ):
            return (
                package.frame_start + 1 + payload_index,
                frame_core.frame_type1_payloads[payload_index],
            )
    return None, None


def _global_signal_source_frame(
    frame_core: ParsedCoreFrames,
) -> tuple[int | None, int | None]:
    for package in frame_core.packages:
        if package.frame_type == 1 and len(frame_core.frame_type1_payloads) >= 2:
            # global_send/global_receive are encoded in config frame type1 word2.
            return package.frame_start + 2, frame_core.frame_type1_payloads[1]
    return None, None


def _build_global_signal_links(
    configs: dict[tuple[int, int], _CoreConfigSource],
) -> list[LinkView]:
    links: list[LinkView] = []
    for coord in sorted(configs):
        x, y = coord
        source = configs[coord]
        if not _coord_in_grid(x, y):
            continue

        send_bits = source.config.get("global_send", 0)
        for direction in global_signal_dirs(send_bits):
            dx, dy = GLOBAL_SIGNAL_DIRECTIONS[direction]
            target = (x + dx, y + dy)
            if not _coord_in_grid(*target):
                _raise_global_signal_error(
                    "global signal send target leaves chip grid",
                    source,
                    owner=coord,
                    signal_kind="global_send",
                    direction=direction,
                    signal_source=coord,
                    signal_target=target,
                )

            expected_receive = reverse_global_signal_dir(direction)
            target_receive_dirs = _global_signal_dir_set(
                configs, target, "global_receive"
            )
            if expected_receive not in target_receive_dirs:
                _raise_global_signal_error(
                    "global signal send missing matching receive",
                    source,
                    owner=coord,
                    signal_kind="global_send",
                    direction=direction,
                    signal_source=coord,
                    signal_target=target,
                    expected_direction=expected_receive,
                )

            links.append(
                LinkView(
                    chip_id=CHIP_ID,
                    source={"x": x, "y": y},
                    target={"x": target[0], "y": target[1]},
                    kind="global_send",
                    direction=direction,
                )
            )

        receive_bits = source.config.get("global_receive", 0)
        for direction in global_signal_dirs(receive_bits):
            dx, dy = GLOBAL_SIGNAL_DIRECTIONS[direction]
            signal_source = (x + dx, y + dy)
            if not _coord_in_grid(*signal_source):
                _raise_global_signal_error(
                    "global signal receive source leaves chip grid",
                    source,
                    owner=coord,
                    signal_kind="global_receive",
                    direction=direction,
                    signal_source=signal_source,
                    signal_target=coord,
                )

            expected_send = reverse_global_signal_dir(direction)
            source_send_dirs = _global_signal_dir_set(
                configs, signal_source, "global_send"
            )
            if expected_send not in source_send_dirs:
                _raise_global_signal_error(
                    "global signal receive missing matching send",
                    source,
                    owner=coord,
                    signal_kind="global_receive",
                    direction=direction,
                    signal_source=signal_source,
                    signal_target=coord,
                    expected_direction=expected_send,
                )

            links.append(
                LinkView(
                    chip_id=CHIP_ID,
                    source={"x": signal_source[0], "y": signal_source[1]},
                    target={"x": x, "y": y},
                    kind="global_receive",
                    direction=direction,
                )
            )
    return links


def _build_global_source_control_paths(
    configs: dict[tuple[int, int], _CoreConfigSource],
    runtime_infos: list[RuntimeInfo],
) -> dict[tuple[int, int], list[ControlPathView]]:
    paths: dict[tuple[int, int], list[ControlPathView]] = {}
    for runtime in runtime_infos:
        source_coord = _coord_from_offset(runtime.root_core_offset)
        if source_coord is None:
            continue
        if not _coord_in_grid(*source_coord):
            raise FrameDecodeError(
                "global signal source root leaves chip grid",
                context={
                    "chip_id": CHIP_ID,
                    "thread_id": runtime.thread_id,
                    "source_x": source_coord[0],
                    "source_y": source_coord[1],
                    "grid_width": GRID_WIDTH,
                    "grid_height": GRID_HEIGHT,
                },
            )

        source = configs.get(source_coord)
        if source is None:
            raise FrameDecodeError(
                "global signal source core is missing frame-derived core config",
                context={
                    "chip_id": CHIP_ID,
                    "thread_id": runtime.thread_id,
                    "source_x": source_coord[0],
                    "source_y": source_coord[1],
                },
            )

        paths.setdefault(source_coord, []).append(
            _build_control_path(runtime.thread_id, source_coord, source)
        )
    return paths


def _coord_from_offset(offset: dict[str, int]) -> tuple[int, int] | None:
    if not {"xy", "x", "y"}.issubset(offset):
        return None
    z = offset.get("xy", 0)
    return z + offset.get("x", 0), z + offset.get("y", 0)


def _build_control_path(
    thread_id: int,
    source_coord: tuple[int, int],
    source: _CoreConfigSource,
) -> ControlPathView:
    offset_z = sign_magnitude_to_int(source.config.get("test_core_xy", 0))
    offset_x = sign_magnitude_to_int(source.config.get("test_core_x", 0))
    offset_y = sign_magnitude_to_int(source.config.get("test_core_y", 0))
    points = _control_route_points(
        thread_id,
        source_coord,
        source,
        offset_z,
        offset_x,
        offset_y,
    )
    target = points[-1]
    if (target.x, target.y) != CONTROL_PATH_TARGET:
        raise FrameDecodeError(
            "global signal source control path does not target CPU",
            frame_index=source.config_word1_index,
            raw_frame=source.config_word1_raw,
            context={
                "chip_id": CHIP_ID,
                "thread_id": thread_id,
                "source_x": source_coord[0],
                "source_y": source_coord[1],
                "target_x": target.x,
                "target_y": target.y,
                "expected_x": CONTROL_PATH_TARGET[0],
                "expected_y": CONTROL_PATH_TARGET[1],
                "test_core_xy": offset_z,
                "test_core_x": offset_x,
                "test_core_y": offset_y,
                "test_core_y_low_frame_index": source.config_word2_index,
            },
        )
    return ControlPathView(
        thread_id=thread_id,
        target_x=target.x,
        target_y=target.y,
        offset_xy=offset_z,
        offset_x=offset_x,
        offset_y=offset_y,
        points=points,
        source_frame_index=source.config_word1_index,
    )


def _control_route_points(
    thread_id: int,
    source_coord: tuple[int, int],
    source: _CoreConfigSource,
    offset_z: int,
    offset_x: int,
    offset_y: int,
) -> list[RoutePointView]:
    x, y = source_coord
    points = [RoutePointView(x=x, y=y)]
    target = (x + offset_z + offset_x, y + offset_z + offset_y)
    for axis, steps in (("z", offset_z), ("x", offset_x), ("y", offset_y)):
        for step in range(abs(steps)):
            dx, dy = _axis_unit(axis, steps)
            x += dx
            y += dy
            if not _coord_in_grid(x, y):
                _raise_control_path_error(
                    source,
                    thread_id=thread_id,
                    source_coord=source_coord,
                    target=target,
                    axis=axis,
                    step=step + 1,
                    illegal=(x, y),
                    offsets=(offset_z, offset_x, offset_y),
                )
            points.append(RoutePointView(x=x, y=y))
    return points


def _axis_unit(axis: str, steps: int) -> tuple[int, int]:
    if steps == 0:
        return 0, 0
    sign = 1 if steps > 0 else -1
    if axis == "z":
        return sign, sign
    if axis == "x":
        return sign, 0
    return 0, sign


def _raise_control_path_error(
    source: _CoreConfigSource,
    *,
    thread_id: int,
    source_coord: tuple[int, int],
    target: tuple[int, int],
    axis: str,
    step: int,
    illegal: tuple[int, int],
    offsets: tuple[int, int, int],
) -> None:
    raise FrameDecodeError(
        "global signal source control path leaves chip grid",
        frame_index=source.config_word1_index,
        raw_frame=source.config_word1_raw,
        context={
            "chip_id": CHIP_ID,
            "thread_id": thread_id,
            "source_x": source_coord[0],
            "source_y": source_coord[1],
            "target_x": target[0],
            "target_y": target[1],
            "illegal_x": illegal[0],
            "illegal_y": illegal[1],
            "route_axis": axis,
            "route_step": step,
            "test_core_xy": offsets[0],
            "test_core_x": offsets[1],
            "test_core_y": offsets[2],
            "test_core_y_low_frame_index": source.config_word2_index,
            "grid_width": GRID_WIDTH,
            "grid_height": GRID_HEIGHT,
        },
    )


def _global_signal_dir_set(
    configs: dict[tuple[int, int], _CoreConfigSource],
    coord: tuple[int, int],
    key: str,
) -> set[str]:
    source = configs.get(coord)
    if source is None:
        return set()
    return set(global_signal_dirs(source.config.get(key, 0)))


def _raise_global_signal_error(
    message: str,
    source: _CoreConfigSource,
    *,
    owner: tuple[int, int],
    signal_kind: str,
    direction: str,
    signal_source: tuple[int, int],
    signal_target: tuple[int, int],
    expected_direction: str | None = None,
) -> None:
    context: dict[str, Any] = {
        "chip_id": CHIP_ID,
        "core_x": owner[0],
        "core_y": owner[1],
        "signal_kind": signal_kind,
        "direction": direction,
        "source_x": signal_source[0],
        "source_y": signal_source[1],
        "target_x": signal_target[0],
        "target_y": signal_target[1],
        "grid_width": GRID_WIDTH,
        "grid_height": GRID_HEIGHT,
    }
    if expected_direction is not None:
        context["expected_direction"] = expected_direction
    raise FrameDecodeError(
        message,
        frame_index=source.frame_index,
        raw_frame=source.raw_frame,
        context=context,
    )


def _validate_frame_content(
    resolved: dict[str, Any], frames: np.ndarray, validation: list[ValidationEntry]
) -> None:
    merged_npy = resolved.get("merged_npy")
    if merged_npy is not None:
        npy_frames = np.load(merged_npy).astype(np.uint64, copy=False)
        if len(npy_frames) != len(frames):
            validation.append(
                ValidationEntry(
                    severity="error",
                    code="merged_npy_frame_count_mismatch",
                    message=(
                        f"loaded frame count {len(frames)} does not match "
                        f"cfg_frames.npy count {len(npy_frames)}"
                    ),
                )
            )
        elif not np.array_equal(npy_frames, frames):
            validation.append(
                ValidationEntry(
                    severity="error",
                    code="merged_npy_content_mismatch",
                    message="loaded frame stream does not match cfg_frames.npy",
                )
            )


def _coord_in_grid(x: int, y: int) -> bool:
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT


def _source_label(has_frame: bool, has_metadata: bool) -> str:
    if has_frame and has_metadata:
        return "frame+metadata"
    if has_frame:
        return "frame"
    if has_metadata:
        return "metadata"
    return "layout"


def _thread_id(core_config: dict[str, int], metadata: dict[str, Any]) -> int | None:
    if "thread_number" in core_config:
        return core_config["thread_number"]
    if "thread_id" in metadata:
        return metadata["thread_id"]
    return None
