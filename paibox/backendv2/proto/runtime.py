from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from paicorelib import AERPacketZXYCopy, CoordXY, CoordZXYOffset, OnlineFrameGenV2
from paicorelib.framelib.base import FramePackageHeaderV2
from paicorelib.framelib.frame_defs import (
    FFV2,
    FrameHeader,
    FramePackageType,
    OnlineConfigFrame1FormatV2,
)

from ..output_route_offsets import terminal_route_side
from .compile_artifacts_pb2 import (
    CompileArtifacts,
    ConfigFrames,
    DataType,
    InputEntry,
    InputTensorMapping,
    OutputTensorMapping,
)

FrameArray = NDArray[np.uint64]
_GroupKey = tuple[int, int, int, int, int, int, int]
_ONLINE_WF1_TS_AXON_OFFSET = 8
_ONLINE_WF1_TS_AXON_MASK = 0xFFFF
_ONLINE_CONFIG_PKG_NFRAMES = 5
_ONLINE_CORE_EXPORT_NFRAMES = 10
_ONLINE_ROUTE_COORD_MIN = 0
_ONLINE_ROUTE_COORD_MAX = 31
_CONFIG_PB_NAME = "config.pb"
_PROTO_DIRNAME = "proto"


@dataclass(frozen=True, slots=True)
class OutputEntryInfo:
    elem_idx: int
    copy_id: int
    bit_width: int
    dtype: int


@dataclass(slots=True)
class OutputMappingTable:
    shape: tuple[int, ...]
    target_lcn: int
    kind: int
    bit_width: int
    entries_by_axon: dict[int, OutputEntryInfo]
    boundary: OnlineOutputBoundaryInfo | None = None


@dataclass(frozen=True, slots=True)
class OnlineOutputBoundaryInfo:
    producer_core_offset: tuple[int, int, int]
    test_core_offset: tuple[int, int, int]
    target_coord: tuple[int, int]
    data_route_offset: tuple[int, int, int] | None
    control_ingress_side: tuple[str, int]
    data_ingress_side: tuple[str, int] | None
    work_mode: int
    output_core: int
    global_send: int
    global_receive: int
    package_type: int = int(FramePackageType.CONF_TESTOUT)


@dataclass(frozen=True, slots=True)
class OnlineOutputPackageHeaderInfo:
    frame_header: int
    core_offset: tuple[int, int, int]
    copy_count: tuple[int, int, int]
    start_addr: int
    package_type: int
    n_payload_frames: int


__all__ = [
    "decode_online_boundary_output_frames",
    "encode_online_boundary_output_frames",
    "encode_online_data_output_frames",
    "package_online_data_output_frames",
    "parse_online_data_output_package_header",
    "OnlineOutputBoundaryInfo",
    "OnlineOutputPackageHeaderInfo",
    "OutputEntryInfo",
    "OutputMappingTable",
    "build_output_mapping_tables",
    "decode_online_data_output_frames",
    "encode_online_input_frames",
    "iter_config_frame_u64",
    "load_compile_artifacts",
    "scatter_data_output",
    "strip_online_data_output_package_header",
]


def _resolve_compile_artifacts_path(pb_path: str | Path) -> Path:
    path = Path(pb_path)
    if path.is_dir():
        direct_pb = path / _CONFIG_PB_NAME
        if direct_pb.exists():
            return direct_pb

        nested_pb = path / _PROTO_DIRNAME / _CONFIG_PB_NAME
        if nested_pb.exists():
            return nested_pb

        raise FileNotFoundError(
            f"cannot find '{_CONFIG_PB_NAME}' under '{path}' or '{path / _PROTO_DIRNAME}'."
        )

    return path


def load_compile_artifacts(pb_path: str | Path) -> CompileArtifacts:
    artifacts = CompileArtifacts()
    artifacts.ParseFromString(_resolve_compile_artifacts_path(pb_path).read_bytes())
    return artifacts


def iter_config_frame_u64(config_frames: ConfigFrames) -> Iterable[int]:
    words = list(config_frames.words)
    if len(words) % 2 != 0:
        raise ValueError("config_frames.words must contain an even number of words.")

    for first, second in zip(words[0::2], words[1::2]):
        if config_frames.word_order == ConfigFrames.HIGH_FIRST:
            high32, low32 = first, second
        else:
            low32, high32 = first, second

        yield (int(high32) << 32) | int(low32)


def _mapping_numel(mapping: InputTensorMapping) -> int:
    return prod(mapping.shape.size) if mapping.shape.size else 1


def _entry_group_key(entry: InputEntry) -> _GroupKey:
    return (
        int(entry.core_offset.xy),
        int(entry.core_offset.x),
        int(entry.core_offset.y),
        int(entry.copy_count.xy),
        int(entry.copy_count.x),
        int(entry.copy_count.y),
        int(entry.target_lcn),
    )


def _mapping_bit_width(mapping: InputTensorMapping | OutputTensorMapping) -> int:
    if not mapping.HasField("bit_width"):
        raise ValueError(f"mapping '{mapping.name}' is missing bit_width.")
    return int(mapping.bit_width)


def _validate_online_entry(
    entry: InputEntry, mapping: InputTensorMapping, mapping_name: str
) -> None:
    if entry.dtype != DataType.FLOAT16:
        raise ValueError(
            f"online input mapping '{mapping_name}' requires FLOAT16 dtype, "
            f"got {DataType.Code.Name(entry.dtype)}."
        )

    if _mapping_bit_width(mapping) != 16:
        raise ValueError(
            f"online input mapping '{mapping_name}' requires 16-bit payload, "
            f"got {_mapping_bit_width(mapping)}."
        )


def _encode_online_input_mapping(
    mapping: InputTensorMapping, flat_data: np.ndarray
) -> FrameArray:
    groups: dict[_GroupKey, list[InputEntry]] = defaultdict(list)
    for entry in mapping.entries:
        _validate_online_entry(entry, mapping, mapping.name)
        groups[_entry_group_key(entry)].append(entry)

    if not groups:
        return np.array([], dtype=np.uint64)

    frame_parts: list[FrameArray] = []
    for key, entries in groups.items():
        core_xy, core_x, core_y, copy_xy, copy_x, copy_y, target_lcn = key
        ordered = sorted(entries, key=lambda item: int(item.elem_idx))
        elem_indices = [int(entry.elem_idx) for entry in ordered]
        timesteps = np.asarray(
            [int(entry.tick_relative) for entry in ordered], dtype=np.uint64
        )
        axons = np.asarray([int(entry.addr_axon) for entry in ordered], dtype=np.uint64)
        payload = flat_data[elem_indices]
        frame_parts.append(
            OnlineFrameGenV2.gen_work_frame1(
                CoordZXYOffset(core_xy, core_x, core_y),
                AERPacketZXYCopy(copy_xy, copy_x, copy_y),
                timesteps,
                axons,
                target_lcn,
                payload,
            )
        )

    return np.concatenate(frame_parts).astype(np.uint64, copy=False)


def encode_online_input_frames(
    artifacts: CompileArtifacts,
    input_name: str,
    data: ArrayLike,
    *,
    thread_id: int | None = None,
) -> FrameArray:
    flat_data = np.asarray(data).reshape(-1)
    frame_parts: list[FrameArray] = []
    mapping_found = False

    for thread in artifacts.io_mapping.threads:
        if thread_id is not None and int(thread.thread_id) != thread_id:
            continue

        for mapping in thread.input_mappings.items:
            if mapping.name != input_name:
                continue

            mapping_found = True
            expected_numel = _mapping_numel(mapping)
            if flat_data.size != expected_numel:
                raise ValueError(
                    f"online input '{input_name}' expects {expected_numel} elements, "
                    f"got {flat_data.size}."
                )

            frame_parts.append(_encode_online_input_mapping(mapping, flat_data))

    if not mapping_found:
        raise KeyError(f"online input mapping not found: {input_name}")

    if not frame_parts:
        return np.array([], dtype=np.uint64)

    return np.concatenate(frame_parts).astype(np.uint64, copy=False)


def _frame_header(frame: int) -> int:
    return (frame >> FFV2.GENERAL_HEADER_OFFSET) & FFV2.GENERAL_HEADER_MASK


def _frame_package_type(frame: int) -> int:
    payload = frame & FFV2.GENERAL_PAYLOAD_MASK
    return (
        payload >> FFV2.GENERAL_PACKAGE_TYPE_OFFSET
    ) & FFV2.GENERAL_PACKAGE_TYPE_MASK


def _frame_package_count(frame: int) -> int:
    return int(frame) & FFV2.GENERAL_PACKAGE_NUM_MASK


def _extract_bits(word: int, offset: int, mask: int) -> int:
    return (word >> offset) & mask


def _sign_magnitude_to_int(value: int, *, nbits: int = 6) -> int:
    sign_bit = 1 << (nbits - 1)
    magnitude = value & (sign_bit - 1)
    return -magnitude if value & sign_bit else magnitude


def _parse_online_output_boundary(
    chunk: np.ndarray,
    producer_core_offset: tuple[int, int, int],
) -> OnlineOutputBoundaryInfo:
    if chunk.shape != (_ONLINE_CORE_EXPORT_NFRAMES,):
        raise ValueError(
            "online core export chunk must contain "
            f"{_ONLINE_CORE_EXPORT_NFRAMES} frames, got {chunk.shape[0]}."
        )

    header = int(chunk[0])
    if _frame_header(header) != FrameHeader.CONFIG_TYPE1.value:
        raise ValueError("online core export chunk must start with CONFIG_TYPE1.")
    if _frame_package_type(header) != FramePackageType.CONF_TESTOUT.value:
        raise ValueError(
            "online core export chunk must use CONF_TESTOUT package headers."
        )
    if _frame_package_count(header) != _ONLINE_CONFIG_PKG_NFRAMES:
        raise ValueError(
            "online core export chunk must contain one CONFIG_TYPE1 package with "
            f"{_ONLINE_CONFIG_PKG_NFRAMES} payload frames."
        )

    expected_ctrl_headers = (
        FrameHeader.CTRL_TYPE1.value,
        FrameHeader.CTRL_TYPE2.value,
        FrameHeader.CTRL_TYPE3.value,
        FrameHeader.CTRL_TYPE4.value,
    )
    actual_ctrl_headers = tuple(_frame_header(int(frame)) for frame in chunk[6:10])
    if actual_ctrl_headers != expected_ctrl_headers:
        raise ValueError(
            "online core export chunk control-frame sequence must be CTRL_TYPE1..4, "
            f"got {actual_ctrl_headers}."
        )

    w1, _, w3, w4, _ = (int(word) for word in chunk[1:6])
    frame_format = OnlineConfigFrame1FormatV2

    test_core_y = (
        _extract_bits(
            w3,
            frame_format.Word3.TEST_CORE_Y_HIGH1_OFFSET,
            frame_format.Word3.TEST_CORE_Y_HIGH1_MASK,
        )
        << 5
    ) | _extract_bits(
        w4,
        frame_format.Word4.TEST_CORE_Y_LOW5_OFFSET,
        frame_format.Word4.TEST_CORE_Y_LOW5_MASK,
    )
    test_core_offset = (
        _sign_magnitude_to_int(
            _extract_bits(
                w3,
                frame_format.Word3.TEST_CORE_XY_OFFSET,
                frame_format.Word3.TEST_CORE_XY_MASK,
            )
        ),
        _sign_magnitude_to_int(
            _extract_bits(
                w3,
                frame_format.Word3.TEST_CORE_X_OFFSET,
                frame_format.Word3.TEST_CORE_X_MASK,
            )
        ),
        _sign_magnitude_to_int(test_core_y),
    )
    producer_coord = _coord_from_offset_tuple(producer_core_offset)
    target_coord = _coord_after_offset(producer_coord, test_core_offset)
    control_ingress_side = _route_side_tuple(test_core_offset)
    data_route_offset = _select_output_route_offset_for_side(
        producer_coord,
        target_coord,
        control_ingress_side,
    )

    return OnlineOutputBoundaryInfo(
        producer_core_offset=producer_core_offset,
        test_core_offset=test_core_offset,
        target_coord=(target_coord.x, target_coord.y),
        data_route_offset=data_route_offset,
        control_ingress_side=control_ingress_side,
        data_ingress_side=(
            _route_side_tuple(data_route_offset)
            if data_route_offset is not None
            else None
        ),
        work_mode=_extract_bits(
            w1,
            frame_format.Word1.WORK_MODE_OFFSET,
            frame_format.Word1.WORK_MODE_MASK,
        ),
        output_core=_extract_bits(
            w1,
            frame_format.Word1.OUTPUT_CORE_OFFSET,
            frame_format.Word1.OUTPUT_CORE_MASK,
        ),
        global_send=_extract_bits(
            w4,
            frame_format.Word4.GLOBAL_SEND_OFFSET,
            frame_format.Word4.GLOBAL_SEND_MASK,
        ),
        global_receive=_extract_bits(
            w4,
            frame_format.Word4.GLOBAL_RECEIVE_OFFSET,
            frame_format.Word4.GLOBAL_RECEIVE_MASK,
        ),
    )


def _coord_from_offset_tuple(offset: tuple[int, int, int]) -> CoordXY:
    return CoordXY(offset[0] + offset[1], offset[0] + offset[2])


def _coord_after_offset(start_coord: CoordXY, offset: tuple[int, int, int]) -> CoordXY:
    return CoordXY(
        start_coord.x + offset[0] + offset[1],
        start_coord.y + offset[0] + offset[2],
    )


def _route_side_tuple(offset: tuple[int, int, int]) -> tuple[str, int]:
    side = terminal_route_side(CoordZXYOffset(*offset))
    return side[0], side[1]


def _select_output_route_offset_for_side(
    producer_coord: CoordXY,
    target_coord: CoordXY,
    control_ingress_side: tuple[str, int],
) -> tuple[int, int, int] | None:
    dcoord = target_coord - producer_coord
    for z in range(-31, 32):
        x = dcoord.x - z
        y = dcoord.y - z
        if not (-31 <= x <= 31 and -31 <= y <= 31):
            continue

        offset = CoordZXYOffset(z, x, y)
        if terminal_route_side(offset) != control_ingress_side:
            continue
        if not _route_stays_in_online_grid(producer_coord, offset, target_coord):
            continue

        return int(offset.z), int(offset.x), int(offset.y)
    return None


def _route_stays_in_online_grid(
    start_coord: CoordXY, offset: CoordZXYOffset, target_coord: CoordXY
) -> bool:
    x, y = start_coord.x, start_coord.y

    def walk(step_count: int, dx: int, dy: int) -> bool:
        nonlocal x, y
        for _ in range(step_count):
            x += dx
            y += dy
            if not (
                _ONLINE_ROUTE_COORD_MIN <= x <= _ONLINE_ROUTE_COORD_MAX
                and _ONLINE_ROUTE_COORD_MIN <= y <= _ONLINE_ROUTE_COORD_MAX
            ):
                return False
        return True

    if offset.z != 0 and not walk(
        abs(offset.z), 1 if offset.z > 0 else -1, 1 if offset.z > 0 else -1
    ):
        return False
    if offset.x != 0 and not walk(abs(offset.x), 1 if offset.x > 0 else -1, 0):
        return False
    if offset.y != 0 and not walk(abs(offset.y), 0, 1 if offset.y > 0 else -1):
        return False

    return x == target_coord.x and y == target_coord.y


def _online_output_boundaries_by_name(
    artifacts: CompileArtifacts, *, thread_id: int | None = None
) -> dict[str, OnlineOutputBoundaryInfo]:
    frames = np.fromiter(
        iter_config_frame_u64(artifacts.config_frames), dtype=np.uint64
    )
    total_core_ticks = sum(
        len(thread.core_ticks) for thread in artifacts.io_mapping.threads
    )
    expected_frames = total_core_ticks * _ONLINE_CORE_EXPORT_NFRAMES
    if total_core_ticks == 0 or frames.size != expected_frames:
        return {}

    boundaries: dict[str, OnlineOutputBoundaryInfo] = {}
    offset = 0
    for thread in artifacts.io_mapping.threads:
        n_core = len(thread.core_ticks)
        thread_frames = frames[offset : offset + n_core * _ONLINE_CORE_EXPORT_NFRAMES]
        offset += n_core * _ONLINE_CORE_EXPORT_NFRAMES

        if thread_id is not None and int(thread.thread_id) != thread_id:
            continue

        chunks = thread_frames.reshape(n_core, _ONLINE_CORE_EXPORT_NFRAMES)
        for core_tick, chunk in zip(thread.core_ticks, chunks):
            boundary = _parse_online_output_boundary(
                chunk,
                (
                    int(core_tick.core_offset.xy),
                    int(core_tick.core_offset.x),
                    int(core_tick.core_offset.y),
                ),
            )
            for node_name in core_tick.nodes:
                boundaries[node_name] = boundary

    return boundaries


def build_output_mapping_tables(
    artifacts: CompileArtifacts, *, thread_id: int | None = None
) -> dict[str, OutputMappingTable]:
    tables: dict[str, OutputMappingTable] = {}
    boundaries = _online_output_boundaries_by_name(artifacts, thread_id=thread_id)

    for thread in artifacts.io_mapping.threads:
        if thread_id is not None and int(thread.thread_id) != thread_id:
            continue

        target_lcn = int(thread.output_mappings.target_lcn)
        for mapping in thread.output_mappings.items:
            if mapping.name in tables:
                raise ValueError(f"duplicate output mapping name: {mapping.name}")

            bit_width = _mapping_bit_width(mapping)
            entries_by_axon: dict[int, OutputEntryInfo] = {}
            for entry in mapping.entries:
                axon_bit_idx = int(entry.axon_bit_idx)
                if axon_bit_idx in entries_by_axon:
                    raise ValueError(
                        f"output mapping '{mapping.name}' has duplicate axon_bit_idx "
                        f"{axon_bit_idx}."
                    )

                dtype = (
                    int(entry.dtype) if entry.HasField("dtype") else DataType.NOT_SET
                )
                entries_by_axon[axon_bit_idx] = OutputEntryInfo(
                    int(entry.elem_idx),
                    int(entry.copy_id),
                    bit_width,
                    dtype,
                )

            kind = (
                int(mapping.kind)
                if mapping.HasField("kind")
                else OutputTensorMapping.DATA
            )
            tables[mapping.name] = OutputMappingTable(
                tuple(mapping.shape.size),
                target_lcn,
                kind,
                bit_width,
                entries_by_axon,
                boundary=boundaries.get(mapping.name),
            )

    return tables


def _require_data_output_table(
    output_tables: dict[str, OutputMappingTable], output_name: str
) -> OutputMappingTable:
    if output_name not in output_tables:
        raise KeyError(f"output mapping not found: {output_name}")

    table = output_tables[output_name]
    if table.kind != OutputTensorMapping.DATA:
        raise ValueError(f"output mapping '{output_name}' is not a DATA output.")

    return table


def _core_offset_value(
    value: tuple[int, int, int] | None, fallback: tuple[int, int, int]
) -> tuple[int, int, int]:
    return fallback if value is None else value


def _normalize_tick_relatives(
    tick_relatives: int | ArrayLike, count: int, output_name: str
) -> np.ndarray:
    if isinstance(tick_relatives, int):
        return np.full(count, tick_relatives, dtype=np.uint64)

    arr = np.asarray(tick_relatives, dtype=np.uint64).reshape(-1)
    if arr.size != count:
        raise ValueError(
            f"output mapping '{output_name}' expects {count} tick_relative values, "
            f"got {arr.size}."
        )
    return arr


def _require_single_data_dtype(
    table: OutputMappingTable, output_name: str
) -> int | None:
    dtype_codes = {entry.dtype for entry in table.entries_by_axon.values()}
    if not dtype_codes:
        return None
    if len(dtype_codes) != 1:
        ordered = [DataType.Code.Name(dtype_code) for dtype_code in sorted(dtype_codes)]
        raise ValueError(
            f"output mapping '{output_name}' contains mixed DATA dtypes: {ordered}."
        )

    return next(iter(dtype_codes))


def _data_dtype_to_numpy(dtype_code: int, output_name: str) -> np.dtype:
    if dtype_code == DataType.FLOAT16:
        return np.dtype(np.float16)

    if dtype_code in (
        DataType.UINT1,
        DataType.UINT2,
        DataType.UINT4,
        DataType.UINT8,
    ):
        return np.dtype(np.uint8)

    if dtype_code in (DataType.INT1, DataType.INT2, DataType.INT4, DataType.INT8):
        return np.dtype(np.int8)

    raise ValueError(
        f"output mapping '{output_name}' has unsupported DATA dtype "
        f"{DataType.Code.Name(dtype_code)}."
    )


def _require_output_boundary_route(
    table: OutputMappingTable,
    output_name: str,
    route_kind: Literal["data", "control"],
) -> tuple[int, int, int]:
    boundary = table.boundary
    if boundary is None:
        raise ValueError(f"output mapping '{output_name}' has no boundary metadata.")

    if route_kind == "control":
        return boundary.test_core_offset

    if boundary.data_route_offset is None:
        raise ValueError(
            f"output mapping '{output_name}' has no aligned data-route offset for "
            "the current output boundary."
        )

    return boundary.data_route_offset


def _parse_signed_triplet(
    frame: int, *, base_offset: int, x_offset: int, y_offset: int
) -> tuple[int, int, int]:
    return (
        _sign_magnitude_to_int(
            _extract_bits(frame, base_offset, FFV2.GENERAL_CORE_XY_ADDR_MASK)
        ),
        _sign_magnitude_to_int(
            _extract_bits(frame, x_offset, FFV2.GENERAL_CORE_X_ADDR_MASK)
        ),
        _sign_magnitude_to_int(
            _extract_bits(frame, y_offset, FFV2.GENERAL_CORE_Y_ADDR_MASK)
        ),
    )


def encode_online_data_output_frames(
    output_tables: dict[str, OutputMappingTable],
    output_name: str,
    data: ArrayLike,
    *,
    tick_relatives: int | ArrayLike = 0,
    core_offset: tuple[int, int, int] | None = None,
    copy_count: tuple[int, int, int] = (0, 0, 0),
    packaged: bool = False,
) -> FrameArray:
    table = _require_data_output_table(output_tables, output_name)
    dtype_code = _require_single_data_dtype(table, output_name)
    if dtype_code is None:
        raise ValueError(f"output mapping '{output_name}' has no DATA dtype.")

    flat_data = np.asarray(
        data,
        dtype=_data_dtype_to_numpy(dtype_code, output_name),
    ).reshape(-1)
    expected_numel = prod(table.shape) if table.shape else 1
    if flat_data.size != expected_numel:
        raise ValueError(
            f"output mapping '{output_name}' expects {expected_numel} elements, "
            f"got {flat_data.size}."
        )

    ordered_entries = sorted(table.entries_by_axon.items())
    axon_indices = np.asarray(
        [axon_bit_idx for axon_bit_idx, _ in ordered_entries], dtype=np.uint64
    )
    payload = flat_data[[entry.elem_idx for _, entry in ordered_entries]]
    ticks = _normalize_tick_relatives(tick_relatives, len(ordered_entries), output_name)
    boundary = table.boundary
    packet_core_offset = _core_offset_value(
        core_offset,
        boundary.producer_core_offset if boundary is not None else (0, 0, 0),
    )
    frames = OnlineFrameGenV2.gen_work_frame1(
        CoordZXYOffset(*packet_core_offset),
        AERPacketZXYCopy(*copy_count),
        ticks,
        axon_indices,
        table.target_lcn,
        payload,
    ).astype(np.uint64, copy=False)

    if packaged:
        return package_online_data_output_frames(
            frames,
            core_offset=packet_core_offset,
            copy_count=copy_count,
        )

    return frames


def encode_online_boundary_output_frames(
    output_tables: dict[str, OutputMappingTable],
    output_name: str,
    data: ArrayLike,
    *,
    route_kind: Literal["data", "control"] = "data",
    tick_relatives: int | ArrayLike = 0,
    copy_count: tuple[int, int, int] = (0, 0, 0),
    packaged: bool = False,
) -> FrameArray:
    table = _require_data_output_table(output_tables, output_name)
    return encode_online_data_output_frames(
        output_tables,
        output_name,
        data,
        tick_relatives=tick_relatives,
        core_offset=_require_output_boundary_route(table, output_name, route_kind),
        copy_count=copy_count,
        packaged=packaged,
    )


def package_online_data_output_frames(
    frames: ArrayLike,
    *,
    core_offset: tuple[int, int, int] = (0, 0, 0),
    copy_count: tuple[int, int, int] = (0, 0, 0),
    n_package: int | None = None,
) -> FrameArray:
    frame_array = np.asarray(frames, dtype=np.uint64).reshape(-1)
    header = FramePackageHeaderV2.make_pkg_header(
        FrameHeader.WORK_TYPE1,
        CoordZXYOffset(*core_offset),
        AERPacketZXYCopy(*copy_count),
        0,
        FramePackageType.CONF_TESTOUT,
        frame_array.size if n_package is None else n_package,
    ).value.astype(np.uint64, copy=False)
    return np.concatenate([header, frame_array]).astype(np.uint64, copy=False)


def _decode_output_payload(
    payload_bytes: list[int], entry: OutputEntryInfo, output_name: str
) -> float | int | np.generic:
    payload = np.asarray(payload_bytes, dtype=np.uint8)
    if entry.dtype == DataType.FLOAT16:
        if payload.size != 2:
            raise ValueError(
                f"output mapping '{output_name}' requires 2 byte lanes for FLOAT16, "
                f"got {payload.size}."
            )
        return payload.view(np.float16)[0]

    value = int(payload[0]) & ((1 << entry.bit_width) - 1)
    if entry.dtype in (
        DataType.UINT1,
        DataType.UINT2,
        DataType.UINT4,
        DataType.UINT8,
    ):
        return value

    if entry.dtype in (DataType.INT1, DataType.INT2, DataType.INT4, DataType.INT8):
        sign_bit = 1 << (entry.bit_width - 1)
        if value & sign_bit:
            value -= 1 << entry.bit_width
        return value

    raise ValueError(
        f"output mapping '{output_name}' has unsupported DATA dtype "
        f"{DataType.Code.Name(entry.dtype)}."
    )


def parse_online_data_output_package_header(
    frames: ArrayLike,
) -> OnlineOutputPackageHeaderInfo:
    frame_array = np.asarray(frames, dtype=np.uint64).reshape(-1)
    if frame_array.size == 0:
        raise ValueError("online output package header requires at least one frame.")

    header = int(frame_array[0])
    frame_type = _frame_header(header)
    if frame_type != FrameHeader.WORK_TYPE1.value:
        raise ValueError(
            "online output package header requires WORK_TYPE1 frame header."
        )

    payload = header & FFV2.GENERAL_PAYLOAD_MASK
    package_type = (
        payload >> FFV2.GENERAL_PACKAGE_TYPE_OFFSET
    ) & FFV2.GENERAL_PACKAGE_TYPE_MASK
    if package_type != FramePackageType.CONF_TESTOUT.value:
        raise ValueError(
            "online output package header requires CONF_TESTOUT package type."
        )

    expected_nframes = payload & FFV2.GENERAL_PACKAGE_NUM_MASK
    actual_nframes = frame_array.size - 1
    if expected_nframes != actual_nframes:
        raise ValueError(
            f"online output package header expects {expected_nframes} payload "
            f"frames, got {actual_nframes}."
        )

    return OnlineOutputPackageHeaderInfo(
        frame_header=frame_type,
        core_offset=_parse_signed_triplet(
            header,
            base_offset=FFV2.GENERAL_CORE_XY_ADDR_OFFSET,
            x_offset=FFV2.GENERAL_CORE_X_ADDR_OFFSET,
            y_offset=FFV2.GENERAL_CORE_Y_ADDR_OFFSET,
        ),
        copy_count=_parse_signed_triplet(
            header,
            base_offset=FFV2.GENERAL_COPY_XY_ADDR_OFFSET,
            x_offset=FFV2.GENERAL_COPY_X_ADDR_OFFSET,
            y_offset=FFV2.GENERAL_COPY_Y_ADDR_OFFSET,
        ),
        start_addr=_extract_bits(
            payload,
            FFV2.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET,
            FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK,
        ),
        package_type=package_type,
        n_payload_frames=expected_nframes,
    )


def strip_online_data_output_package_header(frames: ArrayLike) -> FrameArray:
    frame_array = np.asarray(frames, dtype=np.uint64).reshape(-1)
    if frame_array.size == 0:
        return frame_array

    parse_online_data_output_package_header(frame_array)

    return frame_array[1:]


def _decode_online_data_output_frame_array(
    table: OutputMappingTable, output_name: str, frame_array: np.ndarray
) -> list[tuple[int, float | int | np.generic]]:
    pending: dict[int, list[int]] = defaultdict(list)
    decoded_items: list[tuple[int, float | int | np.generic]] = []

    for frame in frame_array:
        axon_bit_idx = (
            int(frame) >> _ONLINE_WF1_TS_AXON_OFFSET
        ) & _ONLINE_WF1_TS_AXON_MASK
        if axon_bit_idx not in table.entries_by_axon:
            continue

        entry = table.entries_by_axon[axon_bit_idx]
        expected_nbytes = max(1, (entry.bit_width + 7) // 8)
        lane_bytes = pending[axon_bit_idx]
        lane_bytes.append(int(frame) & 0xFF)
        if len(lane_bytes) == expected_nbytes:
            decoded_items.append(
                (axon_bit_idx, _decode_output_payload(lane_bytes, entry, output_name))
            )
            pending.pop(axon_bit_idx)

    if pending:
        waiting = ", ".join(str(axon_bit_idx) for axon_bit_idx in sorted(pending))
        raise ValueError(
            f"output mapping '{output_name}' has incomplete byte lanes for axon_bit_idx "
            f"{waiting}."
        )

    return decoded_items


def _validate_online_boundary_output_package_header(
    table: OutputMappingTable,
    output_name: str,
    header: OnlineOutputPackageHeaderInfo,
    route_kind: Literal["data", "control"],
) -> None:
    expected_core_offset = _require_output_boundary_route(
        table, output_name, route_kind
    )
    if header.core_offset != expected_core_offset:
        raise ValueError(
            f"output mapping '{output_name}' expects boundary route '{route_kind}' "
            f"package core_offset {expected_core_offset}, got {header.core_offset}."
        )


def decode_online_data_output_frames(
    output_tables: dict[str, OutputMappingTable],
    output_name: str,
    frames: ArrayLike,
    *,
    packaged: bool = False,
) -> list[tuple[int, float | int | np.generic]]:
    table = _require_data_output_table(output_tables, output_name)
    _require_single_data_dtype(table, output_name)
    frame_array = (
        strip_online_data_output_package_header(frames)
        if packaged
        else np.asarray(frames, dtype=np.uint64).reshape(-1)
    )
    return _decode_online_data_output_frame_array(table, output_name, frame_array)


def decode_online_boundary_output_frames(
    output_tables: dict[str, OutputMappingTable],
    output_name: str,
    frames: ArrayLike,
    *,
    route_kind: Literal["data", "control"] = "data",
) -> list[tuple[int, float | int | np.generic]]:
    table = _require_data_output_table(output_tables, output_name)
    _require_single_data_dtype(table, output_name)
    header = parse_online_data_output_package_header(frames)
    _validate_online_boundary_output_package_header(
        table,
        output_name,
        header,
        route_kind,
    )
    return _decode_online_data_output_frame_array(
        table,
        output_name,
        np.asarray(frames, dtype=np.uint64).reshape(-1)[1:],
    )


def scatter_data_output(
    output_tables: dict[str, OutputMappingTable],
    output_name: str,
    decoded_items: Iterable[tuple[int, float | int | np.generic]],
) -> np.ndarray:
    table = _require_data_output_table(output_tables, output_name)
    dtype_code = _require_single_data_dtype(table, output_name)
    if dtype_code is None:
        return np.zeros(table.shape or (), dtype=np.float16)

    flat = np.zeros(
        prod(table.shape) if table.shape else 1,
        dtype=_data_dtype_to_numpy(dtype_code, output_name),
    )
    for axon_bit_idx, payload in decoded_items:
        if axon_bit_idx not in table.entries_by_axon:
            continue
        flat[table.entries_by_axon[axon_bit_idx].elem_idx] = payload

    return flat.reshape(table.shape or ())
