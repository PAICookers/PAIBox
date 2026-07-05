from collections.abc import Iterable, Iterator
from typing import Literal, TextIO

from paicorelib import FrameArrayType

from ..coreplacement import CorePlacement

LiteralFormat = Literal["bin", "hex"]
WordOrder = Literal["high_first", "low_first"]
TargetPlatform = Literal["x86", "riscv", "all"]
FrameTriplet = tuple[FrameArrayType, FrameArrayType | None, FrameArrayType | None]
FrameRecord = tuple[CorePlacement, FrameTriplet]
FrameRecords = tuple[FrameRecord, ...]


def make_frame_records(coreplacements: Iterable[CorePlacement]) -> FrameRecords:
    return tuple(
        (core_placement, core_placement.to_frame()) for core_placement in coreplacements
    )


def export_single_framearray(
    frame_array: FrameArrayType, file: TextIO, prefix: str = ""
) -> None:
    lines = []
    for frame in frame_array:
        h = f"{frame:016x}"
        lines.append(f"{prefix}{'_'.join(h[i : i + 4] for i in range(0, 16, 4))}")
    if lines:
        file.write("\n".join(lines) + "\n")


def export_framearray_to_bit(
    frame_array: FrameArrayType,
    file: TextIO,
    prefix: str = "",
    literal_format: LiteralFormat = "bin",
) -> None:
    if literal_format == "bin":
        h_fmt = "0b{:032b}"
        l_fmt = "0b{:032b}"
    elif literal_format == "hex":
        h_fmt = "0x{:08X}"
        l_fmt = "0x{:08X}"
    else:
        raise ValueError("literal_format must be 'bin' or 'hex'")

    lines = [
        f"{prefix}{h_fmt.format((f >> 32) & 0xFFFFFFFF)},{l_fmt.format(f & 0xFFFFFFFF)},"
        for f in frame_array
    ]
    if lines:
        file.write("\n".join(lines) + "\n")


def export_framearray_to_int32(
    frame_array: FrameArrayType, word_order: WordOrder = "high_first"
) -> list[int]:
    n = len(frame_array)
    result = [0] * (n * 2)
    for i, f in enumerate(frame_array):
        lo = f & 0xFFFFFFFF
        hi = (f >> 32) & 0xFFFFFFFF
        if word_order == "high_first":
            result[i * 2] = hi
            result[i * 2 + 1] = lo
        else:
            result[i * 2] = lo
            result[i * 2 + 1] = hi
    return result


def iter_frame_types(frames: FrameTriplet) -> Iterator[tuple[int, FrameArrayType]]:
    for idx, frame_array in enumerate(frames, start=1):
        if frame_array is not None:
            yield idx, frame_array


def iter_frame_arrays_core_major(
    frame_records: FrameRecords,
) -> Iterator[FrameArrayType]:
    for _, frames in frame_records:
        for _, frame_array in iter_frame_types(frames):
            yield frame_array


def resolve_platform_exports(
    target_platform: TargetPlatform, debug: bool
) -> tuple[bool, bool]:
    if target_platform not in ("x86", "riscv", "all"):
        raise ValueError("target_platform must be 'x86', 'riscv', or 'all'")
    return (
        debug or target_platform in ("x86", "all"),
        debug or target_platform in ("riscv", "all"),
    )
