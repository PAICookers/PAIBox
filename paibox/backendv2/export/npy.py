from pathlib import Path

import numpy as np
from paicorelib import FrameArrayType

from .utils import FrameRecords, iter_frame_arrays_core_major, iter_frame_types


def export_frame_npy(
    output_path: str | Path,
    frame_records: FrameRecords,
    export_merged_frames: bool = True,
) -> None:
    """Export frame arrays as explicit little-endian uint64 NumPy files."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    typed_parts: list[list[FrameArrayType]] = [[], [], []]
    typed_counts = [0, 0, 0]
    merged_parts: list[FrameArrayType] | None = [] if export_merged_frames else None
    merged_count = 0
    for _, frames in frame_records:
        for idx, frame_array in iter_frame_types(frames):
            typed_parts[idx - 1].append(frame_array)
            typed_counts[idx - 1] += len(frame_array)
    if merged_parts is not None:
        merged_parts.extend(iter_frame_arrays_core_major(frame_records))
        merged_count = sum(len(frame_array) for frame_array in merged_parts)

    for idx, parts in enumerate(typed_parts, start=1):
        if parts:
            frame_array = np.zeros(typed_counts[idx - 1], dtype="<u8")
            cursor = 0
            for part in parts:
                n = len(part)
                frame_array[cursor : cursor + n] = part
                cursor += n
            np.save(out / f"cfg_frame{idx}.npy", frame_array)

    if merged_parts:
        frame_array = np.zeros(merged_count, dtype="<u8")
        cursor = 0
        for part in merged_parts:
            n = len(part)
            frame_array[cursor : cursor + n] = part
            cursor += n
        np.save(out / "cfg_frames.npy", frame_array)
