from pathlib import Path

import numpy as np
from paicorelib import FrameArrayType

from .utils import FrameRecords, iter_frame_types


def _concat_frames(parts: list[FrameArrayType]) -> np.ndarray:
    return np.concatenate([np.asarray(part, dtype="<u8") for part in parts])


def export_frame_npy(
    output_path: str | Path,
    frame_records: FrameRecords,
    export_merged_frames: bool = True,
) -> None:
    """Export frame arrays as explicit little-endian uint64 NumPy files."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    typed_parts: list[list[FrameArrayType]] = [[], [], []]
    merged_parts: list[FrameArrayType] = []
    for _, frames in frame_records:
        for idx, frame_array in iter_frame_types(frames):
            typed_parts[idx - 1].append(frame_array)
            if export_merged_frames:
                merged_parts.append(frame_array)

    for idx, parts in enumerate(typed_parts, start=1):
        if parts:
            np.save(out / f"cfg_frame{idx}.npy", _concat_frames(parts))

    if export_merged_frames and merged_parts:
        np.save(out / "cfg_frames.npy", _concat_frames(merged_parts))
