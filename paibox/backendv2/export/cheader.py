from contextlib import ExitStack
from pathlib import Path

from ..frame_cheader import write_c_array_close, write_c_array_decl
from .utils import (
    FrameRecords,
    LiteralFormat,
    export_framearray_to_bit,
    iter_frame_arrays_core_major,
    iter_frame_types,
)


def export_cheader_files(
    output_path: str | Path,
    frame_records: FrameRecords,
    literal_format: LiteralFormat = "bin",
) -> None:
    """Export per-type config frames as C header arrays."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        (out / "cfg_frame1.h", "config_frame1"),
        (out / "cfg_frame2.h", "config_frame2"),
        (out / "cfg_frame3.h", "config_frame3"),
    ]
    with ExitStack() as stack:
        files = [stack.enter_context(p.open("w")) for p, _ in paths]
        for f, (_, name) in zip(files, paths):
            write_c_array_decl(f, name)
        for _, frames in frame_records:
            for idx, frame_array in iter_frame_types(frames):
                export_framearray_to_bit(
                    frame_array, files[idx - 1], "\t", literal_format
                )
        for f in files:
            write_c_array_close(f)


def export_cheader_merged(
    output_path: str | Path,
    frame_records: FrameRecords,
    literal_format: LiteralFormat = "bin",
) -> None:
    """Export all config frame types into one merged C header array."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    frame_path = out / "cfg_frames.h"
    with frame_path.open("w") as frame_file:
        write_c_array_decl(frame_file, "config_frame")
        for frame_array in iter_frame_arrays_core_major(frame_records):
            export_framearray_to_bit(frame_array, frame_file, "\t", literal_format)
        write_c_array_close(frame_file)
