from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

from .utils import (
    FrameRecords,
    LiteralFormat,
    export_framearray_to_bit,
    iter_frame_arrays_core_major,
    iter_frame_types,
)

# Each frame array is emitted as a volatile unsigned int array in the
# ``.large_const_data`` section, callable from C firmware on the chip.

C_ARRAY_CLOSE = "};\n"


def _normalize_guard_name(file: TextIO) -> str:
    name = getattr(file, "name", "FRAME_HEADER")
    stem = Path(name).name.upper()
    chars = [ch if ch.isalnum() else "_" for ch in stem]
    return f"_{''.join(chars).removesuffix('_H')}_H"


def write_c_array_decl(
    file: TextIO, name: str, section: str = ".large_const_data"
) -> None:
    """Write the header prologue and opening line of a C array declaration."""
    guard = _normalize_guard_name(file)
    file.write(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")
    file.write(
        f'volatile uint32_t {name}[] __attribute__((section("{section}"))) ={{\n'
    )


def write_c_array_close(file: TextIO) -> None:
    """Write the closing brace of a C array and the header epilogue."""
    guard = _normalize_guard_name(file)
    file.write(C_ARRAY_CLOSE)
    file.write(f"#endif /* {guard} */\n")


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
