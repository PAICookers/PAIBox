"""C header file generation for embedded frame configuration arrays.

Each frame array is emitted as a volatile unsigned int array in the
``.large_const_data`` section, callable from C firmware on the chip.
"""

from pathlib import Path
from typing import TextIO

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
