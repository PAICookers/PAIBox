"""C header file generation for embedded frame configuration arrays.

Each frame array is emitted as a volatile unsigned int array in the
``.large_const_data`` section, callable from C firmware on the chip.
"""

import string
from typing import TextIO

C_ARRAY_DECL = string.Template(
    'volatile uint32_t $name[] __attribute__((section("$section"))) ={\n'
)
C_ARRAY_CLOSE = "};\n"


def write_c_array_decl(
    file: TextIO,
    name: str,
    section: str = ".large_const_data",
) -> None:
    """Write the opening line of a C array declaration."""
    file.write(C_ARRAY_DECL.substitute(name=name, section=section))


def write_c_array_close(file: TextIO) -> None:
    """Write the closing brace of a C array."""
    file.write(C_ARRAY_CLOSE)
