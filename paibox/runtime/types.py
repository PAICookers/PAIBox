import ast
import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from paicorelib.coordinate import Coord, CoordTuple

ChipCoordStr = str
CoordStr = ChipCoordStr
NodeName = str


def coordstr_to_tuple(coord_str: CoordStr) -> CoordTuple:
    try:
        return ast.literal_eval(coord_str)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid coord_str: {coord_str}") from e


class NeuSegAddrKeys(TypedDict):
    n_neuron: int
    """The number of neurons in the segment."""
    addr_offset: int
    """The address offset at which the segment starts in the neuron address space."""
    interval: int
    """The number of times the neuron in the segment is repeated."""
    idx_offset: int
    """The offset of the starting address of this neuron corresponding to the neuron node   \
        in which it is located."""


NeuPhyLoc = dict[ChipCoordStr, dict[CoordStr, list[NeuSegAddrKeys]]]
CoreNeuSegLocType = dict[Coord, NeuSegAddrKeys]

import paicorelib.framelib.types as ftypes

if hasattr(ftypes, "DestInfoKeys"):
    InputProjInfoKeys = ftypes.DestInfoKeys  # type: ignore
    OutputDestInfoKeys = ftypes.DestInfoKeys  # type: ignore
else:

    class InputProjInfoKeys(TypedDict):
        addr_chip_x: int
        addr_chip_y: int
        addr_core_x: int
        addr_core_y: int
        addr_core_x_ex: int
        addr_core_y_ex: int
        tick_relative: list[int]
        addr_axon: list[int]

    OutputDestInfoKeys = InputProjInfoKeys

del ftypes

InputProjInfo = dict[NodeName, InputProjInfoKeys]
OutputDestInfo = dict[NodeName, dict[CoordStr, OutputDestInfoKeys]]
