from enum import Enum, auto

from ._namespace import _IRNamespace


class PAIIR:
    _ir_namespace = _IRNamespace()

    def __init__(self) -> None:
        self.name = self._ir_namespace.create_name(self)


class OpLoc(Enum):
    OFFLINE_CORE = auto()
    ONLINE_CORE = auto()
    CPU = auto()


class ViewIR(PAIIR):
    pass


class InputIR(ViewIR):
    """
    Represents the placeholder.
    """


class OutputIR(ViewIR):
    """
    Represents the output node.
    """
