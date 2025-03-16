import sys
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Optional

import numpy as np

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from paicorelib import WeightWidth as WW

from .collector import Collector
from .mixin import ReceiveInputProj, StatusMemory, TimeRelatedNode
from .naming import get_unique_name, is_name_unique
from .node import NodeDict, NodeList
from .types import WeightType
from .utils import arg_check_non_neg, arg_check_pos

__all__ = []


_IdPathType: TypeAlias = tuple[int, int]


class PAIBoxObject:
    __avoid_name_conflict__: ClassVar[bool] = False

    def __init__(self, name: Optional[str] = None) -> None:
        self._name = self._unique_name(name)

    def __eq__(self, other: "PAIBoxObject") -> bool:
        if not isinstance(other, PAIBoxObject):
            raise TypeError(
                f"cannot compare {type(self).__name__} with {type(other).__name__}."
            )

        if self is other:
            return True

        return type(self) == type(other) and self._name == other._name

    def __hash__(self) -> int:
        return hash((type(self), self._name))

    def _unique_name(
        self, name: Optional[str] = None, _type: Optional[str] = None
    ) -> str:
        if name is None:
            if _type is None:
                __type = self.__class__.__name__
            else:
                __type = _type

            return get_unique_name(__type)

        is_name_unique(name, self, self.__avoid_name_conflict__)
        return name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = self._unique_name(name)

    def nodes(
        self,
        method: Literal["absolute", "relative"] = "absolute",
        level: int = -1,
        include_self: bool = False,
    ) -> Collector[str, "PAIBoxObject"]:
        """Collect all child nodes.

        Args:
            - method: the method to find the nodes.
                - "absolute": the name of the node it is looking for will be `v.name`.
                - "relative": the name will be its attribute name, `x` in `self.x = v`.
            - level: the level at which the search ends. The default value is -1, which indicates   \
                that all levels will be searched.
            - include_self: whether to include the current node itself.
        """
        return self._find_nodes(method, level, include_self)

    def _find_nodes(
        self,
        method: Literal["absolute", "relative"] = "absolute",
        level: int = -1,
        include_self: bool = False,
        _lid: int = 0,
        _paths: Optional[set[_IdPathType]] = None,
    ) -> Collector[str, "PAIBoxObject"]:
        if _paths is None:
            _paths = set()

        gather = Collector()

        if include_self:
            if method == "absolute":
                gather[self.name] = self
            else:
                gather[""] = self

        if (level > -1) and (_lid >= level):
            return gather

        from .simulator import Probe

        def _find_nodes_absolute() -> None:
            for v in self.__dict__.values():
                if isinstance(v, PAIBoxObject):
                    _add_node2(self, v, _paths, gather, nodes)
                elif isinstance(v, NodeList):
                    for v2 in v:
                        if isinstance(v2, PAIBoxObject):
                            _add_node2(self, v2, _paths, gather, nodes)
                elif isinstance(v, NodeDict):
                    for v2 in v.values():
                        if isinstance(v2, PAIBoxObject):
                            _add_node2(self, v2, _paths, gather, nodes)

            # finding nodes recursively
            for v in nodes:
                gather.update(
                    v._find_nodes(
                        method=method,
                        level=level,
                        include_self=include_self,
                        _lid=_lid + 1,
                        _paths=_paths,
                    )
                    if not isinstance(v, Probe)
                    else {}
                )

        def _find_nodes_relative() -> None:
            for k, v in self.__dict__.items():
                if isinstance(v, PAIBoxObject):
                    _add_node1(self, k, v, _paths, gather, nodes)
                elif isinstance(v, NodeList):
                    for i, v2 in enumerate(v):
                        if isinstance(v2, PAIBoxObject):
                            _add_node1(self, f"{k}-{str(i)}", v2, _paths, gather, nodes)
                elif isinstance(v, NodeDict):
                    for k2, v2 in v.items():
                        if isinstance(v2, PAIBoxObject):
                            _add_node1(self, f"{k}.{k2}", v2, _paths, gather, nodes)

            # finding nodes recursively
            for k1, v1 in nodes:
                if not isinstance(v1, Probe):
                    for k2, v2 in v1._find_nodes(
                        method=method,
                        level=level,
                        include_self=include_self,
                        _lid=_lid + 1,
                        _paths=_paths,
                    ).items():
                        if k2:
                            gather[f"{k1}.{k2}"] = v2

        nodes = []

        if method == "absolute":
            _find_nodes_absolute()
        else:
            _find_nodes_relative()

        return gather


def _add_node1(
    obj: Any,
    k: str,
    v: PAIBoxObject,
    _paths: set[_IdPathType],
    gather: Collector[str, PAIBoxObject],
    nodes: list[tuple[str, PAIBoxObject]],
) -> None:
    path = (id(obj), id(v))

    if path not in _paths:
        _paths.add(path)
        gather[k] = v
        nodes.append((k, v))


def _add_node2(
    obj: Any,
    v: PAIBoxObject,
    _paths: set[_IdPathType],
    gather: Collector[str, PAIBoxObject],
    nodes: list[PAIBoxObject],
) -> None:
    path = (id(obj), id(v))

    if path not in _paths:
        _paths.add(path)
        gather[v.name] = v
        nodes.append(v)


class DynamicSys(PAIBoxObject, StatusMemory):
    __gh_build_ignore__: bool = False
    """To indicate whether the backend will take the object into account
        when the network topology information is first constructed"""

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        super(PAIBoxObject, self).__init__()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def reset_state(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def shape_in(self) -> tuple[int, ...]:
        """Actual shape of input."""
        raise NotImplementedError

    @property
    def shape_out(self) -> tuple[int, ...]:
        """Actual shape of output."""
        raise NotImplementedError

    @property
    def num_in(self) -> int:
        raise NotImplementedError

    @property
    def num_out(self) -> int:
        raise NotImplementedError

    @property
    def output(self):
        """Actual output to the sucessors."""
        raise NotImplementedError

    @property
    def feature_map(self):
        raise NotImplementedError

    @property
    def state(self) -> NodeDict:
        """Return the stateful attirbutes in the system."""
        return self._memories


INFINITE_DATAFLOW = 0
T_FIRST_VLD_OUT_OF_RANGE_TEXT = (
    "the {0} output time of the first valid data should be in the working "
    + "time from {1} to {2}, but got {3}."
)


@dataclass
class DataFlowFormat:
    """Describe in detail the format of valid data in the dataflow."""

    t_1st_vld: int = 0
    """Global time or a relative time of the first valid data in the dataflow, determined by `is_local_time`."""
    interval: int = 1
    """The interval of valid data in the dataflow."""
    n_vld: int = INFINITE_DATAFLOW
    """The number of valid data. <0 for infinite dataflow."""

    is_local_time: bool = True
    """Whether the `t_1st_vld` is relative to the local time(tws+T) of the neuron, or   \
        relative to the global time of the external input."""

    def t_at_idx(self, idx: int) -> int:
        """The time of the valid data at the given index."""
        if self.n_vld > INFINITE_DATAFLOW:
            assert 0 <= idx <= self.n_vld - 1

        return self.t_1st_vld + idx * self.interval

    def t_at_n(self, n: int) -> int:
        """The time of the n-th valid data."""
        return self.t_at_idx(n - 1)

    def local2global(self, tws: int):
        assert self.is_local_time
        return type(self)(
            self.get_global_t_1st_vld(tws), self.interval, self.n_vld, False
        )

    @property
    def t_last_vld(self) -> int:
        """The time of the last valid data."""
        assert self.n_vld > INFINITE_DATAFLOW
        return self.t_at_n(self.n_vld)

    def get_global_t_1st_vld(self, tws: int) -> int:
        """Get the global time of the first valid data."""
        arg_check_non_neg(tws)
        return tws + self.t_1st_vld if self.is_local_time else self.t_1st_vld

    def _check_after_assign(self, tws: int, end_tick: int) -> None:
        # The global time of the first valid data is in [tws, end_tick].
        gb_t_1st_vld = self.get_global_t_1st_vld(tws)
        if gb_t_1st_vld < tws or gb_t_1st_vld > end_tick:
            if self.is_local_time:
                raise ValueError(
                    T_FIRST_VLD_OUT_OF_RANGE_TEXT.format(
                        "local", "+0", f"+{end_tick - tws + 1}", self.t_1st_vld
                    )
                )
            else:
                raise ValueError(
                    T_FIRST_VLD_OUT_OF_RANGE_TEXT.format(
                        "global", tws, end_tick, self.t_1st_vld
                    )
                )

        if self.n_vld > INFINITE_DATAFLOW:
            if (
                t_last_vld := gb_t_1st_vld + (self.n_vld - 1) * self.interval
            ) > end_tick:
                raise ValueError(
                    f"valid data is output after the end time. The neuron stops working at "
                    f"{end_tick}, but still needs to output at {t_last_vld}."
                )


class NeuDyn(DynamicSys, ReceiveInputProj, TimeRelatedNode):

    _delay: int
    _tws: int
    """tick_wait_start"""
    _twe: int
    """tick_wait_end"""
    _uf: int
    """unrolling_factor"""

    _oflow_format: DataFlowFormat
    """The format of output data stream"""

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.master_nodes = NodeDict()

    def is_working(self) -> bool:
        return (self.tick_wait_start > 0 and self.timestamp >= 0) and (
            self.tick_wait_end == 0 or self.timestamp + 1 <= self.tick_wait_end
        )

    @property
    def delay_relative(self) -> int:
        return self._delay

    @property
    def tick_wait_start(self) -> int:
        return self._tws

    @property
    def tick_wait_end(self) -> int:
        return self._twe

    @property
    def unrolling_factor(self) -> int:
        return self._uf

    @property
    def end_tick(self) -> int:
        """End time of work."""
        if self.tick_wait_end == 0:
            return 9999  # Never end

        return self.tick_wait_start + self.tick_wait_end - 1

    @unrolling_factor.setter
    def unrolling_factor(self, factor: int) -> None:
        self._uf = arg_check_pos(factor, "'unrolling_factor'")


class SynSys(DynamicSys):
    CFLAG_ENABLE_WP_OPTIMIZATION: ClassVar[bool] = True
    """Compilation flag for weight precision optimization."""

    @property
    def weights(self) -> WeightType:
        raise NotImplementedError

    @property
    def weight_width(self) -> WW:
        raise NotImplementedError

    @property
    def connectivity(self) -> WeightType:
        raise NotImplementedError

    @property
    def n_axon_each(self) -> np.ndarray:
        return np.sum(self.connectivity, axis=0)

    @property
    def num_axon(self) -> int:
        return np.count_nonzero(np.any(self.connectivity, axis=1))

    @property
    def num_dendrite(self) -> int:
        return np.count_nonzero(np.any(self.connectivity, axis=0))
