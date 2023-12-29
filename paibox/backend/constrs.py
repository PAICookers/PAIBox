import sys
from collections import defaultdict
from typing import ClassVar, Dict, FrozenSet, List, Sequence, Tuple

from paibox.base import NeuDyn

from .graphs_types import NodeName

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

BoundedConstrType: TypeAlias = List[FrozenSet[NodeName]]


class Constraints:
    pass


class GraphNodeConstrs(Constraints):
    bound_constrs: ClassVar[List[List[NodeName]]] = []
    conflicted_constrs: Dict[NodeName, Tuple[NodeName, ...]] = defaultdict(tuple)

    @classmethod
    def clear(cls):
        cls.bound_constrs = []
        cls.conflicted_constrs = {}

    @classmethod
    def add_node_constr(
        cls,
        *,
        bounded: Sequence[NodeName] = (),
        conflicted: Dict[NodeName, Sequence[NodeName]] = {},
    ):
        """Add constraints to a node."""
        if len(bounded) > 0:
            cls.bound_constrs.extend(list(bounded))

        if conflicted:
            for k, v in conflicted.items():
                cls.conflicted_constrs[k] = v

    @staticmethod
    def tick_wait_attr_constr(raw_nodes: List[NeuDyn]) -> List[List[int]]:
        """Check whether the neurons to be assigned to a group are "equal" after\
            automatic inference.

        NOTE: Check attributes `tick_wait_start` & `tick_wait_end`. For those   \
            neurons with different attributes, they need to be separated.

        Return: returen the group of indices.
        """
        tw_attrs = [
            (raw_node.tick_wait_start, raw_node.tick_wait_end) for raw_node in raw_nodes
        ]

        if len(tw_attrs_set := set(tw_attrs)) == 1:
            return []
        else:
            constr = []
            pos = []
            for attr in tw_attrs_set:
                pos.clear()
                # Find all positions
                for i, v in enumerate(tw_attrs):
                    if attr == v:
                        pos.append(i)

                constr.append(pos.copy())

            return constr
