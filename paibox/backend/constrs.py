import sys
from collections import defaultdict
from typing import ClassVar, Dict, FrozenSet, List, Sequence, Tuple

from .graphs_types import NodeName, NodeType

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

BoundedConstrType: TypeAlias = List[FrozenSet[NodeName]]


class Constraints:
    pass


class GraphNodeConstrs(Constraints):
    BOUNDED_CONSTRS: ClassVar[List[List[NodeName]]] = []
    CONFLICTED_CONSTRS: ClassVar[Dict[NodeName, Tuple[NodeName, ...]]] = defaultdict(
        tuple
    )

    @classmethod
    def clear(cls) -> None:
        cls.BOUNDED_CONSTRS = []
        cls.CONFLICTED_CONSTRS = {}

    @classmethod
    def add_node_constr(
        cls,
        *,
        bounded: Sequence[NodeName] = (),
        conflicted: Dict[NodeName, Sequence[NodeName]] = {},
    ):
        """Add constraints to a node."""
        if len(bounded) > 0:
            cls.BOUNDED_CONSTRS.append(list(bounded))

        if conflicted:
            for k, v in conflicted.items():
                cls.CONFLICTED_CONSTRS[k] = tuple(v)

    @staticmethod
    def tick_wait_attr_constr(raw_nodes: List[NodeType]) -> List[List[int]]:
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
