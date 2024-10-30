import sys
from collections import defaultdict
from typing import ClassVar

from .types import NodeType

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

NodeIdx: TypeAlias = int
NodeConstrsAttr: TypeAlias = str


class Constraints:
    pass


class GraphNodeConstrs(Constraints):
    node_constr_attrs: ClassVar[list[NodeConstrsAttr]] = [
        "pool_max",
        "tick_wait_start",
        "tick_wait_end",
    ]
    """Node attributes that are actually the parameters of the cores."""

    @classmethod
    def set_constr_attr(cls, attr: NodeConstrsAttr) -> None:
        if attr not in cls.node_constr_attrs:
            cls.node_constr_attrs.append(attr)

    @classmethod
    def remove_constr_attr(cls, attr: NodeConstrsAttr, strict: bool = False) -> None:
        if attr in cls.node_constr_attrs:
            cls.node_constr_attrs.remove(attr)
        elif strict:
            raise ValueError(
                f"attribute {attr} not found in constraint attributes list."
            )

    @staticmethod
    def apply_constrs(raw_nodes: list[NodeType]) -> list[list[NodeIdx]]:
        """Group the nodes by the constraints of the nodes.

        Args:
            raw_nodes: nodes that need to be grouped using core parameter constraints.

        Returns:
            a list of groups of node indices.
        """
        grouped_indices = defaultdict(list)

        for i, node in enumerate(raw_nodes):
            key_lst = []
            for attr in GraphNodeConstrs.node_constr_attrs:
                if (v := getattr(node, attr, None)) is None:
                    raise AttributeError(f"node {node.name} has no attribute {attr}.")

                key_lst.append(v)

            k = tuple(key_lst)
            grouped_indices[k].append(i)

        return list(grouped_indices.values())
