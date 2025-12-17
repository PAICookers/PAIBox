from collections import defaultdict
from typing import ClassVar

from .types import NodeType

NodeIdx = int
NodeConstrsAttr = str


class Constraints:
    pass


class GraphNodeConstrs(Constraints):
    offline_node_constr_attrs: ClassVar[list[NodeConstrsAttr]] = [
        "pool_max",
        "tick_wait_start",
        "tick_wait_end",
    ]
    """Node attributes that are actually the parameters of the cores."""

    online_node_constr_attrs: ClassVar[list[NodeConstrsAttr]] = [
        "tick_wait_start",
        "tick_wait_end",
        "lateral_inhi_target",
        "lateral_inhi_source",
        "lateral_inhi_value",
        "weight_decay_value",
        "upper_weight",
        "lower_weight",
        "lut_random_en",
        "decay_random_en",
        "leak_comparison",
        "online_mode_en",
    ]

    @classmethod
    def set_constr_attr(cls, attr: NodeConstrsAttr) -> None:
        if attr not in cls.offline_node_constr_attrs:
            cls.offline_node_constr_attrs.append(attr)
        if attr not in cls.online_node_constr_attrs:
            cls.online_node_constr_attrs.append(attr)

    @classmethod
    def remove_constr_attr(cls, attr: NodeConstrsAttr, strict: bool = False) -> None:
        if attr in cls.offline_node_constr_attrs:
            cls.offline_node_constr_attrs.remove(attr)
        elif strict:
            raise ValueError(
                f"attribute {attr} not found in constraint attributes list."
            )
        if attr in cls.online_node_constr_attrs:
            cls.online_node_constr_attrs.remove(attr)
        elif strict:
            raise ValueError(
                f"attribute {attr} not found in constraint attributes list."
            )

    @staticmethod
    def apply_constrs(raw_nodes: list[NodeType], online: bool) -> list[list[NodeIdx]]:
        """Group the nodes by the constraints of the nodes.

        Args:
            raw_nodes: nodes that need to be grouped using core parameter constraints.

        Returns:
            a list of groups of node indices.
        """
        grouped_indices = defaultdict(list)

        for i, node in enumerate(raw_nodes):
            key_lst = []
            node_constr_attrs = (
                GraphNodeConstrs.online_node_constr_attrs
                if online
                else GraphNodeConstrs.offline_node_constr_attrs
            )
            for attr in node_constr_attrs:
                if (v := getattr(node, attr, None)) is None:
                    raise AttributeError(f"node {node.name} has no attribute {attr}.")

                key_lst.append(v)

            k = tuple(key_lst)
            grouped_indices[k].append(i)

        return list(grouped_indices.values())
