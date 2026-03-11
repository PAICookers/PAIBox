"""PAIIR computation graph (DAG).

Provides a directed acyclic graph container for organising IR nodes and
their connections.
"""

from collections import deque
from dataclasses import dataclass

from torch import nn

from .ir_base import InputNode, OutputNode, PAIIRNode

__all__ = ["Edge", "PAIIRGraph"]


@dataclass(frozen=True)
class Edge:
    """A directed edge in the computation graph.

    Attributes:
        src: Source node name.
        dst: Destination node name.
        dst_port: Input port index on the destination node (for multi-input
            nodes such as :class:`AccumulateOp` or :class:`AddOp`).
    """

    src: str
    dst: str
    dst_port: int = 0


class PAIIRGraph:
    """PAIIR directed acyclic computation graph.

    Manages a collection of IR nodes and the edges between them, providing
    topological sorting and predecessor / successor queries.

    Example::

        graph = PAIIRGraph("my_model")
        inp = InputNode(shape=(1, 3, 32, 32))
        graph.add_node(inp)

        op = SequentialOp(nn.Conv2d(3, 16, 3), LIFNodeV25())
        graph.add_node(op)
        graph.add_edge(inp.name, op.name)

        out = OutputNode()
        graph.add_node(out)
        graph.add_edge(op.name, out.name)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.nodes: dict[str, PAIIRNode] = {}
        self.edges: list[Edge] = []

    def add_node(self, node: PAIIRNode) -> None:
        """Add a node to the graph."""
        if node.name in self.nodes:
            raise ValueError(f"node '{node.name}' already exists in graph")
        self.nodes[node.name] = node

    def add_edge(self, src: str, dst: str, dst_port: int = 0) -> None:
        """Add a directed edge.

        Args:
            src: Source node name.
            dst: Destination node name.
            dst_port: Input port index on the destination node.
        """
        if src not in self.nodes:
            raise KeyError(f"source node '{src}' not found in graph")
        if dst not in self.nodes:
            raise KeyError(f"destination node '{dst}' not found in graph")
        self.edges.append(Edge(src=src, dst=dst, dst_port=dst_port))

    def input_nodes(self) -> list[InputNode]:
        """Return all input nodes."""
        return [n for n in self.nodes.values() if isinstance(n, InputNode)]

    def output_nodes(self) -> list[OutputNode]:
        """Return all output nodes."""
        return [n for n in self.nodes.values() if isinstance(n, OutputNode)]

    def predecessors(self, name: str) -> list[str]:
        """Return predecessor node names, sorted by ``dst_port``."""
        return [e.src for e in sorted(
            (e for e in self.edges if e.dst == name),
            key=lambda e: e.dst_port,
        )]

    def successors(self, name: str) -> list[str]:
        """Return successor node names."""
        return [e.dst for e in self.edges if e.src == name]

    def incoming_edges(self, name: str) -> list[Edge]:
        """Return all edges going *into* the given node, sorted by port."""
        edges = [e for e in self.edges if e.dst == name]
        edges.sort(key=lambda e: e.dst_port)
        return edges

    def outgoing_edges(self, name: str) -> list[Edge]:
        """Return all edges going *out of* the given node."""
        return [e for e in self.edges if e.src == name]

    def topo_sort(self) -> list[str]:
        """Topological sort. Returns an ordered list of node names.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        in_degree: dict[str, int] = {name: 0 for name in self.nodes}
        for edge in self.edges:
            in_degree[edge.dst] = in_degree.get(edge.dst, 0) + 1

        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for succ in self.successors(node):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(result) != len(self.nodes):
            raise ValueError("graph contains a cycle")
        return result

    def summary(self) -> str:
        """Generate a text summary of the graph."""
        lines = [
            f"PAIIRGraph '{self.name}'",
            f"  Nodes: {len(self.nodes)}",
            f"  Edges: {len(self.edges)}",
            f"  Inputs: {[n.name for n in self.input_nodes()]}",
            f"  Outputs: {[n.name for n in self.output_nodes()]}",
            "",
        ]
        for name, node in self.nodes.items():
            preds = self.predecessors(name)
            succs = self.successors(name)
            lines.append(f"  {name} ({type(node).__name__})")
            if preds:
                lines.append(f"    <- {preds}")
            if succs:
                lines.append(f"    -> {succs}")
        return "\n".join(lines)

    def train(self, mode: bool = True) -> "PAIIRGraph":
        for node in self.nodes.values():
            if isinstance(node, nn.Module):
                node.train(mode)
        return self

    def eval(self) -> "PAIIRGraph":
        return self.train(False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', nodes={len(self.nodes)}, edges={len(self.edges)})"
