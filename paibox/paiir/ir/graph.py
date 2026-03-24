"""PAIIR computation graph (DAG).

Provides a directed acyclic graph container for organising IR nodes and
their connections, plus simulation methods for chip-accurate inference.
"""

import graphlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .add_ops import GeneralAddOp
from .core_neuron import CoreNeuronV25
from .ir_base import InputNode, OutputNode, PAIIRNode
from .op_node import ConcatOp, OfflineCoreOp, OpNode, ReshapeOp

__all__ = ["Edge", "PAIIRGraph"]


@dataclass(frozen=True, slots=True)
class Edge:
    """A directed edge in the computation graph.

    Attributes:
        src: Source node name.
        dst: Destination node name.
        dst_port: Input port index on the destination node (for multi-input
            nodes such as :class:`GeneralAddOp`, :class:`AccumulateOp`, or
            :class:`PotentialAddOp`).
    """

    src: str
    dst: str
    dst_port: int = 0


@dataclass(slots=True)
class _GraphIndex:
    """Derived graph structure used by query helpers."""

    pred_edges_by_dst: dict[str, tuple[Edge, ...]]
    succ_edges_by_src: dict[str, tuple[Edge, ...]]
    topo_order: tuple[str, ...] | None = None


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
        self._index: _GraphIndex | None = None
        self._sim_step: int = 0
        self._active_counts: dict[str, int] = {}

    def _invalidate_structure(self) -> None:
        self._index = None

    def _build_index(self) -> _GraphIndex:
        pred_edges: dict[str, list[Edge]] = {name: [] for name in self.nodes}
        succ_edges: dict[str, list[Edge]] = {name: [] for name in self.nodes}

        for edge in self.edges:
            pred_edges.setdefault(edge.dst, []).append(edge)
            succ_edges.setdefault(edge.src, []).append(edge)

        for incoming in pred_edges.values():
            incoming.sort(key=lambda edge: edge.dst_port)

        return _GraphIndex(
            {name: tuple(pred_edges.get(name, ())) for name in self.nodes},
            {name: tuple(succ_edges.get(name, ())) for name in self.nodes},
        )

    def _get_index(self) -> _GraphIndex:
        if self._index is None:
            self._index = self._build_index()

        return self._index

    def _get_topo_order(self) -> tuple[str, ...]:
        index = self._get_index()
        if index.topo_order is None:
            predecessor_map = {
                name: tuple(edge.src for edge in index.pred_edges_by_dst.get(name, ()))
                for name in self.nodes
            }
            sorter = graphlib.TopologicalSorter(predecessor_map)
            index.topo_order = tuple(sorter.static_order())

        return index.topo_order

    def add_node(self, node: PAIIRNode) -> None:
        """Add a node to the graph."""
        if node.name in self.nodes:
            raise ValueError(f"node '{node.name}' already exists in graph")
        self.nodes[node.name] = node
        self._invalidate_structure()

    def replace_node(self, old_name: str, new_node: PAIIRNode) -> None:
        """Replace a node while preserving graph connectivity.

        All incoming/outgoing edges that referenced ``old_name`` are rewritten to
        the replacement node's current name.
        """
        if old_name not in self.nodes:
            raise KeyError(f"node '{old_name}' not found in graph")

        new_name = new_node.name
        if new_name != old_name and new_name in self.nodes:
            raise ValueError(f"node '{new_name}' already exists in graph")

        self.nodes.pop(old_name)
        self.nodes[new_name] = new_node
        self.edges = [
            Edge(
                new_name if edge.src == old_name else edge.src,
                new_name if edge.dst == old_name else edge.dst,
                edge.dst_port,
            )
            for edge in self.edges
        ]
        self._invalidate_structure()

    def clone_shallow(self) -> "PAIIRGraph":
        """Return a shallow copy of the graph structure.

        Node objects are shared; the node/edge containers and simulation
        bookkeeping are copied.
        """
        cloned = PAIIRGraph(self.name)
        cloned.nodes = dict(self.nodes)
        cloned.edges = list(self.edges)
        cloned._sim_step = self._sim_step
        cloned._active_counts = dict(self._active_counts)
        return cloned

    def remove_node(self, name: str) -> None:
        """Remove a node and all incoming/outgoing edges."""
        if name not in self.nodes:
            raise KeyError(f"node '{name}' not found in graph")

        self.nodes.pop(name)
        self.edges = [e for e in self.edges if e.src != name and e.dst != name]
        self._invalidate_structure()

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
        self._invalidate_structure()

    def input_nodes(self) -> list[InputNode]:
        """Return all input nodes."""
        return [n for n in self.nodes.values() if isinstance(n, InputNode)]

    def output_nodes(self) -> list[OutputNode]:
        """Return all output nodes."""
        return [n for n in self.nodes.values() if isinstance(n, OutputNode)]

    def predecessors(self, name: str) -> list[str]:
        """Return predecessor node names, sorted by ``dst_port``."""
        index = self._get_index()
        return [edge.src for edge in index.pred_edges_by_dst.get(name, ())]

    def successors(self, name: str) -> list[str]:
        """Return successor node names."""
        index = self._get_index()
        return [edge.dst for edge in index.succ_edges_by_src.get(name, ())]

    def incoming_edges(self, name: str) -> list[Edge]:
        """Return all edges going *into* the given node, sorted by port."""
        return list(self._get_index().pred_edges_by_dst.get(name, ()))

    def outgoing_edges(self, name: str) -> list[Edge]:
        """Return all edges going *out of* the given node."""
        return list(self._get_index().succ_edges_by_src.get(name, ()))

    def topo_sort(self) -> list[str]:
        """Topological sort. Returns an ordered list of node names.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        return list(self._get_topo_order())

    def _summary_node_label(self, node: PAIIRNode) -> str:
        """Return the human-readable type label used by :meth:`summary`.

        For compute nodes, prefer the contained compute / activation module
        types over the coarse IR wrapper type. This makes summaries easier to
        scan when the graph contains many generic wrappers such as
        ``SequentialOp`` or ``StandaloneActOp``.
        """
        if isinstance(node, OfflineCoreOp):
            comp_label: str | None = None

            comp = getattr(node, "comp", None)
            if isinstance(comp, nn.Module):
                comp_label = type(comp).__name__
            else:
                comps = getattr(node, "comps", None)
                if comps is not None:
                    comp_types = [type(op).__name__ for op in comps]
                    if comp_types:
                        comp_label = " + ".join(comp_types)

            act = getattr(node, "act", None)
            act_label = type(act).__name__ if act is not None else None

            if comp_label and act_label:
                return f"{comp_label} -> {act_label}"
            if comp_label:
                return comp_label
            if act_label:
                return act_label

        return type(node).__name__

    def summary(self) -> None:
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
            lines.append(f"  {name} ({self._summary_node_label(node)})")
            if preds:
                lines.append(f"    <- {preds}")
            if succs:
                lines.append(f"    -> {succs}")

        print("\n".join(lines))

    def train(self, mode: bool = True) -> "PAIIRGraph":
        for node in self.nodes.values():
            if isinstance(node, nn.Module):
                node.train(mode)
        return self

    def eval(self) -> "PAIIRGraph":
        return self.train(False)

    def verify_before_sim(self) -> None:
        """Verify that the graph is ready for simulation.

        Checks that all :class:`OfflineCoreOp` nodes have the required
        tick parameters for correct simulation.  Call before running
        simulation to catch configuration errors early.

        Required parameters:
        - ``tick_start``: Must be set (not None)
        - ``tick_duration``: Must be non-negative
        - ``tick_initial``: Must be non-negative
        - ``output_shape``: Must be set for zero output generation

        Raises:
            RuntimeError: If any verification check fails.
        """
        errors: list[str] = []

        for name, node in self.nodes.items():
            if not isinstance(node, OfflineCoreOp):
                continue

            cp = node.core_params

            if cp.tick_start is None:
                errors.append(
                    f"'{name}': tick_start is None. "
                    "Run assign_tick_params() before simulation."
                )

            if cp.tick_duration < 0:
                errors.append(
                    f"'{name}': tick_duration must be non-negative, got {cp.tick_duration}"
                )

            if cp.tick_initial < 0:
                errors.append(
                    f"'{name}': tick_initial must be non-negative, got {cp.tick_initial}"
                )

            if not node.output_shape:
                errors.append(
                    f"'{name}': output_shape is not set. "
                    "Pass sample_inputs to torch_to_paiir() for shape inference."
                )

        if errors:
            raise RuntimeError(
                f"PAIIRGraph verification failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def reset(self) -> None:
        """Reset all neuron states and simulation counters.

        Call before starting a new inference sequence.
        """
        for node in self.nodes.values():
            if isinstance(node, OfflineCoreOp):
                act: Any | None = getattr(node, "act", None)
                if act is not None and isinstance(act, CoreNeuronV25):
                    act.reset()

        self._sim_step = 0
        self._active_counts = {name: 0 for name in self.nodes}

    def _is_active(self, node: OfflineCoreOp) -> bool:
        """Check whether *node* is active at the current simulation step."""
        ts = node.core_params.tick_start
        if ts is None:
            ts = 0
        td = node.core_params.tick_duration
        if self._sim_step < ts:
            return False
        if td != 0 and self._sim_step >= ts + td:
            return False
        return True

    def _maybe_reset_node(self, name: str, node: OfflineCoreOp) -> None:
        """Reset neuron state if tick_initial period has elapsed."""
        ti = node.core_params.tick_initial
        if ti == 0:
            return
        count = self._active_counts.get(name, 0)
        if count > 0 and count % ti == 0:
            act = getattr(node, "act", None)
            if act is not None:
                act.reset()

    def _zero_output(self, node: OpNode) -> Tensor:
        """Return a zero tensor matching the node's expected output shape."""
        shape = node.output_shape
        if shape:
            return torch.zeros(shape)
        return torch.zeros(())

    def step(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Execute one time step (one sync_all cycle) through the graph.

        Advances the internal simulation step counter by 1. Each
        :class:`OfflineCoreOp` is active only within its
        ``[tick_start, tick_start + tick_duration)`` window; inactive nodes
        output zero without updating neuron state.

        Execution order mirrors the chip's sync_all protocol:

        1. ``_sim_step`` increments (corresponds to one sync_all pulse).
        2. Each node is visited in topological order — safe because the graph
           is a DAG and every predecessor is resolved before its successor.
        3. For each :class:`OfflineCoreOp`, the activity window is checked
           first.  Inactive nodes produce zero output and are otherwise
           invisible to downstream nodes; no neuron state is modified and
           ``_active_counts`` is not incremented.
        4. For active nodes, ``tick_initial``-triggered state reset fires
           *before* ``forward()`` so that the neuron starts the new work
           cycle from ``init_v``, exactly as the hardware reinitialises the
           membrane register at the boundary between work cycles.
        5. ``_active_counts[name]`` is incremented *after* ``forward()``
           so that ``_maybe_reset_node`` sees the count from the *previous*
           active step, keeping the modulo check aligned with cycle
           boundaries (reset at steps N, 2N, 3N, …).

        Args:
            *inputs: One tensor per :class:`InputNode`, in the order
                returned by :meth:`input_nodes`.

        Returns:
            Output tensor(s) from :class:`OutputNode` (s).
        """
        if not self._active_counts:
            self._active_counts = {name: 0 for name in self.nodes}

        self._sim_step += 1
        node_outputs: dict[str, Tensor] = {}

        # --- Phase 1: bind external inputs to InputNodes ---
        input_nodes = self.input_nodes()
        if len(inputs) != len(input_nodes):
            raise ValueError(f"expected {len(input_nodes)} input(s), got {len(inputs)}")

        for node, x in zip(input_nodes, inputs):
            node_outputs[node.name] = x

        # --- Phase 2: execute nodes in topological order ---
        for name in self.topo_sort():
            node = self.nodes[name]

            if isinstance(node, InputNode):
                # Already bound above; nothing to compute.
                continue

            if isinstance(node, OutputNode):
                # Pass-through: each OutputNode has exactly one predecessor
                preds = self.predecessors(name)
                node_outputs[name] = node_outputs[preds[0]]
                continue

            # Collect predecessor tensors ordered by dst_port.
            # predecessors() returns names sorted by dst_port, so xs[i]
            # corresponds to input port i of multi-input nodes (GeneralAddOp,
            # AccumulateOp, PotentialAddOp, ConcatOp).
            pred_names = self.predecessors(name)
            xs = [node_outputs[p] for p in pred_names]

            if isinstance(node, (ConcatOp, ReshapeOp, GeneralAddOp)):
                # Routing/shape transformation operation.
                # Or frontend/general expression operation.
                # Executes tensor transformation / expression evaluation for simulation.
                node_outputs[name] = node(*xs)
                continue

            # --- OfflineCoreOp: activity window + tick_initial + forward ---
            assert isinstance(node, OfflineCoreOp)

            if not self._is_active(node):
                node_outputs[name] = self._zero_output(node)
                continue

            # Check whether tick_initial requires a state reset before this
            # active step.  Must happen *before* forward() so the neuron
            # begins the step from init_v when a cycle boundary is crossed.
            self._maybe_reset_node(name, node)

            # Execute the core: comp (if any) -> neuron/LUT activation.
            # Single-input nodes (SequentialOp, StandaloneActOp, etc.) use
            # xs[0]; multi-input nodes (AccumulateOp) unpack all xs.
            out = node(xs[0]) if len(xs) == 1 else node(*xs)
            node_outputs[name] = out

            # Increment *after* forward() so _maybe_reset_node sees the
            # completed-step count rather than a look-ahead value.
            self._active_counts[name] = self._active_counts.get(name, 0) + 1

        results = [node_outputs[n.name] for n in self.output_nodes()]
        return results[0] if len(results) == 1 else tuple(results)

    def forward(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Alias for :meth:`step`."""
        return self.step(*inputs)

    def run(
        self, *inputs: Tensor, T: int, reset: bool = True
    ) -> Tensor | tuple[Tensor, ...]:
        """Run the graph for *T* time steps and return stacked outputs.

        Args:
            *inputs: Each tensor has a leading time dimension of size *T*,
                shape ``[T, N, ...]``.
            T: Number of time steps to simulate.
            reset: Call :meth:`reset` before the first step (default True).

        Returns:
            Stacked output(s) with shape ``[T, N, ...]``.
        """
        self.verify_before_sim()

        if reset:
            self.reset()

        output_steps: list[Tensor | tuple[Tensor, ...]] = []
        for t in range(T):
            xs = tuple(x[t] for x in inputs)
            output_steps.append(self.step(*xs))

        if isinstance(output_steps[0], Tensor):
            return torch.stack(output_steps, dim=0)  # type: ignore[arg-type]

        return tuple(
            torch.stack([step[i] for step in output_steps], dim=0)
            for i in range(len(output_steps[0]))
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', nodes={len(self.nodes)}, edges={len(self.edges)})"
