"""PAIIR computation graph (DAG).

Provides a directed acyclic graph container for organising IR nodes and
their connections, plus simulation methods for chip-accurate inference.
"""

import graphlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..exceptions import GraphValidationError
from .add_ops import GeneralAddOp
from .core_neuron import CoreNeuronV25
from .ir_base import InputNode, OutputNode, PAIIRNode, TensorLayout
from .op_node import OfflineCoreOp, OpNode, RoutingOp

__all__ = ["Edge", "PAIIRGraph"]

NodeOutput = Tensor | tuple[Tensor, ...]


@dataclass(frozen=True, slots=True)
class Edge:
    """A directed edge in the computation graph.

    Attributes:
        src: Source node name.
        dst: Destination node name.
        src_port: Output port index on the source node. Single-output nodes
            always use ``0``. Multi-output nodes such as :class:`SplitOp`
            use this to identify which logical branch flows along the edge.
        dst_port: Input port index on the destination node (for multi-input
            nodes such as :class:`GeneralAddOp`, :class:`AccumulateOp`, or
            :class:`PotentialAddOp`).
    """

    src: str
    dst: str
    src_port: int = 0
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
        inp = InputNode(shape=torch.Size((1, 3, 32, 32)))
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
            incoming.sort(key=lambda edge: (edge.dst_port, edge.src_port, edge.src))

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

    def _collect_reachable_nodes(
        self, start_names: list[str], *, reverse: bool
    ) -> set[str]:
        """Collect nodes reachable from *start_names* in one graph direction."""
        seen: set[str] = set()
        stack = list(start_names)

        while stack:
            name = stack.pop()
            if name in seen or name not in self.nodes:
                continue
            seen.add(name)
            neighbors = self.predecessors(name) if reverse else self.successors(name)
            stack.extend(neighbors)

        return seen

    def nodes_on_input_output_paths(self) -> set[str]:
        """Return nodes that lie on at least one input-to-output path."""
        reachable_from_inputs = self._collect_reachable_nodes(
            [node.name for node in self.input_nodes()], reverse=False
        )
        reachable_to_outputs = self._collect_reachable_nodes(
            [node.name for node in self.output_nodes()], reverse=True
        )
        return reachable_from_inputs & reachable_to_outputs

    def disconnected_nodes(self) -> list[str]:
        """Return node names that are not on any input-to-output path."""
        return sorted(set(self.nodes) - self.nodes_on_input_output_paths())

    def lint(self, *, allow_disconnected: bool = False) -> None:
        """Validate structural graph invariants.

        Checks graph/container consistency (node names, edge endpoints,
        duplicate edges, boundary-node constraints, acyclicity), and by default
        also rejects nodes that do not lie on any input-to-output path.

        Args:
            allow_disconnected: When True, skip the input-to-output reachability
                check. This is useful for mid-pipeline cleanup stages that still
                intend to prune disconnected fragments.
        """
        errors: list[str] = []

        input_nodes = self.input_nodes()
        output_nodes = self.output_nodes()
        if not input_nodes:
            errors.append("graph has no input node")
        if not output_nodes:
            errors.append("graph has no output node")

        seen_names: set[str] = set()
        for key, node in self.nodes.items():
            if key != node.name:
                errors.append(
                    f"graph node key '{key}' does not match node.name '{node.name}'"
                )
            if node.name in seen_names:
                errors.append(f"node redefines name '{node.name}'")
            seen_names.add(node.name)

        seen_edges: set[Edge] = set()
        incoming_src_by_port: dict[tuple[str, int], Edge] = {}
        edge_endpoint_errors = False

        for edge in self.edges:
            if edge.src not in self.nodes:
                errors.append(
                    f"edge {edge.src!r} -> {edge.dst!r} references missing source node"
                )
                edge_endpoint_errors = True

            if edge.dst not in self.nodes:
                errors.append(
                    f"edge {edge.src!r} -> {edge.dst!r} references missing destination node"
                )
                edge_endpoint_errors = True

            if edge.dst_port < 0:
                errors.append(
                    f"edge {edge.src!r} -> {edge.dst!r} has negative dst_port={edge.dst_port}"
                )

            if edge.src_port < 0:
                errors.append(
                    f"edge {edge.src!r} -> {edge.dst!r} has negative src_port={edge.src_port}"
                )

            if edge in seen_edges:
                errors.append(
                    f"duplicate edge {edge.src!r} -> {edge.dst!r} "
                    f"(dst_port={edge.dst_port}, src_port={edge.src_port})"
                )
            else:
                seen_edges.add(edge)

            port_key = (edge.dst, edge.dst_port)
            existing_edge = incoming_src_by_port.get(port_key)
            if existing_edge is None:
                incoming_src_by_port[port_key] = edge
            elif existing_edge != edge:
                errors.append(
                    f"node '{edge.dst}' has multiple incoming edges on "
                    f"dst_port {edge.dst_port}: "
                    f"('{existing_edge.src}', src_port={existing_edge.src_port}) and "
                    f"('{edge.src}', src_port={edge.src_port})"
                )

        if not edge_endpoint_errors:
            try:
                self.topo_sort()
            except graphlib.CycleError as exc:
                cycle = exc.args[1] if len(exc.args) > 1 else ()
                if cycle:
                    cycle_desc = " -> ".join(str(name) for name in cycle)
                    errors.append(f"graph contains a cycle: {cycle_desc}")
                else:
                    errors.append("graph contains a cycle")

            for name, node in self.nodes.items():
                preds = self.predecessors(name)
                succs = self.successors(name)

                if isinstance(node, InputNode) and preds:
                    errors.append(
                        f"InputNode '{name}' has predecessors {preds} (expected none)"
                    )

                if isinstance(node, OutputNode) and succs:
                    errors.append(
                        f"OutputNode '{name}' has successors {succs} (expected none)"
                    )

        if (
            not allow_disconnected
            and input_nodes
            and output_nodes
            and not edge_endpoint_errors
        ):
            disconnected = self.disconnected_nodes()
            if disconnected:
                names = ", ".join(disconnected)
                errors.append(f"nodes not on any input-to-output path: {names}")

        if errors:
            raise GraphValidationError(errors)

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
                src=new_name if edge.src == old_name else edge.src,
                dst=new_name if edge.dst == old_name else edge.dst,
                src_port=edge.src_port,
                dst_port=edge.dst_port,
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

    def add_edge(
        self, src: str, dst: str, *, src_port: int = 0, dst_port: int = 0
    ) -> None:
        """Add a directed edge.

        Args:
            src: Source node name.
            dst: Destination node name.
            src_port: Output port index on the source node.
            dst_port: Input port index on the destination node.
        """
        if src not in self.nodes:
            raise KeyError(f"source node '{src}' not found in graph")
        if dst not in self.nodes:
            raise KeyError(f"destination node '{dst}' not found in graph")
        self.edges.append(Edge(src, dst, src_port, dst_port))
        self._invalidate_structure()

    def replace_all_uses_with(
        self, old_name: str, new_name: str, *, delete_old: bool = False
    ) -> None:
        """Rewrite every outgoing use of ``old_name`` to ``new_name``.

        The destination node and destination port are preserved. Duplicate edges
        created by the rewrite are removed. ``old_name`` remains in the graph
        unless ``delete_old`` is explicitly requested.
        """
        if old_name not in self.nodes:
            raise KeyError(f"node '{old_name}' not found in graph")
        if new_name not in self.nodes:
            raise KeyError(f"node '{new_name}' not found in graph")
        if old_name == new_name:
            return

        new_edges: list[Edge] = []
        seen: set[Edge] = set()
        for edge in self.edges:
            updated = (
                Edge(new_name, edge.dst, edge.src_port, edge.dst_port)
                if edge.src == old_name
                else edge
            )
            if updated in seen:
                continue
            seen.add(updated)
            new_edges.append(updated)

        self.edges = new_edges
        self._invalidate_structure()
        if delete_old:
            self.remove_node(old_name)

    def remove_node_and_reconnect(
        self, name: str, *, source_name: str | None = None
    ) -> None:
        """Remove a node and reconnect one predecessor to all of its outgoing uses.

        If ``source_name`` is omitted, the node must have exactly one predecessor.
        When provided explicitly, ``source_name`` must be one of the removed
        node's predecessors.
        """
        if name not in self.nodes:
            raise KeyError(f"node '{name}' not found in graph")

        preds = self.predecessors(name)
        if source_name is None:
            if len(preds) != 1:
                raise ValueError(
                    f"node '{name}' must have exactly one predecessor to reconnect"
                )
            source_name = preds[0]
        elif source_name not in self.nodes:
            raise KeyError(f"source node '{source_name}' not found in graph")
        elif source_name not in preds:
            raise ValueError(
                f"source node '{source_name}' is not a predecessor of '{name}'"
            )

        incoming = self.incoming_edges(name)
        outgoing = self.outgoing_edges(name)
        self.remove_node(name)

        source_src_port = 0
        for edge in incoming:
            if edge.src == source_name:
                source_src_port = edge.src_port
                break

        for edge in outgoing:
            candidate = Edge(source_name, edge.dst, source_src_port, edge.dst_port)
            if candidate in self.edges:
                continue
            self.add_edge(
                source_name, edge.dst, src_port=source_src_port, dst_port=edge.dst_port
            )

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

    @staticmethod
    def _summary_shape_without_batch(shape: torch.Size) -> tuple[int, ...]:
        if not shape:
            return ()
        if len(shape) >= 5 and shape[1] == 1:
            return (shape[0], *shape[2:])
        if shape[0] == 1:
            return tuple(shape[1:])
        return tuple(shape)

    def _summary_node_shape(self, node: PAIIRNode) -> torch.Size:
        if isinstance(node, (InputNode, OutputNode)):
            return node.shape
        if isinstance(node, OpNode) and node.num_outputs == 1:
            return node.output_layouts[0].shape
        return torch.Size()

    @staticmethod
    def _summary_layout_desc(layout: TensorLayout) -> str:
        if not layout.shape:
            return "shape=(), dims=()"
        return f"shape={tuple(layout.shape)}, dims={layout.dims}"

    def get_node_output_layout(self, name: str, port: int = 0) -> TensorLayout:
        node = self.nodes[name]
        if isinstance(node, InputNode):
            return node.layout
        if isinstance(node, OutputNode):
            return node.layout
        if isinstance(node, OpNode):
            return node.output_layouts[port]
        return TensorLayout()

    def get_edge_output_layout(self, edge: Edge) -> TensorLayout:
        return self.get_node_output_layout(edge.src, edge.src_port)

    def summary(self, verbose: bool | int = False) -> None:
        """Print a text summary of the graph.

        Args:
            verbose:
                - ``False`` / ``0``: compact node/edge summary
                - ``True`` / ``1``: include layouts and explicit edge ports
        """
        verbose = int(verbose)
        lines = [
            f"PAIIRGraph '{self.name}'",
            f"  Nodes: {len(self.nodes)}",
            f"  Edges: {len(self.edges)}",
            f"  Inputs: {[n.name for n in self.input_nodes()]}",
            f"  Outputs: {[n.name for n in self.output_nodes()]}",
            "",
        ]
        for name in self.topo_sort():
            node = self.nodes[name]
            incoming = self.incoming_edges(name)
            outgoing = self.outgoing_edges(name)
            line = f"  {name} ({self._summary_node_label(node)})"
            shape = self._summary_node_shape(node)
            if shape:
                line += f" {self._summary_shape_without_batch(shape)}"
            lines.append(line)
            if verbose:
                if isinstance(node, InputNode):
                    lines.append(
                        f"    layout: {self._summary_layout_desc(node.layout)}"
                    )
                elif isinstance(node, OutputNode):
                    lines.append(
                        f"    layout: {self._summary_layout_desc(node.layout)}"
                    )
                elif isinstance(node, OpNode):
                    if node.input_layouts:
                        rendered_inputs = ", ".join(
                            f"in[{i}] {self._summary_layout_desc(layout)}"
                            for i, layout in enumerate(node.input_layouts)
                        )
                        lines.append(f"    {rendered_inputs}")
                    if node.output_layouts:
                        rendered_outputs = ", ".join(
                            f"out[{i}] {self._summary_layout_desc(layout)}"
                            for i, layout in enumerate(node.output_layouts)
                        )
                        lines.append(f"    {rendered_outputs}")
                if incoming:
                    lines.append(
                        "    <- "
                        + ", ".join(
                            f"{edge.src}[src_port={edge.src_port}] -> dst_port={edge.dst_port}"
                            for edge in incoming
                        )
                    )
                if outgoing:
                    lines.append(
                        "    -> "
                        + ", ".join(
                            f"src_port={edge.src_port} -> {edge.dst}[dst_port={edge.dst_port}]"
                            for edge in outgoing
                        )
                    )
            else:
                preds = self.predecessors(name)
                succs = self.successors(name)
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

        First checks structural graph integrity via :meth:`lint`, then checks
        that all :class:`OfflineCoreOp` nodes have the required tick parameters
        for correct simulation. Call before running simulation to catch
        configuration errors early.

        Required parameters:
        - ``tick_start``: Must be set (not None)
        - ``tick_duration``: Must be non-negative
        - ``tick_initial``: Must be non-negative
        - output layout on port 0: Must be set for zero output generation

        Raises:
            RuntimeError: If any verification check fails.
        """
        errors: list[str] = []

        try:
            self.lint()
        except GraphValidationError as exc:
            errors.extend(exc.errors)

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

            if node.num_outputs == 0 or not node.output_layouts[0].shape:
                errors.append(
                    f"'{name}': output layout is not set. "
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
        shape = node.output_layouts[0].shape if node.num_outputs > 0 else torch.Size()
        if shape:
            return torch.zeros(shape)
        return torch.zeros(())

    def _resolve_edge_tensor(
        self, edge: Edge, node_outputs: dict[str, NodeOutput]
    ) -> Tensor:
        value = node_outputs[edge.src]
        if torch.is_tensor(value):
            if edge.src_port != 0:
                raise RuntimeError(
                    f"edge '{edge.src}' -> '{edge.dst}' requests src_port={edge.src_port} "
                    "from a single-output node"
                )
            return value

        src_node = self.nodes[edge.src]
        if not isinstance(src_node, OpNode) or src_node.num_outputs <= 1:
            raise RuntimeError(
                f"node '{edge.src}' produced multiple outputs, but has no multi-output graph contract"
            )

        if len(value) != src_node.num_outputs:
            raise RuntimeError(
                f"node '{edge.src}' produced {len(value)} runtime outputs, "
                f"but metadata declares {src_node.num_outputs}"
            )

        if edge.src_port < 0 or edge.src_port >= len(value):
            raise RuntimeError(
                f"node '{edge.src}' src_port={edge.src_port} is invalid for successor "
                f"'{edge.dst}' dst_port={edge.dst_port}"
            )
        branch = value[edge.src_port]
        if not torch.is_tensor(branch):
            raise RuntimeError(
                f"node '{edge.src}' output branch {edge.src_port} is not a tensor"
            )
        return branch

    def step(self, *inputs: Tensor) -> NodeOutput:
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
        node_outputs: dict[str, NodeOutput] = {}

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
                incoming = self.incoming_edges(name)
                node_outputs[name] = self._resolve_edge_tensor(
                    incoming[0], node_outputs
                )
                continue

            # Collect predecessor tensors ordered by dst_port.
            # predecessors() returns names sorted by dst_port, so xs[i]
            # corresponds to input port i of multi-input nodes (GeneralAddOp,
            # AccumulateOp, PotentialAddOp, ConcatOp).
            incoming = self.incoming_edges(name)
            xs = [self._resolve_edge_tensor(edge, node_outputs) for edge in incoming]

            if isinstance(node, (RoutingOp, GeneralAddOp)):
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

        results: list[Tensor] = []
        for out_node in self.output_nodes():
            value = node_outputs[out_node.name]
            if not torch.is_tensor(value):
                raise RuntimeError(
                    f"OutputNode '{out_node.name}' resolved to non-tensor output"
                )
            results.append(value)

        return results[0] if len(results) == 1 else tuple(results)

    def forward(self, *inputs: Tensor) -> NodeOutput:
        """Alias for :meth:`step`."""
        return self.step(*inputs)

    def run(self, *inputs: Tensor, T: int, reset: bool = True) -> NodeOutput:
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

        output_steps: list[NodeOutput] = []
        for t in range(T):
            xs = tuple(x[t] for x in inputs)
            output_steps.append(self.step(*xs))

        if torch.is_tensor(output_steps[0]):
            return torch.stack(output_steps, dim=0)  # type: ignore[arg-type]

        return tuple(
            torch.stack([step[i] for step in output_steps], dim=0)
            for i in range(len(output_steps[0]))
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', nodes={len(self.nodes)}, edges={len(self.edges)})"
