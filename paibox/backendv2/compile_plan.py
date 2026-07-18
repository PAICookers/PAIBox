"""Planning helpers for caller-partitioned pure PAICORE subgraphs."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

from paibox.paiir.ir import CPUOp, InputNode, OfflineCoreOp, PAIIRGraph


@dataclass(frozen=True, slots=True)
class SubgraphCompilePlan:
    """One namespaced graph and the prefixes assigned to its source graphs."""

    graph: PAIIRGraph
    prefixes: tuple[str, ...]


def build_subgraph_compile_plan(
    subgraphs: Sequence[PAIIRGraph], timesteps: int, auto_reset: bool
) -> SubgraphCompilePlan:
    """Combine independent pure PAICORE graphs without mutating their nodes.

    Graph order defines execution order. Each source graph is deep-copied and
    receives a ``gN/`` name prefix so its I/O and core metadata remain distinct
    in the exported artifact.
    """
    graphs = tuple(subgraphs)
    if not graphs:
        raise ValueError("'subgraphs' must contain at least one graph")
    if timesteps <= 0:
        raise ValueError(f"'timesteps' must be positive, got {timesteps}")

    combined = PAIIRGraph("subgraphs")
    prefixes: list[str] = []
    base_tick = 0
    for index, source in enumerate(graphs):
        if not isinstance(source, PAIIRGraph):
            raise TypeError(
                f"subgraphs[{index}] must be a PAIIRGraph, "
                f"got {type(source).__name__}"
            )
        source.lint()
        unsupported = [
            node.name
            for node in source.nodes.values()
            if isinstance(node, CPUOp)
        ]
        if unsupported:
            raise ValueError(
                f"subgraphs[{index}] must be pure PAICORE; found {unsupported}"
            )

        prefix = f"g{index}/"
        prefixes.append(prefix)
        names = {name: f"{prefix}{name}" for name in source.nodes}
        for name, node in source.nodes.items():
            copied = deepcopy(node)
            copied.name = names[name]
            combined.add_node(copied)
        for edge in source.edges:
            combined.add_edge(
                names[edge.src],
                names[edge.dst],
                src_port=edge.src_port,
                dst_port=edge.dst_port,
            )

        base_tick = _assign_timing(
            combined,
            tuple(names.values()),
            base_tick,
            timesteps,
            auto_reset,
        )

    return SubgraphCompilePlan(combined, tuple(prefixes))


def _assign_timing(
    graph: PAIIRGraph,
    names: tuple[str, ...],
    base_tick: int,
    timesteps: int,
    auto_reset: bool,
) -> int:
    name_set = set(names)
    depth: dict[str, int] = {}
    max_tick = base_tick
    for name in graph.topo_sort():
        if name not in name_set:
            continue
        node = graph.nodes[name]
        if isinstance(node, InputNode):
            depth[name] = base_tick
            continue
        pred_depth = max(
            (depth.get(pred, base_tick) for pred in graph.predecessors(name)),
            default=base_tick,
        )
        node_depth = pred_depth + node.__tick_depth__
        depth[name] = node_depth
        if isinstance(node, OfflineCoreOp):
            node.core_params.tick_start = node_depth
            node.core_params.tick_duration = 0 if auto_reset else timesteps
            node.core_params.tick_initial = timesteps if auto_reset else 0
            node.core_params.validate_tick_params()
            max_tick = max(max_tick, node_depth)
    return max_tick
