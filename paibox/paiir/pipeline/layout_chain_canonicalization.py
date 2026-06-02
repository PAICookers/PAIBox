"""Canonicalize consecutive transform chains before fusion."""

from ..ir.graph import PAIIRGraph
from ..ir.op_node import TransformOp, TransformStage
from .graph_utils import is_order_preserving_transform_node, is_transform_node

__all__ = ["canonicalize_layout_chains"]


def canonicalize_layout_chains(graph: PAIIRGraph) -> PAIIRGraph:
    """Reduce continuous transform segments until no further rewrite applies.

    The pass operates on maximal linear transform slices. When a transform
    chain is interrupted by an observed intermediate output, the chain is cut
    at that observation point and each safe slice is canonicalized
    independently.
    """
    changed = True
    changed_any = False
    while changed:
        changed = False
        for name in graph.topo_sort():
            node = graph.nodes.get(name)
            if not is_transform_node(node):
                continue

            if _try_reduce_transform_segment(graph, name):
                changed = True
                changed_any = True
                break

    return graph.clone_shallow() if changed_any else graph


def _try_reduce_transform_segment(graph: PAIIRGraph, head_name: str) -> bool:
    segment = _collect_transform_segment(graph, head_name)
    if segment is None:
        return False

    if _is_identity_transform(segment[0]):
        _remove_transform_segment(graph, [segment[0]])
        return True

    if _is_identity_segment(segment):
        _remove_transform_segment(graph, segment)
        return True

    if len(segment) == 1:
        return False

    _replace_transform_segment(
        graph, segment, TransformOp(_compose_transform_stages(segment))
    )
    return True


def _collect_transform_segment(
    graph: PAIIRGraph, head_name: str
) -> list[TransformOp] | None:
    """Return one maximal reducible transform slice starting at *head_name*."""
    head = graph.nodes.get(head_name)
    if not is_transform_node(head):
        return None

    if not _is_transform_segment_head(graph, head_name, head):
        return None
    if not _has_concrete_layouts(head):
        return None

    segment: list[TransformOp] = [head]
    current_name = head_name

    while True:
        succs = graph.successors(current_name)
        if len(succs) != 1:
            return segment

        next_name = succs[0]
        next_node = graph.nodes.get(next_name)
        if not is_transform_node(next_node):
            return segment
        if len(graph.predecessors(next_name)) != 1:
            return segment
        if not _is_single_input_output_transform(next_node):
            return segment
        if not _has_concrete_layouts(next_node):
            return segment

        segment.append(next_node)
        current_name = next_name


def _is_transform_segment_head(graph: PAIIRGraph, name: str, node: TransformOp) -> bool:
    """Return whether *name* starts a reducible transform slice."""
    if not _is_single_input_output_transform(node):
        return False

    preds = graph.predecessors(name)
    if len(preds) != 1:
        return False

    pred_name = preds[0]
    pred = graph.nodes.get(pred_name)
    if not is_transform_node(pred):
        return True

    return len(graph.successors(pred_name)) != 1


def _is_single_input_output_transform(node: TransformOp) -> bool:
    return node.num_inputs == 1 and node.num_outputs == 1


def _has_concrete_layouts(node: TransformOp) -> bool:
    return bool(node.input_layouts and node.input_layouts[0].shape) and bool(
        node.output_layouts and node.output_layouts[0].shape
    )


def _is_identity_segment(segment: list[TransformOp]) -> bool:
    head_input_layout = segment[0].input_layouts[0]
    tail_output_layout = segment[-1].output_layouts[0]
    if head_input_layout.shape != tail_output_layout.shape:
        return False

    combined = TransformOp(_compose_transform_stages(segment))
    combined.input_layouts = (head_input_layout,)
    combined.output_layouts = (tail_output_layout,)
    return is_order_preserving_transform_node(combined)


def _is_identity_transform(node: TransformOp) -> bool:
    if not _has_concrete_layouts(node):
        return False

    input_layout = node.input_layouts[0]
    output_layout = node.output_layouts[0]
    if input_layout.shape != output_layout.shape:
        return False

    return is_order_preserving_transform_node(node)


def _remove_transform_segment(graph: PAIIRGraph, segment: list[TransformOp]) -> None:
    """Delete one identity transform slice and reconnect its surrounding edges."""
    head_name = segment[0].name
    tail_name = segment[-1].name
    incoming = graph.incoming_edges(head_name)
    if len(incoming) != 1:
        raise ValueError("transform segment head must have exactly one incoming edge")

    source_edge = incoming[0]
    outgoing = graph.outgoing_edges(tail_name)

    for node in segment:
        graph.remove_node(node.name)

    for edge in outgoing:
        candidate = (source_edge.src, edge.dst, source_edge.src_port, edge.dst_port)
        if any(
            existing.src == candidate[0]
            and existing.dst == candidate[1]
            and existing.src_port == candidate[2]
            and existing.dst_port == candidate[3]
            for existing in graph.edges
        ):
            continue
        graph.add_edge(
            source_edge.src,
            edge.dst,
            src_port=source_edge.src_port,
            dst_port=edge.dst_port,
        )


def _replace_transform_segment(
    graph: PAIIRGraph,
    segment: list[TransformOp],
    replacement: TransformOp,
) -> None:
    """Replace one transform slice with a single composed ``TransformOp``."""
    head_name = segment[0].name
    tail_name = segment[-1].name
    incoming = graph.incoming_edges(head_name)
    if len(incoming) != 1:
        raise ValueError("transform segment head must have exactly one incoming edge")

    replacement.input_layouts = segment[0].input_layouts
    replacement.output_layouts = segment[-1].output_layouts
    graph.add_node(replacement)
    graph.add_edge(
        incoming[0].src,
        replacement.name,
        src_port=incoming[0].src_port,
        dst_port=0,
    )
    graph.replace_all_uses_with(tail_name, replacement.name, delete_old=False)

    for node in segment:
        graph.remove_node(node.name)


def _compose_transform_stages(segment: list[TransformOp]) -> tuple[TransformStage, ...]:
    """Concatenate the stages of one transform slice in graph order."""
    return tuple(stage for node in segment for stage in node.stages)
