"""Generic helpers for graph fusion."""

from ..ir.op_node import SequentialOp, StandaloneActOp, StandaloneCompOp

__all__ = ["_materialize_shared_sequential"]


def _materialize_shared_sequential(
    pred_name: str,
    pred: StandaloneCompOp,
    act_name: str,
    act_node: StandaloneActOp,
    consumed: set[str],
    node_remap: dict[str, str],
) -> SequentialOp:
    """Build and register a shared-core ``SequentialOp`` from comp + act."""
    fused = SequentialOp(pred.comp, act_node.act)
    fused.input_layouts = pred.input_layouts
    fused.output_layouts = act_node.output_layouts

    consumed.add(act_name)
    consumed.add(pred_name)
    node_remap[act_name] = fused.name
    node_remap[pred_name] = fused.name

    return fused
