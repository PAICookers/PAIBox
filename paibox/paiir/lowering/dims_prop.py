"""Dimension propagation for FX graphs.

Propagates axis ordering information through an FX graph so that each node's
``meta["dims"]`` contains the output axis ordering as a ``tuple[int, ...]``.
Identity ordering ``(0, 1, ...)`` means no transpose/permute has been applied.

Requires ``ShapeProp`` to have been run first (needs ``tensor_meta`` for
determining the number of dimensions).
"""

from collections.abc import Sequence
from typing import cast

import torch
from torch import fx

__all__ = ["DimsProp", "DimsType"]

DimsType = tuple[int, ...]

TRANSPOSE_TARGETS = {torch.transpose, "transpose", "transpose_"}
PERMUTE_TARGETS = {torch.permute, "permute", "permute_"}
RESHAPE_TARGETS = {"flatten", "reshape", "view"}
CONTIGUOUS_TARGETS = {"contiguous", "contiguous_"}


class DimsProp:
    """Propagate axis ordering through an FX graph.

    After propagation, ``node.meta["dims"]`` contains the output axis
    ordering as a ``tuple[int, ...]``.  Identity ordering ``(0, 1, ...)``
    means no transpose/permute has been applied.

    Requires ``ShapeProp`` to have been run first (needs ``tensor_meta``
    for determining the number of dimensions).
    """

    KEY = "dims"

    def propagate(self, gm: fx.GraphModule) -> None:
        """Propagate dims through *gm* in topological order.

        Relies on ``gm.graph.nodes`` being topologically sorted (an FX
        invariant), so that every predecessor's dims are available when
        a node is visited.
        """
        for node in gm.graph.nodes:
            dims = self._compute_dims(node)
            node.meta[self.KEY] = dims

    def _compute_dims(self, node: fx.Node) -> DimsType:
        if node.op == "placeholder":
            return self._identity(node)

        if node.op == "output":
            args = (
                node.args[0]
                if isinstance(node.args[0], (tuple, list))
                else [node.args[0]]
            )
            for arg in args:
                if isinstance(arg, fx.Node):
                    return self._get(arg)
            return ()

        if node.op == "call_module":
            return self._identity(node)

        if node.op in ("call_function", "call_method"):
            if node.target in TRANSPOSE_TARGETS:
                return self._handle_transpose(node)
            if node.target in PERMUTE_TARGETS:
                return self._handle_permute(node)
            if node.target in RESHAPE_TARGETS:
                return self._identity(node)
            if node.target in CONTIGUOUS_TARGETS:
                return self._inherit(node)
            return self._inherit(node)

        if node.op == "get_attr":
            return self._identity(node)

        return ()

    def _handle_transpose(self, node: fx.Node) -> DimsType:
        input_dims = self._inherit(node)
        a1, a2 = node.args[1:3] if len(node.args) >= 3 else (None, None)

        if isinstance(a1, int) and isinstance(a2, int):
            dim0, dim1 = a1, a2
        elif "dim0" in node.kwargs and "dim1" in node.kwargs:
            kw0, kw1 = node.kwargs["dim0"], node.kwargs["dim1"]
            if isinstance(kw0, int) and isinstance(kw1, int):
                dim0, dim1 = kw0, kw1
            else:
                return input_dims
        else:
            return input_dims

        dims_list = list(input_dims)
        dims_list[dim0], dims_list[dim1] = dims_list[dim1], dims_list[dim0]
        return tuple(dims_list)

    def _handle_permute(self, node: fx.Node) -> DimsType:
        input_dims = self._inherit(node)
        dims_order: Sequence[int]

        if len(node.args) >= 2:
            perm = node.args[1]
            if isinstance(perm, (tuple, list)):
                dims_order = cast(list[int], perm)
            else:
                dims_order = cast(list[int], list(node.args[1:]))
        elif "dims" in node.kwargs:
            kw = node.kwargs["dims"]
            if isinstance(kw, (tuple, list)):
                dims_order = cast(list[int], kw)
            else:
                return input_dims
        else:
            return input_dims

        return tuple(input_dims[i] for i in dims_order)

    def _identity(self, node: fx.Node) -> DimsType:
        ndim = self._get_ndim(node)
        return tuple(range(ndim)) if ndim > 0 else ()

    def _inherit(self, node: fx.Node) -> DimsType:
        """Inherit dims from first input node."""
        for inp in node.all_input_nodes:
            dims = self._get(inp)
            if dims:
                return dims
        return ()

    def _get(self, node: fx.Node) -> DimsType:
        return node.meta.get(self.KEY, ())

    def _get_ndim(self, node: fx.Node) -> int:
        meta = node.meta.get("tensor_meta")
        if meta is not None and hasattr(meta, "shape"):
            return len(meta.shape)
        return 0
