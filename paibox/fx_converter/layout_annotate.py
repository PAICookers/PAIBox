from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor, fx
from torch.fx.passes.shape_prop import ShapeProp

from paibox import _logging

from ..exceptions import NotSupportedError
from .data_semantic_type import DataSemanticType, merge_data_semantic_types
from .opset import (
    is_module_supported_act,
    is_module_supported_comp,
    is_module_supported_neuron,
)

__all__ = [
    "LayoutAnnotator",
    "is_node_dims_annotated",
    "is_node_semantic_type_annotated",
    "get_node_semantic_type",
]

logger = _logging.get_artifact_logger(__name__, "layout_annotate")

DimsType = tuple[int, ...]


@dataclass
class DimsAnnotation:
    input_dims: dict[fx.Node, DimsType] = field(default_factory=dict)
    output_dims: DimsType | None = None
    _annotated: bool = False


DIMS_ANNOTATION_KEY = "dims_annotation"
SEMANTIC_TYPE_KEY = "semantic_type"


def is_node_dims_annotated(node: fx.Node) -> bool:
    return (
        DIMS_ANNOTATION_KEY in node.meta and node.meta[DIMS_ANNOTATION_KEY]._annotated
    )


def _axis_transpose_fn(axes: DimsType, dim0: int, dim1: int) -> DimsType:
    return tuple(
        axes[dim1] if i == dim0 else axes[dim0] if i == dim1 else ax
        for i, ax in enumerate(axes)
    )


def _axis_permute_fn(axes: DimsType, *dims: int) -> DimsType:
    return tuple(axes[i] for i in dims)


CHANGE_AXES_FN = [
    (
        [torch.transpose, "transpose", "transpose_"],
        _axis_transpose_fn,
    ),
    ([torch.permute, "permute", "permute_"], _axis_permute_fn),
]


def _get_change_axes_fn(node: fx.Node) -> Callable | None:
    for ops, fn in CHANGE_AXES_FN:
        if node.target in ops:
            return fn

    return None


def is_node_semantic_type_annotated(node: fx.Node) -> bool:
    return SEMANTIC_TYPE_KEY in node.meta


def get_node_semantic_type(node: fx.Node) -> DataSemanticType:
    if is_node_semantic_type_annotated(node):
        return node.meta[SEMANTIC_TYPE_KEY]
    else:
        return DataSemanticType.UNKNOWN


class LayoutAnnotator:
    def annotate(self, gm: fx.GraphModule, *input: Tensor) -> None:
        self._shape_annotate(gm, *input)
        self._data_semantic_type_annotate(gm)
        self._dims_annotate(gm)

    def _shape_annotate(self, gm: fx.GraphModule, *input: Tensor) -> None:
        for i in input:
            if (batch := i.shape[0]) != 1:
                raise NotSupportedError(
                    f"only support batch size 1 for now, but got {batch}"
                )

        shape_prop = ShapeProp(gm)
        shape_prop.propagate(*input)

    def _dims_annotate(self, gm: fx.GraphModule) -> None:
        """
        Only annotate nodes that change the order of output dims. The order of output dims of perators
        like nn.Linear is default (0,1,...).
        """

        def _get_prev_dims(node: fx.Node):
            input_dims = {}
            for i_node in node.all_input_nodes:
                if is_node_dims_annotated(i_node):
                    input_dims[i_node] = i_node.meta[DIMS_ANNOTATION_KEY].output_dims
                else:
                    shape = i_node.meta["tensor_meta"].shape
                    input_dims[i_node] = tuple(range(len(shape)))

            return input_dims

        for node in gm.graph.nodes:
            if node.op not in ("call_function", "call_method"):
                continue

            if "tensor_meta" not in node.meta:
                raise RuntimeError(f"cannot get shape of node '{node.name}'.")

            if is_node_dims_annotated(node):
                continue

            input_dims = _get_prev_dims(node)

            if (fn := _get_change_axes_fn(node)) is not None:
                assert len(node.all_input_nodes) == 1
                _, *args = node.args
                output_dims = fn(input_dims[node.all_input_nodes[0]], *args)

                for user in node.users:
                    if user.op == "output":  # no need to annotate output
                        continue
                    if is_node_dims_annotated(user):
                        raise RuntimeError(f"Node '{user.name}' is already annotated.")

                    if DIMS_ANNOTATION_KEY in user.meta:
                        user.meta[DIMS_ANNOTATION_KEY].input_dims |= {node: output_dims}
                    else:
                        user.meta[DIMS_ANNOTATION_KEY] = DimsAnnotation(
                            {node: output_dims}, None, _annotated=False
                        )
            else:
                shape = node.meta["tensor_meta"].shape
                output_dims = tuple(range(len(shape)))

            node.meta[DIMS_ANNOTATION_KEY] = DimsAnnotation(
                input_dims, output_dims, _annotated=True
            )
            logger.debug(
                f"Annotate dims of node '{node.name}': {input_dims} -> {output_dims}"
            )

        # Annotate the rest of modules
        for node in gm.graph.nodes:
            if node.op != "call_module":
                continue

            if "tensor_meta" not in node.meta:
                raise RuntimeError(f"cannot get shape of node '{node.name}'.")

            shape = node.meta["tensor_meta"].shape
            output_dims = tuple(range(len(shape)))

            if DIMS_ANNOTATION_KEY in node.meta:
                node.meta[DIMS_ANNOTATION_KEY].output_dims = output_dims
                node.meta[DIMS_ANNOTATION_KEY]._annotated = True
            else:
                input_dims = _get_prev_dims(node)
                node.meta[DIMS_ANNOTATION_KEY] = DimsAnnotation(
                    input_dims, output_dims, _annotated=True
                )

    def _data_semantic_type_annotate(self, gm: fx.GraphModule) -> None:
        """Annotate each fx node's output semantic type.

        Rules:
        - neuron module -> SPIKE
        - activation module -> ACTIVATION
        - conv/linear/pool module -> POTENTIAL
        - other nodes: propagate previous semantic (single input), or merge semantics by
        compatibility for multi-input nodes (max).
        """

        def _merge_input_semantics(nodes: Iterable[fx.Node]) -> DataSemanticType:
            return merge_data_semantic_types([get_node_semantic_type(n) for n in nodes])

        def _merge_all_inputs(node: fx.Node) -> DataSemanticType:
            return _merge_input_semantics(node.all_input_nodes)

        modules = dict(gm.named_modules())
        for node in gm.graph.nodes:
            if node.op == "output":
                continue

            if node.op in ("placeholder", "get_attr"):
                semantic_type = DataSemanticType.UNKNOWN
            elif node.op == "call_module":
                assert isinstance(node.target, str)
                m = modules[node.target]

                if is_module_supported_neuron(m):
                    semantic_type = DataSemanticType.SPIKE
                elif is_module_supported_act(m):
                    semantic_type = DataSemanticType.ACTIVATION
                elif is_module_supported_comp(m):
                    semantic_type = DataSemanticType.POTENTIAL
                else:  # other modules
                    semantic_type = _merge_all_inputs(node)

            elif node.op in ("call_function", "call_method"):
                semantic_type = _merge_all_inputs(node)
            else:
                semantic_type = _merge_all_inputs(node)

            node.meta[SEMANTIC_TYPE_KEY] = semantic_type
            logger.debug(
                f"Annotate semantic type of node '{node.name}': {semantic_type.name}"
            )
