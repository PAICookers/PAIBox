import operator
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import torch
from paicorelib import (
    AddPotentialMode,
    DataSign,
    DataWidth,
    OutputType,
    PoolingMode,
    SNNMode,
    ThresholdNegMode,
)
from spikingjelly.activation_based.neuron import IFNode, LIFNode
from torch import fx, nn

from .clac_params import NeuV2ClacParams as NeuronParams
from .clac_params import OfflineCoreV2CalcParams as CoreParams
from .data_semantic_type import infer_input_specs_by_semantic_type
from .ir_base import PAIIR, OpLoc
from .layout_annotate import get_node_semantic_type
from .lut_activation import LutActivation, LutReLU, LutSigmoid
from .neuron import NeuronV2, SJIFNode, SJLIFNode
from .opset import (
    is_module_supported_comp,
    is_module_supported_maxpool,
    is_node_supported_comp,
    is_node_supported_neu_act,
)

__all__ = [
    "BaseCoreOp",
    "SeqCoreOp",
    "AccumCoreOp",
    "CalcCoreOp",
    "SingleCompOp",
    "SingleNeuLUTOp",
    "CPUOp",
]


def _infer_input_specs(
    nodes: Sequence[fx.Node] | fx.Node, modules: dict[str, Any]
) -> tuple[DataSign, DataWidth]:
    if isinstance(nodes, fx.Node):
        nodes = [nodes]

    signs = []
    widths = []
    for node in nodes:
        for prev in node.all_input_nodes:
            if prev.op == "call_module":
                assert isinstance(prev.target, str)
                mod = modules[prev.target]
                if isinstance(mod, BaseCoreOp):
                    sign = mod.core_params.output_sign
                    width = mod.core_params.output_width
                else:
                    # Use the semantic type to infer the input semantic type
                    sign, width = infer_input_specs_by_semantic_type(
                        get_node_semantic_type(prev)
                    )

                signs.append(sign)
                widths.append(width)

    if len(signs) == len(widths) == 0:
        return DataSign.SIGNED, DataWidth.WIDTH_8BIT
    return max(signs), max(widths)


def _infer_output_specs(
    op: NeuronV2 | LutActivation,
) -> tuple[DataSign, DataWidth]:
    if isinstance(op, (SJIFNode, SJLIFNode)):
        sign = DataSign.UNSIGNED
        width = DataWidth.WIDTH_1BIT
    elif isinstance(op, NeuronV2):  # other user-defined neuron
        if op.thres_neg_mode == ThresholdNegMode.FIRE:
            # will generate negative spikes
            sign = DataSign.SIGNED
            width = DataWidth.WIDTH_2BIT
        else:
            sign = DataSign.UNSIGNED
            width = DataWidth.WIDTH_1BIT
    elif isinstance(op, (LutReLU, LutSigmoid)):
        sign = DataSign.UNSIGNED
        width = DataWidth.WIDTH_8BIT
    else:  # other lut
        sign = DataSign.SIGNED
        width = DataWidth.WIDTH_8BIT

    return sign, width


def make_op2(
    node: fx.Node, modules: dict[str, Any], **neu_kwargs: Any
) -> NeuronV2 | LutActivation:
    if not is_node_supported_neu_act(node, modules):
        raise TypeError(f"unsupported module: {torch.typename(node)}")

    assert isinstance(node.target, str)
    mod = modules[node.target]

    if isinstance(mod, nn.ReLU):
        op = LutReLU()
    elif isinstance(mod, nn.Sigmoid):
        op = LutSigmoid()
    elif isinstance(mod, IFNode):
        op = SJIFNode(mod.v_threshold, mod.v_reset, **neu_kwargs)
    elif isinstance(mod, LIFNode):
        op = SJLIFNode(
            mod.tau,
            mod.decay_input,
            mod.v_threshold,
            mod.v_reset,
            **neu_kwargs,
        )
    else:
        raise TypeError(f"unsupported module: {torch.typename(mod)}")

    return op


def _get_pooling_mode(op: nn.Module) -> PoolingMode:
    return PoolingMode.MAX if is_module_supported_maxpool(op) else PoolingMode.AVERAGE


def _get_snn_mode(op: NeuronV2 | LutActivation) -> SNNMode:
    return SNNMode.SNN if isinstance(op, NeuronV2) else SNNMode.ANN


def _get_sign(node: fx.Node) -> tuple[int, int]:
    if node.target in (operator.add, torch.add):
        return (1, 1)
    elif node.target in (operator.sub, torch.sub):
        return (1, -1)
    else:
        return (1, 1)


@dataclass
class ComputeParams:
    # Operation signs for accumulation (e.g., (1, 1) for add, (1, -1) for sub)
    op_signs: tuple[int, ...] = field(default_factory=tuple)


class BaseCoreOp(nn.Module, PAIIR):
    def __init__(
        self,
        core_params: CoreParams,
        neuron_params: NeuronParams | None,
        compute_params: ComputeParams | None,
        neuron_op: nn.Module | None,
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
    ) -> None:
        super().__init__()
        super(nn.Module, self).__init__()
        self.core_params = core_params
        self.neuron_params = neuron_params
        self.compute_params = compute_params
        self.neuron_op = neuron_op or nn.Identity()
        self.op_loc = op_loc

        # TODO
        self.t_local = 0
        self.tick_start = core_params.tick_start
        self.tick_duration = core_params.tick_duration
        self.tick_initial = core_params.tick_initial

    def is_working(self) -> bool:
        if self.tick_start == 0 or self.t_local < self.tick_start:
            return False
        if self.tick_duration > 0 and self.t_local >= self.tick_duration:
            return False
        return True

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs): ...

    def get_attrs(self) -> tuple[dict[str, Any], NeuronParams, dict[str, Any]]:
        core_attrs = asdict(self.core_params)

        if self.neuron_params is not None:
            neu_attrs = self.neuron_params
        elif isinstance(self.neuron_op, NeuronV2):
            neu_attrs_dict = self.neuron_op.get_attrs()
            neu_attrs_dict.setdefault("output_type", OutputType.VALUE)
            neu_attrs = NeuronParams(**neu_attrs_dict)
        else:  # Use default parameters
            neu_attrs = NeuronParams()

        if self.compute_params is not None:
            comp_attrs = asdict(self.compute_params)
        else:
            comp_attrs = asdict(ComputeParams())

        return core_attrs, neu_attrs, comp_attrs

    def extra_repr(self) -> str:
        return f"op_loc={self.op_loc.name}"


class SeqCoreOp(BaseCoreOp):
    def __init__(
        self,
        op1: nn.Module,
        op2: NeuronV2 | LutActivation,
        core_params: CoreParams | None = None,
        neuron_params: NeuronParams | None = None,
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
    ) -> None:
        if not is_module_supported_comp(op1):
            raise TypeError(f"unsupported module: {torch.typename(op1)}")

        if core_params is None:
            core_params = CoreParams()

        core_params.snn_ann = _get_snn_mode(op2)
        core_params.max_pooling = _get_pooling_mode(op1)
        out_sign, out_width = _infer_output_specs(op2)
        core_params.output_sign = out_sign
        core_params.output_width = out_width

        super().__init__(core_params, neuron_params, None, op2, op_loc)
        self.op1 = op1

    @classmethod
    def build(
        cls,
        i_node: fx.Node,
        o_node: fx.Node,
        modules: dict[str, fx.GraphModule],
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
        **kwargs,
    ):
        assert isinstance(i_node.target, str)
        op1 = modules[i_node.target]
        op2 = make_op2(o_node, modules, **kwargs.copy())

        core_params = CoreParams()
        in_sign, in_width = _infer_input_specs(i_node, modules)
        core_params.input_sign = in_sign
        core_params.input_width = in_width

        for k, v in kwargs.items():
            if hasattr(core_params, k):
                setattr(core_params, k, v)

        return cls(op1, op2, core_params, None, op_loc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.op1(x.float())
        return self.neuron_op(out)


class AccumCoreOp(BaseCoreOp):
    def __init__(
        self,
        op1: Sequence[nn.Module],
        op2: NeuronV2 | LutActivation,
        core_params: CoreParams | None = None,
        compute_params: ComputeParams | None = None,
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
    ) -> None:
        # TODO check op1 are of same type
        for op in op1:
            if not is_module_supported_comp(op):
                raise TypeError(f"unsupported module: {torch.typename(op)}")

        if core_params is None:
            core_params = CoreParams()

        core_params.snn_ann = _get_snn_mode(op2)
        core_params.max_pooling = _get_pooling_mode(op1[0])
        out_sign, out_width = _infer_output_specs(op2)
        core_params.output_sign = out_sign
        core_params.output_width = out_width

        if compute_params is None:
            compute_params = ComputeParams(op_signs=(1,) * len(op1))
        assert len(compute_params.op_signs) == len(op1)

        super().__init__(core_params, None, compute_params, op2, op_loc)
        self.op1 = nn.ModuleList(op1)
        self.register_buffer(
            "signs", torch.tensor(compute_params.op_signs, dtype=torch.int8)
        )

    @classmethod
    def build(
        cls,
        i_nodes: Sequence[fx.Node],
        o_node: fx.Node,
        modules: dict[str, fx.GraphModule],
        implicit_sum_signs: tuple[int, ...] | None = None,
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
        **kwargs,
    ):
        op1 = []
        for i_node in i_nodes:
            if not is_node_supported_comp(i_node, modules):
                raise TypeError(f"unsupported module: {torch.typename(i_node)}")

            assert isinstance(i_node.target, str)
            op1.append(modules[i_node.target])

        op2 = make_op2(o_node, modules, **kwargs.copy())
        in_sign, in_width = _infer_input_specs(i_nodes, modules)
        core_params = CoreParams()
        core_params.input_sign = in_sign
        core_params.input_width = in_width

        for k, v in kwargs.items():
            if hasattr(core_params, k):
                setattr(core_params, k, v)

        if implicit_sum_signs is None:
            signs = (1,) * len(op1)
        elif len(implicit_sum_signs) != len(op1):
            raise ValueError(
                f"number of signs ({len(implicit_sum_signs)}) must match number of op1 ({len(op1)})"
            )
        else:
            signs = implicit_sum_signs

        compute_params = ComputeParams(op_signs=signs)

        return cls(op1, op2, core_params, compute_params, op_loc)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        s = 0
        for sign, op, x in zip(self.signs, self.op1, xs):
            output = op(x.float())
            s += sign * output
        return self.neuron_op(s)


class CalcCoreOp(BaseCoreOp):
    def __init__(
        self,
        core_params: CoreParams | None = None,
        compute_params: ComputeParams | None = None,
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
    ) -> None:
        if core_params is None:
            core_params = CoreParams()
        core_params.add_potential = AddPotentialMode.DIRECT_ADD

        neuron_params = NeuronParams(output_type=OutputType.POTENTIAL)
        if compute_params is None:
            compute_params = ComputeParams(op_signs=(1, 1))
        assert len(compute_params.op_signs) == 2

        super().__init__(core_params, neuron_params, compute_params, None, op_loc)
        self.register_buffer(
            "signs", torch.tensor(compute_params.op_signs, dtype=torch.int8)
        )

    @classmethod
    def build(
        cls,
        node: fx.Node,
        modules: dict[str, fx.GraphModule],
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
        **kwargs,
    ):
        in_sign, in_width = _infer_input_specs(node, modules)

        core_params = CoreParams()
        core_params.input_sign = in_sign
        core_params.input_width = in_width
        compute_params = ComputeParams(op_signs=_get_sign(node))

        for k, v in kwargs.items():
            if hasattr(core_params, k):
                setattr(core_params, k, v)

        return cls(core_params, compute_params, op_loc)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        s = 0
        for sign, x in zip(self.signs, xs):
            s += sign * x.float()
        return self.neuron_op(s)


class SingleCompOp(BaseCoreOp):
    def __init__(
        self,
        op: nn.Module,
        core_params: CoreParams | None = None,
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
    ) -> None:
        if not is_module_supported_comp(op):
            raise TypeError(f"unsupported module: {torch.typename(op)}")

        if core_params is None:
            core_params = CoreParams()

        core_params.snn_ann = SNNMode.ANN
        core_params.max_pooling = _get_pooling_mode(op)

        neuron_params = NeuronParams(output_type=OutputType.POTENTIAL)
        super().__init__(core_params, neuron_params, None, None, op_loc)
        self.op = op

    @classmethod
    def build(
        cls,
        node: fx.Node,
        modules: dict[str, fx.GraphModule],
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
        **kwargs,
    ):
        assert isinstance(node.target, str)
        op = modules[node.target]

        core_params = CoreParams()
        in_sign, in_width = _infer_input_specs(node, modules)
        core_params.input_sign = in_sign
        core_params.input_width = in_width

        for k, v in kwargs.items():
            if hasattr(core_params, k):
                setattr(core_params, k, v)

        return cls(op, core_params, op_loc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x.float())


class SingleNeuLUTOp(BaseCoreOp):
    @classmethod
    def build(
        cls,
        node: fx.Node,
        modules: dict[str, fx.GraphModule],
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
        **kwargs,
    ):
        op2 = make_op2(node, modules, **kwargs.copy())

        core_params = CoreParams()
        in_sign, in_width = _infer_input_specs(node, modules)
        core_params.input_sign = in_sign
        core_params.input_width = in_width
        core_params.snn_ann = _get_snn_mode(op2)

        out_sign, out_width = _infer_output_specs(op2)
        core_params.output_sign = out_sign
        core_params.output_width = out_width

        for k, v in kwargs.items():
            if hasattr(core_params, k):
                setattr(core_params, k, v)

        return cls(core_params, None, None, op2, op_loc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neuron_op(x.float())


class CPUOp(nn.Module, PAIIR):
    op_loc: ClassVar[OpLoc] = OpLoc.CPU
