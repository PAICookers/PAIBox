from collections.abc import Sequence
from dataclasses import asdict, dataclass
import operator
from typing import Any, ClassVar, Literal

import torch
from torch import fx, nn

from paicorelib import (
    AddPotentialMode,
    InputSignMode,
    OutputSignMode,
    OutputType,
    PoolingMode,
    SNNMode,
    ThresholdNegMode,
    WeightSignMode,
    ZeroOutputMode,
)
from paicorelib.core_defs import WeightWidth
from spikingjelly.activation_based.neuron import IFNode, LIFNode

from .ir_base import PAIIR, OpLoc
from .lut_activation import LutActivation, LutReLU, LutSigmoid
from .neuron import NeuronV2, SJIFNode, SJLIFNode
from .opset import is_node_supported_comp, SUPPORTED_POOL_OPS

__all__ = ["SeqCoreOp", "AccumCoreOp",
           "SingleConvMaxOp", "SingleNeuLUTOp", "CalcCoreOp"]


@dataclass
class CoreParams:
    snn_ann: int | SNNMode = SNNMode.SNN
    max_pooling: int | PoolingMode = PoolingMode.AVERAGE
    add_potential: int | AddPotentialMode = AddPotentialMode.NORMAL
    zero_output: int | ZeroOutputMode = ZeroOutputMode.DISABLE
    input_sign: int | InputSignMode = InputSignMode.SIGNED
    input_width: int | WeightWidth = WeightWidth.WEIGHT_WIDTH_8BIT
    output_sign: int | OutputSignMode = OutputSignMode.SIGNED
    output_width: int | WeightWidth = WeightWidth.WEIGHT_WIDTH_8BIT
    weight_sign: int | WeightSignMode = WeightSignMode.SIGNED
    weight_width: int | WeightWidth = WeightWidth.WEIGHT_WIDTH_8BIT


@dataclass
class ComputeParams:
    # Operation signs for accumulation (e.g., [1, 1] for add, [1, -1] for sub)
    op_signs: list[int] | None = None


@dataclass
class NeuronParams:
    output_type: int | OutputType = OutputType.VALUE


class BaseCoreOp(nn.Module, PAIIR):
    def __init__(
        self,
        core_params: CoreParams,
        neuron_params: NeuronParams,
        compute_params: ComputeParams,
        neuron_op: nn.Module | None,
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
    ):
        super().__init__()
        super(nn.Module, self).__init__()
        self.core_params = core_params
        self.neuron_params = neuron_params
        self.compute_params = compute_params
        self.neuron_op = neuron_op or nn.Identity()
        self.op_loc = op_loc

    def get_attrs(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        neu_attrs = (
            self.neuron_op.get_attrs()
            if hasattr(self.neuron_op, "get_attrs")
            else {}
        )
        neu_params_dict = asdict(self.neuron_params)
        neu_attrs.update(neu_params_dict)

        core_attrs = asdict(self.core_params)
        comp_attrs = asdict(self.compute_params)

        return core_attrs, neu_attrs, comp_attrs

    def extra_repr(self) -> str:
        return f"op_loc={self.op_loc.name}"


def infer_input_specs(
    i_nodes: Sequence[fx.Node], modules: dict[str, Any]
) -> tuple[InputSignMode, WeightWidth]:
    input_signs = []
    input_widths = []

    for node in i_nodes:
        if isinstance(node, fx.Node):
            # If the node itself is a CoreOp (already fused module), check its output specs directly
            if node.target in modules:
                mod = modules[node.target]
                if hasattr(mod, "core_params"):
                    input_signs.append(mod.core_params.output_sign)
                    input_widths.append(mod.core_params.output_width)
                    continue  # Found specs for this input, move to next node
                elif hasattr(mod, "output_sign"):
                    input_signs.append(mod.output_sign)
                    if hasattr(mod, "output_width"):
                        input_widths.append(mod.output_width)
                    continue

            # Fallback: Check input nodes of the compute node (original logic)
            # This logic seems to assume i_nodes are Compute Ops (like Conv) and we look at THEIR inputs.
            # But in SeqCoreOp.build(i_node, o_node), i_node is the Compute Op.
            # Its input specs should determine the 'input_sign/width' of the SeqCoreOp.

            for inp in node.all_input_nodes:
                if inp.target in modules:
                    mod = modules[inp.target]
                    if hasattr(mod, "core_params"):
                        input_signs.append(mod.core_params.output_sign)
                        input_widths.append(mod.core_params.output_width)
                    elif hasattr(mod, "output_sign"):
                        input_signs.append(mod.output_sign)
                        if hasattr(mod, "output_width"):
                            input_widths.append(mod.output_width)
                    elif isinstance(mod, (IFNode, LIFNode, SJIFNode, SJLIFNode)):
                        input_signs.append(InputSignMode.UNSIGNED)
                        input_widths.append(WeightWidth.WEIGHT_WIDTH_1BIT)

    in_sign = (
        InputSignMode.SIGNED
        if (input_signs and max(input_signs) == InputSignMode.SIGNED)
        else InputSignMode.UNSIGNED
    )

    in_width = (
        max(input_widths) if input_widths else WeightWidth.WEIGHT_WIDTH_8BIT
    )
    return in_sign, in_width


class SeqCoreOp(BaseCoreOp):
    def __init__(
        self,
        op1: nn.Module,
        op2: nn.Module,
        core_params: CoreParams,
        neuron_params: NeuronParams,
        compute_params: ComputeParams,
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
    ):
        super().__init__(core_params, neuron_params, compute_params, op2, op_loc)
        self.op1 = op1
        self.register_buffer("sign", torch.tensor([1], dtype=torch.int8))

    @classmethod
    def build(
        cls,
        i_node: fx.Node,
        o_node: fx.Node,
        modules: dict[str, fx.GraphModule],
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
        **kwargs,
    ):
        if not is_node_supported_comp(i_node, modules):
            raise TypeError(f"unsupported module: {torch.typename(i_node)}")

        op1 = modules[i_node.target]
        m_op2 = modules[o_node.target]

        in_sign, in_width = infer_input_specs([i_node], modules)

        neu_kwargs = kwargs.copy()

        # Determine neuron
        if isinstance(m_op2, nn.ReLU):
            op2 = LutReLU()
        elif isinstance(m_op2, nn.Sigmoid):
            op2 = LutSigmoid()
        elif isinstance(m_op2, IFNode):
            op2 = SJIFNode(m_op2.v_threshold, m_op2.v_reset, **neu_kwargs)
        elif isinstance(m_op2, LIFNode):
            op2 = SJLIFNode(
                m_op2.tau,
                m_op2.decay_input,
                m_op2.v_threshold,
                m_op2.v_reset,
                **neu_kwargs,
            )
        else:
            raise TypeError(f"unsupported module: {torch.typename(m_op2)}")

        core_params = CoreParams()
        core_params.input_sign = in_sign
        core_params.input_width = in_width
        core_params.snn_ann = (
            SNNMode.ANN if isinstance(op2, LutActivation) else SNNMode.SNN
        )
        core_params.max_pooling = (
            PoolingMode.MAX
            if isinstance(op1, (nn.MaxPool1d, nn.MaxPool2d))
            else PoolingMode.AVERAGE
        )

        compute_params = ComputeParams()
        compute_params.op_signs = [1]

        if isinstance(op2, (SJIFNode, SJLIFNode)):
            core_params.output_sign = OutputSignMode.UNSIGNED
            core_params.output_width = WeightWidth.WEIGHT_WIDTH_1BIT
        elif isinstance(op2, NeuronV2):
            if op2.thres_neg_mode == ThresholdNegMode.FIRE:
                core_params.output_sign = OutputSignMode.SIGNED
                core_params.output_width = WeightWidth.WEIGHT_WIDTH_2BIT
            else:
                core_params.output_sign = OutputSignMode.UNSIGNED
                core_params.output_width = WeightWidth.WEIGHT_WIDTH_1BIT
        elif isinstance(op2, (LutReLU, LutSigmoid)):
            core_params.output_sign = OutputSignMode.UNSIGNED
            core_params.output_width = WeightWidth.WEIGHT_WIDTH_8BIT
        else:
            core_params.output_sign = OutputSignMode.SIGNED
            core_params.output_width = WeightWidth.WEIGHT_WIDTH_8BIT

        # Set user params
        for k, v in kwargs.items():
            if hasattr(core_params, k):
                setattr(core_params, k, v)

        neuron_params = NeuronParams()
        neuron_params.output_type = OutputType.VALUE  # Default for Seq

        op = cls(op1, op2, core_params, neuron_params, compute_params, op_loc)
        return op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.op1(x.float())
        return self.neuron_op(out)


class AccumCoreOp(BaseCoreOp):
    def __init__(
        self,
        ops: Sequence[nn.Module],
        op2: nn.Module,
        core_params: CoreParams,
        neuron_params: NeuronParams,
        compute_params: ComputeParams,
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
    ):
        super().__init__(core_params, neuron_params, compute_params, op2, op_loc)
        self.ops = nn.ModuleList(ops)
        self.register_buffer(
            "signs", torch.tensor(compute_params.op_signs, dtype=torch.int8)
        )

    @classmethod
    def build(
        cls,
        i_nodes: Sequence[fx.Node],
        o_node: fx.Node,
        modules: dict[str, fx.GraphModule],
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
        implicit_sum_signs: list[Literal[1, -1]] | None = None,
        **kwargs,
    ):
        ops = []
        for i_node in i_nodes:
            if not is_node_supported_comp(i_node, modules):
                raise TypeError(
                    f"unsupported module: {torch.typename(i_node)}")
            ops.append(modules[i_node.target])

        if implicit_sum_signs is None:
            signs = [1 for _ in ops]
        elif len(implicit_sum_signs) != len(ops):
            raise ValueError(
                f"number of signs ({len(implicit_sum_signs)}) must match number of ops ({len(ops)})"
            )
        else:
            signs = implicit_sum_signs

        in_sign, in_width = infer_input_specs(i_nodes, modules)

        m_op2 = modules[o_node.target]
        neu_kwargs = kwargs.copy()

        if isinstance(m_op2, nn.ReLU):
            op2 = LutReLU()
        elif isinstance(m_op2, nn.Sigmoid):
            op2 = LutSigmoid()
        elif isinstance(m_op2, IFNode):
            op2 = SJIFNode(m_op2.v_threshold, m_op2.v_reset, **neu_kwargs)
        elif isinstance(m_op2, LIFNode):
            op2 = SJLIFNode(
                m_op2.tau,
                m_op2.decay_input,
                m_op2.v_threshold,
                m_op2.v_reset,
                **neu_kwargs,
            )
        else:
            raise TypeError(f"unsupported module: {torch.typename(m_op2)}")

        core_params = CoreParams()
        core_params.input_sign = in_sign
        core_params.input_width = in_width
        core_params.snn_ann = (
            SNNMode.ANN if isinstance(op2, LutActivation) else SNNMode.SNN
        )
        core_params.max_pooling = PoolingMode.AVERAGE
        if len(ops) > 0 and isinstance(ops[0], (nn.MaxPool1d, nn.MaxPool2d)):
            core_params.max_pooling = PoolingMode.MAX

        compute_params = ComputeParams()
        compute_params.op_signs = signs

        if isinstance(op2, (SJIFNode, SJLIFNode)):
            core_params.output_sign = OutputSignMode.UNSIGNED
            core_params.output_width = WeightWidth.WEIGHT_WIDTH_1BIT
        elif isinstance(op2, NeuronV2):
            if op2.thres_neg_mode == ThresholdNegMode.FIRE:
                core_params.output_sign = OutputSignMode.SIGNED
                core_params.output_width = WeightWidth.WEIGHT_WIDTH_2BIT
            else:
                core_params.output_sign = OutputSignMode.UNSIGNED
                core_params.output_width = WeightWidth.WEIGHT_WIDTH_1BIT
        elif isinstance(op2, (LutReLU, LutSigmoid)):
            core_params.output_sign = OutputSignMode.UNSIGNED
            core_params.output_width = WeightWidth.WEIGHT_WIDTH_8BIT
        else:
            core_params.output_sign = OutputSignMode.SIGNED
            core_params.output_width = WeightWidth.WEIGHT_WIDTH_8BIT

        for k, v in kwargs.items():
            if hasattr(core_params, k):
                setattr(core_params, k, v)

        neuron_params = NeuronParams()
        neuron_params.output_type = OutputType.VALUE

        op = cls(ops, op2, core_params, neuron_params, compute_params, op_loc)
        return op

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        s = 0
        for sign, op, x in zip(self.signs, self.ops, xs):
            output = op(x.float())
            s += sign * output
        return self.neuron_op(s)


class CalcCoreOp(BaseCoreOp):
    def __init__(
        self,
        core_params: CoreParams,
        neuron_params: NeuronParams,
        compute_params: ComputeParams,
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
    ):
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
        in_sign, in_width = infer_input_specs(node.all_input_nodes, modules)

        # Determine signs based on operation type
        if node.target in [operator.add, torch.add]:
            signs = [1, 1]
        elif node.target in [operator.sub, torch.sub]:
            signs = [1, -1]
        else:
            # Default or fallback
            signs = [1, 1]

        core_params = CoreParams()
        core_params.input_sign = in_sign
        core_params.input_width = in_width
        core_params.snn_ann = SNNMode.ANN
        core_params.max_pooling = PoolingMode.AVERAGE
        core_params.add_potential = AddPotentialMode.DIRECT_ADD

        compute_params = ComputeParams()
        compute_params.op_signs = signs

        # Set user params
        for k, v in kwargs.items():
            if hasattr(core_params, k):
                setattr(core_params, k, v)

        neuron_params = NeuronParams()
        neuron_params.output_type = OutputType.POTENTIAL

        op = cls(core_params, neuron_params, compute_params, op_loc)
        return op

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        s = 0
        for sign, x in zip(self.signs, xs):
            s += sign * x.float()
        return self.neuron_op(s)


class SingleConvMaxOp(BaseCoreOp):
    def __init__(
        self,
        op: nn.Module,
        core_params: CoreParams,
        neuron_params: NeuronParams,
        compute_params: ComputeParams,
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
    ):
        super().__init__(core_params, neuron_params, compute_params, None, op_loc)
        self.op = op

    @classmethod
    def build(
        cls,
        node: fx.Node,
        modules: dict[str, fx.GraphModule],
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
        **kwargs,
    ):
        if not is_node_supported_comp(node, modules):
            raise TypeError(f"unsupported module: {torch.typename(node)}")

        op = modules[node.target]
        in_sign, in_width = infer_input_specs([node], modules)

        core_params = CoreParams()
        core_params.input_sign = in_sign
        core_params.input_width = in_width
        core_params.snn_ann = SNNMode.ANN
        core_params.max_pooling = (
            PoolingMode.MAX
            if isinstance(op, tuple(SUPPORTED_POOL_OPS))
            else PoolingMode.AVERAGE
        )

        compute_params = ComputeParams()
        compute_params.op_signs = [1]

        core_params.output_sign = OutputSignMode.SIGNED
        core_params.output_width = WeightWidth.WEIGHT_WIDTH_8BIT

        for k, v in kwargs.items():
            if hasattr(core_params, k):
                setattr(core_params, k, v)

        neuron_params = NeuronParams()
        neuron_params.output_type = OutputType.POTENTIAL  # As requested

        ret = cls(op, core_params, neuron_params, compute_params, op_loc)
        return ret

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x.float())


class SingleNeuLUTOp(BaseCoreOp):
    def __init__(
        self,
        op2: nn.Module,
        core_params: CoreParams,
        neuron_params: NeuronParams,
        compute_params: ComputeParams,
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
    ):
        super().__init__(core_params, neuron_params, compute_params, op2, op_loc)

    @classmethod
    def build(
        cls,
        node: fx.Node,
        modules: dict[str, fx.GraphModule],
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
        **kwargs,
    ):
        m_op2 = modules[node.target]
        op_prev = node.all_input_nodes[0]
        # Since op_prev might not be a module (e.g., placeholder), we need a helper or handle it.
        # infer_input_specs expects a sequence of nodes (typically compute nodes).
        # We can pass [op_prev] or just treat input as unknown if not a module.
        # Wait, infer_input_specs iterates all_input_nodes of the passed node.
        # So we should pass a dummy node whose input is op_prev?
        # No, infer_input_specs logic: `for comp_node in i_nodes: for inp in comp_node.all_input_nodes:`
        # That logic assumes i_nodes is the current layer (like Conv) and looks at its inputs.
        # Here we don't have a Conv. We are the Neuron. Our input is op_prev.
        # But we want to know the specs of op_prev.
        # So we can look at op_prev directly.

        in_sign = InputSignMode.SIGNED
        in_width = WeightWidth.WEIGHT_WIDTH_8BIT

        if op_prev.target in modules:
            mod = modules[op_prev.target]
            if hasattr(mod, "core_params"):
                in_sign = mod.core_params.output_sign
                in_width = mod.core_params.output_width
            elif hasattr(mod, "output_sign"):
                in_sign = mod.output_sign
                if hasattr(mod, "output_width"):
                    in_width = mod.output_width

        # Determine neuron
        neu_kwargs = kwargs.copy()
        if isinstance(m_op2, nn.ReLU):
            op2 = LutReLU()
        elif isinstance(m_op2, nn.Sigmoid):
            op2 = LutSigmoid()
        elif isinstance(m_op2, IFNode):
            op2 = SJIFNode(m_op2.v_threshold, m_op2.v_reset, **neu_kwargs)
        elif isinstance(m_op2, LIFNode):
            op2 = SJLIFNode(
                m_op2.tau,
                m_op2.decay_input,
                m_op2.v_threshold,
                m_op2.v_reset,
                **neu_kwargs,
            )
        else:
            raise TypeError(f"unsupported module: {torch.typename(m_op2)}")

        core_params = CoreParams()
        core_params.input_sign = in_sign
        core_params.input_width = in_width
        core_params.snn_ann = (
            SNNMode.ANN if isinstance(op2, LutActivation) else SNNMode.SNN
        )
        core_params.max_pooling = PoolingMode.AVERAGE
        core_params.add_potential = AddPotentialMode.DIRECT_ADD

        compute_params = ComputeParams()
        compute_params.op_signs = [1]  # Identity

        if isinstance(op2, (SJIFNode, SJLIFNode)):
            core_params.output_sign = OutputSignMode.UNSIGNED
            core_params.output_width = WeightWidth.WEIGHT_WIDTH_1BIT
        elif isinstance(op2, NeuronV2):
            if op2.thres_neg_mode == ThresholdNegMode.FIRE:
                core_params.output_sign = OutputSignMode.SIGNED
                core_params.output_width = WeightWidth.WEIGHT_WIDTH_2BIT
            else:
                core_params.output_sign = OutputSignMode.UNSIGNED
                core_params.output_width = WeightWidth.WEIGHT_WIDTH_1BIT
        elif isinstance(op2, (LutReLU, LutSigmoid)):
            core_params.output_sign = OutputSignMode.UNSIGNED
            core_params.output_width = WeightWidth.WEIGHT_WIDTH_8BIT
        else:
            # Default fallback
            core_params.output_sign = OutputSignMode.SIGNED
            core_params.output_width = WeightWidth.WEIGHT_WIDTH_8BIT

        # Set user params
        for k, v in kwargs.items():
            if hasattr(core_params, k):
                setattr(core_params, k, v)

        neuron_params = NeuronParams()
        neuron_params.output_type = OutputType.VALUE

        op = cls(op2, core_params, neuron_params, compute_params, op_loc)
        return op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neuron_op(x.float())


class CPUOpNode(nn.Module, PAIIR):
    op_loc: ClassVar[OpLoc] = OpLoc.CPU
