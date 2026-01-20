from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import torch
from paicorelib import (
    AddPotentialMode,
    OutputType,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
)
from spikingjelly.activation_based.neuron import IFNode, LIFNode
from torch import fx, nn

from .ir_base import PAIIR, OpLoc
from .lut_activation import LutActivation, LutReLU, LutSigmoid
from .neuron import NeuronV2, SJIFNode, SJLIFNode
from .opset import is_node_supported_comp


class CoreOpNode(nn.Module, PAIIR):
    _attrs_to_save: tuple[str, ...] = (
        "snn_ann",
        "max_pooling",
        "add_potential",
        "zero_output",
        "input_sign",
        "input_width",
        "output_sign",
        "output_width",
        "weight_sign",
        "weight_width",
        # "tick_start", # ?
        # "tick_duration", # ?
        # "tick_initial", # ?
    )

    def __init__(
        self,
        i_ops: Sequence[nn.Module],
        o_op: LutActivation | NeuronV2,
        implicit_sum_signs: list[Literal[1, -1]] | None = None,  # TODO
        op_loc: OpLoc = OpLoc.OFFLINE_CORE,
        **kwargs,
    ) -> None:
        super().__init__()
        super(nn.Module, self).__init__()
        self.op_loc = op_loc
        self.op1 = nn.ModuleList(i_ops)
        self.op2 = o_op

        if implicit_sum_signs is None:
            signs = [1 for _ in self.op1]
        elif len(implicit_sum_signs) != len(self.op1):
            raise ValueError(
                f"number of signs ({len(implicit_sum_signs)}) must match number of ops ({len(self.op1)})"
            )

        self.register_buffer("signs", torch.tensor(signs, dtype=torch.bool))

        self._paser_core_attrs(**kwargs)

    @classmethod
    def build(
        cls,
        i_nodes: Sequence[fx.Node],
        o_node: fx.Node,
        modules: dict[str, fx.GraphModule],
        op_loc: OpLoc,
        **kwargs,
    ):
        op1 = []
        for i_node in i_nodes:
            if not is_node_supported_comp(i_node, modules):
                raise TypeError(f"unsupported module: {torch.typename(i_node)}")

            op1.append(modules[i_node.target])

        assert isinstance(o_node.target, str)
        m_op2 = modules[o_node.target]

        if isinstance(m_op2, nn.ReLU):
            op2 = LutReLU()
        elif isinstance(m_op2, nn.Sigmoid):
            op2 = LutSigmoid()
        elif isinstance(m_op2, IFNode):
            op2 = SJIFNode(m_op2.v_threshold, m_op2.v_reset, **kwargs)
        elif isinstance(m_op2, LIFNode):
            op2 = SJLIFNode(
                m_op2.tau,
                m_op2.decay_input,
                m_op2.v_threshold,
                m_op2.v_reset,
                **kwargs,
            )
        else:
            raise TypeError(f"unsupported module: {torch.typename(m_op2)}")

        return cls(op1, op2, None, op_loc)

    def _paser_core_attrs(self, **kwargs) -> None:
        self.snn_ann = (
            SNNMode.ANN if isinstance(self.op2, LutActivation) else SNNMode.SNN
        )
        self.max_pooling = (
            PoolingMode.MAX
            if isinstance(self.op1[0], (nn.MaxPool1d, nn.MaxPool2d))
            else PoolingMode.AVERAGE
        )
        self.add_potential = AddPotentialMode.NORMAL
        if self.add_potential == AddPotentialMode.DIRECT_ADD:
            self.zero_output = ZeroOutputMode.ENABLE
            # this is neuron's attribute
            self.output_type = OutputType.POTENTIAL
        else:
            self.zero_output = ZeroOutputMode.DISABLE
            self.output_type = OutputType.VALUE

        # TODO Need pass these attributes from outside
        # input_sign
        # input_width
        # output_sign
        # output_width
        # weight_sign
        # weight_width

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        s = 0
        for sign, op, x in zip(self.signs, self.op1, xs, strict=True):
            output = op(x)
            s += sign * output

        return self.op2(s)

    def get_attrs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(self.op2, LutActivation):
            # XXX do we need configurate neuron if this core is in ANN mode?
            neu_attrs = SJIFNode().get_attrs()
        else:
            neu_attrs = self.op2.get_attrs()

        core_attrs = {}
        for attr in self._attrs_to_save:
            core_attrs[attr] = getattr(self, attr)

        return core_attrs, neu_attrs

    def extra_repr(self) -> str:
        return f"op_loc={self.op_loc.name}"


class SeqCoreOpNode(CoreOpNode):
    @classmethod
    def build(
        cls,
        i_node: fx.Node,
        o_node: fx.Node,
        modules: dict[str, fx.GraphModule],
        op_loc: OpLoc,
        **kwargs,
    ):
        return super().build([i_node], o_node, modules, op_loc, **kwargs)


class OfflineSeqCoreOpNode(SeqCoreOpNode):
    @classmethod
    def build(
        cls,
        i_node: fx.Node,
        o_node: fx.Node,
        modules: dict[str, fx.GraphModule],
        **kwargs,
    ):
        return super().build(i_node, o_node, modules, OpLoc.OFFLINE_CORE, **kwargs)


class CPUOpNode(nn.Module, PAIIR):
    op_loc: ClassVar[OpLoc] = OpLoc.CPU
