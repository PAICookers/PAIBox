from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import torch
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
        else:
            signs = implicit_sum_signs

        self.register_buffer("signs", torch.tensor(signs, dtype=torch.int8))

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

        # -----------------------------
        # 推断输入规格 (input specs)
        # -----------------------------
        # 逻辑：
        # 1. 查看所有计算节点(i_nodes, 例如Conv/Linear)的每前置输入节点。
        # 2. 如果前置节点是 CoreOpNode 或其他有 output_sign/width 属性的节点，则收集其属性。
        # 3. 如果任一输入是有符号的 (output_sign=1)，则当前节点的输入也被认为是有符号的。
        # 4. 输入位宽取所有输入来源中的最大值。

        input_signs = []
        input_widths = []

        # 遍历所有输入计算节点（如 Conv, Linear）
        for comp_node in i_nodes:
            # 遍历每个计算节点的输入（即上游来源）
            # all_input_nodes 包含了所有输入，包括 Tensor 输入
            for inp in comp_node.all_input_nodes:
                if inp.target in modules:
                    mod = modules[inp.target]
                    # 检查上游模块是否有 output_sign 属性
                    if hasattr(mod, "output_sign"):
                        input_signs.append(mod.output_sign)
                    # 检查上游模块是否有 output_width 属性
                    if hasattr(mod, "output_width"):
                        input_widths.append(mod.output_width)

        # 确定 input_sign: 只要有一个输入是有符号的 (SIGNED=1)，整体就是有符号的；否则为 UNSIGNED=0
        in_sign = (
            InputSignMode.SIGNED
            if (input_signs and max(input_signs) == InputSignMode.SIGNED)
            else InputSignMode.UNSIGNED
        )

        # 确定 input_width: 取最大位宽，默认 8
        in_width = max(input_widths) if input_widths else WeightWidth.WEIGHT_WIDTH_8BIT

        # 将推断出的属性通过 kwargs 传递给构造函数，如果 kwargs 已有通过优先使用
        if "input_sign" not in kwargs:
            kwargs["input_sign"] = in_sign
        if "input_width" not in kwargs:
            kwargs["input_width"] = in_width

        # Filter out arguments that are not for neuron constructors
        neu_kwargs = kwargs.copy()
        neu_kwargs.pop("input_sign", None)
        neu_kwargs.pop("input_width", None)

        assert isinstance(o_node.target, str)
        m_op2 = modules[o_node.target]

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

        return cls(op1, op2, None, op_loc, **kwargs)

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

        # input_sign & input_width
        # Default checked during build
        self.input_sign = InputSignMode(kwargs.get("input_sign", InputSignMode.SIGNED))
        self.input_width = WeightWidth(
            kwargs.get("input_width", WeightWidth.WEIGHT_WIDTH_8BIT)
        )

        # weight_sign & weight_width
        self.weight_sign = WeightSignMode.SIGNED
        self.weight_width = WeightWidth.WEIGHT_WIDTH_8BIT

        # output_sign & output_width
        if isinstance(self.op2, LutActivation):
            # For ANN activations
            # ReLU, Sigmoid (0-1) are unsigned
            if isinstance(self.op2, (LutReLU, LutSigmoid)):
                self.output_sign = OutputSignMode.UNSIGNED
                self.output_width = WeightWidth.WEIGHT_WIDTH_8BIT
            # Tanh, Softsign (-1-1) are signed
            else:
                self.output_sign = OutputSignMode.SIGNED
                self.output_width = WeightWidth.WEIGHT_WIDTH_8BIT
        elif isinstance(self.op2, (SJIFNode, SJLIFNode)):
            self.output_sign = OutputSignMode.UNSIGNED
            self.output_width = WeightWidth.WEIGHT_WIDTH_1BIT
        elif isinstance(self.op2, NeuronV2):
            if self.op2.thres_neg_mode == ThresholdNegMode.FIRE:
                self.output_sign = OutputSignMode.SIGNED
                self.output_width = WeightWidth.WEIGHT_WIDTH_2BIT
            else:
                self.output_sign = OutputSignMode.UNSIGNED
                self.output_width = WeightWidth.WEIGHT_WIDTH_1BIT
        else:
            # Fallback
            self.output_sign = OutputSignMode.SIGNED
            self.output_width = WeightWidth.WEIGHT_WIDTH_8BIT

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        s = 0
        for sign, op, x in zip(self.signs, self.op1, xs, strict=True):
            output = op(x.float())
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
