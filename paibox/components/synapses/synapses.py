from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

from paibox.base import NeuDyn
from paibox.types import DataType, NeuOutType, SynOutType

from ..neuron import Neuron, OnlineNeuron
from ..projection import InputProj
from .base import (
    Conv1dSyn,
    Conv2dSyn,
    ConvTranspose1dSyn,
    ConvTranspose2dSyn,
    FullConnSyn,
)
from .conv_types import _KOrder3d, _KOrder4d, _Size1Type, _Size2Type
from .conv_utils import _pair, _single
from .learning import STDPSyn
from .transforms import ConnType

__all__ = [
    "FullConn",
    "MatMul2d",
    "Conv1d",
    "Conv2d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "STDPFullConn",
]


class FullConn(FullConnSyn):
    def __init__(
        self,
        source: NeuDyn | InputProj,
        target: NeuDyn,
        weights: DataType = 1,
        *,
        conn_type: ConnType = ConnType.All2All,
        name: str | None = None,
    ) -> None:
        """Full-connected synapses.

        Args:
            - source: source neuron.
            - target: destination neuron.
            - weights: weights of the synapses. It can be a scalar or `np.ndarray`.
            - conn_type: the type of connection.
            - name: name of the full-connected synapses.
        """
        super().__init__(source, target, weights, conn_type, name=name)


class MatMul2d(FullConnSyn):
    def __init__(
        self,
        source: NeuDyn | InputProj,
        target: NeuDyn,
        weights: np.ndarray,
        name: str | None = None,
    ) -> None:
        """MatMul2d synapses.

        Args:
            - source: source neuron.
            - target: destination neuron.
            - weights: weights of the synapses.
            - name: name of the matmul2d.
        """
        super().__init__(source, target, weights, ConnType.MatConn, name)


class Conv1d(Conv1dSyn):
    def __init__(
        self,
        source: Neuron | InputProj,
        target: Neuron,
        kernel: np.ndarray,
        *,
        stride: _Size1Type = 1,
        padding: _Size1Type = 0,
        dilation: _Size1Type = 1,
        groups: int = 1,
        kernel_order: _KOrder3d = "OIL",
        name: str | None = None,
    ) -> None:
        """1d convolution synapses in fully-unrolled format.

        Args:
            - source: source neuron. The dimensions need to be expressed explicitly as (C,L).
            - target: destination neuron.
            - kernel: convolution kernel. Its dimension order is either (O,I,L) or (I,O,L), depending on the    \
                argument `kernel_order`.
            - stride: the step size of the kernel sliding. It can be a scalar or an integer.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or an integer.
            - dilation: the spacing between kernel elements. It can be a scalar or an integer.
            - groups: number of groups in the convolution.
            - kernel_order: dimension order of kernel, (O,I,L) or (I,O,L). (O,I,L) stands for (output channels, \
                input channels, length).
            - name: name of the 1d convolution.

        NOTE: See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d for details.
        """
        if kernel_order not in ("OIL", "IOL"):
            raise ValueError("kernel order must be 'OIL' or 'IOL'.")

        super().__init__(
            source,
            target,
            kernel,
            _single(stride),
            _single(padding),
            _single(dilation),
            groups,
            kernel_order,
            name,
        )


class Conv2d(Conv2dSyn):
    def __init__(
        self,
        source: Neuron | InputProj,
        target: Neuron,
        kernel: np.ndarray,
        stride: _Size2Type = 1,
        padding: _Size2Type = 0,
        dilation: _Size2Type = 1,
        groups: int = 1,
        kernel_order: _KOrder4d = "OIHW",
        name: str | None = None,
    ) -> None:
        """2d convolution synapses in fully-unrolled format.

        Args:
            - source: source neuron. The dimensions need to be expressed explicitly as (C,H,W).
            - target: destination neuron.
            - kernel: convolution kernel. Its dimension order is either (O,I,H,W) or (I,O,H,W), depending on the\
                argument `kernel_order`.
            - stride: the step size of the kernel sliding. It can be a scalar or a tuple of 2 integers.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 2 integers.
            - dilation: the spacing between kernel elements. It can be a scalar or a tuple of 2 integers.
            - groups: number of groups in the convolution.
            - kernel_order: dimension order of kernel, (O,I,H,W) or (I,O,H,W). (O,I,H,W) stands for (output     \
                channels, input channels, height, width).
            - name: name of the 2d convolution.

        NOTE: See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d for details.
        """
        if kernel_order not in ("OIHW", "IOHW"):
            raise ValueError("kernel order must be 'OIHW' or 'IOHW'.")

        super().__init__(
            source,
            target,
            kernel,
            _pair(stride),
            _pair(padding),
            _pair(dilation),
            groups,
            kernel_order,
            name,
        )


class ConvTranspose1d(ConvTranspose1dSyn):
    def __init__(
        self,
        source: Neuron | InputProj,
        target: Neuron,
        kernel: np.ndarray,
        *,
        stride: _Size1Type = 1,
        padding: _Size1Type = 0,
        output_padding: _Size1Type = 0,
        kernel_order: _KOrder3d = "OIL",
        name: str | None = None,
    ) -> None:
        """1d transposed convolution synapses in fully-unrolled format.

        Args:
            - source: source neuron. The dimensions need to be expressed explicitly as (C,L).
            - target: destination neuron.
            - kernel: convolution kernel. Its dimension order is either (O,I,L) or (I,O,L), depending on the    \
                argument `kernel_order`.
            - stride: stride of the convolution. It can be a scalar or an integer.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or an integer.
            - output_padding: the additional size added to one side of the output shape. It can be a scalar or  \
                an integer.
            - kernel_order: dimension order of kernel, (O,I,L) or (I,O,L). (O,I,L) stands for (output channels, \
                input channels, length).
            - name: name of the 1d transposed convolution.

        NOTE: See https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html#torch.nn.ConvTranspose1d  \
            for details.
        """
        if kernel_order not in ("OIL", "IOL"):
            raise ValueError("kernel order must be 'OIL' or 'IOL'.")

        super().__init__(
            source,
            target,
            kernel,
            _single(stride),
            _single(padding),
            _single(1),
            _single(output_padding),
            kernel_order,
            name,
        )


class ConvTranspose2d(ConvTranspose2dSyn):
    def __init__(
        self,
        source: Neuron | InputProj,
        target: Neuron,
        kernel: np.ndarray,
        *,
        stride: _Size2Type = 1,
        padding: _Size2Type = 0,
        output_padding: _Size2Type = 0,
        kernel_order: _KOrder4d = "OIHW",
        name: str | None = None,
    ) -> None:
        """2d transposed convolution synapses in fully-unrolled format.

        Args:
            - source: source neuron. The dimensions need to be expressed explicitly as (C,H,W) or (H,W,C). The  \
                feature map dimension order is specified by `fm_order`.
            - target: destination neuron.
            - kernel: convolution kernel. Its dimension order must be (O,I,H,W) or (I,O,H,W), depending on the  \
                argument `kernel_order`.
            - stride: stride of the convolution. It can be a scalar or a tuple of 2 integers.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 2 integers.
            - output_padding: the additional size added to one side of the output shape. It can be a scalar or  \
                a tuple of 2 integers.
            - fm_order: dimension order of feature map. The order of input & output feature maps must be        \
                consistent, (C,H,W) or (H,W,C).
            - kernel_order: dimension order of kernel, (O,I,H,W) or (I,O,H,W). (O,I,H,W) stands for (output     \
                channels, input channels, height, width).
            - name: name of the 2d transposed convolution.

        NOTE: See https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d  \
            for details.
        """
        if kernel_order not in ("OIHW", "IOHW"):
            raise ValueError("kernel order must be 'OIHW' or 'IOHW'.")

        super().__init__(
            source,
            target,
            kernel,
            _pair(stride),
            _pair(padding),
            _pair(1),
            _pair(output_padding),
            kernel_order,
            name,
        )


class STDPFullConn(STDPSyn, FullConnSyn):
    CFLAG_ENABLE_WP_OPTIMIZATION: ClassVar[bool] = False

    def __init__(
        self,
        source: NeuDyn | InputProj,
        target: OnlineNeuron,
        weights: DataType,
        weight_decay: int = 0,
        upper_weight: int | None = None,
        lower_weight: int | None = None,
        weight_decay_random: bool = False,
        lut: ArrayLike | None = None,
        lut_offset: int | None = None,
        lut_random: bool | ArrayLike = False,
        random_seed: int = 1,
        *,
        learn_by_default: bool = True,
        name: str | None = None,
    ) -> None:
        super(STDPSyn, self).__init__(source, target, weights, ConnType.MatConn, name)
        super().__init__(
            self,
            weight_decay,
            upper_weight,
            lower_weight,
            weight_decay_random,
            lut,
            lut_offset,
            lut_random,
            random_seed,
            learn_by_default,
        )
        # Store synapse's attributes to the target neuron.
        self.target._set_syn_attrs(**self.attrs())

    def update(self, x: NeuOutType | None = None, *args, **kwargs) -> SynOutType:
        synout = super(STDPSyn, self).update(x)
        if self.target.is_working() and self.learning:
            super().step(self.synin, self.target.spike)

        return synout

    def reset_state(self, *args, **kwargs) -> None:
        super().reset_state(*args, **kwargs)
        super(STDPSyn, self).reset_state(*args, **kwargs)
