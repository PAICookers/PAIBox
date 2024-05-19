import sys
from typing import Optional, Union

import numpy as np

from paibox.base import NeuDyn
from paibox.exceptions import PAIBoxDeprecationWarning
from paibox.types import DataArrayType

from ..neuron import Neuron
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
from .transforms import GeneralConnType as GConnType

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

__all__ = ["FullConn", "Conv1d", "Conv2d", "ConvTranspose1d", "ConvTranspose2d"]


class FullConn(FullConnSyn):
    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: NeuDyn,
        weights: DataArrayType = 1,
        *,
        conn_type: GConnType = GConnType.MatConn,
        name: Optional[str] = None,
    ) -> None:
        """Full-connected synapses.

        Args:
            - source: source neuron.
            - dest: destination neuron.
            - weights: weights of the synapses. It can be a scalar or `np.ndarray`.
            - conn_type: the type of connection.
            - name: name of the full-connected synapses. Optional.
        """
        super().__init__(source, dest, weights, conn_type, name)


@deprecated(
    "'NoDecay' will be removed in a future version. Use 'FullConn' instead.",
    category=PAIBoxDeprecationWarning,
)
class NoDecay(FullConn):
    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: NeuDyn,
        weights: DataArrayType = 1,
        *,
        conn_type: GConnType = GConnType.MatConn,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(source, dest, weights, conn_type=conn_type, name=name)


class Conv1d(Conv1dSyn):
    def __init__(
        self,
        source: Union[Neuron, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        *,
        stride: _Size1Type = 1,
        padding: _Size1Type = 0,
        kernel_order: _KOrder3d = "OIL",
        name: Optional[str] = None,
    ) -> None:
        """1d convolution synapses in fully-unrolled format.

        Args:
            - source: source neuron. The dimensions need to be expressed explicitly as (C,L).
            - dest: destination neuron.
            - kernel: convolution kernel. Its dimension order is either (O,I,L) or (I,O,L), depending on the    \
                argument `kernel_order`.
            - stride: the step size of the kernel sliding. It can be a scalar or an integer.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or an integer.
            - kernel_order: dimension order of kernel, (O,I,L) or (I,O,L). (O,I,L) stands for (output channels, \
                input channels, length).
            - name: name of the 1d convolution. Optional.

        NOTE: See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d for details.
        """
        if kernel_order not in ("OIL", "IOL"):
            raise ValueError(f"kernel order must be 'OIL' or 'IOL'.")

        super().__init__(
            source,
            dest,
            kernel,
            _single(stride),
            _single(padding),
            _single(1),
            kernel_order,
            name,
        )


class Conv2d(Conv2dSyn):
    def __init__(
        self,
        source: Union[Neuron, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        *,
        stride: _Size2Type = 1,
        padding: _Size2Type = 0,
        kernel_order: _KOrder4d = "OIHW",
        name: Optional[str] = None,
    ) -> None:
        """2d convolution synapses in fully-unrolled format.

        Args:
            - source: source neuron. The dimensions need to be expressed explicitly as (C,H,W).
            - dest: destination neuron.
            - kernel: convolution kernel. Its dimension order is either (O,I,H,W) or (I,O,H,W), depending on the\
                argument `kernel_order`.
            - stride: the step size of the kernel sliding. It can be a scalar or a tuple of 2 integers.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 2 integers.
            - kernel_order: dimension order of kernel, (O,I,H,W) or (I,O,H,W). (O,I,H,W) stands for (output     \
                channels, input channels, height, width).
            - name: name of the 2d convolution. Optional.

        NOTE: See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d for details.
        """
        if kernel_order not in ("OIHW", "IOHW"):
            raise ValueError(f"kernel order must be 'OIHW' or 'IOHW'.")

        super().__init__(
            source,
            dest,
            kernel,
            _pair(stride),
            _pair(padding),
            _pair(1),
            kernel_order,
            name,
        )


class ConvTranspose1d(ConvTranspose1dSyn):
    def __init__(
        self,
        source: Union[Neuron, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        *,
        stride: _Size1Type = 1,
        padding: _Size1Type = 0,
        output_padding: _Size1Type = 0,
        kernel_order: _KOrder3d = "OIL",
        name: Optional[str] = None,
    ) -> None:
        """1d transposed convolution synapses in fully-unrolled format.

        Args:
            - source: source neuron. The dimensions need to be expressed explicitly as (C,L).
            - dest: destination neuron.
            - kernel: convolution kernel. Its dimension order is either (O,I,L) or (I,O,L), depending on the    \
                argument `kernel_order`.
            - stride: stride of the convolution. It can be a scalar or an integer.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or an integer.
            - output_padding: the additional size added to one side of the output shape. It can be a scalar or  \
                an integer.
            - kernel_order: dimension order of kernel, (O,I,L) or (I,O,L). (O,I,L) stands for (output channels, \
                input channels, length).
            - name: name of the 1d transposed convolution. Optional.

        NOTE: See https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html#torch.nn.ConvTranspose1d  \
            for details.
        """
        if kernel_order not in ("OIL", "IOL"):
            raise ValueError(f"kernel order must be 'OIL' or 'IOL'.")

        super().__init__(
            source,
            dest,
            kernel,
            _single(stride),
            _single(padding),
            _single(output_padding),
            _single(1),
            kernel_order,
            name,
        )


class ConvTranspose2d(ConvTranspose2dSyn):
    def __init__(
        self,
        source: Union[Neuron, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        *,
        stride: _Size2Type = 1,
        padding: _Size2Type = 0,
        output_padding: _Size2Type = 0,
        kernel_order: _KOrder4d = "OIHW",
        name: Optional[str] = None,
    ) -> None:
        """2d transposed convolution synapses in fully-unrolled format.

        Args:
            - source: source neuron. The dimensions need to be expressed explicitly as (C,H,W) or (H,W,C). The  \
                feature map dimension order is specified by `fm_order`.
            - dest: destination neuron.
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
            - name: name of the 2d transposed convolution. Optional.

        NOTE: See https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d  \
            for details.
        """
        if kernel_order not in ("OIHW", "IOHW"):
            raise ValueError(f"kernel order must be 'OIHW' or 'IOHW'.")

        super().__init__(
            source,
            dest,
            kernel,
            _pair(stride),
            _pair(padding),
            _pair(output_padding),
            _pair(1),
            kernel_order,
            name,
        )
