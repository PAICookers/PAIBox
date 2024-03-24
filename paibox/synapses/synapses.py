import warnings
from typing import Optional, Union

import numpy as np

from paibox.base import NeuDyn
from paibox.neuron import Neuron
from paibox.projection import InputProj
from paibox.types import DataArrayType

from .base import Conv1dSyn, Conv2dSyn, FullConnSyn
from .conv_utils import (
    _KOrder3d,
    _KOrder4d,
    _Order2d,
    _Order3d,
    _pair,
    _single,
    _Size1Type,
    _Size2Type,
)
from .transforms import GeneralConnType as GConnType

__all__ = ["FullConn", "Conv1d", "Conv2d"]


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
        """
        Arguments:
            - source: source neuron(s).
            - dest: destination neuron(s).
            - weights: weights of the synapses. It can be a scalar or `np.ndarray`.
            - conn_type: the type of connection.
            - name: name of this synapses. Optional.
        """
        super().__init__(source, dest, weights, conn_type, name)


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
        warnings.warn(
            "'NoDecay' class will be deprecated in future versions. Use 'FullConn' instead.",
            DeprecationWarning,
        )

        super().__init__(source, dest, weights, conn_type=conn_type, name=name)


class Conv1d(Conv1dSyn):
    def __init__(
        self,
        source: Union[Neuron, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        *,
        stride: _Size1Type = 1,
        # padding: _Size1Type = 0,
        fm_order: _Order2d = "CL",
        kernel_order: _KOrder3d = "OIL",
        name: Optional[str] = None,
    ) -> None:
        if fm_order not in ("CL", "LC"):
            raise ValueError(f"feature map order must be 'CL' or 'LC'.")

        if kernel_order not in ("OIL", "IOL"):
            raise ValueError(f"kernel order must be 'OIL' or 'IOL'.")

        super().__init__(
            source,
            dest,
            kernel,
            _single(stride),
            # _single(padding),
            fm_order,
            kernel_order,
            name=name,
        )


class Conv2d(Conv2dSyn):
    def __init__(
        self,
        source: Union[Neuron, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        *,
        stride: _Size2Type = 1,
        # padding: _Size2Type = 0,
        fm_order: _Order3d = "CHW",
        kernel_order: _KOrder4d = "OIHW",
        name: Optional[str] = None,
    ) -> None:
        """2d convolution synapses in fully-unrolled format.

        Arguments:
            - source: source neuron(s). The dimensions need to be expressed explicitly as (C,H,W) or    \
                (H,W,C). The feature map dimension order is specified by `fm_order`.
            - dest: destination neuron(s).
            - kernel: convolution kernel. Its dimension order must be (O,I,H,W) or (I,O,H,W), depending \
                on the argument `kernel_order`.
            - stride: the step size of the kernel sliding. It can be a scalar or a tuple of 2 integers.
            - fm_order: dimension order of feature map. The order of input & output feature maps must be\
                consistent, (C,H,W) or (H,W,C).
            - kernel_order: dimension order of kernel, (O,I,H,W) or (I,O,H,W). (O,I,H,W) stands for     \
                (output channels, input channels, height, width).
            - name: name of the 2d convolution. Optional.
        """
        if fm_order not in ("CHW", "HWC"):
            raise ValueError(f"feature map order must be 'CHW or 'HWC'.")

        if kernel_order not in ("OIHW", "IOHW"):
            raise ValueError(f"kernel order must be 'OIHW' or 'IOHW'.")

        super().__init__(
            source,
            dest,
            kernel,
            _pair(stride),
            # _pair(padding),
            fm_order,
            kernel_order,
            name=name,
        )
