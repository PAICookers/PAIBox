from enum import Enum, auto, unique
from typing import Type, Union

import numpy as np
from numpy.typing import NDArray
from paicorelib import WeightPrecision as WP

from paibox.exceptions import ShapeError
from paibox.types import DataArrayType, IntScalarType, SynOutType, WeightType
from paibox.utils import is_shape

from .conv_utils import (
    Size1Type,
    Size2Type,
    _conv1d_faster,
    _conv1d_unroll,
    _conv2d_faster,
    _conv2d_unroll,
    _Order2d,
    _Order3d,
)

__all__ = [
    "GeneralConnType",
    "OneToOne",
    "AllToAll",
    "Identity",
    "MaskedLinear",
    "Conv1dForward",
    "Conv2dForward",
]


MAX_INT1 = np.int8(1)
MIN_INT1 = np.int8(0)
MAX_INT2 = np.int8(1)
MIN_INT2 = np.int8(-2)
MAX_INT4 = np.int8(7)
MIN_INT4 = np.int8(-8)
MAX_INT8 = np.iinfo(np.int8).max
MIN_INT8 = np.iinfo(np.int8).min


class ConnType(Enum):
    """Basic connection enum type."""

    pass


@unique
class GeneralConnType(ConnType):
    MatConn = auto()
    """General matrix connection."""

    One2One = auto()
    """One-to-one connection."""

    Identity = auto()
    """Identity connection with scaling factor."""

    All2All = auto()
    """All-to-all connection."""


def _get_weight_precision(weight: np.ndarray, enable_wp_opt: bool) -> WP:
    """Get the actual weight_precision of the weight."""
    _max = np.max(weight, axis=None).astype(np.int32)
    _min = np.min(weight, axis=None).astype(np.int32)

    if _max > MAX_INT8 or _min < MIN_INT8:
        raise ValueError(f"weight precision out of range [{_min}, {_max}].")

    if _max <= MAX_INT1 and _min >= MIN_INT1:
        return WP.WEIGHT_WIDTH_1BIT
    elif enable_wp_opt:
        if _max <= MAX_INT2 and _min >= MIN_INT2:
            return WP.WEIGHT_WIDTH_2BIT
        elif _max <= MAX_INT4 and _min >= MIN_INT4:
            return WP.WEIGHT_WIDTH_4BIT
        else:
            return WP.WEIGHT_WIDTH_8BIT
    else:
        return WP.WEIGHT_WIDTH_8BIT


class Transform:
    def __init__(self, weights: WeightType) -> None:
        self.weights = weights
        """The actual weights in synapse. Must stored in `np.int8` format."""
        self.weights.setflags(write=False)

    def __call__(self, *args, **kwargs) -> SynOutType:
        """Ensure that in all subclasses, the output dimensions are (M,)."""
        raise NotImplementedError

    def _get_wp(self, enable_wp_opt: bool) -> WP:
        """Precision of weights."""
        return _get_weight_precision(self.weights, enable_wp_opt)

    @property
    def conn_dtype(self) -> Union[Type[np.bool_], Type[np.int8]]:
        # The value of `enable_wp_opt` dosen't effect the dtype of `connectivity`.
        if self._get_wp(enable_wp_opt=False) is WP.WEIGHT_WIDTH_1BIT:
            return np.bool_
        else:
            return np.int8

    @property
    def connectivity(self) -> NDArray[Union[np.bool_, np.int8]]:
        """The connectivity matrix in `np.ndarray` format."""
        raise NotImplementedError


class OneToOne(Transform):
    def __init__(self, num: int, weights: DataArrayType) -> None:
        """
        Arguments:
            - num: number of neurons.
            - weights: synaptic weights. The shape must be a scalar or array (num,).
                - weights is a scalar(ndim = 0), the connectivity matrix will be:
                    [[x, 0, 0]
                     [0, x, 0]
                     [0, 0, x]]
                - weights is an array(ndim = 1), [x, y, z] corresponding to the weights \
                    of the post-neurons respectively. The connectivity matrix will be:
                    [[x, 0, 0]
                     [0, y, 0]
                     [0, 0, z]]
        """
        self.num = num

        if isinstance(weights, np.ndarray) and not is_shape(weights, (num,)):
            raise ShapeError(f"expected shape is ({num},), but got {weights.shape}.")

        # The ndim of weights = 0 or 1.
        _w = np.asarray(weights, dtype=np.int8)

        if _w.ndim not in (0, 1):
            raise ShapeError(f"the ndim of weights must be 0 or 1, but got {_w.ndim}.")

        super().__init__(_w)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> SynOutType:
        # (N,) * (N,) -> (N,)
        output = x * self.weights.copy()

        return output.astype(np.int32)

    @property
    def connectivity(self):
        return (
            (self.weights * np.eye(self.num, dtype=np.bool_)).astype(self.conn_dtype)
            if self.weights.ndim == 0
            else np.diag(self.weights).astype(self.conn_dtype)
        )


class Identity(OneToOne):
    def __init__(self, num: int, scaling_factor: IntScalarType = 1) -> None:
        """
        Arguments:
            - num: number of neurons.
            - scaling_factor: scaling factor.
        """
        super().__init__(num, scaling_factor)


class AllToAll(Transform):
    def __init__(self, conn_size: Size2Type, weights: DataArrayType) -> None:
        """
        Arguments:
            - conn_size: size of connections.
            - weights: synaptic weights. The shape must be a scalar or a matrix.
                - when weights is a scalar(ndim = 0), the connectivity matrix will be:  \
                    x * I
                - when weights is a matrix(ndim = 2), the connectivity matrix will be:  \
                    [[a, b, c]
                     [d, e, f]
                     [g, h, i]]
        """
        self.conn_size = conn_size

        if isinstance(weights, np.ndarray) and not is_shape(weights, conn_size):
            raise ShapeError(f"expected shape is {conn_size}, but got {weights.shape}.")

        _w = np.asarray(weights, dtype=np.int8)

        if _w.ndim not in (0, 2):
            raise ShapeError(f"the ndim of weights must be 0 or 2, but got {_w.ndim}.")

        super().__init__(_w)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> SynOutType:
        """
        NOTE:
            - When weights is a scalar, the output is a scalar (sum * w) & repeated     \
                `conn_size[1]` times.
            - When weights is a matrix, the output is the dot product of `x` & weights.
        """
        if self.weights.ndim == 0:
            sum_x = np.sum(x, axis=None, dtype=np.int32)
            # (M,)
            output = np.full((self.conn_size[1],), self.weights * sum_x, dtype=np.int32)
        else:
            # (N,) @ (N, M) -> (M,)
            output = x @ self.weights.copy().astype(np.int32)

        return output

    @property
    def connectivity(self):
        return (
            self.weights.astype(self.conn_dtype)
            if self.weights.ndim == 2
            else (self.weights * np.ones(self.conn_size, dtype=np.bool_)).astype(
                self.conn_dtype
            )
        )


class MaskedLinear(Transform):
    def __init__(
        self,
        conn_size: Size2Type,
        weights: np.ndarray,
    ) -> None:
        if not is_shape(weights, conn_size):
            raise ShapeError(f"expected shape is {conn_size}, but got {weights.shape}.")

        _w = np.asarray(weights, dtype=np.int8)
        super().__init__(_w)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> SynOutType:
        # (N,) @ (N, M) -> (M,)
        output = x @ self.weights.copy().astype(np.int32)

        return output.astype(np.int32)

    @property
    def connectivity(self):
        return self.weights.astype(self.conn_dtype)


class Conv1dForward(Transform):
    def __init__(
        self,
        in_shape: Size1Type,
        out_shape: Size1Type,
        kernel: np.ndarray,
        stride: Size1Type,
        padding: Size1Type,
        fm_order: _Order2d,
    ) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.stride = stride
        self.padding = padding
        self.fm_order = fm_order

        _w = kernel.astype(np.int8)
        super().__init__(_w)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> SynOutType:
        cin = self.weights.shape[1]

        if self.fm_order == "LC":
            # (N,) -> (L, C) -> (C, L)
            _x = x.reshape(self.in_shape + (cin,)).T
        else:
            _x = x.reshape((cin,) + self.in_shape)

        o_conv1d = _conv1d_faster(
            _x,
            self.out_shape,
            self.weights,
            self.stride,
            self.padding,
        )

        return o_conv1d.flatten()

    @property
    def connectivity(self):
        return _conv1d_unroll(
            self.in_shape, self.out_shape, self.weights, self.stride
        ).astype(self.conn_dtype)


class Conv2dForward(Transform):
    def __init__(
        self,
        in_shape: Size2Type,
        out_shape: Size2Type,
        kernel: np.ndarray,
        stride: Size2Type,
        padding: Size2Type,
        fm_order: _Order3d,
    ) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.stride = stride
        self.padding = padding
        self.fm_order = fm_order

        _w = kernel.astype(np.int8)
        super().__init__(_w)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> SynOutType:
        cin = self.weights.shape[1]

        if self.fm_order == "HWC":
            # (N,) -> (H, W, C) -> (C, H, W)
            _x = x.reshape(self.in_shape + (cin,)).transpose(2, 0, 1)
        else:
            _x = x.reshape((cin,) + self.in_shape)

        o_conv2d = _conv2d_faster(
            _x,
            self.out_shape,
            self.weights,
            self.stride,
            self.padding,
        )

        return o_conv2d.flatten()

    @property
    def connectivity(self):
        return _conv2d_unroll(
            self.in_shape, self.out_shape, self.weights, self.stride
        ).astype(self.conn_dtype)
