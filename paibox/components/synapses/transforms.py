import warnings
from enum import Enum, auto, unique
from typing import Literal

import numpy as np
from paicorelib import WeightPrecision as WP

from paibox.exceptions import AutoOptimizationWarning, ShapeError
from paibox.types import DataArrayType, IntScalarType, SpikeType, SynOutType, WeightType
from paibox.utils import is_shape

from .conv_types import Size1Type, Size2Type
from .conv_utils import (
    _conv1d_faster,
    _conv1d_unroll,
    _conv2d_faster,
    _conv2d_unroll,
    _convtranspose1d_faster,
    _convtranspose1d_unroll,
    _convtranspose2d_faster,
    _convtranspose2d_unroll,
    _func_pool2d,
    _pool2d_kernel_unroll,
)

__all__ = [
    "OneToOne",
    "AllToAll",
    "Identity",
    "MaskedLinear",
    "Conv2dForward",
    "ConvTranspose1dForward",
    "ConvTranspose2dForward",
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


def _set_coarse_dtype(raw_w: DataArrayType) -> WeightType:
    """Convert raw weights to `np.ndarray` coarsely (without optimization).

    Description:
        - For weights of type `bool` or `np.bool_`, set `np.bool_` as the dtype.
        - For integer scalar weight, set the dtype according to its value.
        - For array weights, set the dtype according to its minimum & maximum values. For weights in the\
            range of int8, the dtype when declared will be followed (i.e. not optimized).

    NOTE: Only when the weight is input in integer scalar form, the weight precision will be optimized  \
        automatically. 0/1 is treated as bool_ while others are treated as int8. The weights must not   \
        exceed the range of int8.
    """
    if isinstance(raw_w, (bool, np.bool_, int, np.integer)):
        if raw_w > MAX_INT8 or raw_w < MIN_INT8:
            raise ValueError(f"weight out of range int8, got {raw_w}.")

        if raw_w <= MAX_INT1 and raw_w >= MIN_INT1:
            _dtype = np.bool_
        else:
            _dtype = np.int8

        return np.asarray(raw_w, dtype=_dtype)

    # Convert list or tuple to np.ndarray
    _array = np.asarray(raw_w)
    _max = np.max(_array, axis=None)
    _min = np.min(_array, axis=None)

    if _max > MAX_INT8 or _min < MIN_INT8:
        raise ValueError(f"weight out of range int8, got [{_min}, {_max}].")

    if _array.dtype > np.int8:
        # XXX If it is automatically optimized to int8, it cannot be converted using the 'same_kind' rule.
        # if _max <= MAX_INT1 and _min >= MIN_INT1:
        #     warnings.warn(
        #         f"dtype of weight is optimized automatically, {_array.dtype} -> bool.",
        #         AutoOptimizationWarning,
        #     )
        #     _dtype = np.bool_
        # else:
        warnings.warn(
            f"dtype of weight is optimized automatically, {_array.dtype} -> int8.",
            AutoOptimizationWarning,
        )
        _dtype = np.int8

    elif _array.dtype == np.bool_ or _array.dtype == np.int8:
        _dtype = _array.dtype
    else:
        raise TypeError(f"weights must be bool or int8, but got {_array.dtype}.")

    return _array.astype(_dtype, casting="same_kind")


def _get_weight_precision(weight: WeightType, enable_wp_opt: bool) -> WP:
    """Get the actual weight_precision of the weight."""
    _max = np.max(weight, axis=None)
    _min = np.min(weight, axis=None)

    if enable_wp_opt:
        if _max <= MAX_INT1 and _min >= MIN_INT1:
            return WP.WEIGHT_WIDTH_1BIT
        elif _max <= MAX_INT2 and _min >= MIN_INT2:
            return WP.WEIGHT_WIDTH_2BIT
        elif _max <= MAX_INT4 and _min >= MIN_INT4:
            return WP.WEIGHT_WIDTH_4BIT
        else:
            return WP.WEIGHT_WIDTH_8BIT
    else:
        # If weight precision opt is disabled, return WP1 if dtype is np.bool_ else WP8.
        if weight.dtype == np.bool_:
            return WP.WEIGHT_WIDTH_1BIT
        else:
            return WP.WEIGHT_WIDTH_8BIT


class Transform:
    def __init__(self, weights: DataArrayType) -> None:
        self.weights = _set_coarse_dtype(weights)

        """The actual weights in synapses. Stored in `np.bool_` or `np.int8` format."""
        self.weights.setflags(write=False)

    def __call__(self, *args, **kwargs) -> SynOutType:
        """Ensure that in all subclasses, the output dimensions are (M,)."""
        raise NotImplementedError

    def _get_wp(self, enable_wp_opt: bool) -> WP:
        return _get_weight_precision(self.weights, enable_wp_opt)

    @property
    def connectivity(self) -> WeightType:
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

        super().__init__(weights)

        # The ndim of weights = 0 or 1.
        if self.weights.ndim not in (0, 1):
            raise ShapeError(
                f"the ndim of weights must be 0 or 1, but got {self.weights.ndim}."
            )

    def __call__(self, x: SpikeType, *args, **kwargs) -> SynOutType:
        # (N,) * (N,) -> (N,)
        return x * self.weights.astype(np.int32)

    @property
    def connectivity(self):
        return (
            (self.weights * np.identity(self.num, dtype=np.bool_))
            if self.weights.ndim == 0
            else np.diag(self.weights)
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

        super().__init__(weights)

        if self.weights.ndim not in (0, 2):
            raise ShapeError(
                f"the ndim of weights must be 0 or 2, but got {self.weights.ndim}."
            )

    def __call__(self, x: SpikeType, *args, **kwargs) -> SynOutType:
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
            output = x @ self.weights.astype(np.int32)

        return output

    @property
    def connectivity(self):
        return (
            self.weights
            if self.weights.ndim == 2
            else (self.weights * np.ones(self.conn_size, dtype=np.bool_))
        )


class MaskedLinear(Transform):
    def __init__(self, conn_size: Size2Type, weights: np.ndarray) -> None:
        if not is_shape(weights, conn_size):
            raise ShapeError(f"expected shape is {conn_size}, but got {weights.shape}.")

        super().__init__(weights)

    def __call__(self, x: SpikeType, *args, **kwargs) -> SynOutType:
        # (N,) @ (N, M) -> (M,)
        return x @ self.weights.astype(np.int32)

    @property
    def connectivity(self):
        return self.weights


class Conv1dForward(Transform):
    def __init__(
        self,
        in_shape: Size1Type,
        out_shape: Size1Type,
        kernel: np.ndarray,
        stride: Size1Type,
        padding: Size1Type,
        # fm_order: _Order2d,
    ) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.stride = stride
        self.padding = padding
        # self.fm_order = fm_order

        super().__init__(kernel)

    def __call__(self, x: SpikeType, *args, **kwargs) -> SynOutType:
        cin = self.weights.shape[1]

        # if self.fm_order == "LC":
        #     # (N,) -> (L, C) -> (C, L)
        #     _x = x.reshape(self.in_shape + (cin,)).T
        # else:
        _x = x.reshape((cin,) + self.in_shape)

        return _conv1d_faster(
            _x, self.out_shape, self.weights, self.stride, self.padding
        )

    @property
    def connectivity(self):
        return _conv1d_unroll(
            self.in_shape, self.out_shape, self.weights, self.stride, self.padding
        )


class Conv2dForward(Transform):
    def __init__(
        self,
        in_shape: Size2Type,
        out_shape: Size2Type,
        kernel: np.ndarray,
        stride: Size2Type,
        padding: Size2Type,
        # fm_order: _Order3d,
    ) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.stride = stride
        self.padding = padding
        # self.fm_order = fm_order

        super().__init__(kernel)

    def __call__(self, x: SpikeType, *args, **kwargs) -> SynOutType:
        cin = self.weights.shape[1]

        # if self.fm_order == "HWC":
        #     # (N,) -> (H, W, C) -> (C, H, W)
        #     _x = x.reshape(self.in_shape + (cin,)).transpose(2, 0, 1)
        # else:
        _x = x.reshape((cin,) + self.in_shape)

        return _conv2d_faster(
            _x, self.out_shape, self.weights, self.stride, self.padding
        )

    @property
    def connectivity(self):
        return _conv2d_unroll(
            self.in_shape, self.out_shape, self.weights, self.stride, self.padding
        )


class ConvTranspose1dForward(Transform):
    def __init__(
        self,
        in_shape: Size1Type,
        out_shape: Size1Type,
        kernel: np.ndarray,
        stride: Size1Type,
        padding: Size1Type,
        output_padding: Size1Type,
        # fm_order: _Order2d,
    ) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        # self.fm_order = fm_order

        super().__init__(kernel)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> SynOutType:
        cin = self.weights.shape[1]

        # if self.fm_order == "LC":
        #     # (N,) -> (L, C) -> (C, L)
        #     _x = x.reshape(self.in_shape + (cin,)).T
        # else:
        _x = x.reshape((cin,) + self.in_shape)

        return _convtranspose1d_faster(
            _x,
            self.out_shape,
            self.weights,
            self.stride,
            self.padding,
            self.output_padding,
        )

    @property
    def connectivity(self):
        return _convtranspose1d_unroll(
            self.in_shape,
            self.out_shape,
            self.weights,
            self.stride,
            self.padding,
            self.output_padding,
        )


class ConvTranspose2dForward(Transform):
    def __init__(
        self,
        in_shape: Size2Type,
        out_shape: Size2Type,
        kernel: np.ndarray,
        stride: Size2Type,
        padding: Size2Type,
        output_padding: Size2Type,
        # fm_order: _Order3d,
    ) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        # self.fm_order = fm_order

        super().__init__(kernel)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> SynOutType:
        cin = self.weights.shape[1]

        # if self.fm_order == "HWC":
        #     # (N,) -> (H, W, C) -> (C, H, W)
        #     _x = x.reshape(self.in_shape + (cin,)).transpose(2, 0, 1)
        # else:
        _x = x.reshape((cin,) + self.in_shape)

        return _convtranspose2d_faster(
            _x,
            self.out_shape,
            self.weights,
            self.stride,
            self.padding,
            self.output_padding,
        )

    @property
    def connectivity(self):
        return _convtranspose2d_unroll(
            self.in_shape,
            self.out_shape,
            self.weights,
            self.stride,
            self.padding,
            self.output_padding,
        )


class _Pool2dForward(Transform):
    # DO NOT use in the `FullConnectedSyn`
    def __init__(
        self,
        channels: int,
        in_shape: Size2Type,
        out_shape: Size2Type,
        kernel_size: Size2Type,
        stride: Size2Type,
        padding: Size2Type,
        # fm_order: _Order3d,
        pool_type: Literal["avg", "max"],
    ) -> None:
        self.channels = channels
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        # self.fm_order = fm_order
        self.pool_type = pool_type

        super().__init__(np.asarray(1, dtype=np.int8))

    def __call__(self, x: SpikeType, *args, **kwargs) -> SpikeType:
        # if self.fm_order == "HWC":
        #     # (N,) -> (H, W, C) -> (C, H, W)
        #     _x = x.reshape(self.in_shape + (self.channels,)).transpose(2, 0, 1)
        # else:
        _x = x.reshape((self.channels,) + self.in_shape)

        return _func_pool2d(
            _x, self.out_shape, self.ksize, self.stride, self.padding, self.pool_type
        )

    @property
    def connectivity(self):
        return _pool2d_kernel_unroll(
            self.channels,
            self.in_shape,
            self.out_shape,
            self.ksize,
            self.stride,
        )
