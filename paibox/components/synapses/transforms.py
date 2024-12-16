import warnings
from enum import Enum, auto, unique
from typing import Literal, Optional

import numpy as np
from paicorelib import WeightWidth as WW

from paibox.exceptions import AutoOptimizationWarning, ShapeError
from paibox.types import (
    VOLTAGE_DTYPE,
    WEIGHT_DTYPE,
    DataType,
    IntScalarType,
    NeuOutType,
    SynOutType,
    WeightType,
)
from paibox.utils import is_shape, shape2num, typical_round

from .conv_types import Size1Type, Size2Type, SizeAnyType, _SizeAnyType
from .conv_utils import (
    _conv1d_faster,
    _conv1d_unroll,
    _conv2d_faster,
    _conv2d_semifolded_unroll,
    _conv2d_unroll,
    _convtranspose1d_faster,
    _convtranspose1d_unroll,
    _convtranspose2d_faster,
    _convtranspose2d_unroll,
    _func_pool1d,
    _func_pool2d,
    _pool1d_kernel_unroll,
    _pool2d_kernel_unroll,
)

__all__ = [
    "OneToOne",
    "AllToAll",
    "Identity",
    "MaskedLinear",
    "Conv1dForward",
    "Conv2dForward",
    "Conv2dSemiFoldedForward",
    "ConvTranspose1dForward",
    "ConvTranspose2dForward",
    "CompareMax",
]


MAX_INT1 = np.int8(1)
MIN_INT1 = np.int8(0)
MAX_INT2 = np.int8(1)
MIN_INT2 = np.int8(-2)
MAX_INT4 = np.int8(7)
MIN_INT4 = np.int8(-8)
MAX_INT8 = np.iinfo(np.int8).max
MIN_INT8 = np.iinfo(np.int8).min


@unique
class ConnType(Enum):
    MatConn = auto()
    """General matrix connection."""

    One2One = auto()
    """One-to-one connection."""

    Identity = auto()
    """Identity connection with scaling factor."""

    All2All = auto()
    """All-to-all connection."""


def _set_coarse_dtype(raw_w: DataType) -> WeightType:
    """Convert raw weights to `np.ndarray` coarsely (without optimization).

    Description:
        - For weights of type `bool` or `np.bool_`, set `np.int8` as the dtype.
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

        return np.asarray(raw_w, dtype=WEIGHT_DTYPE)

    # Convert list or tuple to np.ndarray
    _array = np.asarray(raw_w)
    _max = np.max(_array, axis=None)
    _min = np.min(_array, axis=None)

    if _max > MAX_INT8 or _min < MIN_INT8:
        raise ValueError(f"weight out of range int8, got [{_min}, {_max}].")

    if _array.dtype > np.int8:
        warnings.warn(
            f"dtype of weight is optimized automatically, {_array.dtype} -> int8.",
            AutoOptimizationWarning,
        )
        _dtype = WEIGHT_DTYPE

    elif _array.dtype == np.bool_ or _array.dtype == np.int8:
        _dtype = WEIGHT_DTYPE
    else:
        raise TypeError(f"weights must be bool or int8, but got {_array.dtype}.")

    return _array.astype(_dtype, casting="same_kind")


def _get_weight_width_inner(weight: WeightType, enable_wp_opt: bool) -> WW:
    """Get the actual width of the weight."""
    _max, _min = np.max(weight), np.min(weight)

    if enable_wp_opt:
        if _max <= MAX_INT1 and _min >= MIN_INT1:
            return WW.WEIGHT_WIDTH_1BIT
        elif _max <= MAX_INT2 and _min >= MIN_INT2:
            return WW.WEIGHT_WIDTH_2BIT
        elif _max <= MAX_INT4 and _min >= MIN_INT4:
            return WW.WEIGHT_WIDTH_4BIT
        else:
            return WW.WEIGHT_WIDTH_8BIT
    else:
        return WW.WEIGHT_WIDTH_8BIT


class Transform:
    def __init__(self, weights: DataType) -> None:
        self.weights = _set_coarse_dtype(weights)
        """The actual weights in synapses. Stored in np.int8 format."""

        self.weights.setflags(write=False)

    def __call__(self, *args, **kwargs) -> SynOutType:
        # Ensure that in all subclasses, the output dimensions are (M,).
        raise NotImplementedError(
            "function '__call__' must be implemented in the subclasses."
        )

    def _get_weight_width(self, enable_wp_opt: bool) -> WW:
        return _get_weight_width_inner(self.weights, enable_wp_opt)

    @property
    def connectivity(self) -> WeightType:
        """The connectivity matrix in `np.ndarray` format."""
        raise NotImplementedError(
            "property 'connectivity' must be implemented in the subclasses."
        )


class OneToOne(Transform):
    def __init__(self, num: int, weights: DataType) -> None:
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

    def __call__(self, x: NeuOutType, *args, **kwargs) -> SynOutType:
        # (N,) * (N,) -> (N,)
        return x * self.weights.astype(VOLTAGE_DTYPE)

    @property
    def connectivity(self):
        return (
            (self.weights * np.identity(self.num, dtype=WEIGHT_DTYPE))
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
    def __init__(self, conn_size: Size2Type, weights: DataType) -> None:
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

    def __call__(self, x: NeuOutType, *args, **kwargs) -> SynOutType:
        """
        NOTE:
            - When weights is a scalar, the output is a scalar (sum * w) & repeated `conn_size[1]` times.
            - When weights is a matrix, the output is the dot product of `x` & weights.
        """
        if self.weights.ndim == 0:
            sum_x = np.sum(x, dtype=VOLTAGE_DTYPE)
            # (M,)
            output = np.full(
                (self.conn_size[1],), self.weights * sum_x, dtype=VOLTAGE_DTYPE
            )
        else:
            # (N,) @ (N, M) -> (M,)
            output = x @ self.weights.astype(VOLTAGE_DTYPE)

        return output

    @property
    def connectivity(self):
        return (
            self.weights
            if self.weights.ndim == 2
            else (self.weights * np.ones(self.conn_size, dtype=WEIGHT_DTYPE))
        )


class MaskedLinear(Transform):
    def __init__(
        self, in_shape: SizeAnyType, out_shape: SizeAnyType, weights: np.ndarray
    ) -> None:
        self.in_shape = (1,) * (2 - len(in_shape)) + in_shape
        self.out_shape = (1,) * (2 - len(out_shape)) + out_shape

        if self.in_shape[0] == weights.shape[0]:
            self.axes = (1, 0)
        elif self.in_shape[1] == weights.shape[0]:
            self.axes = (0, 1)
        else:
            raise ShapeError(
                f"cannot do matmul between shape {in_shape} & {weights.shape}."
            )

        _in_shape = tuple(self.in_shape[i] for i in self.axes)

        if (expected_oshape := _in_shape[:-1] + weights.shape[1:]) != self.out_shape:
            raise ShapeError(
                f"wrong output shape, expected {expected_oshape}, but got {self.out_shape}."
            )

        super().__init__(weights)

    def __call__(self, x: NeuOutType, *args, **kwargs) -> SynOutType:
        # (n?, k) @ (k, m?) -> (n?, m?)
        _x = x.reshape(self.in_shape).transpose(self.axes)

        return _x @ self.weights.astype(VOLTAGE_DTYPE)

    @staticmethod
    def _matmul_unroll(
        in_shape: SizeAnyType,
        out_shape: SizeAnyType,
        weights: WeightType,
        axes: tuple[int, ...],
    ) -> WeightType:
        n_ishape = shape2num(in_shape)
        n_oshape = shape2num(out_shape)
        in_shape_t = tuple(in_shape[i] for i in axes)

        w_unrolled = np.zeros((n_ishape, n_oshape), dtype=WEIGHT_DTYPE)

        orig_idx = np.arange(n_ishape).reshape(in_shape_t)
        mapping_tbl = orig_idx.transpose(np.argsort(axes)).ravel()

        for i in range(in_shape_t[0]):
            w_unrolled[
                i * weights.shape[0] : (i + 1) * weights.shape[0],
                i * weights.shape[1] : (i + 1) * weights.shape[1],
            ] = weights

        return w_unrolled[mapping_tbl]

    @property
    def connectivity(self):
        return self._matmul_unroll(
            self.in_shape, self.out_shape, self.weights, self.axes
        )

    @property
    def is_T(self) -> bool:
        return self.axes == (1, 0)


class _ConvNdForward(Transform):
    def __init__(
        self,
        in_shape: SizeAnyType,
        out_shape: SizeAnyType,
        kernel: np.ndarray,
        stride: _SizeAnyType = 0,
        padding: _SizeAnyType = 0,
        groups: int = 1,
        output_padding: _SizeAnyType = 0,
    ) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.output_padding = output_padding

        super().__init__(kernel)


class Conv1dForward(_ConvNdForward):

    in_shape: Size1Type
    out_shape: Size1Type
    stride: Size1Type
    padding: Size1Type
    groups: int

    def __call__(self, x: NeuOutType, *args, **kwargs) -> SynOutType:
        cin = self.weights.shape[1] * self.groups

        # if self.fm_order == "LC":
        #     # (N,) -> (L, C) -> (C, L)
        #     _x = x.reshape(self.in_shape + (cin,)).T
        # else:
        _x = x.reshape((cin,) + self.in_shape)

        return _conv1d_faster(
            _x, self.out_shape, self.weights, self.stride, self.padding, self.groups
        )

    @property
    def connectivity(self):
        return _conv1d_unroll(
            self.in_shape,
            self.out_shape,
            self.weights,
            self.stride,
            self.padding,
            self.groups,
        )


class Conv2dForward(_ConvNdForward):

    in_shape: Size2Type
    out_shape: Size2Type
    stride: Size2Type
    padding: Size2Type
    groups: int

    def __call__(self, x: NeuOutType, *args, **kwargs) -> SynOutType:
        cin = self.weights.shape[1] * self.groups

        # if self.fm_order == "HWC":
        #     # (N,) -> (H, W, C) -> (C, H, W)
        #     _x = x.reshape(self.in_shape + (cin,)).transpose(2, 0, 1)
        # else:
        _x = x.reshape((cin,) + self.in_shape)

        return _conv2d_faster(
            _x, self.out_shape, self.weights, self.stride, self.padding, self.groups
        )

    @property
    def connectivity(self):
        return _conv2d_unroll(
            self.in_shape,
            self.out_shape,
            self.weights,
            self.stride,
            self.padding,
            self.groups,
        )


class Conv2dSemiFoldedForward(_ConvNdForward):
    in_shape: Size2Type
    out_shape: Size2Type
    stride: Size2Type
    padding: Size2Type
    groups: int

    def __call__(self, x: NeuOutType, *args, **kwargs) -> SynOutType:
        return x @ self.connectivity

    @property
    def connectivity(self):
        return _conv2d_semifolded_unroll(
            self.in_shape,
            self.out_shape,
            self.weights,
            self.stride,
            self.padding,
            self.groups,
        )


class ConvTranspose1dForward(_ConvNdForward):
    in_shape: Size1Type
    out_shape: Size1Type
    stride: Size1Type
    padding: Size1Type
    groups: int
    output_padding: Size1Type

    def __call__(self, x: NeuOutType, *args, **kwargs) -> SynOutType:
        cin = self.weights.shape[1] * self.groups

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


class ConvTranspose2dForward(_ConvNdForward):
    in_shape: Size2Type
    out_shape: Size2Type
    stride: Size2Type
    padding: Size2Type
    groups: int
    output_padding: Size2Type

    def __call__(self, x: NeuOutType, *args, **kwargs) -> SynOutType:
        cin = self.weights.shape[1] * self.groups

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


class _PoolNdForward(Transform):
    def __init__(
        self,
        channels: int,
        in_shape: SizeAnyType,
        out_shape: SizeAnyType,
        kernel_size: SizeAnyType,
        stride: _SizeAnyType,
        padding: _SizeAnyType,
        pool_type: Literal["avg", "max"],
        threshold: Optional[int] = None,
    ) -> None:
        self.channels = channels
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_type = pool_type

        if isinstance(threshold, int):
            self.threshold = threshold
        else:
            self.threshold = typical_round(shape2num(kernel_size) / 2)

        super().__init__(1)


class _Pool1dForward(_PoolNdForward):
    in_shape: Size1Type
    out_shape: Size1Type
    ksize: Size1Type
    stride: Size1Type
    padding: Size1Type

    def __call__(self, x: NeuOutType, *args, **kwargs) -> NeuOutType:
        _x = x.reshape((self.channels,) + self.in_shape)

        return _func_pool1d(
            _x,
            self.out_shape,
            self.ksize,
            self.stride,
            self.padding,
            self.pool_type,
            self.threshold,
        )

    @property
    def connectivity(self):
        return _pool1d_kernel_unroll(
            self.channels,
            self.in_shape,
            self.out_shape,
            self.ksize,
            self.stride,
            self.padding,
        )


class _Pool2dForward(_PoolNdForward):
    in_shape: Size2Type
    out_shape: Size2Type
    ksize: Size2Type
    stride: Size2Type
    padding: Size2Type

    def __call__(self, x: NeuOutType, *args, **kwargs) -> NeuOutType:
        # if self.fm_order == "HWC":
        #     # (N,) -> (H, W, C) -> (C, H, W)
        #     _x = x.reshape(self.in_shape + (self.channels,)).transpose(2, 0, 1)
        # else:
        _x = x.reshape((self.channels,) + self.in_shape)

        return _func_pool2d(
            _x,
            self.out_shape,
            self.ksize,
            self.stride,
            self.padding,
            self.pool_type,
            self.threshold,
        )

    @property
    def connectivity(self):
        return _pool2d_kernel_unroll(
            self.channels,
            self.in_shape,
            self.out_shape,
            self.ksize,
            self.stride,
            self.padding,
        )


class CompareMax(AllToAll):
    def __init__(self, conn_size: Size2Type, mask: DataType) -> None:
        """A transformation that finds the maximum of the input vector according to each column of the  \
            mask matrix.

        NOTE: the value of mask matrix must be either 0 or 1.
        """
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("the mask must be 0 or 1.")

        super().__init__(conn_size, mask)

    def __call__(self, x: NeuOutType, *args, **kwargs) -> SynOutType:
        """The maximum value of the input corresponding to the non-zero columns of the weight matrix is \
            taken as the output.
            x = (x1, x2, ..., xn)
            w = [n*m]
            y = (y1, y2, ..., ym)
        """
        if self.weights.ndim == 0:
            output = np.full(
                (self.conn_size[1],),
                self.weights * np.max(x, axis=None),
                dtype=VOLTAGE_DTYPE,
            )
        else:
            output = np.max(x[:, None] * self.weights, axis=0).astype(VOLTAGE_DTYPE)

        return output
