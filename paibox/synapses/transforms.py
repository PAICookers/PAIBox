from enum import Enum, auto, unique
from typing import Tuple, Type, Union

import numpy as np
from numpy.typing import NDArray
from paicorelib import WeightPrecision as WP

from paibox.exceptions import ShapeError
from paibox.types import DataArrayType, WeightType
from paibox.utils import is_shape

__all__ = ["ConnType", "OneToOne", "ByPass", "AllToAll", "MaskedLinear"]


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

    BYPASS = auto()

    All2All = auto()
    """All-to-all connection."""


def _get_weight_precision(weight: np.ndarray, enable_wp_opt: bool) -> WP:
    """Get the actual weight_precision of the weight."""
    _max = np.max(weight, axis=None).astype(np.int32)
    _min = np.min(weight, axis=None).astype(np.int32)

    if _max > MAX_INT8 or _min < MIN_INT8:
        raise ValueError(f"Weight precision out of range, [{_min}, {_max}]")

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
    weights: WeightType
    """The actual weights in synapse. Must stored in `np.int8` format."""

    def __call__(self, *args, **kwargs) -> NDArray[np.int32]:
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
                If `weights` is a scalar(ndim = 0), the connectivity matrix will be:
                [[x, 0, 0]
                 [0, x, 0]
                 [0, 0, x]]

                Or `weights` is an array(ndim = 1), [x, y, z] corresponding to the weights of \
                    the post-neurons respectively. The connectivity matrix will be:
                [[x, 0, 0]
                 [0, y, 0]
                 [0, 0, z]]
        """
        self.num = num

        if isinstance(weights, np.ndarray) and not is_shape(weights, (num,)):
            raise ShapeError(
                f"Excepted shape is ({num},), but we got shape {weights.shape}"
            )

        # The ndim of weights = 0 or 1.
        self.weights = np.asarray(weights, dtype=np.int8)

        if not self.weights.ndim in (0, 1):
            raise ShapeError(f"The ndim of weights must be 0 or 1.")

    def __call__(self, x: np.ndarray, *args, **kwargs) -> NDArray[np.int32]:
        output = x * self.weights.copy()

        return output.astype(np.int32)

    @property
    def connectivity(self):
        return (
            (self.weights * np.eye(self.num, dtype=np.bool_)).astype(self.conn_dtype)
            if self.weights.ndim == 0
            else np.diag(self.weights).astype(self.conn_dtype)
        )


class ByPass(OneToOne):
    def __init__(self, num: int) -> None:
        """
        Arguments:
            - num: number of neurons.

        NOTE: The weights are always 1.
        """
        super().__init__(num, 1)


class AllToAll(Transform):
    def __init__(self, conn_size: Tuple[int, int], weights: DataArrayType) -> None:
        """
        Arguments:
            - num_in: number of source neurons.
            - num_out: number of destination neurons.
            - weights: synaptic weights. The shape must be a scalar or a matrix.
                If `weights` is a scalar(ndim = 0), the connectivity matrix will be:
                [[x, x, x]
                 [x, x, x]
                 [x, x, x]]

                Or `weights` is a matrix(ndim = 2), then the connectivity matrix will be:
                [[a, b, c]
                 [d, e, f]
                 [g, h, i]]
        """
        self.conn_size = conn_size

        if isinstance(weights, np.ndarray) and not is_shape(weights, conn_size):
            raise ShapeError(
                f"Excepted shape is {conn_size}, but we got shape {weights.shape}"
            )

        self.weights = np.asarray(weights, dtype=np.int8)

        if not self.weights.ndim in (0, 2):
            raise ShapeError(f"The ndim of weights must be 0 or 2.")

    def __call__(self, x: np.ndarray, *args, **kwargs) -> NDArray[np.int32]:
        """
        - When weights is a scalar, the output is a scalar. (Risky, DO NOT USE)
        - When weights is a matrix, the output is the dot product of `x` & `weights`.
        """
        if self.weights.ndim == 0:
            sum_x = np.sum(x, axis=None, dtype=np.int32)
            output = self.weights * np.full((self.conn_size[1],), sum_x, dtype=np.int32)
            # Risky
            # output = self.weights * sum_x
        else:
            output = x @ self.weights.copy().astype(np.int32)

        return output.astype(np.int32)

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
        conn_size: Tuple[int, int],
        weights: np.ndarray,
    ) -> None:
        """
        Arguments:
            - conn: connector. Only support `MatConn`.
            - weights: unmasked weights.
        """
        self.conn_size = conn_size

        if not is_shape(weights, self.conn_size):
            raise ShapeError(
                f"Excepted shape is {conn_size}, but we got {weights.shape}"
            )

        # Element-wise Multiplication
        self.weights = np.asarray(weights, dtype=np.int8)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> NDArray[np.int32]:
        output = x @ self.weights.copy().astype(np.int32)

        return output.astype(np.int32)

    @property
    def connectivity(self):
        return self.weights.astype(self.conn_dtype)
