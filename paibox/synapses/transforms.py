from enum import Enum, unique
from typing import Tuple, Type, Union

import numpy as np

from paibox.exceptions import *
from paibox.utils import is_shape


@unique
class ConnType(Enum):
    MatConn = 0
    One2One = 1
    All2All = 2


def _get_dtype(weight: np.ndarray) -> Union[Type[np.bool_], Type[np.int8]]:
    """Get the actual dtype of the weight.

    Consider when the weight is a scalar:
        - 1. `np.bool_`, 1-bit unsigned.
        - 2. `np.int8`, 8-bit signed. Not fully supported.
    """
    _max = np.max(weight, axis=None).astype(np.int32)
    _min = np.min(weight, axis=None).astype(np.int32)

    if _max <= np.bool_(True) and _min >= np.bool_(False):
        return np.bool_

    if _max <= np.int8(127) and _min >= np.int8(-128):
        # raise NotImplementedError
        return np.int8

    raise OverflowError


class Transform:
    weights: np.ndarray

    @property
    def dtype(self) -> Union[Type[np.bool_], Type[np.int8]]:
        """The dtype of the weight."""
        return _get_dtype(self.weights)

    @property
    def connectivity(self) -> np.ndarray:
        """The connectivity matrix in `np.ndarray` format."""
        raise NotImplementedError


class OneToOne(Transform):
    def __init__(self, num: int, weights: Union[int, np.integer, np.ndarray]) -> None:
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

        assert self.weights.ndim in (0, 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x * self.weights.copy().astype(np.int32)

    @property
    def connectivity(self) -> np.ndarray:
        return (
            (self.weights * np.eye(self.num, dtype=np.bool_)).astype(self.dtype)
            if self.weights.ndim == 0
            else np.diag(self.weights).astype(self.dtype)
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
    def __init__(
        self, conn_size: Tuple[int, int], weights: Union[int, np.integer, np.ndarray]
    ) -> None:
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

        assert self.weights.ndim in (0, 2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
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

        return output

    @property
    def connectivity(self) -> np.ndarray:
        return (
            self.weights.astype(self.dtype)
            if self.weights.ndim == 2
            else (self.weights * np.ones(self.conn_size, dtype=np.bool_)).astype(
                self.dtype
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
                f"Excepted shape is {conn_size}, but we got shape {weights.shape}"
            )

        # Element-wise Multiplication
        self.weights = np.asarray(weights, dtype=np.int8)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights.copy().astype(np.int32)

    @property
    def connectivity(self) -> np.ndarray:
        return self.weights.astype(self.dtype)
