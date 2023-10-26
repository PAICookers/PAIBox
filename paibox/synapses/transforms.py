from abc import ABC, abstractmethod
from typing import Type, Union

import numpy as np

from paibox.utils import is_shape

from .connector import MatConn, One2One, TwoEndConnector


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


class Transform(ABC):
    weights: np.ndarray
    num_in: int
    num_out: int
    mask: np.ndarray
    conn: TwoEndConnector

    def __call__(self, x):
        return x @ self.weights

    @property
    def dtype(self):
        """The dtype of the weight."""
        raise NotImplementedError

    def __init__(
        self,
        conn: TwoEndConnector,
        weights: Union[int, np.integer, np.ndarray],
    ) -> None:
        self.conn = conn
        self.mask = self.conn.build_mat()
        self.num_in = self.conn.source_num
        self.num_out = self.conn.dest_num
        self._init_weights(weights)
        # Element-wise Multiplication
        self.weights = np.asarray(weights, dtype=np.int8) * self.mask

    def _init_weights(self, weights: Union[int, np.integer, np.ndarray]) -> None:
        if isinstance(weights, np.ndarray):
            if is_shape(weights, (self.num_in, self.num_out)):
                self.weights = weights
            elif (
                is_shape(weights, (self.num_in,))
                and self.num_in == self.num_out
                and isinstance(self.conn, One2One)
            ):
                # only one to one connection can use a vector as weights
                self.weights = np.diag(weights)
            else:
                raise ValueError
        elif isinstance(weights, np.integer) or isinstance(weights, int):
            self.weights = np.full((self.num_in, self.num_out), weights)
        else:
            raise ValueError
        self.weights = self.weights.astype(np.int8)

        if self.weights.shape == self.mask.shape:
            self.weights = self.weights * self.mask
        else:
            raise ValueError

    @property
    def connectivity(self) -> np.ndarray:
        """The connectivity matrix in `np.ndarray` format."""
        raise NotImplementedError


# class OneToOne(Transform):
#     def __init__(self, num: int, weights: Union[int, np.integer, np.ndarray]) -> None:
#         """
#         Arguments:
#             - num: number of neurons.
#             - weights: synaptic weights. The shape must be a scalar or array (num,).
#                 If `weights` is a scalar, the connectivity matrix will be:
#                 [[x, 0, 0]
#                  [0, x, 0]
#                  [0, 0, x]]

#                 Or `weights` is an array, [x, y, z] corresponding to the weights of \
#                     the post-neurons respectively. The connectivity matrix will be:
#                 [[x, 0, 0]
#                  [0, y, 0]
#                  [0, 0, z]]
#         """
#         self.num = num

#         if isinstance(weights, np.ndarray) and not is_shape(weights, (num,)):
#             # TODO Error description
#             raise ValueError

#         self.weights = np.asarray(weights, dtype=np.int8)

#     def __call__(self, x: np.ndarray) -> np.ndarray:
#         return x * self.weights

#     @property
#     def dtype(self) -> Union[Type[np.bool_], Type[np.int8]]:
#         return _get_dtype(self.weights)

#     @property
#     def connectivity(self) -> np.ndarray:
#         return (
#             self.weights.astype(self.dtype)
#             if self.weights.ndim == 2
#             else (self.weights * np.eye(self.num, dtype=np.bool_)).astype(self.dtype)
#         )


# class ByPass(OneToOne):
#     def __init__(self, num: int) -> None:
#         """
#         Arguments:
#             - num: number of neurons.

#         NOTE: The weights are always 1.
#         """
#         super().__init__(num, 1)


# class AllToAll(Transform):
#     def __init__(
#         self, num_in: int, num_out: int, weights: Union[int, np.integer, np.ndarray]
#     ) -> None:
#         """
#         Arguments:
#             - num_in: number of source neurons.
#             - num_out: number of destination neurons.
#             - weights: synaptic weights. The shape must be a scalar or a matrix.
#                 If `weights` is a scalar, the connectivity matrix will be:
#                 [[x, x, x]
#                  [x, x, x]
#                  [x, x, x]]

#                 Or `weights` is a matrix, then the connectivity matrix will be:
#                 [[a, b, c]
#                  [d, e, f]
#                  [g, h, i]]
#         """
#         self.num_in = num_in
#         self.num_out = num_out

#         if isinstance(weights, np.ndarray) and not is_shape(weights, (num_in, num_out)):
#             # TODO Error description
#             raise ValueError

#         self.weights = np.asarray(weights, dtype=np.int8)

#     def __call__(self, x: np.ndarray) -> np.ndarray:
#         """
#         When weights is a scalar, the output is a scalar.
#         When weights is a matrix, the output is the dot product of `x` and `weights`.
#         """
#         if self.weights.ndim == 0:
#             # weight is a scalar
#             if x.ndim == 1:
#                 _x = np.sum(x).astype(np.int32)
#                 output = self.weights * _x
#             else:
#                 # TODO
#                 raise ValueError

#         elif self.weights.ndim == 2:
#             output = x @ self.weights
#         else:
#             raise ValueError(f"weights.ndim={self.weights.ndim}")

#         return output

#     @property
#     def dtype(self) -> Union[Type[np.bool_], Type[np.int8]]:
#         return _get_dtype(self.weights)

#     @property
#     def connectivity(self) -> np.ndarray:
#         return (
#             self.weights.astype(self.dtype)
#             if self.weights.ndim == 2
#             else (self.weights * np.ones((self.num_in, self.num_out), np.bool_)).astype(
#                 self.dtype
#             )
#         )


# class MaskedLinear(Transform):
#     def __init__(
#         self,
#         conn: MatConn,
#         weights: Union[int, np.integer, np.ndarray],
#     ) -> None:
#         """
#         Arguments:
#             - conn: connector. Only support `MatConn`.
#             - weights: unmasked weights.

#         TODO to be varified.
#         """
#         self.conn = conn
#         self.mask = self.conn.build_mat()
#         self.num_in = self.conn.source_num
#         self.num_out = self.conn.dest_num

#         if isinstance(weights, np.ndarray) and not is_shape(
#             weights, (self.num_in, self.num_out)
#         ):
#             # TODO Error description
#             raise ValueError

#         # Element-wise Multiplication
#         self.weights = np.asarray(weights, dtype=np.int8) * self.mask

#     def __call__(self, x: np.ndarray) -> np.ndarray:
#         return x @ self.weights

#     @property
#     def dtype(self) -> Union[Type[np.bool_], Type[np.int8]]:
#         return _get_dtype(self.weights)

#     @property
#     def connectivity(self) -> np.ndarray:
#         return self.weights.astype(self.dtype)
