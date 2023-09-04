from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from paibox.utils import is_shape

from .connector import MatConn


class Transform(ABC):
    @abstractmethod
    def __call__(self, x) -> ...:
        raise NotImplementedError


class OneToOne(Transform):
    def __init__(self, num: int, weights: Union[int, np.integer, np.ndarray]) -> None:
        """
        Arguments:
            - num: number of neurons.
            - weights: synaptic weights. The shape must be a scalar or (num,).
        """
        self.num = num

        if isinstance(weights, np.ndarray) and not is_shape(weights, (num,)):
            # TODO Error description
            raise ValueError

        self.weights = weights

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x * self.weights

    @property
    def connectivity(self) -> np.ndarray:
        return (
            self.weights
            if isinstance(self.weights, np.ndarray)
            else self.weights * np.eye(self.num)
        )


class ByPass(OneToOne):
    def __init__(self, num: int) -> None:
        """
        Arguments:
            - num: number of neurons.

        The synaptic weights are always 1.
        """
        super().__init__(num, 1)


class AllToAll(Transform):
    def __init__(
        self, num_in: int, num_out: int, weights: Union[int, np.integer, np.ndarray]
    ) -> None:
        """
        Arguments:
            - num_in: number of source neurons.
            - num_out: number of destination neurons.
            - weights: synaptic weights.
        """
        self.num_in = num_in
        self.num_out = num_out

        if isinstance(weights, np.ndarray) and not is_shape(weights, (num_in, num_out)):
            # TODO Error description
            raise ValueError

        self.weights = np.asarray(weights)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        When weights is a scalar, the output is a scalar.
        When weights is a matrix, the output is the dot product of `x` and `weights`.
        """
        if self.weights.ndim == 0:
            # weight is a scalar
            if x.ndim == 1:
                _x = np.sum(x)
            else:
                raise ValueError

            output = self.weights * _x
        elif self.weights.ndim == 2:
            output = x @ self.weights  # same as np.dot(x, weights)
        else:
            raise ValueError(f"weights.ndim={self.weights.ndim}")

        return output

    @property
    def connectivity(self) -> np.ndarray:
        return (
            self.weights
            if self.weights.ndim == 2
            else self.weights * np.ones((self.num_in, self.num_out))
        )


class MaskedLinear(Transform):
    def __init__(
        self,
        conn: MatConn,
        weights: Union[int, np.integer, np.ndarray],
    ) -> None:
        """
        Arguments:
            - conn: connector. Only support `MatConn`.
            - weights: unmasked weights.
        """
        self.conn = conn
        self.mask = self.conn.build_mat()
        self.num_in = self.conn.source_num
        self.num_out = self.conn.dest_num

        if isinstance(weights, np.ndarray) and not is_shape(
            weights, (self.num_in, self.num_out)
        ):
            # TODO Error description
            raise ValueError

        # Element-wise Multiplication
        self.weights = np.asarray(weights) * self.mask.astype(np.int8)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights

    @property
    def connectivity(self) -> np.ndarray:
        return self.weights
