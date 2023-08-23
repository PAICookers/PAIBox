from typing import Union

import numpy as np

from paibox.synapses.connector import MatConn, TwoEndConnector
from paibox.utils import is_shape


class Transform:
    @property
    def shape_in(self):
        raise NotImplementedError

    @property
    def shape_out(self):
        raise NotImplementedError


class ByPass(Transform):
    def __init__(self, num: int):
        self.num = num
        self.weights = 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    @property
    def shape_in(self) -> int:
        return self.num

    @property
    def shape_out(self) -> int:
        return self.num


class OneToOne(Transform):
    def __init__(self, num: int, weights: Union[int, np.integer, np.ndarray]) -> None:
        """
        Arguments:
            - num: the number of neurons.
            - weights: the synaptic weights.
        """
        self.num = num

        if isinstance(weights, np.ndarray):
            assert is_shape(weights, (num,))

        self.weights = weights

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x * self.weights

    @property
    def shape_in(self) -> int:
        return self.num

    @property
    def shape_out(self) -> int:
        return self.num


class AllToAll(Transform):
    def __init__(
        self, num_in: int, num_out: int, weights: Union[int, np.integer, np.ndarray]
    ) -> None:
        self.num_in = num_in
        self.num_out = num_out

        if isinstance(weights, np.ndarray):
            assert is_shape(weights, (num_in, num_out))

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
    def shape_in(self) -> int:
        return self.num_in

    @property
    def shape_out(self) -> int:
        return self.num_out


class MaskedLinear(Transform):
    def __init__(
        self,
        conn: TwoEndConnector,
        weights: Union[int, np.integer, np.ndarray],
    ) -> None:
        """
        Arguments:
            - conn: only support `MatConn`.
            - weights: unmasked weights.
        """
        self.conn = conn
        self.mask = self.conn.build_mat()

        if isinstance(weights, np.ndarray):
            assert is_shape(weights, (self.conn.source_num, self.conn.dest_num))

        # The weight is fake until with the mask
        self.weights = np.asarray(weights) * self.mask

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights

    @property
    def shape_in(self) -> int:
        return self.conn.source_num

    @property
    def shape_out(self) -> int:
        return self.conn.dest_num
