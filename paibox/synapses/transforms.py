from typing import Union
import numpy as np
from paibox.synapses.connector import TwoEndConnector


class Transform:
    @property
    def shape_in(self):
        raise NotImplementedError

    @property
    def shape_out(self):
        raise NotImplementedError


class PassBy(Transform):
    def __call__(self, x) -> None:
        return x


class OneToOne(Transform):
    def __init__(self, num: int, weights: Union[int, np.integer, np.ndarray]) -> None:
        self.num = num

        weights = np.asarray(weights)
        assert weights.shape == (num,)
        self.weights = weights

    def __call__(self, x):
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

        weights = np.asarray(weights)
        assert weights.shape == (num_in, num_out)
        self.weights = weights

    def __call__(self, x):
        if self.weights.ndim == 0:
            return x * self.weights
        else:
            return x @ self.weights

    @property
    def shape_in(self) -> int:
        return self.num_in

    @property
    def shape_out(self) -> int:
        return self.num_out


class Dense(Transform):
    def __init__(
        self, num_in: int, num_out: int, weights: Union[int, np.integer, np.ndarray]
    ) -> None:
        self.num_in = num_in
        self.num_out = num_out
        self.weights = weights

    def __call__(self, x) -> None:
        return x @ self.weights

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
        self.conn = conn
        self.mask = self.conn.build_mat()

        weights = np.asarray(weights)
        assert weights.shape == (self.conn.source_num, self.conn.dest_num)
        self.weights = weights

    def __call__(self, x) -> None:
        return x @ (self.weights * self.mask)

    @property
    def shape_in(self) -> int:
        return self.conn.source_num

    @property
    def shape_out(self) -> int:
        return self.conn.dest_num
