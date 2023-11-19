from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
from numpy.random import Generator

from paibox._types import Shape
from paibox.base import DynamicSys
from paibox.utils import as_shape, shape2num

__all__ = ["PeriodicEncoder", "PoissonEncoder"]

MAXSEED = np.iinfo(np.uint32).max
MAXINT = np.iinfo(np.int32).max


class Encoder(DynamicSys):
    def __init__(
        self,
        shape_out: Shape = (0,),
        seed: Optional[int] = None,
        *,
        name: Optional[str] = None,
    ) -> None:
        """
        TODO Consider it as a `PAIBoxObject`? Is `run()` useful?
        """
        self._shape_out = as_shape(shape_out)
        self.seed = seed

        super().__init__(name)

    def get_rng(self) -> Generator:
        seed = np.random.randint(MAXINT) if self.seed is None else self.seed
        return np.random.default_rng(seed)

    @property
    def num_out(self) -> int:
        return shape2num(self._shape_out)

    @property
    def shape_in(self) -> Tuple[int, ...]:
        return (0,)

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self._shape_out


class StatelessEncoder(Encoder):
    pass


class StatefulEncoder(Encoder):
    def __init__(self, T: int, shape_out: Shape, **kwargs) -> None:
        super().__init__(shape_out, **kwargs)

        if T < 1:
            raise ValueError(f"T should be > 0, but got {T}")

        self.T = T
        self.set_memory("spike", None)
        self.set_memory("t", 0)

    def __call__(self, x: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if self.spike is None:
            if x is None:
                raise ValueError("Input must be given if spike is None")

            self.single_step_encode(x)

        t = self.t
        self.t += 1

        if self.t >= self.T:
            self.t = 0

        return self.spike[t]

    @abstractmethod
    def single_step_encode(self, x: np.ndarray):
        raise NotImplementedError


class PeriodicEncoder(StatefulEncoder):
    def __init__(self, spike: np.ndarray, **kwargs) -> None:
        """Periodic encoder.

        Args:
            - spike: the input spike. Encode when instantiate itself. \
                T = `.shape[0]` & shape_out = `.shape[1]`.
        """
        super().__init__(spike.shape[0], spike.shape[1], **kwargs)
        self.spike = spike


class PoissonEncoder(StatelessEncoder):
    def __init__(
        self, shape_out: Shape = (0,), seed: Optional[int] = None, **kwargs
    ) -> None:
        super().__init__(shape_out, seed, **kwargs)

    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return np.less_equal(self.get_rng().random(x.shape), x).astype(np.bool_)
