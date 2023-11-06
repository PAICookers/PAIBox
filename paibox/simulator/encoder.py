from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
from numpy.random import Generator

from paibox._types import Shape
from paibox.base import DynamicSys
from paibox.utils import as_shape, shape2num

__all__ = ["PeriodicEncoder", "PoissonEncoder"]


class Encoder(DynamicSys):
    def __init__(
        self,
        shape_out: Shape = (),
        *,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        TODO Consider it as a `PAIBoxObject`? Is `run()` useful?
        """
        self._shape_out = as_shape(shape_out)
        self.seed = seed

        super().__init__(name)

    def run(
        self,
        duration: int,
        dt: int = 1,
        rng: Generator = np.random.default_rng(),
        **kwargs,
    ) -> np.ndarray:
        if duration < 0:
            # TODO
            raise ValueError

        n_steps = int(duration / dt)
        return self.run_steps(n_steps, rng, **kwargs)

    def run_steps(self, n_steps: int, rng: Generator, **kwargs) -> np.ndarray:
        output = np.zeros((n_steps,) + self.shape_out, dtype=np.bool_)

        for i in range(n_steps):
            output[i] = self(**kwargs)

        return output

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
    def __init__(self, T: int) -> None:
        super().__init__((1,))

        if T < 1:
            raise ValueError

        self.T = T
        self.set_memory("spike", None)
        self.set_memory("t", 0)

    def __call__(self, x: Optional[np.ndarray] = None):
        if self.spike is None:
            if x is None:
                raise ValueError

            self.single_step_encode(x)

        t = self.t
        self.t += 1

        if self.t >= self.T:
            self.t = 0

        if self.spike is not None:
            return self.spike[t]

    @abstractmethod
    def single_step_encode(self, x: np.ndarray):
        raise NotImplementedError


class PeriodicEncoder(StatefulEncoder):
    def __init__(self, spike: np.ndarray) -> None:
        """Periodic encoder.

        Args:
            - spike: the input spike.
        """
        super().__init__(spike.shape[0])

    def single_step_encode(self, spike: np.ndarray) -> None:
        self.spike = spike
        self.T = spike.shape[0]


class PoissonEncoder(StatelessEncoder):
    def __init__(self, shape_out: Shape = (), **kwargs) -> None:
        super().__init__(shape_out, **kwargs)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.less_equal(np.random.rand(*x.shape), x).astype(np.bool_)
