from typing import Optional, Tuple

import numpy as np
from numpy.random import Generator

from paibox._types import Shape
from paibox.utils import as_shape, shape2num

__all__ = ["PoissonEncoder"]


class Encoder:
    def __init__(
        self,
        shape_out: Shape,
        seed: Optional[int] = None,
    ) -> None:
        """
        TODO Consider it as a `PAIBoxObject`? Is `run()` useful?
        """
        self._shape_out = as_shape(shape_out)
        self.seed = seed

    def run(
        self,
        duration: int,
        dt: int = 1,
        rng: Generator = np.random.default_rng(),
        **kwargs
    ) -> np.ndarray:
        if duration < 0:
            # TODO
            raise ValueError

        n_steps = int(duration / dt)
        return self.run_steps(n_steps, rng, **kwargs)

    def run_steps(self, n_steps: int, rng: Generator, **kwargs) -> np.ndarray:
        output = np.zeros((n_steps,) + self.shape_out, dtype=np.bool_)

        for i in range(n_steps):
            output[i] = self(i, **kwargs)  # Do `__call__`

        return output

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def num_out(self) -> int:
        return shape2num(self._shape_out)

    @property
    def shape_in(self) -> Tuple[int, ...]:
        return (0,)

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self._shape_out


class PoissonEncoder(Encoder):
    def __init__(self, shape_out: Shape, **kwargs) -> None:
        super().__init__(shape_out, **kwargs)

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return np.less_equal(np.random.rand(*input.shape), input).astype(np.bool_)
