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
        keep_shape: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self._shape_out = as_shape(shape_out)
        self.keep_shape = keep_shape
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
        output = np.zeros((n_steps,) + self.varshape, dtype=np.bool_)

        for i in range(n_steps):
            output[i] = self(i, **kwargs)  # Do `__call__`

        return output

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def varshape(self) -> Tuple[int, ...]:
        return self.shape_out if self.keep_shape else (self.num_out,)

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
    def __init__(
        self, shape_out: Shape = 1, *, keep_shape: bool = False, **kwargs
    ) -> None:
        super().__init__(shape_out, keep_shape, **kwargs)

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return np.less_equal(input, np.random.rand(*input.shape)).astype(np.bool_)
