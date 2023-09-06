import numpy as np
from typing import Optional, Tuple

from ._types import Shape
from .base import PAIBoxObject, Process
from .utils import as_shape


class Distribution(PAIBoxObject):
    def _sample_shape(
        self, n, shape: Optional[Tuple[int, ...]] = None
    ) -> Tuple[int, ...]:
        return (n,) if shape is None else (n,) + shape

    def sample(self, n, shape: Shape):
        raise NotImplementedError


class Uniform(Distribution):
    def __init__(self) -> None:
        super().__init__()

    def sample(self, n: int, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        _shape = self._sample_shape(n, shape)
        return np.random.default_rng().integers(0, 2, _shape)


class UniformGen(Process):
    def __init__(self, shape_out: Shape = 1, **kwargs) -> None:
        """
        Discrete uniform.
        """
        super().__init__(shape_out, **kwargs)
        self.dist = Uniform()
        self.output = np.zeros(self.shape_out, dtype=np.bool_)

    def update(self, **kwargs) -> None:
        self.output = self.dist.sample(1, as_shape(self.shape_out))[0]


class Constant(Process):
    def __init__(self, shape_out: Shape = 1, constant: int = 0, **kwargs) -> None:
        super().__init__(shape_out=shape_out, **kwargs)
        self.output = np.full(self.shape_out, constant)

    def update(self, **kwargs) -> None:
        pass
