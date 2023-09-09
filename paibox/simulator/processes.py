from typing import Optional, Tuple, Union

import numpy as np

from paibox._types import Shape
from paibox.base import PAIBoxObject, Process
from paibox.utils import as_shape


__all__ = ["UniformGen", "Constant"]


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
    def __init__(
        self, shape_out: Shape = 1, *, keep_size: bool = False, **kwargs
    ) -> None:
        """
        Discrete uniform.
        """
        super().__init__(shape_out, keep_size=keep_size, **kwargs)
        self.dist = Uniform()

    def update(self, *args, **kwargs) -> np.ndarray:
        self._output = self.dist.sample(1, as_shape(self.varshape))[0]
        
        return self.state


class Constant(Process):
    def __init__(
        self,
        shape_out: Shape = 1,
        constant: Union[bool, int] = 0,
        *,
        keep_size: bool = False,
        **kwargs
    ) -> None:
        """
        Arguments:
            - shape_out: the shape of the output.
            - constant: the output value. It's always a constant.

        TODO Only support bool constant now.
        """
        super().__init__(shape_out, keep_size=keep_size, **kwargs)
        self._output = np.full(self.varshape, constant)

    def update(self, *args, **kwargs) -> np.ndarray:
        # Do nothing.
        return self.state
