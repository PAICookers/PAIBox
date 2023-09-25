from typing import Callable, Optional, Tuple, Union

import numpy as np

from ._types import Shape
from .base import Projection
from .utils import as_shape, shape2num

__all__ = ["InputProj", "OutputProj"]


class InputProj(Projection):
    def __init__(
        self,
        val_or_func: Union[int, np.integer, np.ndarray, Callable],
        *,
        shape: Optional[Shape] = None,
        keep_size: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Input projection to define an output or a generation function.

        Arguments:
            - val_or_func: the output value(integer, np.ndarray) or a process.
            - shape: the output shape. If not provided, try to use the shape_in of `target`. Otherwise raise error.
            - name: the name of the node. Optional.
        """
        super().__init__(name)
        self.keep_size = keep_size

        if isinstance(val_or_func, (int, np.integer)):
            self.val = int(val_or_func)
            self._shape = (0,)
        elif isinstance(val_or_func, np.ndarray):
            if shape:
                if as_shape(shape) != val_or_func.shape:
                    # TODO
                    raise ValueError

                self._shape = as_shape(shape)
            else:
                self._shape = (0,)
        else:
            self.val = val_or_func
            self._shape = self.val.varshape

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        if isinstance(self.val, Callable):
            self._state = self.val(*args, **kwargs)
        else:
            self._state = np.full(self.shape_out, self.val)

        return self._state

    def reset_state(self) -> None:
        self._state = np.zeros(self._shape, np.int32)

    @property
    def output(self) -> np.ndarray:
        return self._state

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def shape_in(self) -> Tuple[int, ...]:
        return (0,)

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def num_in(self) -> int:
        return 0

    @property
    def num_out(self) -> int:
        return shape2num(self.shape_out)

    @property
    def method(self) -> str:
        return "function" if isinstance(self.val, Callable) else "value"


class OutputProj(Projection):
    pass
