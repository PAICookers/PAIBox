import inspect
from typing import Callable, Optional, Tuple, TypeVar, Union

import numpy as np
from typing_extensions import ParamSpec

from ._types import Shape
from .base import DynamicSys
from .context import _FRONTEND_CONTEXT
from .exceptions import SimulationError
from .utils import as_shape, shape2num

__all__ = ["InputProj"]

T = TypeVar("T")
P = ParamSpec("P")


class Projection(DynamicSys):
    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)


class InputProj(Projection):
    def __init__(
        self,
        input: Union[T, Callable[P, T]],
        shape_out: Shape,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """The input node of network.

        Arguments:
            - input: the output value(int, np.integer, np.ndarray), or  \
                a callable(function or `Encoder`).
            - shape_out: the shape of the output.
            - keep_shape: wether to keep the shape when retieving the   \
                feature map.
            - name: the name of the node. Optional.
        """
        super().__init__(name)
        self._input = input
        self._shape_out = as_shape(shape_out)
        self.keep_shape = keep_shape

        self.set_memory("spike", np.zeros((self.num_out,), dtype=np.bool_))

    def update(self, *args, **kwargs) -> np.ndarray:
        if self.input is None:
            raise SimulationError("The input is not set.")

        if callable(self.input):
            _spike = _call_with_ctx(self.input, *args, **kwargs)
        else:
            _spike = self.input

        if isinstance(_spike, (int, np.integer)):
            self.spike = np.full((self.num_out,), _spike, dtype=np.int32)
        elif isinstance(_spike, np.ndarray):
            self.spike = _spike.reshape((self.num_out,)).astype(np.int32)
        else:
            raise TypeError(
                f"Excepted type int, np.integer or np.ndarray, "
                f"but got {_spike}, type {type(_spike)}"
            )

        return self.spike

    def reset_state(self) -> None:
        self.reset()  # Call reset of `StatusMemory`.

    @property
    def varshape(self) -> Tuple[int, ...]:
        return self.shape_out if self.keep_shape else (self.num_out,)

    @property
    def num_in(self) -> int:
        return 0

    @property
    def num_out(self) -> int:
        return shape2num(self._shape_out)

    @property
    def shape_in(self) -> Tuple[int, ...]:
        return (0,)

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self._shape_out

    @property
    def input(self):
        return self._input

    @property
    def output(self) -> np.ndarray:
        return self.spike

    @property
    def feature_map(self) -> np.ndarray:
        return self.output.reshape(self.varshape)

    @property
    def state(self) -> np.ndarray:
        return self.output


def _call_with_ctx(f: Callable[P, T], *args, **kwargs) -> T:
    try:
        ctx = _FRONTEND_CONTEXT.get_ctx()
        bound = inspect.signature(f).bind(*args, **ctx, **kwargs)
        # warnings.warn(_input_deprecate_msg, UserWarning)
        return f(*bound.args, **bound.kwargs)

    except TypeError:
        return f(*args, **kwargs)
