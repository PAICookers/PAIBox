import inspect
from typing import Callable, Optional, Tuple, TypeVar, Union

import numpy as np

from ._types import DataType, Shape
from .base import DynamicSys
from .context import _FRONTEND_CONTEXT
from .exceptions import SimulationError
from .utils import as_shape, shape2num

__all__ = ["InputProj"]

T = TypeVar("T")


class Projection(DynamicSys):
    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)


class InputProj(Projection):
    def __init__(
        self,
        input: Union[DataType, Callable[..., DataType]],
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
        self._output = np.zeros((self.num_out,), dtype=np.bool_)

    def update(self, *args, **kwargs) -> np.ndarray:
        if self.input is None:
            raise SimulationError("The input is not set.")

        if callable(self.input):
            output = _call_with_ctx(self.input, *args, **kwargs)
        else:
            output = self.input

        if isinstance(output, (int, np.integer)):
            self._output = np.full((self.num_out,), output, dtype=np.int32)
        elif isinstance(output, np.ndarray):
            self._output = output.reshape((self.num_out,)).astype(np.int32)
        else:
            raise TypeError(
                f"Excepted type int, np.integer, np.ndarray, "
                f"but got {output}, type {type(output)}"
            )

        return self.output

    def reset_state(self) -> None:
        self._output = np.zeros((self.num_out,), dtype=np.bool_)

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
        return self._output

    @property
    def feature_map(self) -> np.ndarray:
        return self.output.reshape(self.varshape)

    @property
    def state(self) -> np.ndarray:
        return self.output


def _call_with_ctx(f: Callable[..., T], *args, **kwargs) -> T:
    try:
        ctx = _FRONTEND_CONTEXT.get_ctx()
        bound = inspect.signature(f).bind(*args, **ctx, **kwargs)
        # warnings.warn(_input_deprecate_msg, UserWarning)
        return f(*bound.args, **bound.kwargs)
    except TypeError:
        return f(*args, **kwargs)
