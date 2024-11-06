import inspect
import sys
from collections.abc import Callable
from typing import Literal, Optional, Union

import numpy as np

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

from paibox.base import DynamicSys
from paibox.context import _FRONTEND_CONTEXT
from paibox.exceptions import ShapeError, SimulationError
from paibox.mixin import TimeRelatedNode
from paibox.types import NEUOUT_U8_DTYPE, DataType, NeuOutType, Shape
from paibox.utils import as_shape, shape2num

__all__ = ["InputProj"]

L = Literal
P = ParamSpec("P")


def _func_bypass(x: DataType) -> DataType:
    return x


class Projection(DynamicSys, TimeRelatedNode):
    def __call__(self, *args, **kwargs) -> NeuOutType:
        return self.update(*args, **kwargs)

    @property
    def delay_relative(self) -> int:
        return 1  # Fixed

    @property
    def tick_wait_start(self) -> int:
        return 1  # Fixed

    @property
    def tick_wait_end(self) -> int:
        return 0  # Fixed


class InputProj(Projection):
    # TODO Since the input port can be equivalent to the output of a neuron, is it more appropriate
    # to use a neuron as an input port?

    def __init__(
        self,
        input: Optional[Union[DataType, Callable[P, DataType]]],
        shape_out: Shape,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """The input node of network.

        Arguments:
            - input: the input value of the projection node. It can be a numeric value or a callable function.
            - shape_out: the shape of the output..
            - keep_shape: wether to keep the shape when retieving the feature map.
            - name: the name of the node. Optional.
        """
        super().__init__(name)
        # Compatible with previous version. Will be deprecated in the future.
        if input is None:
            self._num_input = None
            self._func_input = None
        elif callable(input):
            self._num_input = None
            self._func_input = input
        else:  # Numeric input
            self._num_input = input
            self._func_input = None

        self._shape = as_shape(shape_out)
        self.keep_shape = keep_shape
        self.set_memory("_neu_out", np.zeros((self.num_out,), dtype=NEUOUT_U8_DTYPE))

    def update(self, *args, **kwargs) -> NeuOutType:
        _input = self._get_neumeric_input(**kwargs)

        if isinstance(_input, (int, np.bool_, np.integer)):
            self._neu_out = np.full_like(self._neu_out, _input, dtype=NEUOUT_U8_DTYPE)
        elif isinstance(_input, np.ndarray):
            if _input.size != self._neu_out.size:
                raise ShapeError(
                    f"cannot reshape output value from {_input.shape} to {self._neu_out.shape}."
                )
            self._neu_out = _input.ravel().astype(NEUOUT_U8_DTYPE)
        else:
            # should never be reached
            raise TypeError(
                f"expected type int, np.bool_, np.integer or np.ndarray, "
                f"but got {_input}, type {type(_input)}."
            )

        return self._neu_out

    def reset_state(self) -> None:
        self.reset_memory()  # Call reset of `StatusMemory`.

    def _get_neumeric_input(self, **kwargs):
        # If `_func_input` is `None` while `input` is numeric, use `input` as input to the projection.
        # Otherwise, use the output of `_func_input`.
        if self._num_input is None:
            if self._func_input is None:
                raise SimulationError("both numeric & functional input are not set.")
            else:
                return _call_with_ctx(self._func_input, **kwargs)

        elif self._func_input is None:
            return self._num_input
        else:
            return _call_with_ctx(self._func_input, self._num_input, **kwargs)

    @property
    def varshape(self) -> tuple[int, ...]:
        return self.shape_out if self.keep_shape else (self.num_out,)

    @property
    def num_in(self) -> int:
        return 0

    @property
    def num_out(self) -> int:
        return shape2num(self._shape)

    @property
    def shape_in(self) -> tuple[int, ...]:
        return (0,)

    @property
    def shape_out(self) -> tuple[int, ...]:
        return self._shape

    @property
    def input(self):
        return self._get_neumeric_input()

    @input.setter
    def input(self, value: DataType) -> None:
        """Set the input at the beginning of running the simulation."""
        if not isinstance(value, (int, np.bool_, np.integer, np.ndarray)):
            raise TypeError(
                f"expected type int, np.bool_, np.integer or np.ndarray, "
                f"but got {value}, type {type(value)}."
            )

        self._num_input = value

    @property
    def output(self) -> NeuOutType:
        return self._neu_out

    @property
    def spike(self) -> NeuOutType:
        return self._neu_out

    @property
    def feature_map(self) -> NeuOutType:
        return self._neu_out.reshape(self.varshape)

class InputSlice:
    def __init__(self, input:InputProj, index: slice = None):
        self.target = input
        self.index = index
        if index is None:
            self.index = slice(0, input.num_out)

    @property
    def num_out(self) -> int:
        return self.index.stop - self.index.start
    
    @property
    def info(self) -> str:
        return f"InputSlice {self.target.name}[{self.index.start}:{self.index.stop}]"
    
    def __eq__(self, other: "InputSlice") -> bool:
        return self.target == other.target and self.index == other.index
    
    def __hash__(self) -> int:
        return hash((self.target, self.index.start, self.index.stop))


def _call_with_ctx(f: Callable[..., DataType], *args, **kwargs) -> DataType:
    try:
        ctx = _FRONTEND_CONTEXT.get_ctx()
        bound = inspect.signature(f).bind(*args, **ctx, **kwargs)
        return f(*bound.args, **bound.kwargs)
    except TypeError:
        return f(*args, **kwargs)
