import inspect
import sys
from typing import Callable, Optional, Tuple, Union

import numpy as np

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

from paibox.base import DynamicSys
from paibox.context import _FRONTEND_CONTEXT
from paibox.exceptions import ShapeError, SimulationError
from paibox.mixin import TimeRelatedNode
from paibox.types import DataType, Shape, SpikeType
from paibox.utils import as_shape, shape2num

__all__ = ["InputProj"]

P = ParamSpec("P")


def _func_bypass(x: DataType) -> DataType:
    return x


class Projection(DynamicSys):
    def __call__(self, *args, **kwargs) -> SpikeType:
        return self.update(*args, **kwargs)


class InputProj(Projection, TimeRelatedNode):
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
            - input: the input value of the projection node. It can be numeric value or callable\
                function(function or `Encoder`).
            - shape_out: the shape of the output.
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
            self._func_input = _func_bypass

        self._shape = as_shape(shape_out)
        self.keep_shape = keep_shape

        self.set_memory("_inner_spike", np.zeros((self.num_out,), dtype=np.bool_))

    def update(self, **kwargs) -> SpikeType:
        _spike = self._get_neumeric_input(**kwargs)

        if isinstance(_spike, (int, np.bool_, np.integer)):
            self._inner_spike = np.full((self.num_out,), _spike, dtype=np.bool_)
        elif isinstance(_spike, np.ndarray):
            if shape2num(_spike.shape) != self.num_out:
                raise ShapeError(
                    f"cannot reshape output value from {_spike.shape} to ({self.num_out},)."
                )
            self._inner_spike = _spike.ravel().astype(np.bool_)
        else:
            # should never be reached
            raise TypeError(
                f"expected type int, np.bool_, np.integer or np.ndarray, "
                f"but got {_spike}, type {type(_spike)}."
            )

        return self._inner_spike

    def reset_state(self) -> None:
        self.reset_memory()  # Call reset of `StatusMemory`.

    def _get_neumeric_input(self, **kwargs):
        # If `_func_input` is `None` while `input` is numeric, use `input` as input to the projection.
        # Otherwise, use the output of `_func_input`.
        if self._num_input is None:
            if self._func_input is None:
                raise SimulationError(f"both numeric & functional input are not set.")
            else:
                return _call_with_ctx(self._func_input, **kwargs)

        elif self._func_input is None:
            return self._num_input
        else:
            return _call_with_ctx(self._func_input, self._num_input, **kwargs)

    @property
    def varshape(self) -> Tuple[int, ...]:
        return self.shape_out if self.keep_shape else (self.num_out,)

    @property
    def num_in(self) -> int:
        return 0

    @property
    def num_out(self) -> int:
        return shape2num(self._shape)

    @property
    def shape_in(self) -> Tuple[int, ...]:
        return (0,)

    @property
    def shape_out(self) -> Tuple[int, ...]:
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
    def output(self) -> SpikeType:
        return self._inner_spike

    @property
    def spike(self) -> SpikeType:
        return self._inner_spike

    @property
    def feature_map(self) -> SpikeType:
        return self.output.reshape(self.varshape)

    @property
    def delay_relative(self) -> int:
        return 1  # Fixed

    @property
    def tick_wait_start(self) -> int:
        return 1  # Fixed

    @property
    def tick_wait_end(self) -> int:
        return 0  # Fixed


def _call_with_ctx(f: Callable[..., DataType], *args, **kwargs) -> DataType:
    try:
        ctx = _FRONTEND_CONTEXT.get_ctx()
        bound = inspect.signature(f).bind(*args, **ctx, **kwargs)
        return f(*bound.args, **bound.kwargs)
    except TypeError:
        return f(*args, **kwargs)
