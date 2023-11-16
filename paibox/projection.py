from typing import Callable, Optional, Tuple, Union

import numpy as np

from ._types import Shape
from .base import DynamicSys
from .utils import as_shape, shape2num


class Projection(DynamicSys):
    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)


class InputProj(Projection):
    def __init__(
        self,
        input: Optional[
            Union[int, np.integer, np.ndarray, Callable[..., np.ndarray]]
        ] = None,
        *,
        shape_out: Optional[Shape] = None,
        keep_shape: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """The input node of the network.

        Arguments:
            - input: the output value(integer, np.ndarray), or a callable.
            - shape_out: the shape of the output.
            - keep_shape: wether to keep the shape when retieving the \
                feature map.
            - name: the name of the node. Optional.
        """
        super().__init__(name)

        if isinstance(input, (int, np.integer)):
            # A scalar
            self._input = int(input)
            self._shape_out = as_shape(shape_out)

        elif isinstance(input, np.ndarray):
            self._input = input
            self._shape_out = input.shape

        elif callable(input):
            self._input = input
            if shape_out is None:
                if hasattr(input, "shape_out"):
                    self._shape_out = input.shape_out
                else:
                    raise ValueError(
                        "Shape of output is required when input is callable."
                    )
            else:
                self._shape_out = as_shape(shape_out)
        else:
            # when `input` is None, `shape_out` is required
            if shape_out is None:
                raise ValueError("Shape of output is required when input is None.")

            self._input = None
            self._shape_out = as_shape(shape_out)

        self.keep_shape = keep_shape
        self._output = np.zeros((self.num_out,), dtype=np.bool_)

    def update(self, *args, **kwargs) -> np.ndarray:
        if self.input is None:
            raise RuntimeError("The input is not set.")

        if isinstance(self.input, np.ndarray):
            self._output = self.input.reshape((self.num_out,))
        elif isinstance(self.input, int):
            self._output = np.full((self.num_out,), self.input, dtype=np.int32)
        elif callable(self.input):
            self._output = (
                self.input(*args, **kwargs).astype(np.int32).reshape((self.num_out,))
            )
        else:
            # TODO
            raise TypeError(f"Excepted input type is int, np.integer, np.ndarray or Callable[..., np.ndarray], "
                            f"but we got {input}, type {type(input)}")

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

    @input.setter
    def input(self, new_input) -> None:
        if isinstance(new_input, np.ndarray):
            if new_input.shape != self.shape_out:
                raise ValueError("The shape of input is not match.")

        self._input = new_input

    @property
    def output(self) -> np.ndarray:
        return self._output

    @property
    def feature_map(self) -> np.ndarray:
        return self.output.reshape(self.varshape)

    @property
    def state(self) -> np.ndarray:
        return self.output


class OutputProj(Projection):
    pass
