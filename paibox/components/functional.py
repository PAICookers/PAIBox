from functools import partial
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from paicorelib import LCM

from paibox.base import NeuDyn
from paibox.exceptions import ShapeError
from paibox.network import DynSysGroup
from paibox.types import SpikeType
from paibox.utils import as_shape, shape2num

from .modules import FunctionalModule, FunctionalModule2to1, TransposeModule
from .neuron import Neuron
from .neuron.neurons import *
from .projection import InputProj
from .synapses import GeneralConnType as GConnType
from .synapses.synapses import FullConn


__all__ = [
    "BitwiseAND",
    "BitwiseNOT",
    "BitwiseOR",
    "BitwiseXOR",
    "Transpose2d",
    "Transpose3d",
]


class BitwiseAND(FunctionalModule2to1):
    inherent_delay = 0

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise AND module. Do a bitwise AND of the output spike of two neurons & output.
        
        Args:
            - neuron_a: the first operand.
            - neuron_b: the second operand.
            - delay: delay between module & another module(or neuron). Default is 1.
            - tick_wait_start: set the moodule to start at timestep `T`. 0 means not working. Default is 1.
            - tick_wait_end: set the module to turn off at time `T`. 0 means always working. Default is 0.
            - unrolling_factor: argument related to the backend. It represents the degree to which modules  \
                are expanded. The larger the value, the more cores required for deployment, but the lower   \
                the latency & the higher the throughput. Default is 1.
            - keep_shape: whether to maintain size information when recording data in the simulation.       \
                Default is `False`.
            - name: name of the module. Optional.
        
        NOTE: the inherent delay of the module is 0. It means that under the default delay(=1) setting, the \
            input data is input at time T, and the result output at time T+1.
        """
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def op_func(self, x1: SpikeType, x2: SpikeType) -> SpikeType:
        return np.bitwise_and(x1, x2)

    def build(self, network: DynSysGroup, **build_options) -> None:
        # 1. Instantiate neurons & synapses & connect the source
        n1 = Neuron(
            self.shape_out,
            leak_comparison=LCM.LEAK_BEFORE_COMP,
            neg_threshold=0,
            leak_v=-1,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConn(
            self.module_intf.operands[0],
            n1,
            conn_type=GConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConn(
            self.module_intf.operands[1],
            n1,
            conn_type=GConnType.One2One,
            name=f"s1_{self.name}",
        )

        # 2. Connect the source of all backward synapses to output neuron.
        for syn in self.module_intf.output:
            syn.source = n1  # `source.setter` will be called

        # 3. Add the components to the network & remove the module itself.
        network.add_components(n1, syn1, syn2)
        network.remove_component(self)


class BitwiseNOT(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise NOT module. Do a bitwise NOT of the output spike of one neuron & output.

        Args:
            - neuron: the operand.

        NOTE: the inherent delay of the module is 0.
        """
        if keep_shape:
            shape_out = neuron.shape_out
        else:
            shape_out = (neuron.num_out,)

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def op_func(self, x1: SpikeType) -> SpikeType:
        return np.bitwise_not(x1)

    def build(self, network: DynSysGroup, **build_options) -> None:
        n1 = Neuron(
            self.shape_out,
            leak_comparison=LCM.LEAK_BEFORE_COMP,
            neg_threshold=0,
            leak_v=-1,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConn(
            self.module_intf.operands[0],
            n1,
            weights=-1,
            conn_type=GConnType.One2One,
            name=f"s0_{self.name}",
        )

        for syns in self.module_intf.output:
            syns.source = n1

        network.add_components(n1, syn1)
        network.remove_component(self)


class BitwiseOR(FunctionalModule2to1):
    inherent_delay = 0

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise OR module. Do a bitwise OR of the output spike of two neurons & output.

        Args:
            - neuron_a: the first operand.
            - neuron_b: the second operand.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def op_func(self, x1: SpikeType, x2: SpikeType) -> SpikeType:
        return np.bitwise_or(x1, x2)

    def build(self, network: DynSysGroup, **build_options) -> None:
        n1 = SpikingRelu(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConn(
            self.module_intf.operands[0],
            n1,
            conn_type=GConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConn(
            self.module_intf.operands[1],
            n1,
            conn_type=GConnType.One2One,
            name=f"s1_{self.name}",
        )

        for syns in self.module_intf.output:
            syns.source = n1

        network.add_components(n1, syn1, syn2)
        network.remove_component(self)


class BitwiseXOR(FunctionalModule2to1):
    inherent_delay = 1

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise XOR module. Do a bitwise XOR of the output spike of two neurons & output.

        Args:
            - neuron_a: the first operand.
            - neuron_b: the second operand.

        NOTE: the inherent delay of the module is 1. It means that under the default delay(=1) setting, the \
            input data is input at time T, and the result output at time T+2.
        """
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def op_func(self, x1: SpikeType, x2: SpikeType) -> SpikeType:
        return np.bitwise_xor(x1, x2)

    def build(self, network: DynSysGroup, **build_options) -> None:
        # If neuron_a is of shape (h1, w1) = N, and neuron_b is of shape (h2, w2) = N.
        # The output shape of the module is (N,) or (h1, w1)(if h1 == h2).
        # The shape of n1 is (2N,) or (2, h1, w1).
        n1 = SpikingRelu(
            (2,) + self.shape_out,
            delay=1,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=False,
            name=f"n0_{self.name}",
        )
        # The shape of n2 is (N,) or (h1, w1).
        n2 = SpikingRelu(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start + 1,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n1_{self.name}",
        )

        identity = np.identity(self.num_out, dtype=np.int8)
        # weight of syn1, (-1*(N,), 1*(N,))
        syn1 = FullConn(
            self.module_intf.operands[0],
            n1,
            weights=np.hstack([-1 * identity, identity], casting="safe", dtype=np.int8),
            conn_type=GConnType.MatConn,
            name=f"s0_{self.name}",
        )
        # weight of syn2, (1*(N,), -1*(N,))
        syn2 = FullConn(
            self.module_intf.operands[1],
            n1,
            weights=np.hstack([identity, -1 * identity], casting="safe", dtype=np.int8),
            conn_type=GConnType.MatConn,
            name=f"s1_{self.name}",
        )
        # weight of syn3, identity matrix with shape (2N, N)
        syn3 = FullConn(
            n1,
            n2,
            weights=np.vstack([identity, -1 * identity], casting="safe", dtype=np.int8),
            conn_type=GConnType.MatConn,
            name=f"s2_{self.name}",
        )

        for syns in self.module_intf.output:
            syns.source = n2

        network.add_components(n1, n2, syn1, syn2, syn3)
        network.remove_component(self)


class Transpose2d(TransposeModule):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d transpose module.

        Args:
            - neuron: the neuron of which output spike will be transposed.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron,
            _shape_ndim2_check(neuron.shape_out),
            (1, 0),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def op_func(self, x1: SpikeType) -> SpikeType:
        _x1 = x1.reshape(self.shape_in)

        return _x1.T.flatten()

    def build(self, network: DynSysGroup, **build_options) -> None:
        n1 = Neuron(
            self.shape_out,
            neg_threshold=0,
            leak_v=0,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConn(
            self.module_intf.operands[0],
            n1,
            weights=_transpose2d_mapping(self.shape_in),
            conn_type=GConnType.MatConn,
            name=f"s0_{self.name}",
        )

        for syns in self.module_intf.output:
            syns.source = n1

        network.add_components(n1, syn1)
        network.remove_component(self)


class Transpose3d(TransposeModule):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        axes: Optional[Sequence[int]] = None,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """3d transpose module.

        Args:
            - neuron: the neuron of which output spike will be transposed.
            - axes: If specified, it must be a tuple or list which contains a permutation of [0, 1, …, N-1] \
                where N is the number of axes of output shape of neuron. If not specified, defaults to      \
                `range(ndim)[::-1]`, where `ndim` is the dimension of the output shape, which reverses the  \
                order of the axes.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron,
            _shape_ndim3_check(neuron.shape_out),
            axes,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def op_func(self, x1: SpikeType) -> SpikeType:
        _x1 = x1.reshape(self.shape_in)

        return _x1.transpose(self.axes).flatten()

    def build(self, network: DynSysGroup, **build_options) -> None:
        n1 = Neuron(
            self.shape_out,
            leak_comparison=LCM.LEAK_BEFORE_COMP,
            neg_threshold=0,
            leak_v=-1,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConn(
            self.module_intf.operands[0],
            n1,
            weights=_transpose3d_mapping(self.shape_in, self.axes),
            conn_type=GConnType.MatConn,
            name=f"s0_{self.name}",
        )

        for syns in self.module_intf.output:
            syns.source = n1

        network.add_components(n1, syn1)
        network.remove_component(self)


def _shape_check(shape: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
    if len(shape) > ndim:
        raise ShapeError(
            f"expected shape to have dimensions <= {ndim}, but got {len(shape)}."
        )

    return as_shape(shape, min_dim=ndim)


_shape_ndim2_check = partial(_shape_check, ndim=2)
_shape_ndim3_check = partial(_shape_check, ndim=3)


def _transpose2d_mapping(op_shape: Tuple[int, ...]) -> NDArray[np.bool_]:
    """Get the mapping matrix for transpose of 2d array.

    Argument:
        - op_shape: the shape of matrix to be transposed, flattened in (X,Y) order.

    Return: transposed index matrix with shape (X*Y, Y*X).
    """
    size = shape2num(op_shape)
    mt = np.zeros((size, size), dtype=np.bool_)

    for idx in np.ndindex(op_shape):
        mt[idx[0] * op_shape[1] + idx[1], idx[1] * op_shape[0] + idx[0]] = 1

    return mt


def _transpose3d_mapping(
    op_shape: Tuple[int, ...], axes: Tuple[int, ...]
) -> NDArray[np.bool_]:
    """Get the mapping matrix for transpose of 3d array.

    Argument:
        - op_shape: the shape of matrix to be transposed, flattened in (X,Y,Z) order.
        - axes: If specified, it must be a tuple or list which contains a permutation of [0, 1, …, N-1]     \
            where N is the number of axes of a.

    Return: transposed index matrix with shape (N, N) where N=X*Y*Z.
    """
    size = shape2num(op_shape)
    mt = np.zeros((size, size), dtype=np.bool_)

    shape_t = tuple(op_shape[i] for i in axes)

    size12 = op_shape[1] * op_shape[2]
    size12_t = shape_t[1] * shape_t[2]

    for idx in np.ndindex(op_shape):
        mt[
            idx[0] * size12 + idx[1] * op_shape[2] + idx[2],
            idx[axes[0]] * size12_t + idx[axes[1]] * shape_t[2] + idx[axes[2]],
        ] = 1

    return mt
