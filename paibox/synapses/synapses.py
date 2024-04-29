from typing import ClassVar, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from paicorelib import HwConfig
from paicorelib import WeightPrecision as WP

from paibox.base import DynamicSys, NeuDyn
from paibox.exceptions import ShapeError
from paibox.projection import InputProj
from paibox.types import DataArrayType, WeightType

from .transforms import *

__all__ = ["NoDecay"]

RIGISTER_MASTER_KEY_FORMAT = "{0}.output"


class Synapses:
    """A map connected between neurons of the previous `Node`, \
        and axons of the following `Node`.

    User can use connectivity matrix or COO to represent the \
        connectivity of synapses.
    """

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: NeuDyn,
        /,
        conn_type: ConnType,
    ) -> None:
        """
        Args:
            - source: the source group of neurons.
            - dest: the destination group of neurons.
            - conn_type: the type of connection.
        """
        self.source = source
        self.dest = dest
        self._check(conn_type)

    def _check(self, conn_type: ConnType) -> None:
        if conn_type is ConnType.One2One or conn_type is ConnType.BYPASS:
            if self.num_in != self.num_out:
                raise ShapeError(
                    f"The number of source & destination neurons must "
                    f"be equal, but {self.num_in} != {self.num_out}."
                )

    @property
    def shape_in(self) -> Tuple[int, ...]:
        return self.source.shape_out

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self.dest.shape_in

    @property
    def num_in(self) -> int:
        return self.source.num_out

    @property
    def num_out(self) -> int:
        return self.dest.num_in


class SynSys(Synapses, DynamicSys):
    CFLAG_ENABLE_WP_OPTIMIZATION: ClassVar[bool] = True
    """Compilation flag for weight precision optimization."""

    def __call__(self, *args, **kwargs) -> NDArray[np.int32]:
        return self.update(*args, **kwargs)

    @property
    def weights(self) -> WeightType:
        raise NotImplementedError

    @property
    def weight_precision(self) -> WP:
        raise NotImplementedError

    @property
    def connectivity(self) -> NDArray[Union[np.bool_, np.int8]]:
        raise NotImplementedError

    @property
    def n_axon_each(self) -> np.ndarray:
        return np.sum(self.connectivity, axis=0)

    @property
    def num_axon(self) -> int:
        return np.count_nonzero(np.any(self.connectivity, axis=1))

    @property
    def num_dendrite(self) -> int:
        return np.count_nonzero(np.any(self.connectivity, axis=0))


class NoDecay(SynSys):
    """Synapses model with no decay."""

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: NeuDyn,
        weights: DataArrayType = 1,
        *,
        conn_type: ConnType = ConnType.MatConn,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - source: source neuron(s).
            - dest: destination neuron(s).
            - weights: weights of the synapses. It can be a scalar or `np.ndarray`.
            - conn_type: the type of connection.
            - name: name of this synapses. Optional.
        """
        super().__init__(source, dest, conn_type)
        super(Synapses, self).__init__(name)

        if conn_type is ConnType.One2One:
            self.comm = OneToOne(self.num_in, weights)
        elif conn_type is ConnType.BYPASS:
            self.comm = ByPass(self.num_in)
        elif conn_type is ConnType.All2All:
            self.comm = AllToAll((self.num_in, self.num_out), weights)
        else:  # MatConn
            if not isinstance(weights, np.ndarray):
                raise TypeError(
                    f"Expected type int, np.integer or np.ndarray, but got type {type(weights)}"
                )

            self.comm = MaskedLinear((self.num_in, self.num_out), weights)

        self.weights.setflags(write=False)
        self.set_memory("_synout", np.zeros((self.num_out,), dtype=np.int32))

        # Register `self` for the destination `NeuDyn`.
        dest.register_master(RIGISTER_MASTER_KEY_FORMAT.format(self.name), self)

    def update(
        self, spike: Optional[np.ndarray] = None, *args, **kwargs
    ) -> NDArray[np.int32]:
        # Retrieve the spike at index `timestamp` of the dest neurons
        if self.dest.is_working:
            if isinstance(self.source, InputProj):
                synin = self.source.output.copy() if spike is None else spike
            else:
                idx = self.dest.timestamp % HwConfig.N_TIMESLOT_MAX
                synin = self.source.output[idx].copy() if spike is None else spike
        else:
            # Retrieve 0 to the dest neurons if it is not working
            synin = np.zeros_like(self.source.spike, dtype=np.bool_)

        self._synout = self.comm(synin).astype(np.int32)
        return self._synout

    def reset_state(self, *args, **kwargs) -> None:
        # TODO Add other initialization methods in the future.
        self.reset()  # Call reset of `StatusMemory`.

    @property
    def output(self) -> NDArray[np.int32]:
        return self._synout

    @property
    def weights(self):
        return self.comm.weights

    @property
    def weight_precision(self) -> WP:
        return self.comm._get_wp(self.CFLAG_ENABLE_WP_OPTIMIZATION)

    @property
    def connectivity(self):
        """The connectivity matrix in `np.ndarray` format."""
        return self.comm.connectivity

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return (
            f"{name}(name={self.name}, \n"
            f'{" " * len(name)} source={self.source}, \n'
            f'{" " * len(name)} dest={self.dest})'
        )
