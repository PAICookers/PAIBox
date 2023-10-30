from typing import Optional, Tuple, Union

import numpy as np

from paibox.base import DynamicSys, NeuDyn
from paibox.projection import InputProj

from .transforms import ConnType, OneToOne, AllToAll, MaskedLinear

__all__ = ["Synapses", "NoDecay"]


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
        conn_type: ConnType,
    ) -> None:
        """
        Arguments:
            - source: the source group of neurons.
            - dest: the destination group of neurons.
            - conn: the connectivity representation.
            - name: the name of the synapses. Optional.
        """
        self.source = source
        self.dest = dest

        self._check(conn_type)

    def _check(self, conn_type: ConnType) -> None:
        if conn_type is ConnType.One2One:
            if self.num_in != self.num_out:
                raise ValueError(
                    "The number of source and destination neurons must be equal."
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
    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    @property
    def connectivity(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_axon_each(self) -> np.ndarray:
        return np.count_nonzero(self.connectivity, axis=0, keepdims=True)

    @property
    def num_axon(self) -> int:
        return np.count_nonzero(np.any(self.connectivity, axis=1))

    @property
    def num_dentrite(self) -> int:
        return np.count_nonzero(np.any(self.connectivity, axis=0))


class NoDecay(SynSys):
    """Synapses model with no decay."""

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: NeuDyn,
        weights: Union[int, np.integer, np.ndarray] = 1,
        *,
        conn_type: ConnType = ConnType.MatConn,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - source: source neuron(s).
            - dest: destination neuron(s).
            - weights: weights of the synapses. It can be a scalar, \
                `np.ndarray`, or a distribution.
            - conn_type: the type of connection.
            - name: name of this synapses. Optional.
        """
        super().__init__(source, dest, conn_type)
        super(Synapses, self).__init__(name)

        if conn_type is ConnType.One2One:
            self.comm = OneToOne(self.num_in, weights)
        elif conn_type is ConnType.All2All:
            self.comm = AllToAll((self.num_in, self.num_out), weights)
        elif conn_type is ConnType.MatConn:
            if not isinstance(weights, np.ndarray):
                raise TypeError

            self.comm = MaskedLinear((self.num_in, self.num_out), weights)

        self.weights.setflags(write=False)
        self._synout = np.zeros((self.num_in, self.num_out), dtype=np.int32)

        # Register `self` for the destination NeuDyn.
        dest.register_master(f"{self.name}.output", self)

    def update(self, spike: Optional[np.ndarray] = None, **kwargs):
        if spike is None:
            synin = self.source.output
        else:
            synin = spike

        self._synout = self.comm(synin).astype(np.int32)

        # Keep the return for `update` in `Sequential`.
        return self._synout

    def reset_state(self) -> None:
        # TODO Add other initialization methods in the future.
        self._synout = np.zeros((self.num_in, self.num_out), dtype=np.int32)

    @property
    def output(self) -> np.ndarray:
        return self._synout

    @property
    def state(self) -> np.ndarray:
        return self._synout

    @property
    def weights(self) -> np.ndarray:
        return self.comm.weights

    @property
    def connectivity(self) -> np.ndarray:
        """The connectivity matrix in `np.ndarray` format."""
        return self.comm.connectivity

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return (
            f"{name}(name={self.name}, \n"
            f'{" " * len(name)} source={self.source}, \n'
            f'{" " * len(name)} dest={self.dest})'
        )
