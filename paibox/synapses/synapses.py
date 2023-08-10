from typing import Dict, List, Optional, Union

import numpy as np

from paibox.base import PAIBoxObject
from paibox.neuron.group import Group
from paibox.synapses.connector import (
    All2All,
    IndexConn,
    MatConn,
    One2One,
    TwoEndConnector,
)
from paibox.synapses.transforms import AllToAll, MaskedLinear, OneToOne


class Synapses(PAIBoxObject):
    """A map connected between neurons of the previous `Node`, and axons of the following `Node`.

    User can use connectivity matrix or COO to represent the connectivity of synapses.

    NOTE: Be aware that every axon can only be connected once with a neuron,
        while a neuron is able to connect with several axons.
    """

    def __init__(
        self,
        source: Group,
        dest: Group,
        conn: Union[
            TwoEndConnector, np.ndarray, Dict[str, Union[List[int], np.ndarray]]
        ],
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - source: the source group of neurons
            - dest: the destination group of neurons
            - conn: the connectivity representation
            - name: the name of the synapses.
        """
        super(Synapses, self).__init__(name=name)

        self.source = source
        self.dest = dest
        self.conn = self._init_conn(conn)

    def _init_conn(
        self,
        conn: Union[
            TwoEndConnector, np.ndarray, Dict[str, Union[List[int], np.ndarray]]
        ],
    ) -> Union[TwoEndConnector, MatConn, IndexConn]:
        """Build a connector given the arrays or dictionary."""
        if isinstance(conn, TwoEndConnector):
            return conn(self.source.shape_out, self.dest.shape_in)

        if isinstance(conn, np.ndarray):
            conn = MatConn(conn_mat=conn)
        elif isinstance(conn, Dict):
            if not ("i" in conn and "j" in conn):
                raise ValueError("The keys of the dictionary must include 'i' and 'j'.")
            conn = IndexConn(source_ids=conn["i"], dest_ids=conn["j"])
        else:
            raise TypeError(f"Unsupported type: {type(self.conn)}.")

        return conn

    @property
    def shape_in(self):
        return self.source.shape_out

    @property
    def shape_out(self):
        return self.dest.shape_in


class NoDecay(Synapses):
    def __init__(
        self,
        source: Group,
        dest: Group,
        conn: Union[
            TwoEndConnector, np.ndarray, Dict[str, Union[List[int], np.ndarray]]
        ],
        weights: Union[int, np.integer, np.ndarray] = 1,
        name: Optional[str] = None,
    ):
        super().__init__(source, dest, conn, name)

        if isinstance(conn, All2All):
            self.comm = AllToAll(source.num, dest.num, weights)
        elif isinstance(conn, One2One):
            self.comm = OneToOne(source.num, weights)
        elif isinstance(conn, IndexConn):
            self.comm = MaskedLinear(conn, weights)
        else:
            raise ValueError

    def update(self, spike, output) -> None:
        output[...] = self.comm(spike)
