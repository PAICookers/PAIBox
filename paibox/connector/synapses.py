from .identifier import AxonId, NeuronId
from .connector import IndexConn, TwoEndConnector, MatConn
from typing import Dict, List, Union, Type
from paibox.utils import check_elem_unique, count_unique_elem, singleton
from enum import Enum, unique
from dataclasses import dataclass, field


@unique
class PinStatus(Enum):
    UNCONNECTED = False
    CONNECTED = True


@dataclass
class _PinPair:
    pin: Union[NeuronId, AxonId]
    status: PinStatus = PinStatus.UNCONNECTED

    def __repr__(self) -> str:
        return f"[{self.pin.__repr__()}, {self.status}]"


@dataclass
class SourcePinPair(_PinPair):
    pin: NeuronId = field(default_factory=NeuronId.default)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class DestPinPair(_PinPair):
    pin: AxonId = field(default_factory=AxonId.default)

    def __repr__(self) -> str:
        return super().__repr__()



class SynapsesMap:
    def __init__(
        self,
        source_pins: List[NeuronId],
        dest_pins: List[AxonId],
        conn: TwoEndConnector,
        **kwargs,
    ) -> None:
        try:
            assert check_elem_unique(source_pins)
            assert check_elem_unique(dest_pins)
        except AssertionError:
            raise ValueError("The source pins and destination pins must be unique.")

        self._source_pins = source_pins
        self._dest_pins = dest_pins
        self.conn = conn

        if isinstance(self.conn, MatConn):
            self._init_mat()
        elif isinstance(self.conn, IndexConn):
            self._init_coo()

        self.pin_map = self._build_pin_map(type(self.conn))

    def _init_mat(self) -> None:
        self.conn_mat = self.conn.build()
        try:
            assert (
                len(self._source_pins) == self.conn_mat.shape[0]
                and len(self._dest_pins) == self.conn_mat.shape[1]
            )
        except AssertionError:
            raise ValueError(
                "The number of source and destination pins are not equal to the dimension of connection matrix."
            )

    def _init_coo(self) -> None:
        self.source_ids, self.dest_ids = self.conn.build()

        try:
            assert (
                len(self._source_pins) > self.source_ids.max()
                and len(self._dest_pins) > self.dest_ids.max()
            )
        except AssertionError:
            raise ValueError(
                "The number of source and destination pins are not equal to the dimension of coordinates format."
            )

    def _build_pin_map(self, conn_type: Type[TwoEndConnector]) -> Dict[NeuronId, List[AxonId]]:
        if conn_type is MatConn:
            return self._build_from_mat()
        
        if conn_type is IndexConn:
            return self._build_from_coo()
        
        raise TypeError(f"Unsupported type: {type(conn_type)}.")

    def _build_from_mat(self) -> Dict[NeuronId, List[AxonId]]:
        """Build pin map from the connectivity matrix."""
        pin_map = {}

        # Traverse the source pins and find the corresponding destination pins.
        for i in range(len(self._source_pins)):
            syns_to_dest = self.conn_mat[i, :].nonzero()[0]

            # Traverse the synapses to each destination pin.
            for j in range(len(syns_to_dest)):
                dest_pin = syns_to_dest[j]

                if self._source_pins[i] not in pin_map:
                    pin_map[self._source_pins[i]] = []

                pin_map[self._source_pins[i]].append(self._dest_pins[dest_pin])
        
        return pin_map

    def _build_from_coo(self) -> Dict[NeuronId, List[AxonId]]:
        """Build pin map from the coordinates format."""
        pin_map = {}

        # Traverse the source indices in source pins
        for i in range(len(self.source_ids)):
            if self._source_pins[self.source_ids[i]] not in pin_map:
                pin_map[self._source_pins[self.source_ids[i]]] = []

            pin_map[self._source_pins[self.source_ids[i]]].append(
                self._dest_pins[self.dest_ids[i]]
            )

        return pin_map
