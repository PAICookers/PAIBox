from abc import ABC, abstractmethod
import numpy as np

__all__ = ["TwoEndConnector", "MatConn", "IndexConn"]


class Connector(ABC):
    """Connector of a corelet.

    NOTE: Every corelets have a connector. The connector consists of bipartite mapping lists.
    1. A list of source pins map, length of the #axons.
        The source pins are PUBLIC to the external, while the other are connected internally which are PRIVATE.

    2. A list of destination pins map, length of the #neurons.
        The destination pins are PUBLIC to the external, while the other are connected internally which are PRIVATE.
    """

    @abstractmethod
    def build(self):
        raise NotImplementedError


class TwoEndConnector(Connector):
    def __init__(self, source_size: int, dest_size: int) -> None:
        """
        Arguments:
            - source_num: number of source neurons
            - dest_num: number of destination neurons

        NOTE: don't care the specific connectivity at this time. It's defined by the connection matrix.
        """
        self.source_size = source_size
        self.dest_size = dest_size

    def __repr__(self) -> str:
        return self.__class__.__name__


class MatConn(TwoEndConnector):
    """Connector built from the dense connection matrix."""

    def __init__(
        self, source_size: int, dest_size: int, *, conn_mat: np.ndarray, **kwargs
    ) -> None:
        super(MatConn, self).__init__(source_size, dest_size, **kwargs)

        assert isinstance(conn_mat, np.ndarray) and conn_mat.ndim == 2

        self.source_num, self.dest_num = conn_mat.shape
        self.conn_mat = np.asarray(conn_mat, dtype=np.bool_)

        assert self.source_size == self.source_num
        assert self.dest_size == self.dest_num

    def build(self):
        return self._build_mat()

    def _build_mat(self):
        return self.conn_mat


class IndexConn(TwoEndConnector):
    def __init__(
        self,
        source_size: int,
        dest_size: int,
        *,
        source_ids: np.ndarray,
        dest_ids: np.ndarray,
        **kwargs,
    ) -> None:
        super(IndexConn, self).__init__(source_size, dest_size, **kwargs)

        assert isinstance(source_ids, np.ndarray) and source_ids.ndim == 1
        assert isinstance(dest_ids, np.ndarray) and dest_ids.ndim == 1
        assert source_ids.size == dest_ids.size

        self.source_ids = np.asarray(source_ids, dtype=np.int16)
        self.dest_ids = np.asarray(dest_ids, dtype=np.int16)
        self.max_source_num = self.source_ids.max()
        self.max_dest_num = self.dest_ids.max()

    def build(self):
        return self._build_coo()

    def _build_coo(self):
        self._check()
        return self.source_ids, self.dest_ids

    def _check(self):
        if self.max_source_num >= self.source_size:
            raise ValueError(
                f"Out of range: source_size should be greater than the maximum id ({self.max_source_num}) of source_ids"
            )

        if self.max_dest_num >= self.dest_size:
            raise ValueError(
                f"Out of range: dest_size should be greater than the maximum id ({self.max_dest_num}) of dest_ids"
            )
