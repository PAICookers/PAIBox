from abc import ABC
from typing import List, Union

import numpy as np

from paibox._types import Shape
from paibox.utils import shape2num

__all__ = ["TwoEndConnector", "One2One", "MatConn", "IndexConn"]


class Connector(ABC):
    """Connector"""

    pass


class TwoEndConnector(Connector):
    """Basic two-end connector."""

    def __init__(self, source_shape: Shape = 0, dest_shape: Shape = 0) -> None:
        """
        Arguments:
            - source_shape: shape of source neurons
            - dest_shape: shape of destination neurons
        """
        if isinstance(source_shape, int):
            self.source_shape = (source_shape,)
        else:
            self.source_shape = tuple(source_shape)

        if isinstance(dest_shape, int):
            self.dest_shape = (dest_shape,)
        else:
            self.dest_shape = tuple(dest_shape)

        self.source_num = shape2num(source_shape)
        self.dest_num = shape2num(dest_shape)

    def __call__(self, source_shape: Shape, dest_shape: Shape):
        self.source_shape = source_shape
        self.dest_shape = dest_shape
        self.source_num = shape2num(source_shape)
        self.dest_num = shape2num(dest_shape)
        return self

    def __repr__(self) -> str:
        return self.__class__.__name__

    def build_coo(self) -> ...:
        raise NotImplementedError

    def build_mat(self) -> np.ndarray:
        raise NotImplementedError


class One2One(TwoEndConnector):
    """Connect two groups of neurons one-by-one."""

    def __init__(self) -> None:
        """
        Arguments:
            - source_shape: shape of source neurons
            - dest_shape: shape of destination neurons

        NOTE: The number of source and destination neurons must be equal.
        """
        super(One2One, self).__init__()

        try:
            assert self.source_num == self.dest_num
        except AssertionError:
            raise ValueError(
                "The number of source and destination neurons must be equal."
            )

    def __call__(self, source_shape: Shape, dest_shape: Shape):
        super(One2One, self).__call__(source_shape, dest_shape)
        try:
            assert self.source_num == self.dest_num
        except AssertionError:
            raise ValueError(
                "The number of source and destination neurons must be equal."
            )

        return self

    def build_coo(self):
        return np.arange(self.source_num, dtype=np.uint16), np.arange(
            self.dest_num, dtype=np.uint16
        )

    def build_mat(self):
        return np.eye(self.source_num, self.dest_num, dtype=np.bool_)


class All2All(TwoEndConnector):
    """Connect every source neuron with every destination neuron."""

    def __init__(self, source_shape: Shape = 0, dest_shape: Shape = 0) -> None:
        super(All2All, self).__init__(source_shape, dest_shape)

    def build_mat(self):
        return np.ones((self.source_num, self.dest_num), dtype=np.bool_)


class MatConn(TwoEndConnector):
    """Connector built from the dense connection matrix."""

    def __init__(
        self, source_shape: Shape = 0, dest_shape: Shape = 0, *, conn_mat: np.ndarray
    ) -> None:
        """
        Arguments:
            - source_shape: shape of source neurons
            - dest_shape: shape of destination neurons
            - conn_mat: dense connection matrix
        """
        super(MatConn, self).__init__(source_shape, dest_shape)

        assert isinstance(conn_mat, np.ndarray) and conn_mat.ndim == 2

        self.source_num, self.dest_num = conn_mat.shape
        self.conn_mat = np.asarray(conn_mat, dtype=np.bool_)

    def __call__(self, source_shape: Shape, dest_shape: Shape):
        assert self.source_num == shape2num(source_shape)
        assert self.dest_num == shape2num(dest_shape)
        return self

    def build_mat(self):
        return self.conn_mat


class IndexConn(TwoEndConnector):
    """Connector built from the indices of source and destination neurons."""

    def __init__(
        self,
        source_shape: Shape = 0,
        dest_shape: Shape = 0,
        *,
        source_ids: Union[List[int], np.ndarray],
        dest_ids: Union[List[int], np.ndarray],
    ) -> None:
        """
        Arguments:
            - source_shape: shape of source neurons
            - dest_shape: shape of destination neurons
            - source_ids: indices of source neurons
            - dest_ids: indices of destination neurons
        """
        super(IndexConn, self).__init__(source_shape, dest_shape)

        source_ids = np.asarray(source_ids)
        dest_ids = np.asarray(dest_ids)

        assert source_ids.ndim == 1
        assert dest_ids.ndim == 1
        assert source_ids.size == dest_ids.size

        self.source_ids = np.asarray(source_ids, dtype=np.uint16)
        self.dest_ids = np.asarray(dest_ids, dtype=np.uint16)
        self.max_source_num = self.source_ids.max()
        self.max_dest_num = self.dest_ids.max()

        self._check()

    def __call__(self, source_shape: Shape, dest_shape: Shape):
        super(IndexConn, self).__call__(source_shape, dest_shape)
        self._check()

        return self

    def build_coo(self):
        self._check()
        return self.source_ids, self.dest_ids

    def _check(self):
        if self.max_source_num >= self.source_num:
            raise ValueError(
                f"Out of range: source_shape should be greater than the maximum id ({self.max_source_num}) of source_ids"
            )

        if self.max_dest_num >= self.dest_num:
            raise ValueError(
                f"Out of range: dest_shape should be greater than the maximum id ({self.max_dest_num}) of dest_ids"
            )
