import warnings
from typing import overload

import numpy as np
from numpy.typing import ArrayLike
from paicorelib import LUT_DTYPE, LUTDataType, OnCoreCfg

from paibox.exceptions import PAIBoxWarning, ParamNotSimulatedWarning

__all__ = ["LUT"]

LUT_LEN = OnCoreCfg.LUT_LEN


class LUT:
    """LUT class for STDP learning."""

    def __init__(
        self,
        lut: ArrayLike | None = None,
        offset: int | None = None,
        lut_random: bool | ArrayLike = False,
    ) -> None:
        if lut is None:
            lut = np.zeros((LUT_LEN,), dtype=LUT_DTYPE)
        else:
            lut = np.asarray(lut, dtype=LUT_DTYPE)

        if lut.ndim != 1:
            raise ValueError("LUT must be a 1D array")
        if lut.size > LUT_LEN:
            raise ValueError(f"LUT length exceeds maximum of {LUT_LEN}")
        elif lut.size < LUT_LEN:
            lut = np.pad(
                lut, (0, LUT_LEN - lut.size), mode="constant", constant_values=0
            )
            warnings.warn(
                f"LUT length is less than {LUT_LEN}, padding with zeros at the end.",
                PAIBoxWarning,
            )

        self.lut = lut

        if offset is None:
            offset = self.size // 2
        if offset < 0 or offset >= self.size:
            raise ValueError(
                f"Offset must be between 0 and {self.size - 1}, but got {offset}"
            )
        else:
            self.offset = offset

        # TODO This feature needs LFSR support
        if isinstance(lut_random, bool):
            self.lut_random_en = np.full(self.size, lut_random, dtype=bool)
        else:
            lut_random_en = np.asarray(lut_random, dtype=bool)
            if lut_random_en.size != self.size:
                raise ValueError(
                    f"LUT random enable array must be of size {self.size}, but got {lut_random_en.size}"
                )
            self.lut_random_en = lut_random_en

        if np.any(self.lut_random_en):
            warnings.warn(
                "LUT random enable will not be simulated.", ParamNotSimulatedWarning
            )

    @overload
    def __getitem__(self, index: int) -> LUT_DTYPE: ...

    @overload
    def __getitem__(self, index: np.ndarray) -> LUTDataType: ...

    def __getitem__(self, index: int | np.ndarray) -> LUT_DTYPE | LUTDataType:
        return self.lut[np.clip(index + self.offset, 0, self.size - 1)]

    def lookup(self, index: int | np.ndarray) -> LUT_DTYPE | LUTDataType:
        return self.__getitem__(index)

    @property
    def size(self) -> int:
        return len(self.lut)

    @property
    def ltp(self) -> LUTDataType:
        return self.lut[self.offset :]

    @ltp.setter
    def ltp(self, ltp_lut: ArrayLike) -> None:
        """Reassign the long-term potentiation part of the LUT."""
        ltp_lut = np.asarray(ltp_lut, dtype=LUT_DTYPE)

        if ltp_lut.size > self.size - self.offset:
            raise ValueError(
                f"LTP LUT length exceeds maximum of {self.size - self.offset}"
            )

        self.lut[self.offset : self.offset + ltp_lut.size] = ltp_lut

    @property
    def ltd(self) -> LUTDataType:
        return self.lut[: self.offset]

    @ltd.setter
    def ltd(self, ltd_lut: ArrayLike) -> None:
        ltd_lut = np.asarray(ltd_lut, dtype=LUT_DTYPE)

        if ltd_lut.size > self.offset:
            raise ValueError(f"LTD LUT length exceeds maximum of {self.offset}")

        self.lut[self.offset - ltd_lut.size : self.offset] = ltd_lut
