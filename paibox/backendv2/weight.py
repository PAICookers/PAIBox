from __future__ import annotations

import numpy as np
from paicorelib import (
    AddPotentialMode,
    DataWidth,
    FrameArrayType,
    OfflineFrameGenV2,
    WeightCompressType,
)

N_WEIGHTS_PER_SRAM = {
    DataWidth.WIDTH_1BIT: 7,
    DataWidth.WIDTH_2BIT: 7,
    DataWidth.WIDTH_4BIT: 6,
    DataWidth.WIDTH_8BIT: 5,
}


class Weight:
    def __init__(
        self,
        data: np.ndarray | list[int],
        compress_type: WeightCompressType,
        weight_width: DataWidth,
        input_width: DataWidth,
        AddPotential: AddPotentialMode = AddPotentialMode.NORMAL,
    ):
        if AddPotential == AddPotentialMode.DIRECT_ADD:
            # in direct add mode, weight width should be at least 4 bit to avoid overflow
            weight_width = DataWidth.WIDTH_1BIT
            input_width = DataWidth.WIDTH_32BIT
            if isinstance(data, list):
                data = np.array(data, dtype=np.int16)
            data = data.astype(np.int16)  # ensure weight is in int16 to avoid overflow
            data = np.repeat(
                data, 32
            )  # each one in original weight repeat 32 times to form mask for direct add mode

        if isinstance(data, np.ndarray):
            self.raw_weights: list[int] = data.tolist()
        else:
            self.raw_weights: list[int] = list(data)

        self.weight_width = weight_width
        self.input_width = input_width

        self.compress: bool = (
            compress_type == WeightCompressType.SPARSE
        )  # whether the weight is compressed

        # processed weights remove zero at the end if not compressed
        self.processed_weights: list[int] = self.raw_weights.copy()
        # remove zero at the end of raw weights
        while len(self.processed_weights) > 0 and self.processed_weights[-1] == 0:
            self.processed_weights.pop()

        self.n_sram_required = self.n_sram_required_()

    def n_sram_required_(self) -> int:
        if not self.compress:
            return (len(self.processed_weights) * (2**self.weight_width) + 127) // 128

        else:
            # non zero weight in raw weights
            n_non_zero = sum(1 for w in self.raw_weights if w != 0)
            n_weight_per_sram = N_WEIGHTS_PER_SRAM.get(self.weight_width, 0)
            if n_weight_per_sram == 0:
                raise ValueError(
                    f"Unsupported weight width for compression: {self.weight_width}"
                )
            return (n_non_zero + n_weight_per_sram - 1) // n_weight_per_sram

    def to_package(self) -> FrameArrayType:
        frames = OfflineFrameGenV2.gen_config_frame3_weight_pkg(
            weight=np.array(self.processed_weights),
            input_width=self.input_width,
            weight_width=self.weight_width,
            csc_compress=self.compress,
        )
        return frames
