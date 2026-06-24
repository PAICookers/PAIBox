from collections.abc import Sequence
from math import ceil

import numpy as np
from numpy.typing import ArrayLike
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
        data: ArrayLike,
        compress_type: WeightCompressType,
        weight_width: DataWidth,
        input_width: DataWidth,
        AddPotential: AddPotentialMode = AddPotentialMode.NORMAL,
    ) -> None:
        raw_weights = np.asarray(data, dtype=np.int16).ravel()

        if AddPotential == AddPotentialMode.DIRECT_ADD:
            # in direct add mode, weight width should be at least 4 bit to avoid overflow
            weight_width = DataWidth.WIDTH_1BIT
            input_width = DataWidth.WIDTH_1BIT
            # each one in original weight repeat 32 times to form mask for direct add mode
            raw_weights = np.repeat(raw_weights, 32)

        self.raw_weights = raw_weights
        self.weight_width = weight_width
        self.input_width = input_width

        # whether the weight is compressed
        self.compress = compress_type == WeightCompressType.SPARSE

        nonzero_indices = np.flatnonzero(self.raw_weights)
        n_nonzero = nonzero_indices.size
        end = nonzero_indices[-1] + 1 if n_nonzero > 0 else 0

        self.processed_weights = self.raw_weights[:end]
        self.n_sram_required = self.get_n_sram_required(n_nonzero)

    def get_n_sram_required(self, n_nonzero: int) -> int:
        if self.compress:
            n_weight_per_sram = N_WEIGHTS_PER_SRAM.get(self.weight_width, 0)
            if n_weight_per_sram == 0:
                raise ValueError(
                    f"Unsupported weight width for compression: {self.weight_width.name}"
                )
            return ceil(n_nonzero / n_weight_per_sram)
        else:
            n_weight_per_sram = 128 >> self.weight_width
            return ceil(self.processed_weights.size / n_weight_per_sram)

    def to_package(self, weight_skews: Sequence[int] | None = None) -> FrameArrayType:
        return OfflineFrameGenV2.gen_config_frame3_weight_pkg(
            self.processed_weights,
            weight_width=self.weight_width,
            input_width=self.input_width,
            csc_compress=self.compress,
            weight_skews=weight_skews,
        )
