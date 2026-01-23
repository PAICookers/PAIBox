from __future__ import annotations

from typing import Literal

import numpy as np
from paicorelib import FRAME_DTYPE, FrameArrayType, OfflineFrameGenV2

N_WEIGHTS_PER_SRAM = {1: 7, 2: 7, 4: 6, 8: 5}


class Weight:
    def __init__(self):
        self.raw_weights: list[int] = []  # the raw weight
        self.weight_width: Literal[1, 2, 4, 8] = 1  # bit width of each weight
        self.compress: bool = False  # whether the weight is compressed

        # processed weights remove zero at the end if not compressed
        self.processed_weights: list[int] = self.raw_weights.copy()
        # remove zero at the end of raw weights
        while len(self.processed_weights) > 0 and self.processed_weights[-1] == 0:
            self.processed_weights.pop()

    def n_sram_required(self) -> int:
        if not self.compress:
            return (len(self.processed_weights) * self.weight_width + 127) // 128

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
            weight_width=self.weight_width,
            csc_compress=self.compress,
        )
        return frames
