from __future__ import annotations

N_WEIGHTS_PER_SRAM = {1: 7, 2: 7, 4: 6, 8: 5}


class Weight:
    def __init__(self):
        self.raw_weights: list[int] = []  # the raw weight
        self.weight_width: int = 0  # bit width of each weight
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

    def to_package(self):
        raise NotImplementedError("to_package method is not implemented yet.")
