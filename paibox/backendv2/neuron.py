from __future__ import annotations

import numpy as np
from paicorelib import (
    FRAME_DTYPE,
    FrameArrayType,
    NeuDestInfoV2,
    NeuronType,
    OfflineFrameGenV2,
    OfflineNeuDestInfoV2,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFoldedAttrsV2Part2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
)

from .op_node import Neuron


class NeuronPlacement:
    def __init__(self, neu: list[Neuron]) -> None:
        self.raw_neus: list[Neuron] = neu
        self.dest_info: NeuDestInfoV2 | None = None


class OfflineNeuronPlacement(NeuronPlacement):
    def __init__(
        self,
        neu: list[Neuron],
        attrs_part1: OfflineNeuFullAttrsV2Part1,
        attrs_part2: OfflineNeuFullAttrsV2Part2,
    ):
        super().__init__(neu)
        self.neu_attrs_part1: OfflineNeuFullAttrsV2Part1 = attrs_part1
        self.neu_attrs_part2: OfflineNeuFullAttrsV2Part2 | None = (
            attrs_part2 if attrs_part1.neuron_type == NeuronType.FULL else None
        )
        self.folded_neu_attrs_part1: OfflineNeuFoldedAttrsV2Part1 | None = None
        self.folded_neu_attrs_part2s: list[OfflineNeuFoldedAttrsV2Part2] = []
        self.n_sram_required: int = self.n_sram_required_()

    def n_sram_required_(self) -> int:
        n_sram = 0
        if self.neu_attrs_part1 is not None:
            n_sram += 1
        if self.neu_attrs_part2 is not None:
            n_sram += 1
        if self.folded_neu_attrs_part1 is not None:
            n_sram += 1
        n_sram += len(self.folded_neu_attrs_part2s)
        return n_sram

    def to_package(self) -> FrameArrayType:
        if self.dest_info is None:
            raise ValueError("dest_info has not been set yet.")
        if not isinstance(self.dest_info, OfflineNeuDestInfoV2):
            raise TypeError("dest_info must be of type OfflineNeuDestInfoV2.")

        half_neu, full_neu, fold_neu = OfflineFrameGenV2.gen_config_frame3_pkg_neu(
            dest_info=self.dest_info,
            full_attrs1=self.neu_attrs_part1,
            full_attrs2=self.neu_attrs_part2,
            folded_attrs1=self.folded_neu_attrs_part1,
            folded_attrs2_=self.folded_neu_attrs_part2s,
        )

        if self.neuron_type == NeuronType.HALF:
            frame_list: FrameArrayType = np.concatenate(
                [half_neu, fold_neu], axis=0
            ).astype(FRAME_DTYPE)
        else:
            frame_list: FrameArrayType = np.concatenate(
                [full_neu, fold_neu], axis=0
            ).astype(FRAME_DTYPE)
        return frame_list

    @property
    def neuron_type(self) -> NeuronType:
        return self.neu_attrs_part1.neuron_type
