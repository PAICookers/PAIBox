from __future__ import annotations

from typing import Optional

import numpy as np
from paicorelib import (
    FRAME_DTYPE,
    FrameArrayType,
    NeuDestInfoV2,
    NeuronType,
    OfflineCoreRegV2,
    OfflineFrameGenV2,
    OfflineNeuDestInfoV2,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFoldedAttrsV2Part2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
)

from .core_config import Inherited_Core_Config


class CoreOpNode:
    def attrs_part2(self) -> OfflineNeuFullAttrsV2Part2:
        raise NotImplementedError("attrs_part2 method is not implemented yet.")

    def core_config(self) -> Inherited_Core_Config:
        raise NotImplementedError("core_config method is not implemented yet.")

    def __hash__(self) -> int:
        return hash(id(self))


class CustomIndex:
    def __init__(self):
        self.idx = 0

    def __hash__(self) -> int:
        return hash(self.idx)


class Neuron:
    def __init__(
        self, target: Optional[CoreOpNode] = None, index: Optional[CustomIndex] = None
    ):
        if target is None or index is None:
            raise ValueError("target and index must be provided.")
        self.target = target
        self.index = index

    def attrs_part2(self) -> OfflineNeuFullAttrsV2Part2:
        if self.target is not None:
            return self.target.attrs_part2()
        else:
            raise ValueError("target has not been set yet.")

    def core_config(self) -> Inherited_Core_Config:
        if self.target is not None:
            return self.target.core_config()
        else:
            raise ValueError("target has not been set yet.")

    def __hash__(self) -> int:
        return hash((self.index, self.target))


class NeuronPlacement:
    def __init__(self) -> None:
        self.raw_neus: list[Neuron] = []
        self.dest_info: Optional[OfflineNeuDestInfoV2] = None


class OfflineNeuronPlacement(NeuronPlacement):
    def __init__(self, neu, attrs):
        super().__init__(neu)
        self.neu_attrs_part1: Optional[OfflineNeuFullAttrsV2Part1] = None
        self.neu_attrs_part2: Optional[OfflineNeuFullAttrsV2Part2] = attrs
        self.folded_neu_attrs_part1: Optional[OfflineNeuFoldedAttrsV2Part1] = None
        self.folded_neu_attrs_part2s: list[OfflineNeuFoldedAttrsV2Part2] = []
        self.neuron_type = NeuronType.FULL

    def n_sram_required(self) -> int:
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

        half_neu, full_neu, fold_neu = OfflineFrameGenV2.gen_config_frame3_pkg_neu(
            dest_info=self.dest_info,
            full_attrs1=self.neu_attrs_part1,
            full_attrs2=self.neu_attrs_part2,
            folded_attrs1=self.folded_neu_attrs_part1,
            folded_attrs2_=self.folded_neu_attrs_part2s,
        )

        frame_list: FrameArrayType = np.concatenate(
            [half_neu, full_neu, fold_neu], axis=0
        ).astype(FRAME_DTYPE)
        return frame_list
