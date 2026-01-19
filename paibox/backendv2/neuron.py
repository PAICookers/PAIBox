from __future__ import annotations

from typing import Optional

from paicorelib import (
    FoldedNeuAttrsV2Part1,
    NeuDestInfoV2,
    OfflineCoreRegV2,
    OfflineFoldedNeuAttrsV2Part2,
    OfflineNeuDestInfoV2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
)

from core_config import Inherited_Core_Config


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
    def __init__(self):
        self.neu_attrs_part1: Optional[OfflineNeuFullAttrsV2Part1] = None
        self.neu_attrs_part2: Optional[OfflineNeuFullAttrsV2Part2] = None
        self.folded_neu_attrs_part1: Optional[FoldedNeuAttrsV2Part1] = None
        self.folded_neu_attrs_part2s: list[OfflineFoldedNeuAttrsV2Part2] = []

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

    def to_package(self):
        raise NotImplementedError("to_package method is not implemented yet.")
