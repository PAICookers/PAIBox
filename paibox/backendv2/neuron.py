from __future__ import annotations
from paicorelib import OfflineNeuFullAttrsV2Part1, OfflineNeuFullAttrsV2Part2, FoldedNeuAttrsV2Part1, OfflineFoldedNeuAttrsV2Part2, NeuDestInfoV2, OfflineNeuDestInfoV2
from typing import Optional
from paicorelib import OfflineCoreRegV2
from core_config import Inherited_Core_Config


class CoreOpNode:
    def attrs_part2(self) -> OfflineNeuFullAttrsV2Part2:
        raise NotImplementedError("attrs_part2 method is not implemented yet.")

    def core_config(self) -> Inherited_Core_Config:
        raise NotImplementedError("core_config method is not implemented yet.")

class CustomIndex:
    pass


class Neuron:
    def __init__(self):
        self.target: Optional[CoreOpNode] = None
        self.index: Optional[CustomIndex] = None
    
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

class NeuronPlacement:
    pass

class OfflineNeuronPlacement(NeuronPlacement):
    def __init__(self):
        self.dest_info: Optional[OfflineNeuDestInfoV2] = None
        self.neu_attrs_part1: Optional[OfflineNeuFullAttrsV2Part1] = None
        self.neu_attrs_part2: Optional[OfflineNeuFullAttrsV2Part2] = None
        self.folded_neu_attrs_part1: Optional[FoldedNeuAttrsV2Part1] = None
        self.folded_neu_attrs_part2s: list[OfflineFoldedNeuAttrsV2Part2] = []
        self.raw_neus: list[Neuron] = []
        
    
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