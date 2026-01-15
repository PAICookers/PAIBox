from __future__ import annotations
from paicorelib import CoordXY, OfflineCoreRegV2
from neuron import NeuronPlacement, OfflineNeuronPlacement
from routing import RoutingGroup
from typing import Optional
from weight import Weight
from abc import abstractmethod

class CorePlacement:
    def __init__(
        self, 
    ) -> None:
        self.coord: Optional[CoordXY] = None
        self._core_config: Optional[OfflineCoreRegV2] = None
        self.neus: list[NeuronPlacement] = [] # full or half neu, depending on neu allocation
        self.weights: list[Weight] = [] # weight of each single neu, can reuse for different single neu
        self.neu_weight_map: dict[int, int] = {} # map from neu index to weight index
    
    def max_input_num(self) -> int:
        max_input_num = 0
        for weight in self.weights:
            input_num = len(weight.processed_weights)
            max_input_num = max(max_input_num, input_num)
        return max_input_num

class OfflineCorePlamentV2(CorePlacement):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.neus: list[OfflineNeuronPlacement] = []
    
    def n_sram_required(self) -> int:
        n_sram = 0
        for neu in self.neus:
            n_sram += neu.n_sram_required()
        for weight in self.weights:
            n_sram += weight.n_sram_required()
        return n_sram
    
    def to_frame(self):    
        packages = []
        for neu in self.neus:
            packages.extend(neu.to_package())
        for weight in self.weights:
            packages.extend(weight.to_package())

        def gen_frame():
            raise NotImplementedError("gen_frame method is not implemented yet.")
        
        start_frame = gen_frame()
        
        raise NotImplementedError("to_frame method is not implemented yet.")
        
        
        