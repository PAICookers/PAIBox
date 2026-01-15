from __future__ import annotations
from coreplacement import CorePlacement
from paicorelib import AERPacketZXYCopy, CoordXY, LCN_EX
from typing import Optional
from neuron import Neuron
import numpy as np


FANIN_BASE = 512

def get_raw_weights(raw_neus: list[Neuron], input_neus: list[Neuron]) -> np.ndarray:
    n_output = len(raw_neus)
    n_input = len(input_neus)
    # random weights for testing, shape (n_output, n_input)
    weights = np.random.randint(-128, 127, size=(n_output, n_input), dtype=np.int8)
    return weights

class RoutingGroup:
    # core_blocks in the same routing group share the same following properties
    # lcn, input_sign, input_width
    def __init__(self):
        # set by generate_routing_groups
        self.raw_neus: list[Neuron] = []
        
        # set by set_rough_dest
        self.dests: list["RoutingGroup"] = []
        self.input_list: list[Neuron] = [] # input_list can be reordered later
        self.lcn: LCN_EX = LCN_EX.LCN_1X
        
        # self.core_blocks: list[CoreBlock] = []
        self.core_placements: list[CorePlacement] = []
        self.n_core_required: int = -1
        
        self.muticast_config: Optional[AERPacketZXYCopy] = None
        self.base_coord: Optional[CoordXY] = None
    
    def set_lcn(self):
        input_widths = set([neu.core_config().input_width for neu in self.raw_neus])
        assert len(input_widths) == 1, "All neurons in the routing group must have the same input width."
        # if input width, input sign weight are different, need to split routing group before calling this function
        input_width = input_widths.pop()
        max_axon_addr = len(self.input_list) * input_width
        lcn = ((max_axon_addr - 1) // FANIN_BASE).bit_length()
        self.lcn = LCN_EX(lcn)
        
    
    def allocate_neurons(self):
        """core placement generation"""
        # you can get neu_attrs_part2, inherited_core_config, target_lcn, lcn for each neuron in self.raw_neus, like:
        # all the attrs in neu_attrs_part2 are valid except weight compress, you should set weight compress according to your weight storage strategy
        # inherited core_config are valid except weight_width, you can set weight_width larger than or equal to the original value for optimization
        
        for neu, dest in zip(self.raw_neus, self.dests):
            attrs = neu.attrs_part2()
            inherited_core_config = neu.core_config()
            target_lcn = dest.lcn
            lcn = self.lcn
        
        # implement your core placement generation logic here
        # you need to create OfflineCorePlamentV2 instances, set their properties including:
        # coreplacement._core_config, coreplacment.neus, coreplacment.weights, coreplacment.neu_weight_map
        # remember the neurons in the same core placement share the same core config, so 
        
        # the number of core placements created should be set to self.n_core_required
        # dest_info in NeuronPlacement should remain unset, it will be set later during routing.
        # auto_core_config in CorePlacement should remain unset, it will be set later during core allocation.
        
        # each Coreplacement has 4096 sram, you need to make sure the total sram required by neu and weight in each coreplacement does not exceed 4096
        # you can get the sram required by each neuron using OfflineNeuronPlacement.n_sram_required() function
        # you can get the sram required by each weight using Weight.n_sram_required() function
        
        # you can use get_raw_weights function to get the raw weights between self.raw_neus and self.input_list
        # then you can reorder the input_list as well as the weight's rows according to your core placement strategy
        
        # you can think about the following optimizations:
        # 1. for neurons in the same Coreplacement with the same attrs_part2, you can use half neuron to save SRAM
        # 2. you can reorder the input_list, to move as many as possible zero weights to the end of each weight row, so that you can reduce the weight storage requirement
        # ... etc.
                
        raise NotImplementedError("allocate_neurons method is not implemented yet.")
        
        
            