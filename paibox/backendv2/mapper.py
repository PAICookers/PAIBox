
from __future__ import annotations
from routing import RoutingGroup

class Mapper():
    def __init__(self):
        pass
    
    def generate_routing_groups(self) -> list[RoutingGroup]:
        raise NotImplementedError("generate_routing_groups method is not implemented yet.")
    
    def set_rough_dest(self):
        raise NotImplementedError("set_rough_dest method is not implemented yet.")
        
    def routing(self):
        raise NotImplementedError("routing method is not implemented yet.")
    
    def export(self):
        raise NotImplementedError("export method is not implemented yet.")
    
    def compile(self):
        
        # determine raw_neus in routing groups, other properties remain unset
        self.routing_groups = self.generate_routing_groups()
        
        # determine which rg each neuron sends to
        # dests and input_list set
        # other properties remain unset
        self.set_rough_dest()
        
        for rg in self.routing_groups:
            rg.allocate_neurons()
        
        # set core placements' coord, and generate detailed dest info for each neuron
        self.routing()
        
        # export to hardware executable format
        self.export()
        