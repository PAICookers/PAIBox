from paibox.components.projection import InputSlice
from paibox.components.neuron import NeuronSlice
from paibox.components.synapses import EdgeSlice
from .types import NodeSliceType, EdgeSliceType, SourceSliceType, DestSliceType, NodeType, InputProj, Neuron
from paicorelib import Coord, CoreMode

from paibox.components import Neuron
from .types import NeuSegment, NodeDegree, NodeType, SourceNodeType, is_iw8, NeuSegOfCoreBlock, SourceSliceType, AxonSegment

def slice_overlap(slice1: slice, slice2: slice) -> bool:
    """Check if two slices overlap."""
    return slice1.start < slice2.stop and slice2.start < slice1.stop

def node_slice_overlap(node1: NodeSliceType, node2: NodeSliceType) -> bool:
    return node1.target == node2.target and slice_overlap(node1.index, node2.index)

def overlap(node_list1: list[NodeSliceType], node_list2: list[NodeSliceType]) -> bool:
    for node1 in node_list1:
        for node2 in node_list2:
            if node_slice_overlap(node1, node2):
                return True
    return False

def node_overlap(node: NodeType, node_list: list[NodeSliceType]) -> bool:
    if isinstance(node, InputProj):
        node_slice = InputSlice(node)
    elif isinstance(node, Neuron):
        node_slice = NeuronSlice(node)
    for node in node_list:
        if node_slice_overlap(node_slice, node):
            return True
    return False

# judge if node_slice1 is part of node_slice2
def node_covered(node_slice1: NodeSliceType, node_slice2: NodeSliceType) -> bool:
    return node_slice1.index.start >= node_slice2.index.start and node_slice1.index.stop <= node_slice2.index.stop


def cover(slice1: slice, slice2: slice) -> bool:
    return slice1.start <= slice2.start and slice1.stop >= slice2.stop

def fusion(nums: list[int]):
    base = nums[0]
    rid  = 0
    for num in nums[1:]:
        rid |= base ^ num
    base &= ~rid
    return base, rid

class SliceDest:
    def __init__(self, dest_axon: AxonSegment, dest_coords: list[Coord], dest_chip_coord: Coord, time_slot: int, mode: CoreMode) -> None:
        self.dest_axon: AxonSegment = dest_axon
        self.dest_coords: list[Coord] = dest_coords
        self.dest_chip_coord: Coord = dest_chip_coord
        self.base_coord: Coord = Coord(0, 0)
        self.rid: Coord = Coord(0, 0)
        self.time_slot: int = time_slot
        self.rt_mode: CoreMode = mode
    def fusion(self):
        base, rid = fusion([coord.x for coord in self.dest_coords])
        self.base_coord.x = base
        self.rid.x = rid
        base, rid = fusion([coord.y for coord in self.dest_coords])
        self.base_coord.y = base
        self.rid.y = rid


class SourceDest: 
    def __init__(self) -> None:
        self.slices: list[slice] = list()
        self.dests: list[SliceDest] = list()
        self.cut_points: list[int] = list()
    
    def add_dest(self, dest_slice: SourceSliceType, dest_axon: AxonSegment, coreblock: "CoreBlock"):
        dest_coords = coreblock.core_coords.copy()
        dest_chip_coord = coreblock.chip_coord
        time_slot = coreblock.n_timeslot
        mode = coreblock.rt_mode
        if dest_slice.index not in self.slices:
            self.slices.append(dest_slice.index)
            slice_dest = SliceDest(dest_axon, dest_coords, dest_chip_coord, time_slot, mode)
            self.dests.append(slice_dest)
        else:
            idx = self.slices.index(dest_slice.index)
            slice_dest = self.dests[idx]
            assert slice_dest.dest_axon == dest_axon
            assert slice_dest.dest_chip_coord == dest_chip_coord
            assert slice_dest.time_slot == time_slot
            assert slice_dest.rt_mode == mode
            slice_dest.dest_coords.extend(dest_coords)

    def fusion_dest(self):
        for slice_dest in self.dests:
            slice_dest.fusion()
        
        sorted_lists = sorted(zip(self.slices, self.dests), key=lambda x: x[0].start)
        sorted_slices, sorted_dests = zip(*sorted_lists)
        self.slices = list(sorted_slices)
        self.dests = list(sorted_dests)
        for slice in self.slices:
            self.cut_points.append(slice.stop)
            
    def dest_info(self, index: slice = None) -> SliceDest:
        if index is None:
            if len(self.dests) != 1:
                raise ValueError("Multiple destinations")
            else:
                return self.dests[0]
        for source_index in self.slices:
            if cover(source_index, index):
                idx = self.slices.index(source_index)
                return self.dests[idx]
    
    def slice_dest(self, nue_seg: NeuSegment) -> tuple[list[NeuSegment], list[SliceDest]]:
        neu_seg_list:list[NeuSegment] = list()
        dest_list: list[SliceDest] = list()
        start = nue_seg.index.start
        stop = nue_seg.index.stop
        for i, cut_point in enumerate(self.cut_points):
            if cut_point <= start:
                continue
            
            elif cut_point > start and cut_point < stop:
                neu_seg_list.append(NeuSegment(nue_seg.target, slice(start, cut_point), nue_seg.repeat))
                dest_list.append(self.dests[i])
                start = cut_point
                
            elif cut_point >= stop:
                neu_seg_list.append(NeuSegment(nue_seg.target, slice(start, stop), nue_seg.repeat))
                dest_list.append(self.dests[i])
                break
        return neu_seg_list, dest_list
                
        
            
        