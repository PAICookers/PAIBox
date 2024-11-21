from paibox.components.projection import InputSlice
from paibox.components.neuron import NeuronSlice
from paibox.components import Neuron
from typing import Union
from .types import NodeSliceType, NodeType, InputProj, Neuron


def overlap(slice1: slice, slice2: slice) -> bool:
    """Check if two slices overlap."""
    return slice1.start < slice2.stop and slice2.start < slice1.stop

def NN_overlap(node1: NodeSliceType, node2: NodeSliceType) -> bool:
    return node1.target == node2.target and overlap(node1.index, node2.index)

def LL_overlap(node_list1: list[NodeSliceType], node_list2: list[NodeSliceType]) -> bool:
    for node1 in node_list1:
        for node2 in node_list2:
            if NN_overlap(node1, node2):
                return True
    return False

def NL_overlap(node: Union[NodeType, NodeSliceType], node_list: list[NodeSliceType]) -> bool:
    if isinstance(node, InputProj):
        node_slice = InputSlice(node)
    elif isinstance(node, Neuron):
        node_slice = NeuronSlice(node)
    else:
        node_slice = node

    for _node in node_list:
        if NN_overlap(node_slice, _node):
            return True
    return False

def cover(part_slice: slice, whole_slice: slice) -> bool:
    return whole_slice.start <= part_slice.start and whole_slice.stop >= part_slice.stop

def NN_cover(part_node_slice: NodeSliceType, whole_node_slice: NodeSliceType) -> bool:
    return part_node_slice.target == whole_node_slice.target and cover(part_node_slice.index, whole_node_slice.index)

def NL_cover(node: Union[NodeType, NodeSliceType], node_list: list[NodeSliceType]) -> bool:
    if isinstance(node, InputProj):
        node_slice = InputSlice(node)
    elif isinstance(node, Neuron):
        node_slice = NeuronSlice(node)
    else:
        node_slice = node

    for _node in node_list:
        if NN_cover(node_slice, _node):
            return True
    return False


     
