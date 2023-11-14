from typing import List, NamedTuple, Sequence, Union
from paibox.base import NeuDyn
from paibox.projection import InputProj
from paibox.libpaicore import LCN_EX
import math
class AxonSegment(NamedTuple):
    index: slice
    """指示这段轴突对应的下标范围（一定连续分配）"""
    tick_relative: int
    """分配到的tick relative"""
    addr_axon: slice
    """分配到的轴突坐标范围（一定连续分配）"""
    
   
def get_axon_segments(axons: Sequence[Union[NeuDyn, InputProj]], lcn_ex: LCN_EX) -> List[List[AxonSegment]]:
    #假设了对于入度大于一的节点，其前驱节点的出度都是1。
    axon_segments_list= []
    attribute_used = 0
    num_tick = 1 << lcn_ex
    print(num_tick)
    for axon in axons:
        num_out = axon.num_out
        attribute_need = math.ceil(num_out / num_tick)
        axon_segments = []
        start_addr = attribute_used
        for i in range(num_tick):
            start_index = i * attribute_need
            end_index = min(start_index + attribute_need, num_out)
            end_addr = start_addr + end_index - start_index
            axon_segments.append(AxonSegment(slice(start_index, end_index, 1), i, slice(start_addr, end_addr, 1)))
        attribute_used += attribute_need
        axon_segments_list.append(axon_segments)
    if attribute_used > 1152:
        raise ValueError("attributes needed out of range")
    return axon_segments_list
      
    