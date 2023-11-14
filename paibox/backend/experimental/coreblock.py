from typing import List, NamedTuple, NewType, Sequence

from paibox.base import NeuDyn


class NeuronSegment(NamedTuple):
    parent: NeuDyn
    """指示这个对象是描述的哪个神经元组"""
    index: slice
    """指示这段神经元对应的下标范围（一定连续分配）"""
    addr_ram: slice
    """分配到在物理核内的RAM坐标范围（一定连续分配）"""


NeuronSegments = NewType("NeuronSegments", List[NeuronSegment])  # one_core


def get_neuron_segments_1(
    neurons: Sequence[NeuDyn], capacity: int
) -> List[NeuronSegments]:
    result = []

    for n in neurons:
        num = n.num_out
        i = 0

        while i < (num - 1) // capacity:
            segment = NeuronSegment(
                n, slice(i * capacity, capacity * (i + 1), 1), slice(0, capacity, 1)
            )

            result.append([segment])
            i += 1

        segment = NeuronSegment(
            n, slice(i * capacity, num, 1), slice(0, num - (i * capacity), 1)
        )
        result.append([segment])

    return result


def get_neuron_segments_2(
    neurons: Sequence[NeuDyn], capacity: int
) -> List[NeuronSegments]:
    result = []
    segments_of_neurons = get_neuron_segments_1(neurons, capacity)
    temp = []

    sum = 0
    for segs in segments_of_neurons:
        if segs[0].addr_ram.stop < capacity:
            temp.append(segs[0])
            sum += segs[0].addr_ram.stop
        else:
            result.append(segs)

    temp.sort(key=lambda seg: seg.addr_ram.stop)

    i = 0  # 剩余部分可组成的物理核个数
    j = 0  # 有剩余的的物理核
    while i < (sum - 1) // capacity + 1:
        segments = NeuronSegments([])
        full = 0
        empty = capacity - full

        while empty > 0 and j < len(temp):
            if empty >= temp[j].addr_ram.stop:
                segment = NeuronSegment(
                    temp[j].parent,
                    temp[j].index,
                    slice(full, full + temp[j].addr_ram.stop, 1),
                )
                segments.append(segment)
                full += temp[j].addr_ram.stop
                empty = capacity - full
                j += 1
            else:
                segment = NeuronSegment(
                    temp[j].parent,
                    slice(temp[j].index.start, temp[j].index.start + empty, 1),
                    slice(full, capacity, 1),
                )
                segments.append(segment)
                temp[j] = NeuronSegment(
                    temp[j].parent,
                    slice(temp[j].index.start + empty, temp[j].index.stop, 1),
                    slice(0, temp[j].addr_ram.stop - empty, 1),
                )
                full += capacity
                empty = 0

        i += 1
        result.append(segments)

    return result
