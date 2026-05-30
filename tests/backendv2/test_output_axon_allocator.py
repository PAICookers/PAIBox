from typing import cast

import pytest
from paicorelib import LCN_EX

from paibox.backendv2.op_node import SourceElem
from paibox.backendv2.routing import OutputAxonAllocator


class _Elem:
    def __init__(self, bit_width: int, name: str = "elem") -> None:
        self.output_bit_num = bit_width
        self.name = name

    def __repr__(self) -> str:
        return self.name


def _source_elem(bit_width: int, name: str = "elem") -> SourceElem:
    return cast(SourceElem, _Elem(bit_width, name))


def test_allocate_voltage_base_addresses_follow_bank_layout():
    allocator = OutputAxonAllocator()

    bases = [allocator.allocate(_source_elem(32, f"v{i}")) for i in range(10)]

    assert bases == [0, 1, 2, 3, 4, 5, 6, 7, 32, 33]
    assert {0, 8, 16, 24, 32, 40, 48, 56}.issubset(allocator.used_bits)


def test_allocate_data_last_address_is_valid():
    allocator = OutputAxonAllocator()
    allocator.lowest_free_bit = allocator.max_axon_bit

    axon_bit = allocator.allocate(_source_elem(8, "last_data"))

    assert axon_bit == allocator.max_axon_bit


def test_allocator_capacity_uses_target_lcn():
    allocator = OutputAxonAllocator(LCN_EX.LCN_1X)

    assert allocator.max_axon_bit == 511

    allocator.lowest_free_bit = allocator.max_axon_bit
    assert allocator.allocate(_source_elem(8, "small_lcn_last_data")) == 511

    with pytest.raises(ValueError, match="exhausted"):
        allocator.allocate(_source_elem(8, "small_lcn_overflow"))


def test_allocator_returns_existing_address_for_same_element():
    allocator = OutputAxonAllocator(LCN_EX.LCN_1X)
    elem = _source_elem(8, "data")

    first = allocator.allocate(elem)
    second = allocator.allocate(elem)

    assert second == first
    assert allocator.axon_by_elem == {elem: first}
    assert allocator.axon_infos == [(first, elem)]
    assert allocator.lowest_free_bit == first + 1


def test_allocator_retarget_rejects_existing_addresses_outside_new_capacity():
    allocator = OutputAxonAllocator(LCN_EX.LCN_2X)
    allocator.lowest_free_bit = 512
    allocator.allocate(_source_elem(8, "data_above_lcn1"))

    with pytest.raises(ValueError, match="Cannot retarget"):
        allocator.retarget(LCN_EX.LCN_1X)

    assert allocator.target_lcn == LCN_EX.LCN_2X
    assert allocator.max_axon_bit == 1023


def test_allocate_voltage_rejects_lane_overflow():
    allocator = OutputAxonAllocator()
    allocator.lowest_free_bit = allocator.max_axon_bit

    with pytest.raises(ValueError, match="exhausted"):
        allocator.allocate(_source_elem(32, "overflow_voltage"))
