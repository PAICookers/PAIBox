import pytest

from paibox.backendv2.routing import OutputAxonAllocator


class _Elem:
    def __init__(self, bit_width: int, name: str = "elem") -> None:
        self.output_bit_num = bit_width
        self.name = name

    def __repr__(self) -> str:
        return self.name


def test_allocate_voltage_base_addresses_follow_bank_layout():
    allocator = OutputAxonAllocator()

    bases = [allocator.allocate(_Elem(32, f"v{i}")) for i in range(10)]

    assert bases == [0, 1, 2, 3, 4, 5, 6, 7, 32, 33]
    assert {0, 8, 16, 24, 32, 40, 48, 56}.issubset(allocator.used_bits)


def test_allocate_data_last_address_is_valid():
    allocator = OutputAxonAllocator()
    allocator.lowest_free_bit = allocator.MAX_AXON_BIT

    axon_bit = allocator.allocate(_Elem(8, "last_data"))

    assert axon_bit == allocator.MAX_AXON_BIT


def test_allocate_voltage_rejects_lane_overflow():
    allocator = OutputAxonAllocator()
    allocator.lowest_free_bit = allocator.MAX_AXON_BIT

    with pytest.raises(ValueError, match="exhausted"):
        allocator.allocate(_Elem(32, "overflow_voltage"))
