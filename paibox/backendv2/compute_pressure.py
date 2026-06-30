"""Shared compute-pressure helpers for backendv2 placement and visualizer views."""

from paicorelib import DataWidth

from .weight import N_WEIGHTS_PER_SRAM

SRAM_RECORD_BITS = 128


def dense_weight_slots_per_sram(weight_width: DataWidth | int) -> int:
    """Return dense weight slots stored in one 128-bit SRAM record."""
    return SRAM_RECORD_BITS >> weight_width


def csc_weight_slots_per_sram(weight_width: DataWidth | int) -> int:
    """Return CSC-compressed weight slots stored in one SRAM record.

    CSC storage reserves part of the SRAM record for sparse index metadata, so
    its slot count is hardware-defined by weight width instead of being a simple
    `128 / weight_bits` division.
    """
    return N_WEIGHTS_PER_SRAM[DataWidth(weight_width)]


def compute_weight_pressure(
    fold_count: int,
    input_width: DataWidth | int,
    weight_width: DataWidth | int,
    weight_sram_records: int,
    is_csc: bool,
    weight_slots: int | None = None,
) -> int:
    """Return storage-pressure SOPS for one neuron placement and weight range.

    The pressure is:
    `fold_count * input_bits * weight_bits * weight_slots`.

    By default `weight_slots` is derived from `weight_sram_records`. Dense
    weights use all values that fit in 128 bits. CSC weights use the hardware CSC
    capacity `{1: 7, 2: 7, 4: 6, 8: 5}` per SRAM record. That means this metric
    is storage-padded SOPS: CSC padding slots count as pressure. Callers that
    need an unpadded user-facing value may pass an explicit non-padding
    `weight_slots` count while keeping the same bit-scaling rule.
    """
    input_bits = 1 << input_width
    weight_bits = 1 << weight_width
    if weight_slots is None:
        slots_per_sram = (
            csc_weight_slots_per_sram(weight_width)
            if is_csc
            else dense_weight_slots_per_sram(weight_width)
        )
        weight_slots = weight_sram_records * slots_per_sram
    return fold_count * input_bits * weight_bits * weight_slots
