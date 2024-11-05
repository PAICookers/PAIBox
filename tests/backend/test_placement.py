import math
import sys
from contextlib import nullcontext
from functools import partial
from typing import Literal, Optional

import numpy as np
import pytest
from paicorelib import LCN_EX, Coord, HwConfig, NeuronAttrs
from paicorelib import ReplicationId as RId
from paicorelib import WeightWidth as WW
from paicorelib.framelib import OfflineFrameGen

import paibox as pb
from paibox.backend.placement import CorePlacement, FANOUT_IW8
from paibox.backend.types import (
    WRAM_PACKED_DTYPE,
    WRAM_UNPACKED_DTYPE,
    NeuSegment,
    WRAMPackedType,
    WRAMUnpackedType,
)
from paibox.exceptions import ResourceError
from paibox.types import WEIGHT_DTYPE, WeightType

from .test_conf_exporting import _gen_random_neuron_dest_info

if hasattr(HwConfig, "WEIGHT_BITORDER"):
    W_BITORDER = HwConfig.WEIGHT_BITORDER
else:
    W_BITORDER = "little"

N_BIT_PACKED_WEIGHT = np.iinfo(WRAM_PACKED_DTYPE).bits
if hasattr(CorePlacement, "WRAM_BASE_SHAPE"):
    WRAM_BASE_SHAPE = CorePlacement.WRAM_BASE_SHAPE
else:
    WRAM_BASE_SHAPE = (HwConfig.ADDR_AXON_MAX + 1, HwConfig.ADDR_RAM_MAX + 1)


NEURON_PARAMS_BIT_LENGTH = 214
N_NEURON_PARAM_IN_COL = HwConfig.N_FANIN_PER_DENDRITE_MAX // NEURON_PARAMS_BIT_LENGTH


def _packbits_ref(bits: np.ndarray, count: Optional[int] = None) -> int:
    """Pack unsigned bits (from LSB to MSB) into a signed integer.

    Args:
        bits (np.ndarray): an array of bits from LSB to MSB(sign bit).
        count (int, optional): `bits` is an N-bit signed integer. If not provided, it is  \
            assumed to be the same as `bits.size`.
    """
    if count is None:
        count = bits.size

    if count == 1:
        return bits[0]

    _bits = np.append(bits[: count - 1], bits[-1])

    result = sum(bit << i for i, bit in enumerate(_bits))
    result -= _bits[-1] << count

    return result


packbits1 = partial(_packbits_ref, count=1)
packbits2 = partial(_packbits_ref, count=2)
packbits4 = partial(_packbits_ref, count=4)
packbits8 = partial(_packbits_ref, count=8)


def _nbit_limit(nbit: int) -> tuple[int, int]:
    hi = 2 if nbit == 1 else 1 << (nbit - 1)
    lo = 0 if nbit == 1 else -hi
    return lo, hi


def test_get_raw_weight(fixed_rng: np.random.Generator):
    w1 = fixed_rng.integers(-128, 128, size=(10, 20), dtype=WEIGHT_DTYPE)
    w2 = fixed_rng.integers(-128, 128, size=(10, 30), dtype=WEIGHT_DTYPE)

    w_of_neurons = [w1, w2]

    n1 = pb.LIF((20,), 1)
    n2 = pb.LIF((30,), 1)

    dest = [n1, n2]

    neuron_segs_of_cb = [
        [
            NeuSegment(n1, slice(0, 20, 1), 0),
            NeuSegment(n2, slice(0, 5, 1), 20),
        ],
        [NeuSegment(n2, slice(5, 30, 1), 0)],
    ]

    w_of_neu_segs_of_cb = []
    for neu_segs in neuron_segs_of_cb:
        w_of_neu_segs = []
        for neu_seg in neu_segs:
            w = w_of_neurons[dest.index(neu_seg.target)][  # type: ignore
                :, neu_seg.index
            ].copy()
            w.setflags(write=False)
            w_of_neu_segs.append(w)

        w_of_neu_segs_of_cb.append(w_of_neu_segs)


def _get_max_fanout(iw: int, dendr_comb_rate: int) -> int:
    if iw == 1:
        return HwConfig.N_DENDRITE_MAX_SNN >> dendr_comb_rate
    else:
        return FANOUT_IW8[dendr_comb_rate]


class TestWeightUnpackAndPack:
    def test_signed_unpackbits(self):
        for wp in WW:
            nbit = 1 << wp
            _low, _high = _nbit_limit(nbit)

            if nbit == 1:
                assert (_low, _high) == (0, 2)
            elif nbit == 2:
                assert (_low, _high) == (-2, 2)
            elif nbit == 4:
                assert (_low, _high) == (-8, 8)
            else:
                assert (_low, _high) == (-128, 128)

            actual_array = np.arange(_low, _high, dtype=np.int8)

            for actual_signed in actual_array:
                unpacked = np.unpackbits(
                    np.uint8(actual_signed), axis=0, count=nbit, bitorder=W_BITORDER
                )
                assert actual_signed == _packbits_ref(unpacked, nbit)

    @pytest.mark.skipif(sys.byteorder != W_BITORDER, reason=f"not {W_BITORDER}-endian")
    def test_uint8_unpackbits_scalar(self):
        x1 = np.int8(101)  # 01100101
        assert x1 == 0b01100101
        x2 = np.int8(-27)  # 11100101

        assert np.uint8(x2) == 0b11100101

        y1 = np.unpackbits(np.uint8(x1), bitorder=W_BITORDER)
        y2 = np.unpackbits(np.uint8(x2), bitorder=W_BITORDER)

        assert np.array_equal(y1, np.array([1, 0, 1, 0, 0, 1, 1, 0], dtype=np.uint8))
        assert np.array_equal(y2, np.array([1, 0, 1, 0, 0, 1, 1, 1], dtype=np.uint8))

    @pytest.mark.parametrize(
        "shape, wp, lcn_ex, iw",
        [
            ((600, 100), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_4X, 1),
            ((1000, 32), WW.WEIGHT_WIDTH_8BIT, LCN_EX.LCN_2X, 1),
            ((120, 800), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_4X, 8),
            ((16, 16), WW.WEIGHT_WIDTH_4BIT, LCN_EX.LCN_2X, 8),
            ((80, 48), WW.WEIGHT_WIDTH_4BIT, LCN_EX.LCN_16X, 8),
            ((100, 510), WW.WEIGHT_WIDTH_8BIT, LCN_EX.LCN_1X, 8),
            ((99, 32), WW.WEIGHT_WIDTH_8BIT, LCN_EX.LCN_2X, 8),
            ((100, 32), WW.WEIGHT_WIDTH_8BIT, LCN_EX.LCN_8X, 8),
        ],
    )
    def test_unpacked_weight_pack(
        self, shape, wp, lcn_ex, iw, fixed_rng: np.random.Generator
    ):
        assert shape[1] <= _get_max_fanout(iw, wp + lcn_ex)

        nbit = 1 << wp
        nfold = 1 << lcn_ex
        _low, _high = _nbit_limit(nbit)
        # Generate the unpacked weight, folded
        test_weight = fixed_rng.integers(_low, _high, size=shape, dtype=WEIGHT_DTYPE)
        w_packed_u64 = self._weight_pack(test_weight, nbit, nfold, iw)

        assert w_packed_u64.shape[0] == WRAM_BASE_SHAPE[1]

    @staticmethod
    def _weight_pack(
        w: WeightType, nbit: int, nfold: int, iw: Literal[1, 8]
    ) -> WRAMPackedType:
        """This prototype function is used to pack the unpacked uint8 weight of size `WRAM_BASE_SHAPE` into \
            a packed uint64 weight of size (WRAM_BASE_SHAPE[1], WRAM_BASE_SHAPE[0]//64).
        """
        wram_base_shape = np.zeros(WRAM_BASE_SHAPE, dtype=WRAM_UNPACKED_DTYPE)

        # -> 1152*512 uint8
        wram_unpacked = TestWeightRamMapping._weight_ram_mapping(w, nbit, nfold, iw)

        wram_base_shape[:, : wram_unpacked.shape[1]] = wram_unpacked

        # -> 512*1152 -> (512*18)*64
        w_unpacked_aligned = wram_base_shape.T.reshape((-1, N_BIT_PACKED_WEIGHT))

        # -> (512*18)*8 uint8
        w_packed_u8 = np.packbits(w_unpacked_aligned, axis=1, bitorder=W_BITORDER)
        assert w_packed_u8.shape[1] == 8

        _n_u64 = WRAM_BASE_SHAPE[0] // N_BIT_PACKED_WEIGHT
        # -> (512*18)*1 uint64 -> 512*18 uint64
        w_packed_u64 = w_packed_u8.view(WRAM_PACKED_DTYPE).reshape((-1, _n_u64))

        # -> 512*144 -> 512*18 uint64
        a = np.packbits(wram_base_shape.T, axis=1, bitorder=W_BITORDER)
        b = np.ascontiguousarray(a).view(WRAM_PACKED_DTYPE)

        # Use the method in the `CorePlacement`, return the weight part only.
        # TODO If everything is OK, just keep this method.
        wram_packed_u64 = CorePlacement._weight_pack(wram_unpacked)

        assert np.array_equal(w_packed_u64, b)
        assert np.array_equal(
            wram_packed_u64, w_packed_u64[: wram_packed_u64.shape[0], :]
        )

        return w_packed_u64


class TestWeightRamMapping:
    @pytest.mark.parametrize("expected_row", [3, 5, 7])
    def test_nfold_weight(self, expected_row):
        """A prototype function of `CorePlacement._nfold_weight` to test the weight folding."""
        original_matrix = np.arange(1, 25, dtype=WEIGHT_DTYPE).reshape(8, 3)
        nfold = 3

        assert nfold <= expected_row
        result = CorePlacement._nfold_weight(original_matrix, expected_row, nfold)

        expected_folded = np.array(
            [
                [1, 10, 19, 2, 11, 20, 3, 12, 21],
                [4, 13, 22, 5, 14, 23, 6, 15, 24],
                [7, 16, 0, 8, 17, 0, 9, 18, 0],
            ],
            dtype=WEIGHT_DTYPE,
        )
        expected = np.pad(expected_folded, ((0, expected_row - nfold), (0, 0)))

        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "shape, wp, lcn_ex",
        [
            ((1200, 200), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_2X),
            ((1000 * 4, 24), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_4X),
            ((1000 * 8, 50), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_8X),
            ((1152 * 2, 120), WW.WEIGHT_WIDTH_2BIT, LCN_EX.LCN_2X),
            ((16, 16), WW.WEIGHT_WIDTH_4BIT, LCN_EX.LCN_2X),
            ((80, 5), WW.WEIGHT_WIDTH_4BIT, LCN_EX.LCN_16X),
            ((800, 60), WW.WEIGHT_WIDTH_8BIT, LCN_EX.LCN_1X),
            ((800, 32), WW.WEIGHT_WIDTH_8BIT, LCN_EX.LCN_2X),
            ((1100 * 8, 8), WW.WEIGHT_WIDTH_8BIT, LCN_EX.LCN_8X),
        ],
    )
    def test_weight_ram_mapping_iw1(
        self, shape, wp, lcn_ex, fixed_rng: np.random.Generator
    ):
        """A prototype function for testing weight RAM mapping for 1-bit input width.

        NOTE: The shape of unpacked weight mapped in WRAM `wram_unpacked` is (1152(WRAM_BASE_SHAPE[0]), x),    \
            where x <= 512 (WRAM_BASE_SHAPE[1]).
        """
        iw = 1
        nbit = 1 << wp
        nfold = 1 << lcn_ex
        # Check the shape[1] is legal
        assert shape[1] <= _get_max_fanout(iw, wp + lcn_ex)

        if shape[0] % nfold > 0:
            expected_h = shape[0] // nfold + 1
        else:
            expected_h = shape[0] // nfold

        expected_shape = (expected_h, shape[1] * nfold)

        # Generate the original weight with shape
        _low, _high = _nbit_limit(nbit)
        test_weight = fixed_rng.integers(_low, _high, size=shape, dtype=WEIGHT_DTYPE)

        # 1. Fold, return the folded weight after padding.
        w_folded = CorePlacement._nfold_weight(test_weight, expected_shape[0], nfold)

        # 2. Map to the WRAM.
        wram_unpacked = np.zeros(WRAM_BASE_SHAPE, dtype=WRAM_UNPACKED_DTYPE)
        wram_weight = self._weight_ram_mapping(w_folded, nbit, nfold, iw)
        wram_unpacked[:, : wram_weight.shape[1]] = wram_weight

        # 3. Check
        self._wram_mapping_check_iw1(test_weight, w_folded, wram_unpacked, nbit, nfold)

    @pytest.mark.parametrize(
        "shape, wp, lcn_ex",
        [
            # E*W < 8
            ((240, 1200), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_2X),
            ((500, 800), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_4X),
            ((200, 800), WW.WEIGHT_WIDTH_2BIT, LCN_EX.LCN_2X),
            ((144, 876), WW.WEIGHT_WIDTH_4BIT, LCN_EX.LCN_1X),
            # E*W >= 8
            ((30, 30), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_8X),
            ((2200, 100), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_16X),
            ((30, 24), WW.WEIGHT_WIDTH_2BIT, LCN_EX.LCN_4X),
            ((100, 15), WW.WEIGHT_WIDTH_2BIT, LCN_EX.LCN_8X),
            ((30, 24), WW.WEIGHT_WIDTH_4BIT, LCN_EX.LCN_2X),
            ((550, 40), WW.WEIGHT_WIDTH_4BIT, LCN_EX.LCN_4X),
            ((1001, 100), WW.WEIGHT_WIDTH_4BIT, LCN_EX.LCN_8X),
            ((30, 24), WW.WEIGHT_WIDTH_8BIT, LCN_EX.LCN_1X),
            ((200, 100), WW.WEIGHT_WIDTH_8BIT, LCN_EX.LCN_2X),
            ((480, 100), WW.WEIGHT_WIDTH_8BIT, LCN_EX.LCN_4X),
            ((4200, 8), WW.WEIGHT_WIDTH_8BIT, LCN_EX.LCN_32X),
        ],
    )
    def test_weight_ram_mapping_iw8(
        self, shape, wp, lcn_ex, fixed_rng: np.random.Generator
    ):
        """A prototype function for testing weight RAM mapping for 8-bit input width.

        NOTE: The shape of unpacked weight mapped in WRAM `wram_unpacked` is (1152(WRAM_BASE_SHAPE[0]), x),    \
            where x <= 512 (WRAM_BASE_SHAPE[1]).
        """
        iw = 8
        nbit = 1 << wp
        nfold = 1 << lcn_ex
        # Check the shape[1] is legal
        assert shape[1] <= _get_max_fanout(iw, wp + lcn_ex)

        if shape[0] % nfold > 0:
            expected_h = shape[0] // nfold + 1
        else:
            expected_h = shape[0] // nfold

        expected_shape = (expected_h, shape[1] * nfold)

        # Generate the original weight with shape
        _low, _high = _nbit_limit(nbit)
        test_weight = fixed_rng.integers(_low, _high, size=shape, dtype=WEIGHT_DTYPE)

        # 1. Fold, return the folded weight after padding.
        w_folded = CorePlacement._nfold_weight(test_weight, expected_shape[0], nfold)

        # 2. Map to the NRAM.
        # (1152, 512)
        wram_unpacked_total = np.zeros(WRAM_BASE_SHAPE, dtype=WRAM_UNPACKED_DTYPE)
        wram_weight_unpacked = self._weight_ram_mapping(w_folded, nbit, nfold, iw)
        wram_unpacked_total[:, : wram_weight_unpacked.shape[1]] = wram_weight_unpacked

        n_col_used_total = wram_weight_unpacked.shape[1]
        wram_nparams_unpacked = None
        # NOTE: While mapping extra neuron parameters to the WRAM occurs
        # during the configuration frame export phase, it is tested here.
        if (n_extra_neurons := shape[1] - WRAM_BASE_SHAPE[1]) > 0:
            wram_nparams_unpacked = self._gen_wram_for_neurons(
                n_extra_neurons, wp, lcn_ex
            )

            n_col_used_total += wram_nparams_unpacked.shape[1]
            wram_unpacked_total[:, wram_weight_unpacked.shape[1] : n_col_used_total] = (
                wram_nparams_unpacked
            )

            # Get the used columns of wram_unpacked_total after `np.hstack`.
            wram_unpacked_total2 = np.hstack(
                [wram_weight_unpacked, wram_nparams_unpacked]
            )
            assert np.array_equal(
                wram_unpacked_total[:, :n_col_used_total], wram_unpacked_total2
            )

        # 3. Check
        assert n_col_used_total <= WRAM_BASE_SHAPE[1]
        self._wram_mapping_check_iw8(
            test_weight, w_folded, wram_unpacked_total, nbit, nfold
        )

        # 4. Pack
        wram_weight_packed = CorePlacement._weight_pack(wram_weight_unpacked)
        assert wram_weight_packed.shape[1] <= WRAM_BASE_SHAPE[1]

        if wram_nparams_unpacked is not None:
            wram_nparams_packed = CorePlacement._weight_pack(wram_nparams_unpacked)
            assert wram_nparams_packed.shape[1] <= WRAM_BASE_SHAPE[1]

    @staticmethod
    def _weight_ram_mapping(
        w_folded: WeightType, n_bit: int, n_fold: int, iw: Literal[1, 8]
    ) -> WRAMUnpackedType:
        if iw == 1:
            # The length of slot for each bit of input data
            bit_slot_length = HwConfig.N_FANIN_PER_DENDRITE_SNN
        else:
            # N_FANIN_PER_DENDRITE_SNN // iw
            bit_slot_length = HwConfig.N_FANIN_PER_DENDRITE_ANN

        folded_row, folded_col = w_folded.shape
        n_dendrite_comb = n_bit * n_fold
        # oc * e / (8/w) = oc * d / 8
        orig_col = folded_col // n_fold
        result_col = math.ceil(orig_col * n_dendrite_comb / iw)
        # Units are divided into small blocks of columns, fan-in extension
        # (oc, lcn, nbit, 144 or 1152)
        cew_block = np.zeros(
            (orig_col, n_fold, n_bit, bit_slot_length), dtype=WRAM_UNPACKED_DTYPE
        )
        # [N*M] -> [M*N*1]
        w_folded_3d = np.expand_dims(w_folded.T, axis=2).view(WRAM_UNPACKED_DTYPE)
        for c in range(orig_col):
            for lcn in range(n_fold):
                # Unpack the array [N*1] -> [N*n_bit], LSB->MSB
                unpacked = np.unpackbits(
                    w_folded_3d[c * n_fold + lcn, :, :],
                    axis=1,
                    count=n_bit,
                    bitorder=W_BITORDER,
                )

                for bit in range(n_bit):
                    cew_block[c, lcn, bit, :folded_row] = unpacked[:, bit].squeeze()

        if n_dendrite_comb >= iw:  # For SNN mode, it must go into this case
            # At least 1 fan-in is required to be combined in one column
            result = cew_block.reshape((result_col, -1)).T
        else:
            # 2/4/8 original columns are combined in one column
            n_col_comb_in_col = iw // n_dendrite_comb
            cew_block = cew_block.reshape((orig_col, -1))

            if (r := orig_col % n_col_comb_in_col) > 0:
                cew_block = np.pad(cew_block, ((0, n_col_comb_in_col - r), (0, 0)))

            # Now, length of padded columns is a multiple of 'n_col_comb_in_col'
            assert cew_block.shape[0] % n_col_comb_in_col == 0
            result = cew_block.reshape((cew_block.shape[0] // n_col_comb_in_col, -1)).T

            # For n_dendrite_comb = 1, the #C columns of result <= FANOUT_IW8[0]/8
            # For n_dendrite_comb = 2, #C <= FANOUT_IW8[1]/4
            # For n_dendrite_comb = 4, #C <= FANOUT_IW8[2]/2
            assert (
                result.shape[1]
                <= FANOUT_IW8[n_dendrite_comb.bit_length() - 1] // n_col_comb_in_col
            )

        assert np.max(result, axis=None) <= 1
        assert np.min(result, axis=None) >= 0

        return result

    @pytest.mark.parametrize(
        "shape, wp, lcn_ex, expectation",
        [
            # E*W=1
            ((120, 1888), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_1X, nullcontext()),
            (
                (120, 1889),
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                pytest.raises(AssertionError),
            ),
            # E*W=2
            ((288, 1364), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_2X, nullcontext()),
            (
                (144, 1365),
                WW.WEIGHT_WIDTH_2BIT,
                LCN_EX.LCN_1X,
                pytest.raises(AssertionError),
            ),
            # E*W=4
            ((144 * 4, 876), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_4X, nullcontext()),
            (
                (144, 877),
                WW.WEIGHT_WIDTH_4BIT,
                LCN_EX.LCN_1X,
                pytest.raises(AssertionError),
            ),
            ((120, 876), WW.WEIGHT_WIDTH_4BIT, LCN_EX.LCN_1X, nullcontext()),
            (
                (240, 877),
                WW.WEIGHT_WIDTH_2BIT,
                LCN_EX.LCN_2X,
                pytest.raises(AssertionError),
            ),
        ],
    )
    def test_weight_ram_mapping_neurons_limit(
        self, shape, wp, lcn_ex, expectation, fixed_rng: np.random.Generator
    ):
        """Test cases about neurons limit, only for 8-bit input width & #N of combined dendrites < 8."""
        assert wp + lcn_ex <= 2
        iw = 8
        nbit = 1 << wp
        nfold = 1 << lcn_ex

        if shape[0] % nfold > 0:
            expected_h = shape[0] // nfold + 1
        else:
            expected_h = shape[0] // nfold

        expected_shape = (expected_h, shape[1] * nfold)

        # Generate the original weight with shape
        _low, _high = _nbit_limit(nbit)
        test_weight = fixed_rng.integers(_low, _high, size=shape, dtype=WEIGHT_DTYPE)

        # 1. Fold, return the folded weight after padding.
        w_folded = CorePlacement._nfold_weight(test_weight, expected_shape[0], nfold)

        # 2. Map to the NRAM.
        with expectation:
            w_mapped = self._weight_ram_mapping(w_folded, nbit, nfold, iw)

    @staticmethod
    def _weight_ram_mapping_iw8(
        folded_weights: WeightType,
        n_bit: int,
        n_fold: int,
        wbit_slot_length: int = HwConfig.N_FANIN_PER_DENDRITE_ANN,
    ):
        """A prototype function for weight ram mapping for 8-bit input width."""
        row, col = folded_weights.shape
        orig_col = col // n_fold
        _n_block_in_row = 8  # iw = 8
        dendrite_comb_rate = n_bit * n_fold
        # oc * e / (8/w) = oc * d / 8
        result_col = math.ceil(orig_col * dendrite_comb_rate / _n_block_in_row)
        result = np.zeros(
            (_n_block_in_row * wbit_slot_length, result_col), dtype=np.uint8
        )
        # Units are divided into small blocks of columns, fan-in extension
        # Each block contains N-bits * 144 (slot length)
        cew_block = np.zeros(
            (orig_col, n_fold, n_bit, wbit_slot_length), dtype=np.uint8
        )
        # [N*M] -> [M*N*1]
        folded_weights_3d = np.expand_dims(folded_weights.T, axis=2).view(np.uint8)

        for c in range(orig_col):
            for lcn in range(n_fold):
                # For every m in M, unpack the array [N*1] -> [N*8]
                # [0,:] -> [row,:]: A[0] -> A[row-1]
                # [:,0] -> [:,7]: LSB->MSB
                unpacked = np.unpackbits(
                    folded_weights_3d[c * n_fold + lcn, :, :],
                    axis=1,
                    count=n_bit,
                    bitorder=W_BITORDER,
                )

                for bit in range(n_bit):
                    cew_block[c, lcn, bit, :row] = unpacked[:, bit].squeeze()

        # if n_bit < 8:
        #     if dendrite_comb_rate > _n_block_in_row:  # W<8, E*W>8
        #         # How many fan-ins are combined in one column
        #         n_lcn_comb_in_col = _n_block_in_row // n_bit  # <n_fold
        #         # For all fan-ins on the original column, how many columns are needed to accommodate
        #         n_col_lcn_accom, r = divmod(n_fold, n_lcn_comb_in_col)
        #         assert r == 0
        #         result3 = cew_block.reshape((result_col, -1)).T
        #         cew_block = cew_block.reshape((orig_col, n_col_lcn_accom, -1))

        #         for c, l in np.ndindex(cew_block.shape[:2]):
        #             result[:, c * n_col_lcn_accom + l] = cew_block[c, l, :].ravel()

        #         result2 = cew_block.reshape((result_col, -1)).T
        #         assert np.array_equal(result, result2)
        #         assert np.array_equal(result, result3)
        #     else:  # W<8, E*W<=8
        #         # How many original columns are combined in one column
        #         n_col_comb_in_col = _n_block_in_row // dendrite_comb_rate  # 1 < x <= 8
        #         cew_block = cew_block.reshape((orig_col, -1))

        #         for c in range(cew_block.shape[0]):
        #             col_idx, row_idx = divmod(c, n_col_comb_in_col)
        #             result[
        #                 row_idx
        #                 * cew_block.shape[-1] : (row_idx + 1)
        #                 * cew_block.shape[-1],
        #                 col_idx,
        #             ] = cew_block[c, :].ravel()

        #         if (r := orig_col % n_col_comb_in_col) > 0:
        #             cew_block = np.pad(cew_block, ((0, n_col_comb_in_col - r), (0, 0)))

        #         # Now, length of padded columns is a multiple of 'n_col_comb_in_col'
        #         assert cew_block.shape[0] % n_col_comb_in_col == 0
        #         result2 = cew_block.reshape(
        #             (cew_block.shape[0] // n_col_comb_in_col, -1)
        #         ).T
        #         assert np.array_equal(result, result2)
        # else:  # W=8, EW>=8
        #     result2 = cew_block.reshape((result_col, -1)).T
        #     cew_block = cew_block.reshape((orig_col, n_fold, -1))
        #     result = cew_block.reshape((orig_col * n_fold, -1)).T

        #     assert np.array_equal(result, result2)

        if dendrite_comb_rate >= _n_block_in_row:
            # At least 1 fan-in is required to be combined in one column
            result999 = cew_block.reshape((result_col, -1)).T
        else:
            # 2/4/8 original columns are combined in one column
            n_col_comb_in_col = _n_block_in_row // dendrite_comb_rate
            cew_block = cew_block.reshape((orig_col, -1))

            if (r := orig_col % n_col_comb_in_col) > 0:
                cew_block = np.pad(cew_block, ((0, n_col_comb_in_col - r), (0, 0)))

            # Now, length of padded columns is a multiple of 'n_col_comb_in_col'
            assert cew_block.shape[0] % n_col_comb_in_col == 0
            result999 = cew_block.reshape(
                (cew_block.shape[0] // n_col_comb_in_col, -1)
            ).T

        # assert np.max(result, axis=None) <= 1
        # assert np.min(result, axis=None) >= 0

        assert np.max(result999, axis=None) <= 1
        assert np.min(result999, axis=None) >= 0

        return result

    # at commit 67054d8
    @staticmethod
    def _weight_ram_mapping_iw1_old(folded_weights: np.ndarray, n_bit: int):
        """Old weight ram mapping for 1-bit input width."""
        row, col = folded_weights.shape
        result = np.zeros((row, col * n_bit), dtype=np.uint8)

        # [N*M] -> [M*N*1]
        folded_weights_3d = np.expand_dims(folded_weights.T, axis=2).astype(np.uint8)

        for i in range(col):
            # For every m in M, unpack the array [N*1] -> [N*8]
            unpacked = np.unpackbits(
                folded_weights_3d[i], axis=1, count=n_bit, bitorder=W_BITORDER
            )

            result[:, n_bit * i : n_bit * (i + 1)] = unpacked

        assert np.max(result, axis=None) <= 1
        assert np.min(result, axis=None) >= 0

        return result

    @staticmethod
    def _wram_mapping_check_iw1(
        test_data: WeightType,
        w_folded: WeightType,
        w_unpacked: WRAMUnpackedType,
        nbit: int,
        nfold: int,
    ) -> None:
        n_in_col = w_folded.shape[0]
        for i, j in np.ndindex(test_data.shape):
            offset_j, now_i = divmod(i, n_in_col)
            now_j = offset_j + j * nfold

            wij = w_unpacked[now_i, now_j * nbit : (now_j + 1) * nbit]
            wij_packed = _packbits_ref(wij, nbit)

            assert test_data[i, j] == wij_packed

    @staticmethod
    def _wram_mapping_check_iw8(
        test_data: WeightType,
        w_folded: WeightType,
        w_unpacked: WRAMUnpackedType,
        nbit: int,
        nfold: int,
    ) -> None:
        n_in_col = w_folded.shape[0]
        n_lcn_in_col = 8 // nbit  # The amount of lcn in one coloumn

        for i, j in np.ndindex(test_data.shape):
            # Get the coordinate (i_folded, j_folded) in the folded weight
            offset_j, i_folded = divmod(i, n_in_col)
            j_folded = offset_j + j * nfold
            # Get the index of E-block
            e_j, e_i = divmod(j_folded, n_lcn_in_col)
            # Just get `nbit` bits
            wij = w_unpacked[i_folded :: HwConfig.N_FANIN_PER_DENDRITE_ANN, e_j][
                e_i * nbit : (e_i + 1) * nbit
            ]
            wij_packed = _packbits_ref(wij, nbit)

            assert test_data[i, j] == wij_packed

    @pytest.mark.parametrize(
        "shape, wp, lcn_ex",
        [
            # E*W < 8
            ((240, 1200), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_2X),
            ((500, 800), WW.WEIGHT_WIDTH_1BIT, LCN_EX.LCN_4X),
            ((200, 800), WW.WEIGHT_WIDTH_2BIT, LCN_EX.LCN_2X),
            ((200, 811), WW.WEIGHT_WIDTH_2BIT, LCN_EX.LCN_2X),
            ((144, 876), WW.WEIGHT_WIDTH_4BIT, LCN_EX.LCN_1X),
        ],
    )
    def test_weight_ram_mapping_for_neurons(self, shape, wp, lcn_ex):
        """This test is for extra neurons parameters mapping on the WRAM for 8-bit input width."""
        assert wp + lcn_ex <= 2

        n_extra_neurons = shape[1] - WRAM_BASE_SHAPE[1]
        wram_neurons = self._gen_wram_for_neurons(n_extra_neurons, wp, lcn_ex)

    @staticmethod
    def _gen_wram_for_neurons(
        n_extra_neurons: int, wp: WW, lcn_ex: LCN_EX
    ) -> WRAMUnpackedType:
        """A prototype function for mapping extra neurons parameters on the WRAM for 8-bit input width.

        NOTE: The shape of final result is (1152(WRAM_BASE_SHAPE[0]), x), where x <= 512 (WRAM_BASE_SHAPE[1]).
        """
        extra_neurons = pb.ANNNeuron(n_extra_neurons, bit_trunc=15)
        # extra_neurons = pb.ANNBypassNeuron(n_extra_neurons)
        dest_info = _gen_random_neuron_dest_info(n_extra_neurons)

        # TODO Current APIs are not enough to generate the parameters of neurons directly.
        frame3 = OfflineFrameGen.gen_config_frame3(
            Coord(31, 31),
            Coord(0, 0),
            RId(0, 0),
            0,
            n_extra_neurons,
            NeuronAttrs.model_validate(extra_neurons.attrs(all=False)),
            dest_info,
            1,
        )

        neuron_params_214b = np.zeros(
            (n_extra_neurons, NEURON_PARAMS_BIT_LENGTH), dtype=WRAM_UNPACKED_DTYPE
        )

        for i in range(n_extra_neurons):
            # A neuron's parameters are packed in 4 single packages
            params = frame3.packages[i * 4 : (i + 1) * 4]
            # [0:NEURON_PARAMS_BIT_LENGTH]:LSB to MSB + [NEURON_PARAMS_BIT_LENGTH:]:0
            neuron_params_214b[i, :] = np.unpackbits(
                params.view(WRAM_UNPACKED_DTYPE), axis=0, bitorder=W_BITORDER
            )[:NEURON_PARAMS_BIT_LENGTH]

        n_col_avail = math.ceil(
            (_get_max_fanout(8, wp + lcn_ex) - WRAM_BASE_SHAPE[1])
            / N_NEURON_PARAM_IN_COL
        )
        # Slow method
        # TODO Remove it if allright
        wram_neurons_slow = np.zeros(
            (WRAM_BASE_SHAPE[0], n_col_avail), dtype=WRAM_UNPACKED_DTYPE
        )
        for i in range(n_extra_neurons):
            idx_col, idx_in_col = divmod(i, N_NEURON_PARAM_IN_COL)
            wram_neurons_slow[
                idx_in_col
                * NEURON_PARAMS_BIT_LENGTH : (idx_in_col + 1)
                * NEURON_PARAMS_BIT_LENGTH,
                idx_col,
            ] = neuron_params_214b[i, :].squeeze()
        # Slow method ends

        # Pad the row of neuron parameters to a multiple of `N_NEURON_PARAM_IN_COL`
        if (r := neuron_params_214b.shape[0] % N_NEURON_PARAM_IN_COL) > 0:
            neuron_params_214b = np.pad(
                neuron_params_214b, ((0, N_NEURON_PARAM_IN_COL - r), (0, 0))
            )

        n_bit_nparams = NEURON_PARAMS_BIT_LENGTH * N_NEURON_PARAM_IN_COL

        neuron_params_214b = neuron_params_214b.reshape((-1, n_bit_nparams))
        _n_col_occupied = neuron_params_214b.shape[0]

        result = np.zeros((WRAM_BASE_SHAPE[0], n_col_avail), dtype=WRAM_UNPACKED_DTYPE)
        result[:n_bit_nparams, :_n_col_occupied] = neuron_params_214b.T

        assert np.array_equal(result, wram_neurons_slow)

        return result

    def test_weight_ram_mapping_8bits(self):
        binary_conn = np.zeros((6, 8 * 5), dtype=np.bool_)
        wp = WW.WEIGHT_WIDTH_8BIT

        array = np.random.randint(-128, 128, size=(4, 4), dtype=WEIGHT_DTYPE)

        y = np.unpackbits(np.uint8(array), axis=1, bitorder=W_BITORDER)
        assert y.shape == (4, (1 << wp) * 4)

        binary_conn[: y.shape[0], : y.shape[1]] = y

        for i, j in np.ndindex((4, 4)):
            expected = array[i, j]
            wij = y[i, j * (1 << wp) : (j + 1) * (1 << wp)]
            r = packbits8(wij)

            assert expected == r

    def test_weight_ram_mapping_4bits(self):
        binary_conn = np.zeros((6, 4 * 5), dtype=np.bool_)
        wp = WW.WEIGHT_WIDTH_4BIT

        array = np.random.randint(-8, 8, size=(4, 4), dtype=WEIGHT_DTYPE)
        y = np.zeros((4, 16), dtype=np.uint8)

        for i in range(4):
            ual = np.uint8(np.expand_dims(array[:, i], axis=1))
            a = np.unpackbits(ual, axis=1, count=4, bitorder=W_BITORDER)
            y[: a.shape[0], (1 << wp) * i : (1 << wp) * (i + 1)] = a

        assert y.shape == (4, (1 << wp) * 4)

        binary_conn[: y.shape[0], : y.shape[1]] = y

        for i, j in np.ndindex(array.shape):
            expected = array[i, j]
            wij = y[i, j * (1 << wp) : (j + 1) * (1 << wp)]
            r = packbits4(wij)

            assert expected == r

    def test_weight_ram_mapping_2bits(self):
        binary_conn = np.zeros((6, 4 * 5), dtype=np.bool_)
        wp = WW.WEIGHT_WIDTH_2BIT

        array = np.random.randint(-2, 2, size=(4, 4), dtype=WEIGHT_DTYPE)
        y = np.zeros((4, 8), dtype=np.uint8)

        for i in range(4):
            ual = np.uint8(np.expand_dims(array[:, i], axis=1))
            a = np.unpackbits(ual, axis=1, count=2, bitorder=W_BITORDER)
            y[: a.shape[0], (1 << wp) * i : (1 << wp) * (i + 1)] = a

        assert y.shape == (4, (1 << wp) * 4)

        binary_conn[: y.shape[0], : y.shape[1]] = y

        for i, j in np.ndindex(array.shape):
            expected = array[i, j]
            wij = y[i, j * (1 << wp) : (j + 1) * (1 << wp)]
            r = packbits2(wij)

            assert expected == r


def test_n_axon2lcn_ex():
    from .conftest import n_axon2lcn_ex_proto

    lcn_ex = n_axon2lcn_ex_proto(
        HwConfig.N_FANIN_PER_DENDRITE_SNN * 18 + 1, HwConfig.N_FANIN_PER_DENDRITE_SNN
    )
    assert lcn_ex == LCN_EX.LCN_32X

    lcn_ex = n_axon2lcn_ex_proto(
        HwConfig.N_FANIN_PER_DENDRITE_ANN * 3 + 20, HwConfig.N_FANIN_PER_DENDRITE_ANN
    )
    assert lcn_ex == LCN_EX.LCN_4X

    with pytest.raises(ValueError):
        lcn_ex = n_axon2lcn_ex_proto(0, HwConfig.N_FANIN_PER_DENDRITE_SNN)

    with pytest.raises(ResourceError):
        lcn_ex = n_axon2lcn_ex_proto(
            HwConfig.N_FANIN_PER_DENDRITE_SNN << LCN_EX.LCN_64X + 1,
            HwConfig.N_FANIN_PER_DENDRITE_SNN,
        )
