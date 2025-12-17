import math
from typing import Literal

import numpy as np
import pytest
from paicorelib import LCN_EX, HwConfig
from paicorelib import WeightWidth as WW

from paibox.backend.placement import FANOUT_IW8
from paibox.backend.types import WRAM_UNPACKED_DTYPE, WRAMUnpackedType
from paibox.types import WEIGHT_DTYPE, WeightType

fixed_rng = np.random.default_rng(42)


def packbits_ref(bits: np.ndarray, count: int | None = None) -> int:
    """Pack unsigned bits (from LSB to MSB) into a signed integer.

    Args:
        - bits: an array of bits from LSB to MSB(sign bit).
        - count: `bits` is an N-bit signed integer. If not provided, it is  \
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


def _nbit_limit(nbit: int) -> tuple[int, int]:
    hi = 2 if nbit == 1 else 1 << (nbit - 1)
    lo = 0 if nbit == 1 else -hi
    return lo, hi


class TestWeightRamMapping:
    @staticmethod
    def _get_max_fanout(iw: int, dendr_comb_rate: int) -> int:
        if iw == 1:
            return HwConfig.N_DENDRITE_MAX_SNN >> dendr_comb_rate
        else:
            return FANOUT_IW8[dendr_comb_rate]

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
    def test_weight_ram_mapping_iw1(self, shape, wp, lcn_ex):
        """A prototype function for testing weight RAM mapping for 1-bit input width.

        NOTE: The shape of unpacked weight mapped in WRAM `w_unpacked` is (1152, x), where x <= 512.
        """
        iw = 1
        # Check the shape[1] is legal
        assert shape[1] <= self._get_max_fanout(iw, wp + lcn_ex)

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
        w_folded = self._fold_raw_weight_single(test_weight, expected_shape[0], nfold)

        # 2. Map to the WRAM.
        wram_unpacked = np.zeros((1152, 512), dtype=WRAM_UNPACKED_DTYPE)
        w_mapped = self._weight_ram_mapping(w_folded, nbit, nfold, shape[1], iw)
        wram_unpacked[:, : w_mapped.shape[1]] = w_mapped

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
    def test_weight_ram_mapping_iw8(self, shape, wp, lcn_ex):
        """Only mapping for weight. Extra neurons are NOT included."""
        iw = 8
        # Check the shape[1] is legal
        assert shape[1] <= self._get_max_fanout(iw, wp + lcn_ex)

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
        w_folded = self._fold_raw_weight_single(test_weight, expected_shape[0], nfold)

        # 2. Map to the NRAM.
        wram_unpacked = np.zeros((1152, 512), dtype=WRAM_UNPACKED_DTYPE)
        w_mapped = self._weight_ram_mapping(w_folded, nbit, nfold, shape[1], iw)
        wram_unpacked[:, : w_mapped.shape[1]] = w_mapped

        # Extra neurons part XXX don't do it now
        if (n_extra_neuron := shape[1] - HwConfig.ADDR_RAM_MAX + 1) > 0:
            pass

        # TODO how to check

    @staticmethod
    def _weight_ram_mapping(
        folded_weights: WeightType,
        n_bit: int,
        n_fold: int,
        orig_col: int,
        iw: Literal[1, 8],
    ):
        if iw == 1:
            # The length of slot for each bit of input data
            bit_slot_length = HwConfig.N_FANIN_PER_DENDRITE_SNN
        else:
            # N_FANIN_PER_DENDRITE_SNN // iw
            bit_slot_length = HwConfig.N_FANIN_PER_DENDRITE_ANN

        folded_row, _ = folded_weights.shape
        n_dendrite_comb = n_bit * n_fold
        # oc * e / (8/w) = oc * d / 8
        result_col = math.ceil(orig_col * n_dendrite_comb / iw)
        # Units are divided into small blocks of columns, fan-in extension
        # (oc, lcn, nbit, 144/1152)
        cew_block = np.zeros((orig_col, n_fold, n_bit, bit_slot_length), dtype=np.uint8)
        # [N*M] -> [M*N*1]
        folded_weights_3d = np.expand_dims(folded_weights.T, axis=2).astype(np.uint8)
        for c in range(orig_col):
            for lcn in range(n_fold):
                # Unpack the array [N*1] -> [N*8]
                # [0, :]-> [folded_row, :]: A[0] -> A[folded_row-1]
                # [:, 0]->[:,7]: LSB->MSB
                unpacked = np.unpackbits(
                    folded_weights_3d[c * n_fold + lcn, :, :],
                    axis=1,
                    count=n_bit,
                    bitorder="little",
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
        folded_weights_3d = np.expand_dims(folded_weights.T, axis=2).astype(np.uint8)

        for c in range(orig_col):
            for lcn in range(n_fold):
                # For every m in M, unpack the array [N*1] -> [N*8]
                # [0, :]-> [row, :]: A[0] -> A[row-1]
                # [:, 0]->[:,7]: LSB->MSB
                unpacked = np.unpackbits(
                    folded_weights_3d[c * n_fold + lcn, :, :],
                    axis=1,
                    count=n_bit,
                    bitorder="little",
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

    @staticmethod
    def _fold_raw_weight_single(raw_weight: WeightType, expected_row: int, nfold: int):
        raw_row, raw_col = raw_weight.shape

        if (r := raw_row % nfold) > 0:
            _padding = nfold - r
            assert expected_row * nfold == raw_row + _padding

            w_padding = np.pad(raw_weight, ((0, _padding), (0, 0)))
        else:
            w_padding = raw_weight

        split = np.vsplit(w_padding, nfold)
        w_folded = np.zeros((expected_row, raw_col * nfold), dtype=WEIGHT_DTYPE)

        for i, j in np.ndindex((nfold, raw_col)):
            w_col = split[i][:, j]
            w_folded[:, j * nfold + i] = w_col

        return w_folded

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
                folded_weights_3d[i], axis=1, count=n_bit, bitorder="little"
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
        for i, j in np.ndindex(test_data.shape):
            n_in_col = w_folded.shape[0]
            offset_j, now_i = divmod(i, n_in_col)
            now_j = offset_j + j * nfold

            wij = w_unpacked[now_i, now_j * nbit : (now_j + 1) * nbit]

            wij_packed = packbits_ref(wij, nbit)
            assert test_data[i, j] == wij_packed

    @staticmethod
    def _wram_mapping_check_iw8(
        test_data: WeightType,
        w_folded: WeightType,
        w_unpacked: WRAMUnpackedType,
        nbit: int,
        nfold: int,
    ) -> None:
        pass
