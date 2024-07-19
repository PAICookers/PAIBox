import numpy as np
import pytest
from paicorelib import HwConfig, LCN_EX
from paicorelib import WeightWidth as WW

import paibox as pb
from paibox.backend.placement import CorePlacement
from paibox.backend.types import NeuSegment, WRAMUnpackedType, WRAM_PACKED_DTYPE
from paibox.exceptions import ResourceError
from paibox.types import WEIGHT_DTYPE, WeightType


def packbits_ref(bits: np.ndarray, count: int) -> int:
    """Pack unsigned bits into a signed integer.

    This is a test of the prototype of the original function.
    """
    _bits = np.append(bits[: count - 1], bits[-1])

    result = sum(bit << i for i, bit in enumerate(_bits))
    result -= _bits[-1] << count

    return result


def test_get_raw_weight_ref(random_fixture):
    w1 = np.random.randint(-128, 128, size=(10, 20), dtype=WEIGHT_DTYPE)
    w2 = np.random.randint(-128, 128, size=(10, 30), dtype=WEIGHT_DTYPE)

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


@pytest.mark.parametrize(
    "input, n_col_groups, expected",
    [
        (
            np.arange(1, 17, dtype=np.int8).reshape(8, 2),
            2,
            np.array(
                [
                    [1, 13, 2, 14],
                    [3, 15, 4, 16],
                    [5, 0, 6, 0],
                    [7, 0, 8, 0],
                    [9, 0, 10, 0],
                    [11, 0, 12, 0],
                ],
                dtype=np.int8,
            ),
        ),
        (
            np.arange(1, 13, dtype=np.int8).reshape(6, 2),
            3,
            np.array([[1, 5, 9, 2, 6, 10], [3, 7, 11, 4, 8, 12]], dtype=np.int8),
        ),
        (
            np.arange(1, 25, dtype=np.int8).reshape(8, 3),
            3,
            np.array(
                [
                    [1, 10, 19, 2, 11, 20, 3, 12, 21],
                    [4, 13, 22, 5, 14, 23, 6, 15, 24],
                    [7, 16, 0, 8, 17, 0, 9, 18, 0],
                ],
                dtype=np.int8,
            ),
        ),
    ],
)
def test_weight_ram_mapping(input, n_col_groups, expected):
    """Convert a weight matirx into a standard binary connectivity.

    This is a test of the prototype of the original function.
    """
    cur_shape = input.shape
    row, _ = expected.shape
    o_matrix = np.zeros(expected.shape, dtype=np.int8)

    for i in range(cur_shape[1]):
        w_col = input[:, i]
        col_group = 0

        while (n_rest_axon := cur_shape[0] - row * col_group) > row:
            o_matrix[:, n_col_groups * i + col_group] = w_col[
                row * col_group : row * (col_group + 1)
            ]
            col_group += 1

        o_matrix[:, n_col_groups * i + col_group] = np.pad(
            w_col[row * col_group :],
            pad_width=(0, row - n_rest_axon),
            mode="constant",
            constant_values=0,
        )

    assert np.array_equal(o_matrix, expected)


def test_nfold_weight_ref():
    original_matrix = np.arange(1, 25, dtype=WEIGHT_DTYPE).reshape(8, 3)
    nfold = 3

    if original_matrix.shape[0] % nfold > 0:
        _padding = nfold - original_matrix.shape[0] % nfold
        w_padding = np.append(
            original_matrix,
            values=np.zeros((_padding, original_matrix.shape[1]), dtype=WEIGHT_DTYPE),
            axis=0,
        )
    else:
        w_padding = original_matrix

    split = np.vsplit(w_padding, nfold)

    result = np.zeros(
        (w_padding.shape[0] // nfold, original_matrix.shape[1] * nfold),
        dtype=WEIGHT_DTYPE,
    )

    for i, j in np.ndindex((nfold, original_matrix.shape[1])):
        g = split[i][:, j]
        result[:, j * nfold + i] = g

    assert np.array_equal(
        result,
        np.array(
            [
                [1, 10, 19, 2, 11, 20, 3, 12, 21],
                [4, 13, 22, 5, 14, 23, 6, 15, 24],
                [7, 16, 0, 8, 17, 0, 9, 18, 0],
            ],
            dtype=WEIGHT_DTYPE,
        ),
    )


class TestWeightUnpack:
    @pytest.mark.parametrize(
        "wp",
        [
            WW.WEIGHT_WIDTH_8BIT,
            WW.WEIGHT_WIDTH_4BIT,
            WW.WEIGHT_WIDTH_2BIT,
            WW.WEIGHT_WIDTH_1BIT,
        ],
    )
    def test_signed_unpackbits(self, wp):
        count = 1 << wp
        actual_array = np.arange(-(1 << (count - 1)), (1 << (count - 1)), dtype=np.int8)

        for actual_signed in actual_array:
            unpacked = np.unpackbits(
                np.uint8(actual_signed), axis=0, count=count, bitorder="little"
            )
            assert actual_signed == packbits_ref(unpacked, count)

    def test_uint8_unpackbits_scalar(self):
        import sys

        # Little endian on x86_64
        assert sys.byteorder == "little"

        x1 = np.int8(101)  # 01100101
        assert x1 == 0b01100101
        x2 = np.int8(-27)  # 11100101

        assert np.uint8(x2) == 0b11100101

        y1 = np.unpackbits(np.uint8(x1), bitorder="little")
        y2 = np.unpackbits(np.uint8(x2), bitorder="little")

        assert np.array_equal(y1, np.array([1, 0, 1, 0, 0, 1, 1, 0], dtype=np.uint8))
        assert np.array_equal(y2, np.array([1, 0, 1, 0, 0, 1, 1, 1], dtype=np.uint8))

    @pytest.mark.parametrize(
        "shape, wp, nfold, is_iw8",
        [
            ((8, 8), WW.WEIGHT_WIDTH_8BIT, 2, False),
            ((32, 32), WW.WEIGHT_WIDTH_8BIT, 2, False),
            ((16, 16), WW.WEIGHT_WIDTH_4BIT, 4, False),
            ((30, 24), WW.WEIGHT_WIDTH_4BIT, 4, False),
            ((32, 24), WW.WEIGHT_WIDTH_2BIT, 3, False),
            ((32, 24), WW.WEIGHT_WIDTH_1BIT, 3, False),
            ((31, 23), WW.WEIGHT_WIDTH_8BIT, 5, False),
            ((1200, 200), WW.WEIGHT_WIDTH_1BIT, 2, False),
            ((800, 64), WW.WEIGHT_WIDTH_8BIT, 2, False),
            ((8, 8), WW.WEIGHT_WIDTH_8BIT, 2, True),
            ((32, 32), WW.WEIGHT_WIDTH_8BIT, 2, True),
            ((16, 16), WW.WEIGHT_WIDTH_4BIT, 4, True),
            ((200, 32), WW.WEIGHT_WIDTH_8BIT, 2, True),
            ((30, 24), WW.WEIGHT_WIDTH_4BIT, 4, True),
            ((32, 24), WW.WEIGHT_WIDTH_2BIT, 3, True),
        ],
    )
    def test_weight_ram_mapping(self, shape, wp, nfold, is_iw8):
        nbit = 1 << wp

        if shape[0] % nfold > 0:
            expected_h = shape[0] // nfold + 1
        else:
            expected_h = shape[0] // nfold

        expected_shape = (expected_h, shape[1] * nfold)

        # Generate the original weight with shape
        _low = 0 if nbit == 1 else -(1 << (nbit - 1))
        _high = 1 << (nbit - 1)
        test_weight = np.random.randint(_low, _high, size=shape, dtype=WEIGHT_DTYPE)

        # 1. Fold, return the folded weight after padding.
        w_folded = self._nfold_weight_ref(test_weight, expected_shape[0], nfold)

        # 2. Unpack, get the weight ram.
        # The real interval is HwConfig.N_FANIN_PER_DENDRITE_ANN
        _fake_interval = w_folded.shape[0] * 2
        w_unpacked = self._weight_ram_mapping_ref(
            w_folded, nbit, is_iw8, _fake_interval
        )
        w_unpacked.setflags(write=False)

        # 3. Check
        self._check(
            test_weight, w_folded, w_unpacked, nbit, nfold, is_iw8, _fake_interval
        )

    @staticmethod
    def _nfold_weight_ref(raw_weight: WeightType, expected_row: int, nfold: int):
        raw_row, raw_col = raw_weight.shape

        if raw_row % nfold > 0:
            _padding = nfold - raw_row % nfold
            assert expected_row * nfold == raw_row + _padding

            w_padding = np.append(
                raw_weight,
                values=np.zeros((_padding, raw_col), dtype=WEIGHT_DTYPE),
                axis=0,
            )
        else:
            w_padding = raw_weight

        split = np.vsplit(w_padding, nfold)
        w_folded = np.zeros((expected_row, raw_col * nfold), dtype=WEIGHT_DTYPE)

        for i, j in np.ndindex((nfold, raw_col)):
            w_col = split[i][:, j]
            w_folded[:, j * nfold + i] = w_col

        return w_folded

    @staticmethod
    def _weight_ram_mapping_ref(
        folded_weights: WeightType,
        n_bit: int,
        is_iw8: bool,
        fake_interval: int,
    ):
        row, col = folded_weights.shape
        # if iw = 1, the row of result is the same as the row of folded_weights
        if not is_iw8:
            result_row = row
        else:
            result_row = 8 * fake_interval

        result = np.zeros((result_row, col * n_bit), dtype=np.uint8)

        if n_bit == 1:
            result[:row, :col] = folded_weights
            return result

        # [N*M] -> [M*N*1]
        folded_weights_3d = np.expand_dims(folded_weights.T, axis=2).astype(np.uint8)

        for i in range(col):
            # For every m in M, unpack the array [N*1] -> [N*8]
            unpacked = np.unpackbits(
                folded_weights_3d[i], axis=1, count=n_bit, bitorder="little"
            )

            if not is_iw8:
                result[:row, n_bit * i : n_bit * (i + 1)] = unpacked
            else:
                for bit in range(n_bit):
                    result[bit * fake_interval : bit * fake_interval + row, i] = (
                        unpacked[:, bit]
                    )

        assert np.max(result, axis=None) <= 1
        assert np.min(result, axis=None) >= 0

        return result

    @staticmethod
    def _check(
        test_data: WeightType,
        w_folded: WeightType,
        w_unpacked: WRAMUnpackedType,
        nbit: int,
        nfold: int,
        is_iw8: bool,
        fake_interval: int = 0,
    ) -> None:
        for i, j in np.ndindex(test_data.shape):
            n_in_col = w_folded.shape[0]
            now_i = i % n_in_col
            offset_j = i // n_in_col
            now_j = offset_j + j * nfold

            if not is_iw8:
                wij = w_unpacked[now_i, now_j * nbit : (now_j + 1) * nbit]
            else:
                # From LSB to MSB
                bits = [
                    w_unpacked[i * fake_interval + now_i, now_j] for i in range(nbit)
                ]
                wij = np.asarray(bits, dtype=np.uint8)

            wij_packed = packbits_ref(wij, nbit)
            assert test_data[i, j] == wij_packed

    def test_CorePlacement_weight_pack_shape(self):
        # Mock unpacked weight
        w_unpacked = np.zeros(CorePlacement.WRAM_BASE_SHAPE, dtype=np.uint8)
        w_packed_u64 = CorePlacement._weight_pack(w_unpacked)

        assert w_packed_u64.shape == (
            (HwConfig.ADDR_RAM_MAX + 1),
            (HwConfig.ADDR_AXON_MAX + 1) // (WRAM_PACKED_DTYPE(1).nbytes * 8),
        )

    def test_packbits_to_mapping_form(self, random_fixture):
        def _weight_ram_T(weight_ram_mapped: np.ndarray):
            _w = weight_ram_mapped.T.reshape(-1, 64)
            w_packed_u8 = np.packbits(_w, axis=-1, bitorder="little")

            return w_packed_u8

        w = np.random.randint(-8, 8, size=(1152, 64), dtype=WEIGHT_DTYPE)

        # 1152 * 512
        w1 = self._weight_ram_mapping_ref(w, 8, False, 0)

        # -> 512 * 1152 -> 512 * 144 (uint8)
        wT = _weight_ram_T(w1)

        ww = wT.view(np.uint64).reshape(-1, 18)
        ww.setflags(write=False)
        assert 1

    def test_weight_ram_mapping_8bits(self, packbits8):
        binary_conn = np.zeros((6, 8 * 5), dtype=np.bool_)
        wp = WW.WEIGHT_WIDTH_8BIT

        array = np.random.randint(-128, 128, size=(4, 4), dtype=WEIGHT_DTYPE)

        y = np.unpackbits(np.uint8(array), axis=1, bitorder="little")
        assert y.shape == (4, (1 << wp) * 4)

        binary_conn[: y.shape[0], : y.shape[1]] = y

        for i, j in np.ndindex((4, 4)):
            expected = array[i, j]
            wij = y[i, j * (1 << wp) : (j + 1) * (1 << wp)]
            r = packbits8(wij)

            assert expected == r

    def test_weight_ram_mapping_4bits(self, packbits4):
        binary_conn = np.zeros((6, 4 * 5), dtype=np.bool_)
        wp = WW.WEIGHT_WIDTH_4BIT

        array = np.random.randint(-8, 8, size=(4, 4), dtype=WEIGHT_DTYPE)
        y = np.zeros((4, 16), dtype=np.uint8)

        for i in range(4):
            ual = np.uint8(np.expand_dims(array[:, i], axis=1))
            a = np.unpackbits(ual, axis=1, count=4, bitorder="little")
            y[: a.shape[0], (1 << wp) * i : (1 << wp) * (i + 1)] = a

        assert y.shape == (4, (1 << wp) * 4)

        binary_conn[: y.shape[0], : y.shape[1]] = y

        for i, j in np.ndindex(array.shape):
            expected = array[i, j]
            wij = y[i, j * (1 << wp) : (j + 1) * (1 << wp)]
            r = packbits4(wij)

            assert expected == r

    def test_weight_ram_mapping_2bits(self, packbits2):
        binary_conn = np.zeros((6, 4 * 5), dtype=np.bool_)
        wp = WW.WEIGHT_WIDTH_2BIT

        array = np.random.randint(-2, 2, size=(4, 4), dtype=WEIGHT_DTYPE)
        y = np.zeros((4, 8), dtype=np.uint8)

        for i in range(4):
            ual = np.uint8(np.expand_dims(array[:, i], axis=1))
            a = np.unpackbits(ual, axis=1, count=2, bitorder="little")
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
