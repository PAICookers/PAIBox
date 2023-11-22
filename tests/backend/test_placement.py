from functools import partial
import numpy as np
import pytest

import paibox as pb
from paibox.backend.placement import (
    NeuSeg,
    aligned_coords,
    get_axon_segments,
    get_neu_segments,
    n_axon2lcn_ex,
)
from paibox.exceptions import ResourceError
from paibox.libpaicore.v2 import AxonCoord, AxonSegment, NeuronSegment
from paibox.libpaicore.v2.reg_types import WeightPrecisionType as WP


def test_get_raw_weight_ref():
    rng = np.random.RandomState(seed=1)
    w1 = rng.randint(-128, 128, size=(10, 20), dtype=np.int8)
    w2 = rng.randint(-128, 128, size=(10, 30), dtype=np.int8)

    w_of_neurons = [w1, w2]

    n1 = pb.neuron.LIF((20,), 1)
    n2 = pb.neuron.LIF((30,), 1)

    dest = [n1, n2]

    neuron_segs_of_cb = [
        [
            NeuSeg(n1, NeuronSegment(slice(0, 20, 1), 0)),
            NeuSeg(n2, NeuronSegment(slice(0, 5, 1), 20)),
        ],
        [NeuSeg(n2, NeuronSegment(slice(5, 30, 1), 0))],
    ]

    w_of_neu_segs_of_cb = []
    for neu_segs in neuron_segs_of_cb:
        w_of_neu_segs = []
        for neu_seg in neu_segs:
            w = w_of_neurons[dest.index(neu_seg.parent)][  # type: ignore
                :, neu_seg.segment.index
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
    expected_shape = expected.shape
    row, col = expected.shape
    o_matrix = np.zeros(expected_shape, dtype=np.int8)

    for i in range(cur_shape[1]):
        w_col = input[:, i]
        col_group = 0

        while (n_rest_axon := cur_shape[0] - row * col_group) > row:
            o_matrix[:, n_col_groups * i + col_group] = w_col[
                row * col_group : row * (col_group + 1)
            ]
            col_group += 1

            print(o_matrix)

        o_matrix[:, n_col_groups * i + col_group] = np.pad(
            w_col[row * col_group :],
            pad_width=(0, row - n_rest_axon),
            mode="constant",
            constant_values=0,
        )

        print(o_matrix)

    assert np.array_equal(o_matrix, expected)


def test_nfold_weight_ref():
    original_matrix = np.arange(1, 25, dtype=np.int8).reshape(8, 3)
    nfold = 3

    if original_matrix.shape[0] % nfold > 0:
        _padding = nfold - original_matrix.shape[0] % nfold
        w_padding = np.append(
            original_matrix,
            values=np.zeros((_padding, original_matrix.shape[1]), dtype=np.int8),
            axis=0,
        )
    else:
        w_padding = original_matrix

    splited = np.vsplit(w_padding, nfold)

    result = np.zeros(
        (w_padding.shape[0] // nfold, original_matrix.shape[1] * nfold), dtype=np.int8
    )

    for i, j in np.ndindex((nfold, original_matrix.shape[1])):
        g = splited[i][:, j]
        result[:, j * nfold + i] = g

    assert np.array_equal(
        result,
        np.array(
            [
                [1, 10, 19, 2, 11, 20, 3, 12, 21],
                [4, 13, 22, 5, 14, 23, 6, 15, 24],
                [7, 16, 0, 8, 17, 0, 9, 18, 0],
            ],
            dtype=np.int8,
        ),
    )


def packbits_ref(bits: np.ndarray, count: int) -> int:
    """Pack unsigned bits into a signed integer.

    This is a test of the prototype of the original function.
    """
    _bits = np.append(bits[: count - 1], bits[-1])

    result = sum(bit << i for i, bit in enumerate(_bits))
    result -= _bits[-1] << count

    return result


@pytest.fixture
def packbits8():
    return partial(packbits_ref, count=8)


@pytest.fixture
def packbits4():
    return partial(packbits_ref, count=4)


@pytest.fixture
def packbits2():
    return partial(packbits_ref, count=2)


@pytest.fixture
def packbits1():
    return partial(packbits_ref, count=1)


class TestWeightUnpack:
    @pytest.mark.parametrize(
        "wp",
        [
            WP.WEIGHT_WIDTH_8BIT,
            WP.WEIGHT_WIDTH_4BIT,
            WP.WEIGHT_WIDTH_2BIT,
            WP.WEIGHT_WIDTH_1BIT,
        ],
    )
    def test_signed_unpackbits(self, wp):
        count = 1 << wp
        actual_array = np.arange(
            -(1 << (count - 1)), (1 << (count - 1)), 1, dtype=np.int8
        )

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
        "shape, wp, nfold",
        [
            ((8, 8), WP.WEIGHT_WIDTH_8BIT, 2),
            ((32, 32), WP.WEIGHT_WIDTH_8BIT, 2),
            ((16, 16), WP.WEIGHT_WIDTH_4BIT, 4),
            ((30, 24), WP.WEIGHT_WIDTH_4BIT, 4),
            ((32, 24), WP.WEIGHT_WIDTH_2BIT, 3),
            ((32, 24), WP.WEIGHT_WIDTH_1BIT, 3),
            ((31, 23), WP.WEIGHT_WIDTH_8BIT, 5),
            ((1200, 200), WP.WEIGHT_WIDTH_1BIT, 2),
            ((800, 64), WP.WEIGHT_WIDTH_8BIT, 2),
        ],
    )
    def test_weight_ram_mapping(self, shape, wp, nfold):
        nbit = 1 << wp

        if shape[0] % nfold > 0:
            expected_h = shape[0] // nfold + 1
        else:
            expected_h = shape[0] // nfold

        expected_shape = (expected_h, shape[1] * nfold)

        # Generate the original weight with shape
        _low = 0 if nbit == 1 else -(1 << (nbit - 1))
        _high = 1 << (nbit - 1)
        array = np.random.randint(_low, _high, size=shape, dtype=np.int8)

        # 1. Fold, return the folded weight after padding.
        w_folded = self._fold_raw_weight_ref(array, expected_shape[0], nfold)

        # 2. Unpack, get the weight ram.
        if nbit > 1:
            w_unpacked = self._weight_ram_mapping_ref(w_folded, nbit)
        else:
            w_unpacked = w_folded.copy().astype(np.bool_)
        
        w_unpacked.setflags(write=False)

        # 3. Check
        for i, j in np.ndindex(shape):
            n_in_col = w_folded.shape[0]
            now_i = i % n_in_col
            offset_j = i // n_in_col

            now_j = offset_j + j * nfold

            expected = array[i, j]
            wij = w_unpacked[now_i, now_j * nbit : (now_j + 1) * nbit]
            packed = packbits_ref(wij, nbit)

            assert expected == packed

    @staticmethod
    def _weight_ram_mapping_ref(folded_weights: np.ndarray, n_bit: int):
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

        return result.astype(np.bool_)

    @staticmethod
    def _fold_raw_weight_ref(raw_weight: np.ndarray, expected_row: int, nfold: int):
        raw_row, raw_col = raw_weight.shape

        if raw_row % nfold > 0:
            _padding = nfold - raw_row % nfold
            assert expected_row * nfold == raw_row + _padding

            w_padding = np.append(
                raw_weight,
                values=np.zeros((_padding, raw_col), dtype=np.int8),
                axis=0,
            )
        else:
            w_padding = raw_weight.copy()

        splited = np.vsplit(w_padding, nfold)
        assert w_padding.shape[0] == expected_row * nfold

        w_folded = np.zeros((expected_row, raw_col * nfold), dtype=np.int8)

        for i, j in np.ndindex((nfold, raw_col)):
            w_col = splited[i][:, j]
            w_folded[:, j * nfold + i] = w_col

        return w_folded

    def test_weight_ram_mapping_8bits(self, packbits8):
        binary_conn = np.zeros((6, 8 * 5), dtype=np.bool_)
        wp = WP.WEIGHT_WIDTH_8BIT

        array = np.random.randint(-128, 128, size=(4, 4), dtype=np.int8)

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
        wp = WP.WEIGHT_WIDTH_4BIT

        array = np.random.randint(-8, 8, size=(4, 4), dtype=np.int8)
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
        wp = WP.WEIGHT_WIDTH_2BIT

        array = np.random.randint(-2, 2, size=(4, 4), dtype=np.int8)
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


class TestGetNeuronSegments:
    def test_get_neu_segments_catagory(
        self,
        neu_segs_test_data,
        neu_segs_expected_catagory,
    ):
        for data, expected in zip(neu_segs_test_data, neu_segs_expected_catagory):
            neu_ins, capacity, wp, lcn_ex = data
            neu_segs = get_neu_segments(
                neu_ins,
                capacity,
                weight_precision=wp,
                lcn_ex=lcn_ex,
                method="catagory",
            )

            assert neu_segs == expected
            assert neu_segs[0][0].segment.interval == (1 << wp) * (1 << lcn_ex)

    def test_get_neu_segments_dense(
        self,
        neu_segs_test_data,
        neu_segs_expected_dense,
    ):
        for data, expected in zip(neu_segs_test_data, neu_segs_expected_dense):
            neu_ins, capacity, wp, lcn_ex = data
            neu_segs = get_neu_segments(
                neu_ins,
                capacity,
                weight_precision=wp,
                lcn_ex=lcn_ex,
                method="dense",
            )

            assert neu_segs == expected
            assert neu_segs[0][0].segment.interval == (1 << wp) * (1 << lcn_ex)


@pytest.mark.parametrize(
    "axons",
    [
        [pb.neuron.LIF(600, 2), pb.neuron.LIF(800, 2), pb.neuron.LIF(256, 2)],
        [pb.neuron.LIF(384, 3), pb.neuron.LIF(383, 3), pb.neuron.LIF(385, 3)],
        [pb.neuron.LIF(1153, 2)],
        [pb.neuron.LIF(2222, 1), pb.neuron.LIF(2378, 1)],
    ],
)
def test_get_axon_segments(axons):
    lcn_ex = n_axon2lcn_ex(sum(axon.num_out for axon in axons), 1152)

    tr_max = 1 << lcn_ex

    axon_segs = get_axon_segments(axons, tr_max, 1152)

    for axon_seg in axon_segs.values():
        assert axon_seg.addr_offset <= 1152


@pytest.mark.parametrize(
    "axons",
    [
        [pb.neuron.LIF(1151, 2), pb.neuron.LIF(1153, 2)],
        [pb.neuron.LIF(1151 * 2, 2), pb.neuron.LIF(1153 * 2, 2)],
    ],
)
def test_get_axon_segments_boundary(axons):
    """Illegal boundary cases."""
    lcn_ex = n_axon2lcn_ex(sum(axon.num_out for axon in axons), 1152)
    tr_max = 1 << lcn_ex

    with pytest.raises(ResourceError):
        axon_segs = get_axon_segments(axons, tr_max, 1152)


@pytest.mark.parametrize(
    "neu_index, axon_seg, expected",
    [
        (
            slice(5, 8),
            AxonSegment(12, 3, 0),
            [
                AxonCoord(1, 2),
                AxonCoord(2, 0),
                AxonCoord(2, 1),
            ],
        ),
        (
            slice(0, 3),
            AxonSegment(12, 3, 0),
            [
                AxonCoord(0, 0),
                AxonCoord(0, 1),
                AxonCoord(0, 2),
            ],
        ),
        (
            slice(1, 5),
            AxonSegment(12, 3, 0),
            [
                AxonCoord(0, 1),
                AxonCoord(0, 2),
                AxonCoord(1, 0),
                AxonCoord(1, 1),
            ],
        ),
        (
            slice(1, 6),
            AxonSegment(12, 3, 0),
            [
                AxonCoord(0, 1),
                AxonCoord(0, 2),
                AxonCoord(1, 0),
                AxonCoord(1, 1),
                AxonCoord(1, 2),
            ],
        ),
        (
            slice(3, 10),
            AxonSegment(16, 4, 4),
            [
                AxonCoord(0, 4 + 3),
                AxonCoord(1, 4 + 0),
                AxonCoord(1, 4 + 1),
                AxonCoord(1, 4 + 2),
                AxonCoord(1, 4 + 3),
                AxonCoord(2, 4 + 0),
                AxonCoord(2, 4 + 1),
            ],
        ),
    ],
)
def test_aligned_segments(neu_index, axon_seg, expected):
    axon_coords = aligned_coords(neu_index, axon_seg)
    assert axon_coords == expected
