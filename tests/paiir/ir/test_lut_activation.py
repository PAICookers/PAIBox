import copy
import math

import pytest
import torch
from paicorelib import DataSign, DataWidth

from paibox.paiir.ir.calc_params import LUT_TABLE_SIZE
from paibox.paiir.ir.lut_activation import (
    LutCustom,
    LutLinear,
    LutReLU,
    LutReLUSymmetric,
    LutSigmoid,
    LutSoftsign,
    LutTanh,
    _lookup_hw_lut_data,
)

_DEFAULT_LUT_CASES = [
    (LutReLU, False, torch.uint8),
    (LutReLUSymmetric, False, torch.uint8),
    (LutLinear, True, torch.int8),
    (LutSigmoid, False, torch.uint8),
    (LutTanh, True, torch.int8),
    (LutSoftsign, True, torch.int8),
]

_NONLINEAR_CASES = [
    (LutSigmoid, lambda x: 1 / (1 + math.exp(-x)), False, 255),
    (LutTanh, math.tanh, True, 127),
    (LutSoftsign, lambda x: x / (1 + abs(x)), True, 127),
]


def _low_range_identity_lut() -> LutCustom:
    levels = torch.arange(5, dtype=torch.int32)
    return LutCustom.from_intervals(levels, levels, output_signed=False)


def _assert_value_block(values: torch.Tensor, start: int, stop: int, value: int) -> None:
    assert values[start:stop].tolist() == [value] * (stop - start)


class TestPublicSignednessAndBuffers:
    @pytest.mark.parametrize(
        "cls, expected_signed, expected_dtype",
        _DEFAULT_LUT_CASES,
        ids=[cls.__name__ for cls, _, _ in _DEFAULT_LUT_CASES],
    )
    def test_default_output_signed_controls_int_buffers(
        self, cls, expected_signed, expected_dtype
    ):
        lut = cls(min_val=-100, max_val=100)

        assert lut.output_signed is expected_signed
        assert lut.thresholds.shape == (LUT_TABLE_SIZE,)
        assert lut.lut_values.shape == (LUT_TABLE_SIZE,)
        assert lut.thresholds.dtype == torch.int32
        assert lut.lut_values.dtype == expected_dtype
        assert torch.all(lut.thresholds[1:] >= lut.thresholds[:-1])

    def test_float_mode_uses_float_buffers(self):
        lut = LutReLU(min_val=-10, max_val=10, output_signed=True, is_float=True)

        assert lut.output_signed is True
        assert lut.thresholds.dtype == torch.float32
        assert lut.lut_values.dtype == torch.bfloat16

    def test_lutcustom_float_mode_uses_float_buffers(self):
        thresholds = torch.linspace(-1.0, 1.0, LUT_TABLE_SIZE)
        values = thresholds.square()

        lut = LutCustom(thresholds, values, is_float=True)

        assert lut.output_signed is False
        assert lut.thresholds.dtype == torch.float32
        assert lut.lut_values.dtype == torch.bfloat16

    @pytest.mark.parametrize(
        "output_signed, values, expected_signed, expected_dtype",
        [
            (
                False,
                torch.arange(LUT_TABLE_SIZE, dtype=torch.int32),
                False,
                torch.uint8,
            ),
            (True, torch.arange(-128, 128, dtype=torch.int32), True, torch.int8),
            (None, torch.arange(LUT_TABLE_SIZE, dtype=torch.int32), False, torch.uint8),
            (None, torch.arange(-128, 128, dtype=torch.int32), True, torch.int8),
        ],
        ids=[
            "explicit_unsigned",
            "explicit_signed",
            "inferred_unsigned",
            "inferred_signed",
        ],
    )
    def test_lutcustom_output_signed_controls_or_infers_dtype(
        self, output_signed, values, expected_signed, expected_dtype
    ):
        thresholds = torch.arange(LUT_TABLE_SIZE, dtype=torch.int32)

        lut = LutCustom(thresholds, values, output_signed=output_signed)

        assert lut.output_signed is expected_signed
        assert lut.lut_values.dtype == expected_dtype

    @pytest.mark.parametrize(
        "thresholds, values, is_float, match",
        [
            (
                torch.arange(LUT_TABLE_SIZE, dtype=torch.float32),
                torch.arange(LUT_TABLE_SIZE, dtype=torch.int32),
                False,
                "integer LUT thresholds",
            ),
            (
                torch.arange(LUT_TABLE_SIZE, dtype=torch.int32),
                torch.arange(LUT_TABLE_SIZE, dtype=torch.float32),
                False,
                "integer LUT values",
            ),
            (
                torch.arange(LUT_TABLE_SIZE, dtype=torch.int32),
                torch.arange(LUT_TABLE_SIZE, dtype=torch.float32),
                True,
                "float LUT thresholds",
            ),
            (
                torch.arange(LUT_TABLE_SIZE, dtype=torch.float32),
                torch.arange(LUT_TABLE_SIZE, dtype=torch.int32),
                True,
                "float LUT values",
            ),
        ],
        ids=[
            "integer_thresholds_reject_float",
            "integer_values_reject_float",
            "float_thresholds_reject_integer",
            "float_values_reject_integer",
        ],
    )
    def test_lutcustom_rejects_wrong_tensor_kinds(
        self, thresholds, values, is_float, match
    ):
        with pytest.raises(TypeError, match=match):
            LutCustom(thresholds, values, is_float=is_float)

    @pytest.mark.parametrize(
        "threshold_count, value_count, match",
        [
            (100, LUT_TABLE_SIZE, f"{LUT_TABLE_SIZE} thresholds"),
            (LUT_TABLE_SIZE, 100, f"{LUT_TABLE_SIZE} values"),
        ],
    )
    def test_lutcustom_rejects_wrong_table_lengths(
        self, threshold_count, value_count, match
    ):
        with pytest.raises(ValueError, match=match):
            LutCustom(
                torch.zeros(threshold_count, dtype=torch.int32),
                torch.zeros(value_count, dtype=torch.int32),
            )

    @pytest.mark.parametrize(
        "output_signed, values, match",
        [
            (False, torch.arange(-128, 128, dtype=torch.int32), "unsigned 8-bit"),
            (True, torch.arange(LUT_TABLE_SIZE, dtype=torch.int32), "signed 8-bit"),
        ],
        ids=["unsigned_negative", "signed_too_large"],
    )
    def test_explicit_signedness_rejects_out_of_range_values(
        self, output_signed, values, match
    ):
        thresholds = torch.arange(LUT_TABLE_SIZE, dtype=torch.int32)

        with pytest.raises(ValueError, match=match):
            LutCustom(thresholds, values, output_signed=output_signed)

    def test_lutcustom_from_intervals_builds_padded_logical_lut(self):
        lut = LutCustom.from_intervals(
            starts=torch.tensor([0, 2, 4]),
            values=torch.tensor([0, 1, 2]),
        )

        assert lut.output_signed is False
        assert lut.thresholds[:5].tolist() == [0, 2, 4, 4, 4]
        assert lut.lut_values[:5].tolist() == [0, 1, 2, 2, 2]
        values, indices = lut.lookup(torch.tensor([-1, 0, 1, 2, 3, 4, 5]))
        assert indices.tolist() == [
            0,
            0,
            0,
            1,
            1,
            LUT_TABLE_SIZE - 1,
            LUT_TABLE_SIZE - 1,
        ]
        assert values.tolist() == [0, 0, 0, 1, 1, 2, 2]

    @pytest.mark.parametrize(
        "starts, values, error_type, match",
        [
            ([], [], ValueError, "must not be empty"),
            ([0, 1], [0], ValueError, "same length"),
            (
                list(range(LUT_TABLE_SIZE + 1)),
                list(range(LUT_TABLE_SIZE + 1)),
                ValueError,
                "at most",
            ),
            ([0, 0], [0, 1], ValueError, "strictly increasing"),
            (
                torch.tensor([0.0, 1.0]),
                torch.tensor([0, 1]),
                TypeError,
                "LUT interval starts",
            ),
            (
                torch.tensor([0, 2**40], dtype=torch.int64),
                torch.tensor([0, 1]),
                ValueError,
                "fit int32",
            ),
        ],
        ids=[
            "empty",
            "mismatched_length",
            "too_many_intervals",
            "non_increasing",
            "non_integer_starts",
            "int32_overflow",
        ],
    )
    def test_lutcustom_from_intervals_rejects_invalid_inputs(
        self, starts, values, error_type, match
    ):
        with pytest.raises(error_type, match=match):
            LutCustom.from_intervals(starts, values)


class TestLookupSemantics:
    def test_lookup_uses_logical_bucketize_semantics(self):
        lut = _low_range_identity_lut()
        x = torch.tensor([-10, 0, 1, 2, 3, 4, 5], dtype=torch.int32)

        values, indices = lut.lookup(x)

        assert indices.tolist() == [
            0,
            0,
            1,
            2,
            3,
            LUT_TABLE_SIZE - 1,
            LUT_TABLE_SIZE - 1,
        ]
        assert values.tolist() == [0, 0, 1, 2, 3, 4, 4]
        assert torch.equal(values, lut(x))

    def test_logical_lut_data_returns_cloned_table(self):
        lut = LutReLU(min_val=-100, max_val=100)

        logical = lut.logical_lut_data

        assert torch.equal(logical.thresholds, lut.thresholds)
        assert torch.equal(logical.values, lut.lut_values)
        assert logical.thresholds is not lut.thresholds
        assert logical.values is not lut.lut_values
        assert logical.is_float == lut.is_float

    def test_forward_preserves_shape_and_uses_lut_entries(self):
        lut = LutTanh(min_val=-500, max_val=500)
        x = torch.linspace(-500, 500, 24).reshape(2, 3, 4)

        y = lut(x)

        assert y.shape == x.shape
        assert set(y.flatten().tolist()).issubset(set(lut.lut_values.tolist()))


class TestHardwareLutExport:
    @pytest.mark.parametrize(
        "cls",
        [
            LutReLU,
            LutReLUSymmetric,
            LutLinear,
            LutSigmoid,
            LutTanh,
            LutSoftsign,
        ],
    )
    def test_default_presets_have_exportable_monotonic_integer_thresholds(self, cls):
        lut = cls()
        output_data_sign = DataSign.SIGNED if lut.output_signed else DataSign.UNSIGNED

        width = lut.infer_hw_output_width(output_data_sign)

        assert width == DataWidth.WIDTH_8BIT

    def test_to_hw_lut_data_reencodes_low_range_identity_u4_for_sar(self):
        lut = _low_range_identity_lut()
        logical = lut.logical_lut_data

        hardware = lut.to_hw_lut_data(DataSign.UNSIGNED, DataWidth.WIDTH_4BIT)
        values, indices = _lookup_hw_lut_data(
            hardware,
            DataWidth.WIDTH_4BIT,
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        )

        assert indices.tolist() == [0, 16, 32, 48, 64]
        assert values.tolist() == [0, 1, 2, 3, 4]
        assert not torch.equal(hardware.thresholds, logical.thresholds)
        assert torch.equal(
            hardware.thresholds[torch.tensor([16, 32, 48, 64])],
            torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        )
        _assert_value_block(hardware.values, 0, 16, 0)
        _assert_value_block(hardware.values, 16, 32, 1)
        _assert_value_block(hardware.values, 32, 48, 2)
        _assert_value_block(hardware.values, 48, 64, 3)
        _assert_value_block(hardware.values, 64, LUT_TABLE_SIZE, 4)

    def test_to_hw_lut_data_reencodes_full_u4_identity_blocks(self):
        levels = torch.arange(16, dtype=torch.int32)
        lut = LutCustom.from_intervals(levels, levels, output_signed=False)

        hardware = lut.to_hw_lut_data(DataSign.UNSIGNED, DataWidth.WIDTH_4BIT)
        values, indices = _lookup_hw_lut_data(
            hardware, DataWidth.WIDTH_4BIT, levels
        )

        assert indices.tolist() == list(range(0, LUT_TABLE_SIZE, 16))
        assert values.tolist() == list(range(16))
        assert hardware.thresholds[torch.arange(16, 256, 16)].tolist() == list(
            range(1, 16)
        )
        for value, start in enumerate(range(0, LUT_TABLE_SIZE, 16)):
            _assert_value_block(hardware.values, start, start + 16, value)

    def test_signed_identity_hardware_lut_preserves_negative_values(self):
        levels = torch.tensor([-1, 0, 1], dtype=torch.int32)
        lut = LutCustom.from_intervals(levels, levels, output_signed=True)

        hardware = lut.to_hw_lut_data(DataSign.SIGNED, DataWidth.WIDTH_2BIT)
        values, indices = _lookup_hw_lut_data(
            hardware,
            DataWidth.WIDTH_2BIT,
            torch.tensor([-1, 0, 1], dtype=torch.int32),
        )

        assert indices.tolist() == [0, 64, 128]
        assert values.tolist() == [-1, 0, 1]
        _assert_value_block(hardware.values, 0, 64, -1)
        _assert_value_block(hardware.values, 64, 128, 0)
        _assert_value_block(hardware.values, 128, LUT_TABLE_SIZE, 1)

    def test_narrow_hardware_lut_rejects_too_many_effective_intervals(self):
        thresholds = torch.arange(LUT_TABLE_SIZE, dtype=torch.int32)
        values = torch.arange(LUT_TABLE_SIZE, dtype=torch.int32).remainder(2)
        lut = LutCustom(thresholds, values, output_signed=False, is_float=False)

        with pytest.raises(ValueError, match="effective intervals"):
            lut.to_hw_lut_data(DataSign.UNSIGNED, DataWidth.WIDTH_4BIT)

    def test_narrow_hardware_lut_rejects_values_outside_output_width(self):
        lut = LutCustom.from_intervals(
            torch.tensor([0, 1]),
            torch.tensor([0, 2]),
            output_signed=False,
        )

        with pytest.raises(ValueError, match="unsigned width_1bit range"):
            lut.to_hw_lut_data(DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)


class TestActivationBehavior:
    @pytest.mark.parametrize("cls", [LutReLU, LutReLUSymmetric])
    @pytest.mark.parametrize(
        "output_signed, max_out", [(False, 255), (True, 127)], ids=["u8", "s8"]
    )
    def test_relu_variants_respect_signedness_and_monotonic_positive_region(
        self, cls, output_signed, max_out
    ):
        lut = cls(min_val=-500, max_val=500, output_signed=output_signed)

        assert torch.all(lut(torch.tensor([-100, -1])) == 0)
        y = lut(torch.linspace(-500, 500, 100))
        assert torch.all((y >= 0) & (y <= max_out))
        assert lut(torch.tensor([500])).item() >= max_out - 5

    def test_symmetric_relu_keeps_uniform_thresholds(self):
        lut = LutReLUSymmetric(min_val=-500, max_val=500, output_signed=True)

        diffs = (lut.thresholds[1:] - lut.thresholds[:-1]).float()

        assert torch.allclose(
            diffs,
            torch.full_like(diffs, diffs[0].item(), dtype=torch.float32),
            atol=1.0,
        )

    def test_linear_default_signed_mapping_preserves_endpoints(self):
        lut = LutLinear(min_val=-100, max_val=100)

        y = lut(torch.tensor([-100, 0, 100]))

        assert abs(y[0] - (-128)) <= 2
        assert abs(y[1]) <= 2
        assert abs(y[2] - 127) <= 2

    @pytest.mark.parametrize(
        "cls, ref_fn, output_signed, scale",
        _NONLINEAR_CASES,
        ids=[cls.__name__ for cls, _, _, _ in _NONLINEAR_CASES],
    )
    def test_nonlinear_luts_are_monotonic_and_reasonably_accurate(
        self, cls, ref_fn, output_signed, scale
    ):
        min_val, max_val, act_range = -500, 500, 10
        lut = cls(
            min_val=min_val,
            max_val=max_val,
            output_signed=output_signed,
            act_range=act_range,
        )

        y = lut(torch.linspace(min_val, max_val, 200))
        assert torch.all(y[1:] - y[:-1] >= -1e-5)

        hw_scale = (max_val - min_val) / (2 * act_range)
        norm_points = [-4, -1, 0, 1, 4]
        hw_points = [min_val + (xn + act_range) * hw_scale for xn in norm_points]
        sampled = lut(torch.tensor(hw_points))
        for xn, actual in zip(norm_points, sampled.tolist()):
            expected = ref_fn(xn) * scale
            assert abs(actual - expected) <= 15

    @pytest.mark.parametrize(
        "cls, ref_fn, output_signed, scale",
        _NONLINEAR_CASES,
        ids=[cls.__name__ for cls, _, _, _ in _NONLINEAR_CASES],
    )
    def test_nonlinear_float_mode_approximates_math_function(
        self, cls, ref_fn, output_signed, scale
    ):  # noqa: ARG002
        lut = cls(min_val=-500, max_val=500, output_signed=output_signed, is_float=True)
        x = torch.linspace(-5, 5, 20)
        ref = torch.tensor([ref_fn(xi.item()) for xi in x])

        y = lut(x)
        assert torch.allclose(y.float(), ref, atol=0.15)


class TestCopying:
    @pytest.mark.parametrize(
        "lut",
        [
            LutReLU(min_val=-100, max_val=100),
            LutCustom(
                torch.arange(LUT_TABLE_SIZE, dtype=torch.int32),
                torch.arange(LUT_TABLE_SIZE, dtype=torch.int32),
            ),
        ],
        ids=["generated", "custom"],
    )
    def test_deepcopy_clones_buffers_without_aliasing(self, lut):
        cloned = copy.deepcopy(lut)

        assert cloned is not lut
        assert type(cloned) is type(lut)
        assert torch.equal(cloned.thresholds, lut.thresholds)
        assert torch.equal(cloned.lut_values, lut.lut_values)
        assert cloned.thresholds is not lut.thresholds
        assert cloned.lut_values is not lut.lut_values
