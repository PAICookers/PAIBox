import math

import pytest
import torch

from paibox.paiir.ir.lut_activation import (
    LutCustom,
    LutLinear,
    LutReLU,
    LutReLUSymmetric,
    LutSigmoid,
    LutSoftsign,
    LutTanh,
)

_ALL_LUT_CLASSES = (
    LutReLU,
    LutReLUSymmetric,
    LutLinear,
    LutSigmoid,
    LutTanh,
    LutSoftsign,
)


class TestLutActivationBase:
    """Tests for shared LutActivation infrastructure (dtype, shape, lookup)."""

    def test_int_mode_dtypes(self):
        """Int mode: thresholds int32, values uint8 (unsigned) or int8 (signed)."""
        lut_u = LutReLU(min_val=-100, max_val=100, output_sign=0)
        assert lut_u.thresholds.dtype == torch.int32
        assert lut_u.lut_values.dtype == torch.uint8

        lut_s = LutTanh(min_val=-100, max_val=100, output_sign=1)
        assert lut_s.thresholds.dtype == torch.int32
        assert lut_s.lut_values.dtype == torch.int8

    def test_float_mode_dtypes(self):
        """Float mode: thresholds float32, values bfloat16."""
        lut = LutReLU(min_val=-10, max_val=10, output_sign=0, is_float=True)
        assert lut.thresholds.dtype == torch.float32
        assert lut.lut_values.dtype == torch.bfloat16

    def test_buffer_shape_256(self):
        """All LUT activations produce 256-entry buffers."""
        for cls in _ALL_LUT_CLASSES:
            lut = cls(min_val=-100, max_val=100, output_sign=0)
            assert lut.thresholds.shape == (256,)
            assert lut.lut_values.shape == (256,)

    def test_thresholds_monotonic(self):
        """Thresholds must be monotonically non-decreasing."""
        for cls in _ALL_LUT_CLASSES:
            lut = cls(min_val=-500, max_val=500, output_sign=0)
            diff = lut.thresholds[1:] - lut.thresholds[:-1]
            assert torch.all(diff >= 0), f"{cls.__name__} thresholds not monotonic"

    def test_lookup_values_and_indices(self):
        """lookup() returns (values, indices) consistent with forward()."""
        lut = LutReLU(min_val=-100, max_val=100, output_sign=0)
        x = torch.tensor([50, -50])
        values, indices = lut.lookup(x)
        assert values.shape == indices.shape == (2,)
        assert torch.all((indices >= 0) & (indices <= 255))
        assert torch.equal(values, lut(x))

    def test_forward_preserves_shape(self):
        """Forward preserves input shape."""
        lut = LutReLU(min_val=-100, max_val=100, output_sign=0)
        for shape in [(2, 3, 4), (8, 16)]:
            assert lut(torch.randn(*shape)).shape == shape

    def test_forward_outputs_in_lut(self):
        """Forward output values are always entries from lut_values."""
        for cls in _ALL_LUT_CLASSES:
            lut = cls(min_val=-500, max_val=500, output_sign=0)
            y = lut(torch.linspace(-500, 500, 100))
            lut_set = set(lut.lut_values.tolist())
            for val in y.tolist():
                assert val in lut_set, f"{cls.__name__}: {val} not in lut_values"


class TestLutReLU:
    def test_relu_behavior(self):
        """ReLU: negative -> 0, positive -> scaled, monotonic."""
        lut = LutReLU(min_val=-500, max_val=512, output_sign=0)
        assert torch.all(lut(torch.tensor([-100, -1])) == 0)
        y_pos = lut(torch.tensor([100, 200]))
        assert torch.all(y_pos > 0)
        assert y_pos[1] > y_pos[0]

    @pytest.mark.parametrize("sign, max_out", [(0, 255), (1, 127)])
    def test_output_range(self, sign, max_out):
        """Output range respects sign mode; max input maps close to max output."""
        lut = LutReLU(min_val=-500, max_val=500, output_sign=sign)
        y = lut(torch.linspace(-500, 500, 100))
        assert torch.all((y >= 0) & (y <= max_out))
        assert lut(torch.tensor([500])).item() >= max_out - 5

    def test_threshold_concentration_positive(self):
        """254 thresholds in the positive region when min < 0 < max."""
        lut = LutReLU(min_val=-500, max_val=500, output_sign=0)
        assert (lut.thresholds > 0).sum().item() == 254

    def test_float_mode(self):
        """Float mode returns unquantized ReLU-like values."""
        lut = LutReLU(min_val=-10, max_val=10, output_sign=0, is_float=True)
        y = lut(torch.tensor([-5, 0, 5]))
        assert y[0] == 0
        assert abs(y[2] - 5) < 0.5


class TestLutReLUSymmetric:
    def test_symmetric_behavior(self):
        """Symmetric ReLU maps negatives to 0 but keeps uniform bins."""
        lut = LutReLUSymmetric(min_val=-500, max_val=512, output_sign=1)
        assert torch.all(lut(torch.tensor([-100, -1])) == 0)
        y_pos = lut(torch.tensor([100, 200]))
        assert torch.all(y_pos > 0)
        assert y_pos[1] > y_pos[0]

    @pytest.mark.parametrize("sign, max_out", [(0, 255), (1, 127)])
    def test_output_range(self, sign, max_out):
        """Output range respects sign mode."""
        lut = LutReLUSymmetric(min_val=-500, max_val=500, output_sign=sign)
        y = lut(torch.linspace(-500, 500, 100))
        assert torch.all((y >= 0) & (y <= max_out))
        assert lut(torch.tensor([500])).item() >= max_out - 5

    def test_uniform_thresholds(self):
        """Thresholds should be uniformly distributed across min_val and max_val."""
        lut = LutReLUSymmetric(min_val=-500, max_val=500, output_sign=1)
        diffs = (lut.thresholds[1:] - lut.thresholds[:-1]).float()
        assert torch.allclose(
            diffs, torch.tensor([diffs[0].item()] * 255, dtype=torch.float32), atol=1.0
        )

    def test_float_mode(self):
        """Float mode returns unquantized ReLU-like values."""
        lut = LutReLUSymmetric(min_val=-10, max_val=10, output_sign=1, is_float=True)
        y = lut(torch.tensor([-5, 0, 5]))
        assert y[0] == 0
        assert abs(y[2] - 5) < 0.5


class TestLutLinear:
    def test_signed_mapping_endpoints(self):
        """Linear maps endpoints close to [-128, 127]."""
        lut = LutLinear(min_val=-100, max_val=100, output_sign=1)
        y = lut(torch.tensor([-100, 0, 100]))
        assert abs(y[0] - (-128)) <= 2
        assert abs(y[1] - 0) <= 2
        assert abs(y[2] - 127) <= 2

    def test_float_mode_identity(self):
        """Float mode linear is approximately identity."""
        lut = LutLinear(min_val=-100, max_val=100, output_sign=1, is_float=True)
        y = lut(torch.tensor([-50, 0, 50]))
        for val, ref in zip(y.tolist(), [-50, 0, 50]):
            assert abs(val - ref) < 1


_NONLINEAR_PARAMS = [
    (LutSigmoid, lambda x: 1 / (1 + math.exp(-x)), 0, 255),
    (LutTanh, math.tanh, 1, 127),
    (LutSoftsign, lambda x: x / (1 + abs(x)), 1, 127),
]


class TestNonlinearActivations:
    """Tests for Sigmoid, Tanh, and Softsign LUT activations."""

    @pytest.mark.parametrize("cls, ref_fn, sign, scale", _NONLINEAR_PARAMS)
    @pytest.mark.parametrize("is_float", [False, True])
    def test_monotonic(self, cls, ref_fn, sign, scale, is_float):  # noqa: ARG002
        """Non-linear LUT activations are monotonically non-decreasing."""
        lut = cls(min_val=-500, max_val=500, output_sign=sign, is_float=is_float)
        y = lut(torch.linspace(-500, 500, 200))
        assert torch.all(y[1:] - y[:-1] >= -1e-5)

    @pytest.mark.parametrize("cls, ref_fn, sign, scale", _NONLINEAR_PARAMS)
    def test_float_accuracy(self, cls, ref_fn, sign, scale):  # noqa: ARG002
        """Float mode approximates the math function in [-5, 5]."""
        lut = cls(min_val=-500, max_val=500, output_sign=sign, is_float=True)
        x = torch.linspace(-5, 5, 20)
        y = lut(x)
        ref = torch.tensor([ref_fn(xi.item()) for xi in x])
        assert torch.allclose(
            y.float(), ref, atol=0.15
        ), f"{cls.__name__} max error = {(y.float() - ref).abs().max().item():.4f}"

    @pytest.mark.parametrize("cls, ref_fn, sign, scale", _NONLINEAR_PARAMS)
    def test_int_accuracy(self, cls, ref_fn, sign, scale):
        """Int mode: quantised accuracy at normalised-domain points."""
        min_val, max_val, act_range = -500, 500, 10
        hw_scale = (max_val - min_val) / (2 * act_range)
        lut = cls(min_val=min_val, max_val=max_val, output_sign=sign)

        norm_points = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
        hw_points = [min_val + (xn + act_range) * hw_scale for xn in norm_points]
        y = lut(torch.tensor(hw_points))

        for xn, yi in zip(norm_points, y.tolist()):
            expected = ref_fn(xn) * scale
            assert (
                abs(yi - expected) <= 15
            ), f"{cls.__name__}: at x_norm={xn}, got {yi}, expected ~{expected:.1f}"

    @pytest.mark.parametrize("cls, ref_fn, sign, scale", _NONLINEAR_PARAMS)
    def test_threshold_density_near_zero(
        self, cls, ref_fn, sign, scale
    ):  # noqa: ARG002
        """Non-uniform binning concentrates thresholds near x=0."""
        thresholds = cls(min_val=-500, max_val=500, output_sign=sign).thresholds
        inner_count = ((thresholds >= -50) & (thresholds <= 50)).sum().item()
        assert (
            inner_count > 50
        ), f"{cls.__name__}: only {inner_count} thresholds in [-50, 50]"

    @pytest.mark.parametrize("cls, ref_fn, sign, scale", _NONLINEAR_PARAMS)
    def test_saturation(self, cls, ref_fn, sign, scale):  # noqa: ARG002
        """Activation saturates at extremes (float and int modes)."""
        lut_f = cls(min_val=-500, max_val=500, output_sign=sign, is_float=True)
        lut_i = cls(min_val=-500, max_val=500, output_sign=sign)

        # Float mode: close to math limits
        y_lo_f = lut_f(torch.tensor([-490])).float().item()
        y_hi_f = lut_f(torch.tensor([490])).float().item()
        assert abs(y_lo_f - ref_fn(-490)) < 0.1
        assert abs(y_hi_f - ref_fn(490)) < 0.1

        # Int mode: close to output range limits
        y_lo_i = lut_i(torch.tensor([-490])).item()
        y_hi_i = lut_i(torch.tensor([490])).item()
        if sign == 0:
            assert y_lo_i <= 5
            assert y_hi_i >= 250
        else:
            assert y_lo_i <= -110
            assert y_hi_i >= 110


class TestLutCustom:
    def test_dtypes(self):
        """LutCustom converts to correct dtypes in both modes."""
        thresholds = torch.arange(256, dtype=torch.float32)
        values = torch.arange(256, dtype=torch.float32)

        lut_int = LutCustom(thresholds, values, output_sign=0)
        assert lut_int.thresholds.dtype == torch.int32
        assert lut_int.lut_values.dtype == torch.uint8

        lut_float = LutCustom(thresholds, values, is_float=True)
        assert lut_float.thresholds.dtype == torch.float32
        assert lut_float.lut_values.dtype == torch.bfloat16

    @pytest.mark.parametrize(
        "n_thres, n_vals, match",
        [(100, 256, "256 thresholds"), (256, 100, "256 values")],
    )
    def test_reject_wrong_length(self, n_thres, n_vals, match):
        """Raises ValueError for non-256 threshold or value length."""
        with pytest.raises(ValueError, match=match):
            LutCustom(torch.zeros(n_thres), torch.zeros(n_vals))

    def test_lookup_and_boundary(self):
        """LutCustom maps inputs correctly, including boundary clamping."""
        thresholds = torch.arange(256, dtype=torch.float32)
        values = torch.arange(256, dtype=torch.float32)
        lut = LutCustom(thresholds, values)

        assert lut(torch.tensor([100.5])).item() == 100
        assert lut(torch.tensor([-100])).item() == 0  # below min
        assert lut(torch.tensor([999])).item() == 255  # above max

    def test_step_function(self):
        """Non-monotonic step function: first half -> 0, second half -> 100."""
        thresholds = torch.arange(256, dtype=torch.float32)
        values = torch.cat([torch.zeros(128), torch.full((128,), 100)])
        lut = LutCustom(thresholds, values)
        y = lut(torch.tensor([50, 180]))
        assert y[0].item() == 0
        assert y[1].item() == 100

    def test_auto_output_sign(self):
        """Auto-detect output_sign from values; explicit override works."""
        lut_signed = LutCustom(
            torch.arange(256, dtype=torch.float32), torch.linspace(-128, 127, 256)
        )
        assert lut_signed.output_sign == 1

        lut_unsigned = LutCustom(
            torch.arange(256, dtype=torch.float32), torch.linspace(0, 255, 256)
        )
        assert lut_unsigned.output_sign == 0

        lut_override = LutCustom(
            torch.arange(256, dtype=torch.float32),
            torch.linspace(-128, 127, 256),
            output_sign=0,
        )
        assert lut_override.output_sign == 0
