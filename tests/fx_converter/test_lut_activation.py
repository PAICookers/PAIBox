import math
import os

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

from paibox.fx_converter.lut_activation import (
    LutLinear,
    LutReLU,
    LutSigmoid,
    LutSoftsign,
    LutTanh,
)

matplotlib.use("Agg")

# Toggle for plotting and CSV export
ENABLE_VISUALIZATION = False
OUTPUT_DIR = "output_test_ops"


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def plot_activation(
    activation_cls, name, min_val, max_val, output_sign, plot_range=None, is_float=False
):
    if not ENABLE_VISUALIZATION:
        return

    ensure_output_dir()

    mode_str = "Float" if is_float else "Int"
    print(f"Plotting {name} ({mode_str})...", end=" ")

    # Initialize implementation
    try:
        activation = activation_cls(
            min_val=min_val, max_val=max_val, output_sign=output_sign, is_float=is_float
        )
    except TypeError:
        # Fallback for classes that might not support is_float yet if any
        activation = activation_cls(
            min_val=min_val, max_val=max_val, output_sign=output_sign
        )

    if plot_range is None:
        plot_min, plot_max = min_val, max_val
    else:
        plot_min, plot_max = plot_range

    # Create input tensor (dense range for smooth plotting)
    x = torch.linspace(plot_min, plot_max, 1000)

    # Get output
    with torch.no_grad():
        y = activation(x)

    # Plot
    plt.figure(figsize=(10, 6))
    if is_float:
        plt.plot(x.numpy(), y.float().numpy(), label=f"{name} ({mode_str} LUT)")
    else:
        plt.plot(x.numpy(), y.numpy(), label=f"{name} ({mode_str} LUT)")

    plt.title(
        f"{name} Activation ({mode_str} LUT)\nRange: [{min_val}, {max_val}], Output Sign: {output_sign}"
    )
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save plot
    suffix = "_float" if is_float else ""
    filename = os.path.join(OUTPUT_DIR, f"lut_activation_{name.lower()}{suffix}.png")
    plt.savefig(filename)
    plt.close()

    print(f"Saved to {filename}")


def export_lut_table(
    activation_cls, name, min_val, max_val, output_sign, is_float=False
):
    if not ENABLE_VISUALIZATION:
        return

    ensure_output_dir()

    suffix = "_float" if is_float else ""
    filename = os.path.join(OUTPUT_DIR, f"lut_table_{name.lower()}{suffix}.csv")
    mode_str = "Float" if is_float else "Int"
    print(f"Exporting LUT table for {name} ({mode_str}) to {filename}...", end=" ")

    # Initialize
    try:
        activation = activation_cls(
            min_val=min_val, max_val=max_val, output_sign=output_sign, is_float=is_float
        )
    except TypeError:
        activation = activation_cls(
            min_val=min_val, max_val=max_val, output_sign=output_sign
        )

    thresholds = activation.thresholds.tolist()
    lut_values = activation.lut_values.tolist()

    # We want 256 rows.
    # thresholds has 256 items. T[0]...T[255].
    # lut_values has 256 items. L[0]...L[255].
    # Row i: T[i], L[i]

    with open(filename, "w") as f:
        f.write("Membrane_Potential,Output_Value\n")

        for i in range(256):
            current_threshold = thresholds[i]
            if is_float:
                val = lut_values[i]
                f.write(f"{current_threshold},{val}\n")
            else:
                val = int(lut_values[i])
                f.write(f"{current_threshold},{val}\n")

    print(f"Saved to {filename}")


class TestLutReLU:
    def test_init(self):
        lut = LutReLU(min_val=-500, max_val=512, output_sign=0)
        assert lut.thresholds.shape == (256,)
        assert lut.lut_values.shape == (256,)
        # Check that thresholds are monotonically increasing
        diff = lut.thresholds[1:] - lut.thresholds[:-1]
        assert torch.all(diff >= 0)

    def test_init_float(self):
        lut = LutReLU(min_val=-500, max_val=512, output_sign=0, is_float=True)
        assert lut.thresholds.shape == (256,)
        assert lut.lut_values.shape == (256,)
        assert lut.thresholds.dtype == torch.float32
        assert lut.lut_values.dtype == torch.bfloat16

    def test_forward_unsigned(self):
        min_val = -500
        max_val = 512  # Output unsigned [0, 255].
        # ReLU: x < 0 -> 0. x > 0 -> x * (255/127)
        lut = LutReLU(min_val=min_val, max_val=max_val, output_sign=0)
        # Plotting
        plot_activation(LutReLU, "ReLU", min_val, max_val, output_sign=0)
        export_lut_table(LutReLU, "ReLU", min_val, max_val, output_sign=0)

        input_tensor = torch.tensor([-5.0, 0.0, 6.0, 9.0])
        output = lut(input_tensor)

        # Expected:
        # -5 -> 0
        # 0 -> 0
        # 5 -> 5 * (255/127) = 10.04 -> 10 (approx)
        # 10 -> 20 (approx)
        assert output[0] == 0
        assert 0 <= output[1] <= 1  # Near 0
        assert output[2] == 3
        assert output[3] == 4

    def test_forward_float(self):
        # Float mode
        min_val = -10
        max_val = 10
        lut = LutReLU(min_val=min_val, max_val=max_val, output_sign=0, is_float=True)

        plot_activation(LutReLU, "ReLU", min_val, max_val, output_sign=0, is_float=True)
        export_lut_table(
            LutReLU, "ReLU", min_val, max_val, output_sign=0, is_float=True
        )

        input_tensor = torch.tensor([-5.0, 5.0, 7.58])
        output = lut(input_tensor)

        # Float mode returns unquantized values (but looked up from LUT)
        # ReLU(-5) = 0
        # ReLU(5) = 5
        assert output[0] == 0
        assert abs(output[1] - 5.0) < 0.1


class TestLutLinear:
    def test_linear_mapping(self):
        min_val = -100
        max_val = 100
        # Map [-100, 100] to [-128, 127]
        lut = LutLinear(min_val=min_val, max_val=max_val, output_sign=1)

        # Plotting
        plot_activation(LutLinear, "Linear", min_val, max_val, output_sign=1)
        export_lut_table(LutLinear, "Linear", min_val, max_val, output_sign=1)

        input_tensor = torch.tensor([-100, 0, 100])
        output = lut(input_tensor)

        # -10 -> -128
        # 0 -> -0.5 (approx 0)
        # 10 -> 127

        # Tolerances due to LUT quantization
        assert abs(output[0] - (-128)) <= 2
        assert abs(output[1] - 0) <= 2
        assert abs(output[2] - 127) <= 2

    def test_linear_mapping_float(self):
        min_val = -100
        max_val = 100
        lut = LutLinear(min_val=min_val, max_val=max_val, output_sign=1, is_float=True)

        plot_activation(
            LutLinear, "Linear", min_val, max_val, output_sign=1, is_float=True
        )
        export_lut_table(
            LutLinear, "Linear", min_val, max_val, output_sign=1, is_float=True
        )

        input_tensor = torch.tensor([-50.0, 0.0, 50.0])
        output = lut(input_tensor)

        # Float mode Linear is Identity mapping for now (based on lut_ops.py implementation analysis)
        # lut_ops.py:
        #   midpoint = self.min_val + (i + 0.5) * step
        #   values.append(midpoint)
        # So it just reconstructs input ~ output

        # -50 -> -50
        # 0 -> 0

        assert abs(output[0] - (-50.0)) < 1.0  # Tolerance for binning
        assert abs(output[1] - 0.0) < 1.0


class TestAdaptiveActivations:
    @pytest.mark.parametrize(
        "act_cls, func, min_v, max_v",
        [
            (LutSigmoid, lambda x: 1 / (1 + math.exp(-x)), -500, 500),
            (LutTanh, math.tanh, -500, 500),
            (LutSoftsign, lambda x: x / (1 + abs(x)), -500, 500),
        ],
    )
    @pytest.mark.parametrize("is_float", [False, True])
    def test_shape(self, act_cls, func, min_v, max_v, is_float):
        # Use unsigned for sigmoid (0-1 -> 0-255)
        # Use signed for others (-1-1 -> -128-127)
        if act_cls == LutSigmoid:
            lut = act_cls(
                min_val=min_v, max_val=max_v, output_sign=0, is_float=is_float
            )
            target_scale = 255.0
            target_min = 0
            # Generate visualization for Sigmoid
            plot_activation(
                LutSigmoid, "Sigmoid", min_v, max_v, output_sign=0, is_float=is_float
            )
            export_lut_table(
                LutSigmoid, "Sigmoid", min_v, max_v, output_sign=0, is_float=is_float
            )
        else:
            lut = act_cls(
                min_val=min_v, max_val=max_v, output_sign=1, is_float=is_float
            )
            target_scale = 127.0  # Tanh/Softsign map to ~[-127, 127]
            target_min = -128
            # Generate visualization for others
            act_name = act_cls.__name__.replace("Lut", "")
            plot_activation(
                act_cls, act_name, min_v, max_v, output_sign=1, is_float=is_float
            )
            export_lut_table(
                act_cls, act_name, min_v, max_v, output_sign=1, is_float=is_float
            )

        # Test monotonic behavior
        x = torch.linspace(min_v, max_v, 50)
        y = lut(x)

        # Check monotonicity
        diff = y[1:] - y[:-1]

        # Allow some flat regions (diff=0), but shouldn't be very negative
        # In float mode, it should be strictly monotonic or flat.
        # In int mode, quantization noise might cause flat regions.
        assert torch.all(diff >= -1e-5)  # Small epsilon for float tolerances

        # Check midpoint
        mid_val = (min_v + max_v) / 2
        # Usually 0 for tanh/softsign/sigmoid symmetric inputs
        if abs(mid_val) < 1e-5:
            mid_out = lut(torch.tensor([0.0]))
            val = mid_out.item()

            if act_cls == LutSigmoid:
                # Sigmoid(0) = 0.5
                # Int: 127.5 -> 127 or 128
                # Float: 0.5
                if is_float:
                    assert 0.45 <= val <= 0.55
                else:
                    assert 120 <= val <= 135
            else:
                # Tanh(0) = 0
                if is_float:
                    assert abs(val) <= 0.1
                else:
                    assert abs(val) <= 2

    def test_sigmoid_range(self):
        lut = LutSigmoid(min_val=-500, max_val=500, output_sign=0)
        # Large negative -> 0
        # Use values closer to the boundary (-500)
        assert lut(torch.tensor([-490.0])).item() <= 5
        # Large positive -> 255
        assert lut(torch.tensor([490.0])).item() >= 250

    def test_sigmoid_range_float(self):
        lut = LutSigmoid(min_val=-500, max_val=500, output_sign=0, is_float=True)
        assert lut(torch.tensor([-490.0])).item() <= 0.01
        assert lut(torch.tensor([490.0])).item() >= 0.99

    def test_tanh_range(self):
        lut = LutTanh(min_val=-500, max_val=500, output_sign=1)  # Signed
        # Tanh(-500) ~ -1 -> -127
        assert lut(torch.tensor([-490.0])).item() <= -120
        # Tanh(500) ~ 1 -> 127
        assert lut(torch.tensor([490.0])).item() >= 120

    def test_tanh_range_float(self):
        lut = LutTanh(min_val=-500, max_val=500, output_sign=1, is_float=True)
        assert lut(torch.tensor([-490.0])).item() <= -0.9
        assert lut(torch.tensor([490.0])).item() >= 0.9

    def test_softsign_range(self):
        lut = LutSoftsign(min_val=-500, max_val=500, output_sign=1)
        # Softsign(-500) ~ -1 -> -127
        assert lut(torch.tensor([-490.0])).item() <= -110
        # Softsign(500) ~ 1 -> 127
        assert lut(torch.tensor([490.0])).item() >= 110

    def test_softsign_range_float(self):
        lut = LutSoftsign(min_val=-500, max_val=500, output_sign=1, is_float=True)
        # Softsign(-490) ~ -1
        assert lut(torch.tensor([-490.0])).item() <= -0.9
        assert lut(torch.tensor([490.0])).item() >= 0.9
