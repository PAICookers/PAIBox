
import math
import torch
import pytest
import matplotlib.pyplot as plt
from paibox.ops.lut_ops import (
    LutReLU,
    LutLinear,
    LutSigmoid,
    LutTanh,
    LutSoftsign
)
import matplotlib
matplotlib.use('Agg')

# Toggle for plotting and CSV export
ENABLE_VISUALIZATION = False


def plot_activation(activation_cls, name, min_val, max_val, output_sign, plot_range=None):
    if not ENABLE_VISUALIZATION:
        return

    print(f"Plotting {name}...", end=" ")

    # Initialize implementation
    activation = activation_cls(
        min_val=min_val, max_val=max_val, output_sign=output_sign)

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
    plt.plot(x.numpy(), y.numpy(), label=f'{name} (LUT)')
    plt.title(
        f'{name} Activation (LUT Approximation)\nRange: [{min_val}, {max_val}], Output Sign: {output_sign}')
    plt.xlabel('Input')
    plt.ylabel('Output (8-bit quantized)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save plot
    filename = f"lut_activation_{name.lower()}.png"
    plt.savefig(filename)
    plt.close()

    print(f"Saved to {filename}")


def export_lut_table(activation_cls, name, min_val, max_val, output_sign):
    if not ENABLE_VISUALIZATION:
        return

    filename = f"lut_table_{name.lower()}.csv"
    print(f"Exporting LUT table for {name} to {filename}...", end=" ")

    # Initialize
    activation = activation_cls(
        min_val=min_val, max_val=max_val, output_sign=output_sign)

    thresholds = activation.thresholds.tolist()
    lut_values = activation.lut_values.tolist()

    # We want 256 rows.
    # The thresholds list has 255 values, separating the 256 bins.
    # Bin i corresponds to the range [thresholds[i-1], thresholds[i]).
    # (With Bin 0 being [min_val, thresholds[0]), and Bin 255 being [thresholds[254], max_val])
    # For export, let's use the start value of the bin range as "Membrane_Potential".

    with open(filename, 'w') as f:
        f.write("Membrane_Potential,Output_Value\n")

        # Bin 0
        current_threshold = min_val
        val = int(lut_values[0])
        f.write(f"{current_threshold},{val}\n")

        # Bins 1 to 255
        for i in range(1, 256):
            current_threshold = thresholds[i-1]
            val = int(lut_values[i])
            f.write(f"{(current_threshold)},{val}\n")

    print(f"Saved to {filename}")


class TestLutReLU:
    def test_init(self):
        lut = LutReLU(min_val=-128, max_val=127, output_sign=0)
        assert lut.thresholds.shape == (255,)
        assert lut.lut_values.shape == (256,)
        # Check that thresholds are monotonically increasing
        diff = lut.thresholds[1:] - lut.thresholds[:-1]
        assert torch.all(diff >= 0)

    def test_forward_unsigned(self):
        min_val = -128
        max_val = 255  # Output unsigned [0, 255].
        # ReLU: x < 0 -> 0. x > 0 -> x * (255/127)
        lut = LutReLU(min_val=min_val, max_val=max_val, output_sign=0)
        # Plotting
        plot_activation(LutReLU, "ReLU", min_val, max_val, output_sign=0)
        export_lut_table(LutReLU, "ReLU", min_val, max_val, output_sign=0)

        input_tensor = torch.tensor([-5.0, 0.0, 5.0, 10.0])
        output = lut(input_tensor)

        # Expected:
        # -5 -> 0
        # 0 -> 0
        # 5 -> 5 * (255/127) = 10.04 -> 10 (approx)
        # 10 -> 20 (approx)
        assert output[0] == 0
        assert 0 <= output[1] <= 1  # Near 0
        assert output[2] == 5
        assert output[3] == 10

    def test_forward_signed(self):
        # min_val=-10, max_val=10. Output signed [-128, 127].
        # ReLU maps 10 -> 127.
        lut = LutReLU(min_val=-10, max_val=10, output_sign=1)

        input_tensor = torch.tensor([-5.0, 5.0, 10.0])
        output = lut(input_tensor)

        assert output[0] == 0  # ReLu is 0 for neg
        assert 60 <= output[1] <= 70  # Half of 127 approx
        assert output[2] == 127


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


class TestAdaptiveActivations:
    @pytest.mark.parametrize("act_cls, func, min_v, max_v", [
        (LutSigmoid, lambda x: 1 / (1 + math.exp(-x)), -500, 500),
        (LutTanh, math.tanh, -500, 500),
        (LutSoftsign, lambda x: x / (1 + abs(x)), -500, 500)
    ])
    def test_shape(self, act_cls, func, min_v, max_v):
        # Use unsigned for sigmoid (0-1 -> 0-255)
        # Use signed for others (-1-1 -> -128-127)
        if act_cls == LutSigmoid:
            lut = act_cls(min_val=min_v, max_val=max_v, output_sign=0)
            target_scale = 255.0
            target_min = 0
            # Generate visualization for Sigmoid
            plot_activation(LutSigmoid, "Sigmoid", min_v, max_v, output_sign=0)
            export_lut_table(LutSigmoid, "Sigmoid",
                             min_v, max_v, output_sign=0)
        else:
            lut = act_cls(min_val=min_v, max_val=max_v, output_sign=1)
            target_scale = 127.0  # Tanh/Softsign map to ~[-127, 127]
            target_min = -128
            # Generate visualization for others
            act_name = act_cls.__name__.replace('Lut', '')
            plot_activation(act_cls, act_name, min_v, max_v, output_sign=1)
            export_lut_table(act_cls, act_name, min_v, max_v, output_sign=1)

        # Test monotonic behavior
        x = torch.linspace(min_v, max_v, 50)
        y = lut(x)

        # Check monotonicity
        diff = y[1:] - y[:-1]
        # Allow some flat regions (diff=0), but shouldn't be very negative
        assert torch.all(diff >= -1)

        # Check midpoint
        mid_val = (min_v + max_v) / 2
        # Usually 0 for tanh/softsign/sigmoid symmetric inputs
        if abs(mid_val) < 1e-5:
            mid_out = lut(torch.tensor([0.0]))
            if act_cls == LutSigmoid:
                # Sigmoid(0) = 0.5 -> 127.5
                assert 120 <= mid_out.item() <= 135
            else:
                # Tanh(0) = 0
                assert abs(mid_out.item()) <= 2

    def test_sigmoid_range(self):
        lut = LutSigmoid(min_val=-500, max_val=500, output_sign=0)
        # Large negative -> 0
        # Use values closer to the boundary (-500)
        assert lut(torch.tensor([-490.0])).item() <= 5
        # Large positive -> 255
        assert lut(torch.tensor([490.0])).item() >= 250

    def test_tanh_range(self):
        lut = LutTanh(min_val=-500, max_val=500, output_sign=1)  # Signed
        # Tanh(-500) ~ -1 -> -127
        assert lut(torch.tensor([-490.0])).item() <= -120
        # Tanh(500) ~ 1 -> 127
        assert lut(torch.tensor([490.0])).item() >= 120

    def test_softsign_range(self):
        lut = LutSoftsign(min_val=-500, max_val=500, output_sign=1)
        # Softsign(-500) ~ -1 -> -127
        assert lut(torch.tensor([-490.0])).item() <= -110
        # Softsign(500) ~ 1 -> 127
        assert lut(torch.tensor([490.0])).item() >= 110
