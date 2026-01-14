import math

import torch
import torch.nn as nn

__all__ = [
    "LutActivation",
    "LutAdaptiveActivation",
    "LutReLU",
    "LutLinear",
    "LutSigmoid",
    "LutTanh",
    "LutSoftsign",
    "LutAdaptiveReLU",
]


class LutActivation(nn.Module):
    def __init__(
        self,
        min_val: int = -2147483648,
        max_val: int = 2147483647,
        output_sign: int = 0,
    ):
        """
        Base class for LUT-based activation functions.
        Approximates activation using a Lookup Table with 256 bins.

        Args:
            min_val: Minimum input value for the range subdivision.
            max_val: Maximum input value for the range subdivision.
            output_sign: 0 for unsigned output [0, 255], 1 for signed output [-128, 127].
        """
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.output_sign = output_sign

        # Register buffers for thresholds (255 values) and LUT values (256 values)
        # 255 thresholds separate the 256 bins.
        self.register_buffer("thresholds", torch.zeros(255))
        self.register_buffer("lut_values", torch.zeros(256))

        # Generate the LUT on initialization
        self.generate_lut()

    def _clamp_value(self, value: float) -> int:
        """Clamps the value to the target 8-bit range."""
        if self.output_sign == 0:
            # Unsigned: 0 to 255
            return max(0, min(255, int(round(value))))
        else:
            # Signed: -128 to 127
            return max(-128, min(127, int(round(value))))

    def _generate_uniform_thresholds(self):
        """Generates uniformly spaced thresholds between min_val and max_val."""
        # 256 bins -> 256 steps.
        # Thresholds are at min + step, min + 2*step, ... min + 255*step
        step = (self.max_val - self.min_val) / 256.0
        # Round thresholds to nearest integer
        thresholds = [round(self.min_val + (i + 1) * step) for i in range(255)]
        self.thresholds.copy_(torch.tensor(thresholds, dtype=self.thresholds.dtype))
        return step

    def generate_lut(self):
        """Generates thresholds and lut_values. Must be implemented by subclasses."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulates the hardware LUT lookup.
        1. Finds the bin index for each input element.
        2. Retrieves the output value from the LUT.
        """
        # torch.bucketize finds indices such that self.thresholds[i-1] < x <= self.thresholds[i]
        # (with right=True).
        # If x <= thresholds[0], index is 0.
        # This matches the behavior where thresholds represent the upper bound of the bin.
        indices = torch.bucketize(x, self.thresholds, right=True)
        return self.lut_values[indices]


class LutReLU(LutActivation):
    def generate_lut(self):
        # ReLU behaves differently for x < 0 and x > 0.
        # x < 0: Output is always 0.
        # x > 0: Output is linear.
        # To avoid wasting bins on the negative side (which all map to 0),
        # we concentrate the thresholds in the positive region [0, max_val].

        # Strategy:
        # Bin 0: Covers [min_val, 0].
        # Bins 1-255: Uniformly divide [0, max_val].

        thresholds = []

        # Assuming standard case where min_val < 0 and max_val > 0.
        if self.min_val < 0 and self.max_val > 0:
            # 1. First threshold at 0.
            # Ranges:
            # Bin 0: [min_val, 0) -> Output 0
            # Bin 1: [0, t1) -> Output > 0
            # ...

            # We have 255 thresholds to place.
            # Let's place one at 0 (or very close to it).
            thresholds.append(0.0)

            # Remaining 254 thresholds divide [0, max_val] uniformly.
            # This creates 255 intervals in [0, max_val].
            # Total bins: 1 (negative) + 255 (positive) = 256. Perfect.

            num_pos_intervals = 255
            step = self.max_val / num_pos_intervals
            for i in range(1, 255):  # 1 to 254 (254 thresholds)
                t = round(i * step)
                thresholds.append(t)

        else:
            # Fallback to uniform if range is all positive or all negative
            step = (self.max_val - self.min_val) / 256.0
            thresholds = [round(self.min_val + (i + 1) * step) for i in range(255)]

        self.thresholds.copy_(torch.tensor(thresholds, dtype=self.thresholds.dtype))

        # Calculate Scale
        # ReLU mapping:
        # If output_sign=0 (Unsigned), map max_val -> 255.
        # If output_sign=1 (Signed), map max_val -> 127.
        if self.max_val > 0:
            if self.output_sign == 0:
                scale = 255.0 / self.max_val
            else:
                scale = 127.0 / self.max_val
        else:
            scale = 0.0

        # Compute LUT Values based on bins defined by thresholds
        full_boundaries = [self.min_val] + thresholds + [self.max_val]
        values = []

        for i in range(256):
            low = full_boundaries[i]
            high = full_boundaries[i + 1]
            if low >= high:
                mid_input = low
            else:
                mid_input = (low + high) / 2.0

            # ReLU function
            val = max(0.0, mid_input)
            val = val * scale
            # Use floor to avoid even/odd steps from banker's rounding at x.5
            # This ensures smooth 0, 1, 2... steps instead of 0, 2, 2, 4...
            val_floor = int(val)
            values.append(self._clamp_value(val_floor))

        self.lut_values.copy_(torch.tensor(values, dtype=self.lut_values.dtype))


class LutLinear(LutActivation):
    def generate_lut(self):
        # Linear (Liner) Requirement:
        # - Map all int (the interval [min_val, max_val]) to -128 * 127.
        # - Usually implies linear mapping from input range to output range.

        step = self._generate_uniform_thresholds()

        # Mapping [min_val, max_val] -> [-128, 127]
        # Slope = (OutMax - OutMin) / (InMax - InMin)
        #       = (127 - (-128)) / (max_val - min_val)
        #       = 255 / (max_val - min_val)

        out_min = -128.0
        out_max = 127.0

        if self.max_val > self.min_val:
            slope = (out_max - out_min) / (self.max_val - self.min_val)
        else:
            slope = 0.0

        values = []
        for i in range(256):
            midpoint = self.min_val + (i + 0.5) * step

            # Apply Linear Mapping
            # val = out_min + (midpoint - min_val) * slope
            val = out_min + (midpoint - self.min_val) * slope

            values.append(self._clamp_value(val))

        self.lut_values.copy_(torch.tensor(values, dtype=self.lut_values.dtype))


class LutAdaptiveActivation(LutActivation):
    """
    Base class for activations that use non-uniform binning based on an inverse function.
    This ensures better precision in regions where the function changes rapidly.
    """

    def __init__(
        self,
        min_val: int = -2147483648,
        max_val: int = 2147483647,
        output_sign: int = 0,
        act_range: float = 10.0,
    ):
        self.act_range = act_range
        super().__init__(min_val, max_val, output_sign)

    def get_math_range(self):
        """Returns the output range of the mathematical function (min, max)."""
        raise NotImplementedError

    def get_forward_func(self):
        """Returns the forward mathematical function y = f(x)."""
        raise NotImplementedError

    def get_inverse_func(self):
        """Returns the inverse function x = f^(-1)(y)."""
        raise NotImplementedError

    def get_output_scale_func(self):
        """Returns a function to map math value to LUT integer value."""
        raise NotImplementedError

    def normalize_input(self, inp, scale):
        """Maps input value to normalized domain [-act_range, act_range]."""
        # normalized = -R + (inp - min) * (2R / range)
        # R = self.act_range
        # scale = range / (2R) = 1 / (2R/range)
        # So (inp - min) / scale -> normalized but shifted by R
        # Normalized = (inp - min) / scale - R
        return (inp - self.min_val) / scale - self.act_range

    def generate_lut(self):
        """
        Generate LUT with non-uniform binning.
        """
        y_min, y_max = self.get_math_range()
        inverse_func = self.get_inverse_func()
        forward_func = self.get_forward_func()
        output_scale_func = self.get_output_scale_func()

        # 1. Determine thresholds in the normalized domain `[-act_range, act_range]`
        # We want outputs uniformly distributed in [y_min, y_max]

        normalized_thresholds = []
        for i in range(1, 256):
            # Fraction of the full range
            frac = i / 256.0
            p = y_min + frac * (y_max - y_min)

            # Compute inverse to find x such that f(x) = p
            # We must clamp p slightly to avoid domain errors if y_min/max are asymptotes
            p_clamped = max(y_min + 1e-6, min(y_max - 1e-6, p))

            t_norm = inverse_func(p_clamped)
            normalized_thresholds.append(t_norm)

        # 2. Map normalized thresholds to input domain [min_val, max_val]
        R = self.act_range
        input_range = self.max_val - self.min_val

        if input_range <= 0:
            self.thresholds.fill_(self.min_val)
            self.lut_values.fill_(0)
            return

        scale = input_range / (2.0 * R)

        thresholds = []
        for t_norm in normalized_thresholds:
            t_in = self.min_val + (t_norm + R) * scale
            t_in = max(self.min_val, min(self.max_val, t_in))
            thresholds.append(round(t_in))

        self.thresholds.copy_(torch.tensor(thresholds, dtype=self.thresholds.dtype))

        # 3. Compute LUT values
        full_boundaries = [self.min_val] + thresholds + [self.max_val]
        values = []

        for i in range(256):
            low = full_boundaries[i]
            high = full_boundaries[i + 1]
            if low >= high:
                mid_input = low
            else:
                mid_input = (low + high) / 2.0

            # Map input to normalized domain
            mid_norm = self.normalize_input(mid_input, scale)

            # Evaluate function
            # Handle out of range for math functions if needed (e.g. exp overflow)
            # Usually strict bounds [-R, R] prevent this if R is reasonable (<=10)
            if mid_norm > 20:
                mid_norm = 20
            elif mid_norm < -20:
                mid_norm = -20

            y_val = forward_func(mid_norm)

            # Scale and Clamp
            val = output_scale_func(y_val)

            values.append(self._clamp_value(val))

        self.lut_values.copy_(torch.tensor(values, dtype=self.lut_values.dtype))


class LutSigmoid(LutAdaptiveActivation):
    def get_math_range(self):
        return (0.0, 1.0)

    def get_forward_func(self):
        return lambda x: 1.0 / (1.0 + math.exp(-x))

    def get_inverse_func(self):
        # logit
        return lambda p: math.log(p / (1.0 - p))

    def get_output_scale_func(self):
        return lambda x: x * 255.0


class LutTanh(LutAdaptiveActivation):
    def get_math_range(self):
        return (-1.0, 1.0)

    def get_forward_func(self):
        return math.tanh

    def get_inverse_func(self):
        return math.atanh

    def get_output_scale_func(self):
        # Maps [-1, 1] to [-127, 127] roughly for output calculation.
        # But _clamp_value will handle sign eventually.
        # Tanh output is [-1, 1].
        # If output_sign=1 (Signed), we want [-128, 127]. So *127 or *128? Usually *127 for symmetry or 128 for full.
        # Let's align with Linear: * 127.
        # If output_sign=0 (Unsigned), Tanh [-1, 1] -> [-127, 127] -> Clamp(0, 255) -> [0, 127].
        # Negative part is clipped. This is standard ReLU-like behavior if unsigned requested.
        return lambda x: x * 127.0


class LutSoftsign(LutAdaptiveActivation):
    def get_math_range(self):
        return (-1.0, 1.0)

    def get_forward_func(self):
        # Softsign: x / (1 + |x|)
        return lambda x: x / (1.0 + abs(x))

    def get_inverse_func(self):
        # y = x / (1 + |x|)
        # If y > 0, y = x / (1 + x)  => y + yx = x => y = x(1-y) => x = y/(1-y)
        # If y < 0, y = x / (1 - x)  => y - yx = x => y = x(1+y) => x = y/(1+y)
        # Combined: x = y / (1 - |y|)
        return lambda y: y / (1.0 - abs(y))

    def get_output_scale_func(self):
        # Range (-1, 1) -> (-127, 127)
        return lambda x: x * 127.0


class LutAdaptiveReLU(LutAdaptiveActivation):
    def get_math_range(self):
        # ReLU range: [max(0, min_val), max(0, max_val)]
        # If max_val < 0, range is [0, 0]
        y_min = max(0.0, self.min_val)
        y_max = max(0.0, self.max_val)
        return (y_min, y_max)

    def _get_scale(self):
        rng = self.max_val - self.min_val
        if rng <= 0:
            return 1.0
        return rng / (2.0 * self.act_range)

    def get_forward_func(self):
        scale = self._get_scale()
        # f(x_norm) -> y_real
        # x_norm domain is [-act_range, act_range] approx
        # x_real = min_val + (x_norm + act_range) * scale
        return lambda x_norm: max(0.0, self.min_val + (x_norm + self.act_range) * scale)

    def get_inverse_func(self):
        scale = self._get_scale()
        # f^-1(y_real) -> x_norm
        # For ReLU, if y > 0, x = y.
        # So x_real = y_real.
        # x_norm = (x_real - min_val)/scale - act_range
        return lambda y_real: (y_real - self.min_val) / scale - self.act_range

    def get_output_scale_func(self):
        # We need to map y_real to quantized int [0..255] or [-128..127]
        # Standard logic: map max_val to max_int (if linear)
        if self.max_val > 0:
            if self.output_sign == 0:
                s = 255.0 / self.max_val
            else:
                s = 127.0 / self.max_val
        else:
            s = 0.0
        return lambda y: y * s
