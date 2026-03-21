"""LUT activation functions.

Discretise continuous activation functions into 256-bin lookup tables,
simulating the v2.5 chip LUT computation. Used as the ``lut`` component
inside :class:`~paibox.paiir.ir.core_neuron.CoreNeuronV25` for ANN mode.

Example::

    from paibox.paiir import ANNNodeV25, LutReLU

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        ANNNodeV25(LutReLU()),
    )
"""

import math
from abc import abstractmethod
from collections.abc import Callable

import torch
from paicorelib.utils import _mask
from torch import Tensor, nn

from .calc_params import LutData

_DEFAULT_MIN = ~_mask(31)
_DEFAULT_MAX = _mask(31)

__all__ = [
    "LutActivation",
    "LutAdaptiveActivation",
    "LutCustom",
    "LutReLU",
    "LutLinear",
    "LutSigmoid",
    "LutTanh",
    "LutSoftsign",
]


class LutActivation(nn.Module):
    thresholds: Tensor
    lut_values: Tensor

    def __init__(
        self,
        min_val: float = _DEFAULT_MIN,
        max_val: float = _DEFAULT_MAX,
        output_sign: int = 0,
        is_float: bool = False,
    ) -> None:
        """Base class for 256-bin LUT activation functions.

        Partitions the input space into 256 bins using thresholds, each bin
        mapping to a pre-computed output value.

        Subclasses override :meth:`generate_lut` to return a ``(thresholds,
        values)`` pair of 256-entry tensors.  The base ``__init__`` registers
        them as buffers with the correct dtype:

        * ``is_float=True``: thresholds ``float32``, values ``bfloat16``.
        * ``is_float=False``: thresholds ``int32``, values ``uint8``
          (unsigned) or ``int8`` (signed).

        Args:
            min_val: Lower bound of the input domain.
            max_val: Upper bound of the input domain.
            output_sign: 0 = unsigned [0, 255], 1 = signed [-128, 127].
            is_float: Use float32 thresholds and bfloat16 values when True.
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.output_sign = output_sign
        self.is_float = is_float

        thresholds, values = self.generate_lut()

        if self.is_float:
            self.register_buffer("thresholds", thresholds.to(torch.float32))
            self.register_buffer("lut_values", values.to(torch.bfloat16))
        else:
            vals_dtype = torch.uint8 if self.output_sign == 0 else torch.int8
            self.register_buffer("thresholds", thresholds.round().to(torch.int32))
            self.register_buffer("lut_values", values.round().to(vals_dtype))

    def _clamp_value(self, value: float) -> int | float:
        """Clamp output to the target 8-bit range."""
        if self.is_float:
            return value
        if self.output_sign == 0:
            # Unsigned: 0 to 255
            return max(0, min(255, int(round(value))))
        else:
            # Signed: -128 to 127
            return max(-128, min(127, int(round(value))))

    def generate_lut(self) -> tuple[Tensor, Tensor]:
        """Generate thresholds and LUT values.

        Subclasses must override this method to return a ``(thresholds,
        values)`` pair of 256-entry float tensors.  The base ``__init__``
        converts them to the appropriate dtype.

        Returns:
            A ``(thresholds, values)`` tuple of 256-entry tensors.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    def lookup(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Simulate hardware LUT lookup, returning both values and indices.

        ``torch.bucketize`` with ``right=True`` finds indices such that
        ``thresholds[i-1] <= x < thresholds[i]``.  Subtracting 1 gives the
        bin index, so if *x* falls in ``[thresholds[i], thresholds[i+1])``
        bucketize returns ``i+1`` and ``(i+1) - 1 = i``, yielding the
        correct LUT entry.

        Returns:
            A ``(values, indices)`` tuple where *values* are the LUT outputs
            and *indices* are the bin indices used for chip-accurate reset.
        """
        indices = torch.bucketize(x, self.thresholds, right=True) - 1
        indices = indices.clamp(0, 255)
        return self.lut_values[indices], indices

    def forward(self, x: Tensor) -> Tensor:
        """nn.Module forward — delegates to :meth:`lookup`."""
        values, _ = self.lookup(x)
        return values

    def export_lut(self) -> LutData:
        """Export LUT table data for backend consumption."""
        return LutData(
            thresholds=self.thresholds.clone(),
            values=self.lut_values.clone(),
            is_float=self.is_float,
        )


class LutReLU(LutActivation):
    """LUT-based ReLU activation.

    To avoid wasting bins on the negative side (which all map to 0),
    thresholds are concentrated in the positive region ``[0, max_val]``.
    """

    def generate_lut(self) -> tuple[Tensor, Tensor]:
        # ReLU: x < 0 -> 0, x > 0 -> linear.
        # Concentrate thresholds on the positive side for better resolution.
        if self.min_val < 0 and self.max_val > 0:
            # t[0] = min_val (covers the entire negative range)
            # t[1] = 0 (positive start)
            # Remaining 254 thresholds subdivide (0, max_val] uniformly.
            pos_thres = torch.linspace(0, self.max_val, 256)[
                1:
            ]  # 255 points: skip 0, include max
            thres_t = torch.cat(
                [torch.tensor([float(self.min_val), 0.0]), pos_thres[:254]]
            )
        else:
            thres_t = torch.linspace(self.min_val, self.max_val, 256)

        if self.is_float:
            return thres_t, torch.relu(thres_t)

        # ReLU scaling:
        # Unsigned (output_sign=0) maps max_val -> 255.
        # Signed   (output_sign=1) maps max_val -> 127.
        scale = (
            (255 if self.output_sign == 0 else 127) / self.max_val
            if self.max_val > 0
            else 0
        )

        # Compute LUT values using the midpoint of each bin.
        # Use floor instead of round to avoid even/odd stepping artefacts
        # from banker's rounding at x.5 boundaries.
        boundaries = torch.cat([thres_t, torch.tensor([self.max_val])])
        low = boundaries[:-1]
        high = boundaries[1:]
        mids = torch.where(low < high, (low + high) / 2, low)
        values = (mids.clamp(min=0) * scale).to(torch.int64)

        if self.output_sign == 0:
            values.clamp_(0, 255)
        else:
            values.clamp_(-128, 127)

        return thres_t, values.float()


class LutLinear(LutActivation):
    """LUT-based linear mapping activation.

    Maps ``[min_val, max_val]`` to ``[-128, 127]``.
    """

    def generate_lut(self) -> tuple[Tensor, Tensor]:
        step = (self.max_val - self.min_val) / 256
        thres_t = torch.linspace(self.min_val, self.max_val, 256)

        if self.is_float:
            return thres_t, thres_t.clone()

        # slope = (out_max - out_min) / (in_max - in_min) = 255 / input_range
        slope = (
            255 / (self.max_val - self.min_val) if self.max_val > self.min_val else 0
        )
        midpoints = self.min_val + (torch.arange(256) + 0.5) * step
        values = (-128 + (midpoints - self.min_val) * slope).round().to(torch.int64)
        values.clamp_(-128, 127)

        return thres_t, values.float()


class LutAdaptiveActivation(LutActivation):
    """Base class for activations that use non-uniform binning.

    Uses the inverse function to place thresholds where the activation
    changes rapidly, providing higher resolution in those regions.

    Subclasses must implement ``get_math_range``, ``get_forward_func``,
    ``get_inverse_func``, and ``get_output_scale_func``.
    """

    def __init__(
        self,
        min_val: float = _DEFAULT_MIN,
        max_val: float = _DEFAULT_MAX,
        output_sign: int = 0,
        act_range: float = 10.0,
        is_float: bool = False,
    ) -> None:
        # Set act_range before super().__init__ because it calls generate_lut()
        if is_float:
            self.act_range = max(abs(min_val), abs(max_val))
            if self.act_range == 0:
                self.act_range = 10.0
        else:
            self.act_range = act_range

        super().__init__(min_val, max_val, output_sign, is_float=is_float)

    @abstractmethod
    def get_math_range(self) -> tuple[float, float]:
        """Return the mathematical output range ``(y_min, y_max)``."""
        raise NotImplementedError

    @abstractmethod
    def get_forward_func(self) -> Callable[[float], float]:
        """Return the forward function ``y = f(x)``."""
        raise NotImplementedError

    @abstractmethod
    def get_inverse_func(self) -> Callable[[float], float]:
        """Return the inverse function ``x = f^{-1}(y)``."""
        raise NotImplementedError

    @abstractmethod
    def get_output_scale_func(self) -> Callable[[float], float]:
        """Return a function that maps mathematical output to LUT integer."""
        raise NotImplementedError

    def _compute_normalized_thresholds(
        self, inverse_fn: Callable[[float], float], y_min: float, y_max: float
    ) -> list[float]:
        """Compute thresholds in normalised domain via inverse function.

        We want outputs uniformly distributed in ``[y_min, y_max]``.
        For each of the 256 fractions ``y = y_min + i/256 * (y_max - y_min)``,
        the inverse function maps *y* back to the normalised input *x*,
        placing thresholds densely where the activation curve is steep.
        """
        thres = []
        for i in range(256):
            y_frac = y_min + (i / 256) * (y_max - y_min)
            # Clamp slightly to avoid domain errors at asymptotes
            y_clamped = max(y_min + 1e-6, min(y_max - 1e-6, y_frac))
            thres.append(inverse_fn(y_clamped))

        return thres

    def _map_to_input_domain(self, thres: list[float], scale: float) -> Tensor:
        """Map normalised thresholds ``[-act_range, act_range]`` to hardware
        input domain ``[min_val, max_val]``."""
        t = torch.as_tensor(thres, dtype=torch.float64)
        mapped = self.min_val + (t + self.act_range) * scale
        mapped.clamp_(self.min_val, self.max_val)
        return mapped

    def _normalize_input(self, x: float, scale: float) -> float:
        """Map a hardware-domain input back to normalised domain
        ``[-act_range, act_range]``."""
        return (x - self.min_val) / scale - self.act_range

    def _compute_lut_values(
        self,
        thresholds: Tensor,
        scale: float,
        forward_fn: Callable[[float], float],
        output_scale_fn: Callable[[float], float],
    ) -> list[int | float]:
        """Compute LUT values by evaluating the forward function at bin
        midpoints, then scaling to the output range."""
        boundaries = torch.cat(
            [thresholds, torch.tensor([self.max_val], dtype=thresholds.dtype)]
        )
        low = boundaries[:-1]
        high = boundaries[1:]
        mids = torch.where(low < high, (low + high) / 2, low)

        values: list[int | float] = []
        for i in range(256):
            x_mid = self._normalize_input(mids[i].item(), scale)
            if not self.is_float:
                x_mid = max(-20, min(20, x_mid))

            y_val = forward_fn(x_mid)
            values.append(self._clamp_value(output_scale_fn(y_val)))

        return values

    def generate_lut(self) -> tuple[Tensor, Tensor]:
        """Generate LUT with non-uniform (adaptive) binning.

        Orchestrates three stages:

        1. Compute normalised thresholds via the inverse function.
        2. Map normalised thresholds to the hardware input domain.
        3. Evaluate the forward function at bin midpoints to produce
           LUT values.
        """
        y_min, y_max = self.get_math_range()
        inverse_fn = self.get_inverse_func()
        forward_fn = self.get_forward_func()

        if self.is_float:
            output_scale_fn = lambda x: x  # noqa: E731
        else:
            output_scale_fn = self.get_output_scale_func()

        input_range = self.max_val - self.min_val
        if input_range <= 0:
            return (torch.full((256,), self.min_val), torch.zeros(256))

        # scale maps normalised [-R, R] to hardware [min_val, max_val].
        scale = input_range / (2 * self.act_range)

        # Stage 1: place thresholds in normalised domain via inverse function
        normalized_thres = self._compute_normalized_thresholds(inverse_fn, y_min, y_max)
        # Stage 2: map to hardware input domain [min_val, max_val]
        thresholds = self._map_to_input_domain(normalized_thres, scale)

        # Stage 3: compute LUT values from bin midpoints
        values = self._compute_lut_values(
            thresholds, scale, forward_fn, output_scale_fn
        )

        return thresholds.float(), torch.as_tensor(values, dtype=torch.float32)


class LutSigmoid(LutAdaptiveActivation):
    """LUT-based Sigmoid activation."""

    def get_math_range(self) -> tuple[float, float]:
        return 0.0, 1.0

    def get_forward_func(self) -> Callable[[float], float]:
        return lambda x: 1 / (1 + math.exp(-x))

    def get_inverse_func(self) -> Callable[[float], float]:
        # logit: p -> log(p / (1 - p))
        return lambda p: math.log(p / (1 - p))

    def get_output_scale_func(self) -> Callable[[float], float]:
        return lambda x: x * 255.0


class LutTanh(LutAdaptiveActivation):
    """LUT-based Tanh activation."""

    def get_math_range(self) -> tuple[float, float]:
        return -1.0, 1.0

    def get_forward_func(self) -> Callable[[float], float]:
        return math.tanh

    def get_inverse_func(self) -> Callable[[float], float]:
        return math.atanh

    def get_output_scale_func(self) -> Callable[[float], float]:
        # tanh [-1, 1] -> [-127, 127] for signed output
        return lambda x: x * 127.0


class LutSoftsign(LutAdaptiveActivation):
    """LUT-based Softsign activation."""

    def get_math_range(self) -> tuple[float, float]:
        return -1.0, 1.0

    def get_forward_func(self) -> Callable[[float], float]:
        return lambda x: x / (1 + abs(x))

    def get_inverse_func(self) -> Callable[[float], float]:
        # y = x / (1 + |x|)  =>  x = y / (1 - |y|)
        return lambda y: y / (1 - abs(y))

    def get_output_scale_func(self) -> Callable[[float], float]:
        # softsign [-1, 1] -> [-127, 127] for signed output
        return lambda x: x * 127.0


class LutCustom(LutActivation):
    """LUT from user-provided threshold and value arrays.

    The caller supplies pre-computed 256-entry threshold and value tensors
    directly.  Dtype conversion is handled by the base class.

    Args:
        thresholds: 256-entry threshold tensor.
        values: 256-entry output value tensor.
        output_sign: 0 = unsigned, 1 = signed, None = auto-detect from values.
        is_float: Use float32 thresholds and bfloat16 values when True.
    """

    def __init__(
        self,
        thresholds: Tensor,
        values: Tensor,
        output_sign: int | None = None,
        is_float: bool = False,
    ) -> None:
        if thresholds.numel() != 256:
            raise ValueError(f"Expected 256 thresholds, got {thresholds.numel()}")
        if values.numel() != 256:
            raise ValueError(f"Expected 256 values, got {values.numel()}")

        if output_sign is None:
            output_sign = 1 if torch.any(values < 0) else 0

        # Store before super().__init__ which calls generate_lut()
        self._init_thresholds = thresholds.float()
        self._init_values = values.float()
        # min/max derived from the threshold range
        min_val = float(thresholds[0].item())
        max_val = float(thresholds[-1].item())
        super().__init__(min_val, max_val, output_sign, is_float)
        # Clean up temporary storage
        del self._init_thresholds, self._init_values

    def generate_lut(self) -> tuple[Tensor, Tensor]:
        return self._init_thresholds, self._init_values
