"""LUT activation functions.

Discretise continuous activation functions into fixed-size lookup tables,
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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch
from paicorelib import DataSign, DataWidth
from torch import Tensor, nn

from .calc_params import LUT_TABLE_SIZE, LutData
from .value_code import (
    SIGNED_VALUE_CODE_RANGES,
    UNSIGNED_VALUE_CODE_RANGES,
    code_range_for_format,
)

__all__ = [
    "LutActivation",
    "LutAdaptiveActivation",
    "LutCustom",
    "LutReLU",
    "LutReLUSymmetric",
    "LutLinear",
    "LutSigmoid",
    "LutTanh",
    "LutSoftsign",
]

_INT32_MAX = torch.iinfo(torch.int32).max
_INT32_MIN = torch.iinfo(torch.int32).min
_UINT8_MAX = torch.iinfo(torch.uint8).max
_INT8_MAX = torch.iinfo(torch.int8).max
_INT8_MIN = torch.iinfo(torch.int8).min


@dataclass(frozen=True)
class _PreparedIntegerLut:
    thresholds: Tensor
    values: Tensor


@dataclass(frozen=True)
class _EffectiveIntegerIntervals:
    starts: list[int]
    values: list[int]


def _width_bits(output_width: DataWidth) -> int:
    if output_width > DataWidth.WIDTH_8BIT:
        raise ValueError(f"unsupported ANN LUT output width: {output_width}")
    return 1 << output_width


def _sar_leaf_count(output_width: DataWidth) -> int:
    return 1 << _width_bits(output_width)


def _sar_block_size(output_width: DataWidth) -> int:
    return LUT_TABLE_SIZE // _sar_leaf_count(output_width)


def _logical_lut_lookup(
    thresholds: Tensor, values: Tensor, x: Tensor
) -> tuple[Tensor, Tensor]:
    """Evaluate a PAIIR logical LUT with bucketize semantics."""
    indices = torch.bucketize(x, thresholds, right=True) - 1
    indices = indices.clamp(0, LUT_TABLE_SIZE - 1)
    return values[indices], indices


def _lookup_hw_lut_data(
    hw_lut: LutData, output_width: DataWidth, x: Tensor
) -> tuple[Tensor, Tensor]:
    """Evaluate integer hardware LUT data with PAICORE 2.5 SAR semantics."""
    prepared = _prepare_integer_lut(hw_lut, require_monotonic=False)
    return _hardware_sar_lut_lookup(prepared, output_width, x)


def _flat_integer_tensor(name: str, x: Tensor) -> Tensor:
    prepared = x.detach().cpu().flatten()
    if prepared.is_floating_point():
        raise TypeError(f"{name} must be an integer tensor")
    return prepared


def _flat_floating_tensor(name: str, x: Tensor) -> Tensor:
    prepared = x.detach().cpu().flatten()
    if not prepared.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    return prepared


def _validate_int32_range(name: str, values: Tensor) -> None:
    if values.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32):
        return
    if values.numel() and (
        bool(torch.any(values < _INT32_MIN)) or bool(torch.any(values > _INT32_MAX))
    ):
        raise ValueError(f"{name} must fit int32")


def _prepare_integer_lut(
    table: LutData, *, require_monotonic: bool
) -> _PreparedIntegerLut:
    if table.thresholds.numel() != LUT_TABLE_SIZE:
        raise ValueError(
            f"expected {LUT_TABLE_SIZE} LUT thresholds, got {table.thresholds.numel()}"
        )
    if table.values.numel() != LUT_TABLE_SIZE:
        raise ValueError(
            f"expected {LUT_TABLE_SIZE} LUT values, got {table.values.numel()}"
        )
    if table.is_float:
        raise ValueError("float LUT data is not supported for hardware SRAM conversion")

    # Hardware export works from a detached CPU int32 view; downstream interval
    # compression, SAR table construction, and equivalence probes reuse it.
    thresholds = _flat_integer_tensor("integer LUT thresholds", table.thresholds)
    values = _flat_integer_tensor("integer LUT values", table.values)
    _validate_int32_range("integer LUT thresholds", thresholds)
    _validate_int32_range("integer LUT values", values)
    thresholds = thresholds.to(torch.int32)
    values = values.to(torch.int32)
    if require_monotonic and not torch.all(thresholds[1:] >= thresholds[:-1]):
        raise ValueError("integer LUT thresholds must be monotonic")
    return _PreparedIntegerLut(thresholds, values)


def _value_range_for_sign(sign: DataSign) -> dict[DataWidth, tuple[int, int]]:
    if sign == DataSign.SIGNED:
        return SIGNED_VALUE_CODE_RANGES
    return UNSIGNED_VALUE_CODE_RANGES


def _signedness_name(output_signed: bool) -> str:
    return "signed" if output_signed else "unsigned"


def _lut_value_dtype(output_signed: bool) -> torch.dtype:
    return torch.int8 if output_signed else torch.uint8


def _lut_value_range(output_signed: bool) -> tuple[int, int]:
    if output_signed:
        return _INT8_MIN, _INT8_MAX
    return 0, _UINT8_MAX


def _validate_lut_tensor_values(values: Tensor, output_signed: bool) -> None:
    lo, hi = _lut_value_range(output_signed)
    bad = values[(values < lo) | (values > hi)]
    if bad.numel():
        raise ValueError(
            f"LUT values [{int(bad.min().item())}, {int(bad.max().item())}] exceed "
            f"{_signedness_name(output_signed)} 8-bit range [{lo}, {hi}]"
        )


def _validate_output_values(
    values: Iterable[int], output_data_sign: DataSign, output_width: DataWidth
) -> None:
    lo, hi = code_range_for_format(output_data_sign, output_width)
    bad = [value for value in values if value < lo or value > hi]
    if bad:
        raise ValueError(
            f"LUT values [{min(bad)}, {max(bad)}] exceed "
            f"{output_data_sign.name.lower()} {output_width.name.lower()} range [{lo}, {hi}]"
        )


def _compress_logical_intervals(
    prepared: _PreparedIntegerLut,
) -> _EffectiveIntegerIntervals:
    """Compress a monotonic logical LUT into reachable integer intervals."""
    thresholds = prepared.thresholds.tolist()
    values = prepared.values.tolist()

    starts = [_INT32_MIN]
    interval_values = [int(values[0])]

    # Duplicate thresholds are not independently reachable with bucketize
    # right=True; only the last value for a repeated threshold can take effect.
    idx = 0
    while idx < LUT_TABLE_SIZE:
        threshold = int(thresholds[idx])
        last = idx
        while last + 1 < LUT_TABLE_SIZE and int(thresholds[last + 1]) == threshold:
            last += 1

        value = int(values[last])
        if threshold == _INT32_MIN:
            interval_values[-1] = value
        elif value != interval_values[-1]:
            starts.append(threshold)
            interval_values.append(value)

        idx = last + 1

    return _EffectiveIntegerIntervals(starts, interval_values)


def _hardware_sar_lut_lookup(
    prepared_hw_lut: _PreparedIntegerLut, output_width: DataWidth, x: Tensor
) -> tuple[Tensor, Tensor]:
    """Simulate the chip's successive-approximation LUT lookup."""
    bits = _width_bits(output_width)
    block_size = _sar_block_size(output_width)
    thresholds = prepared_hw_lut.thresholds.to(device=x.device, dtype=x.dtype)
    values = prepared_hw_lut.values.to(device=x.device)
    prefix = torch.zeros_like(x, dtype=torch.int32)

    # Hardware probes the threshold SRAM in SAR order. For k-bit output,
    # candidate prefixes map to 8-bit LUT addresses at each block start.
    for bit_pos in range(bits - 1, -1, -1):
        candidate = prefix | (1 << bit_pos)
        probe_addr = candidate * block_size
        probe_thresholds = thresholds[probe_addr]
        prefix = torch.where(x >= probe_thresholds, candidate, prefix)

    final_addr = prefix * block_size
    return values[final_addr], final_addr


def _build_sar_lut_data_from_intervals(
    intervals: _EffectiveIntegerIntervals,
    output_data_sign: DataSign,
    output_width: DataWidth,
) -> tuple[LutData, _PreparedIntegerLut]:
    block_size = _sar_block_size(output_width)
    leaf_count = _sar_leaf_count(output_width)
    interval_count = len(intervals.values)
    if interval_count > leaf_count:
        raise ValueError(
            f"LUT has {interval_count} effective intervals, but "
            f"{output_width.name.lower()} hardware SAR LUT can express at most "
            f"{leaf_count}"
        )

    _validate_output_values(intervals.values, output_data_sign, output_width)

    # Store interval starts at the SAR probe addresses. Unused threshold slots
    # are filled high so they are never taken by normal int32 inputs.
    hw_thresholds = torch.full((LUT_TABLE_SIZE,), _INT32_MAX, dtype=torch.int32)
    for leaf_idx, start in enumerate(intervals.starts[1:], start=1):
        probe_addr = leaf_idx * block_size
        hw_thresholds[probe_addr] = start

    # Activation SRAM is indexed by the final 8-bit comparison address. Low
    # precision outputs therefore occupy repeated address blocks.
    padded_values = intervals.values + [intervals.values[-1]] * (
        leaf_count - interval_count
    )
    hw_values_i32 = torch.full((LUT_TABLE_SIZE,), padded_values[-1], dtype=torch.int32)
    for leaf_idx, value in enumerate(padded_values):
        start = leaf_idx * block_size
        hw_values_i32[start : start + block_size] = value

    output_signed = output_data_sign == DataSign.SIGNED
    value_dtype = _lut_value_dtype(output_signed)
    hw_values = hw_values_i32.to(value_dtype)
    hw_lut = LutData(hw_thresholds, hw_values, is_float=False)
    hw_prepared = _PreparedIntegerLut(hw_thresholds, hw_values_i32)
    return hw_lut, hw_prepared


def _probe_points(*threshold_tables: Tensor) -> Tensor:
    points: set[int] = {_INT32_MIN, _INT32_MAX}
    for thresholds in threshold_tables:
        for raw in thresholds.tolist():
            threshold = int(raw)
            points.add(threshold)
            if threshold > _INT32_MIN:
                points.add(threshold - 1)
            if threshold < _INT32_MAX:
                points.add(threshold + 1)

    return torch.tensor(sorted(points), dtype=torch.int32)


def _first_lut_mismatch(
    logical_lut: _PreparedIntegerLut,
    hw_lut: _PreparedIntegerLut,
    output_width: DataWidth,
) -> tuple[int, int, int, int, int] | None:
    # Boundary probes are enough for monotonic integer interval LUTs: every
    # interval can only change at a logical or hardware threshold.
    probes = _probe_points(logical_lut.thresholds, hw_lut.thresholds)
    logical_values, logical_indices = _logical_lut_lookup(
        logical_lut.thresholds, logical_lut.values, probes
    )
    hw_values, hw_indices = _hardware_sar_lut_lookup(hw_lut, output_width, probes)
    mismatch = logical_values != hw_values
    if not bool(mismatch.any()):
        return None

    first = int(torch.nonzero(mismatch, as_tuple=False)[0].item())
    return (
        int(probes[first].item()),
        int(logical_values[first].item()),
        int(hw_values[first].item()),
        int(logical_indices[first].item()),
        int(hw_indices[first].item()),
    )


def _build_hw_lut_data_from_prepared(
    prepared: _PreparedIntegerLut,
    output_data_sign: DataSign,
    output_width: DataWidth,
) -> LutData:
    intervals = _compress_logical_intervals(prepared)
    hw_lut, hw_prepared = _build_sar_lut_data_from_intervals(
        intervals, output_data_sign, output_width
    )
    if (
        mismatch := _first_lut_mismatch(prepared, hw_prepared, output_width)
    ) is not None:
        point, expected, actual, logical_idx, hw_idx = mismatch
        raise ValueError(
            "logical LUT cannot be represented by "
            f"{output_width.name.lower()} hardware SAR LUT: at input {point}, "
            f"logical[{logical_idx}]={expected}, hardware[{hw_idx}]={actual}"
        )
    return hw_lut


def _build_hw_lut_data_from_logical_lut(
    logical_lut: LutData, output_data_sign: DataSign, output_width: DataWidth
) -> LutData:
    """Convert logical integer LUT data to PAICORE 2.5 SAR SRAM LUT data."""
    prepared = _prepare_integer_lut(logical_lut, require_monotonic=True)
    return _build_hw_lut_data_from_prepared(prepared, output_data_sign, output_width)


class LutActivation(nn.Module):
    thresholds: Tensor
    lut_values: Tensor

    def __init__(
        self,
        min_val: float = _INT32_MIN,
        max_val: float = _INT32_MAX,
        output_signed: bool = False,
        is_float: bool = False,
    ) -> None:
        """Base class for fixed-size LUT activation functions.

        Partitions the input space into LUT bins using thresholds, each bin
        mapping to a pre-computed output value.

        Subclasses override :meth:`generate_lut` to return a ``(thresholds,
        values)`` pair of fixed-size tensors. The base ``__init__`` registers
        them as buffers with the correct dtype.

        * ``is_float=True``: thresholds ``float32``, values ``bfloat16``.
        * ``is_float=False``: thresholds ``int32``, values ``uint8``
          (unsigned) or ``int8`` (signed).

        Args:
            min_val: Lower bound of the input domain.
            max_val: Upper bound of the input domain.
            output_signed: Whether integer LUT outputs use signed int8 codes.
            is_float: Use float32 thresholds and bfloat16 values when True.
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.output_signed = output_signed
        self.is_float = is_float

        thresholds, values = self.generate_lut()

        if self.is_float:
            self.register_buffer("thresholds", thresholds.to(torch.float32))
            self.register_buffer("lut_values", values.to(torch.bfloat16))
        else:
            integer_values = _flat_integer_tensor("LUT values", values)
            _validate_lut_tensor_values(integer_values, self.output_signed)
            vals_dtype = _lut_value_dtype(self.output_signed)
            integer_thresholds = _flat_integer_tensor("LUT thresholds", thresholds)
            _validate_int32_range("LUT thresholds", integer_thresholds)
            self.register_buffer("thresholds", integer_thresholds.to(torch.int32))
            self.register_buffer("lut_values", integer_values.to(vals_dtype))

    def _clamp_value(self, value: float) -> int | float:
        """Clamp output to the target 8-bit range."""
        if self.is_float:
            return value
        lo, hi = _lut_value_range(self.output_signed)
        return max(lo, min(hi, round(value)))

    def generate_lut(self) -> tuple[Tensor, Tensor]:
        """Generate thresholds and LUT values.

        Subclasses must override this method to return a ``(thresholds,
        values)`` pair of fixed-size float tensors.  The base ``__init__``
        converts them to the appropriate dtype.

        Returns:
            A ``(thresholds, values)`` tuple of fixed-size tensors.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    def lookup(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate the PAIIR logical LUT, returning values and indices.

        ``torch.bucketize`` with ``right=True`` finds indices such that
        ``thresholds[i-1] <= x < thresholds[i]``.  Subtracting 1 gives the
        bin index, so if *x* falls in ``[thresholds[i], thresholds[i+1])``
        bucketize returns ``i+1`` and ``(i+1) - 1 = i``, yielding the
        correct LUT entry.

        Returns:
            A ``(values, indices)`` tuple where *values* are the LUT outputs
            and *indices* are the logical bin indices.
        """
        return _logical_lut_lookup(self.thresholds, self.lut_values, x)

    def forward(self, x: Tensor) -> Tensor:
        """nn.Module forward — delegates to :meth:`lookup`."""
        values, _ = self.lookup(x)
        return values

    @property
    def logical_lut_data(self) -> LutData:
        """Logical LUT table used by PAIIR graph simulation."""
        return LutData(self.thresholds.clone(), self.lut_values.clone(), self.is_float)

    def to_hw_lut_data(
        self, output_data_sign: DataSign, output_width: DataWidth
    ) -> LutData:
        """Convert the logical LUT to PAICORE 2.5 hardware SRAM LUT data."""
        logical_lut = LutData(self.thresholds, self.lut_values, self.is_float)
        return _build_hw_lut_data_from_logical_lut(
            logical_lut, output_data_sign, output_width
        )

    def infer_hw_output_width(self, output_data_sign: DataSign) -> DataWidth:
        """Return the narrowest output width with equivalent hardware SAR export."""
        if self.is_float:
            return DataWidth.WIDTH_8BIT

        logical_lut = LutData(self.thresholds, self.lut_values, self.is_float)
        prepared = _prepare_integer_lut(logical_lut, require_monotonic=True)
        ranges = _value_range_for_sign(output_data_sign)
        last_error: ValueError | None = None

        for width in ranges:
            try:
                _build_hw_lut_data_from_prepared(prepared, output_data_sign, width)
                return width
            except ValueError as exc:
                last_error = exc

        value_min = int(prepared.values.min().item())
        value_max = int(prepared.values.max().item())
        raise ValueError(
            f"LUT activation range [{value_min}, {value_max}] cannot be exported as "
            f"{output_data_sign.name.lower()} ANN output data"
        ) from last_error


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
            # Remaining thresholds subdivide (0, max_val] uniformly.
            pos_thres = torch.linspace(
                0, self.max_val, LUT_TABLE_SIZE, dtype=torch.float64
            )[
                1:
            ]  # skip 0, include max
            thres_t = torch.cat(
                [
                    torch.tensor([float(self.min_val), 0.0], dtype=pos_thres.dtype),
                    pos_thres[: LUT_TABLE_SIZE - 2],
                ]
            )
        else:
            thres_t = torch.linspace(
                self.min_val, self.max_val, LUT_TABLE_SIZE, dtype=torch.float64
            )

        if self.is_float:
            return thres_t, torch.relu(thres_t)

        # ReLU scaling maps max_val to the positive output limit.
        scale = (
            (_INT8_MAX if self.output_signed else _UINT8_MAX) / self.max_val
            if self.max_val > 0
            else 0
        )

        # Compute LUT values using the midpoint of each bin.
        # Use floor instead of round to avoid even/odd stepping artefacts
        # from banker's rounding at x.5 boundaries.
        boundaries = torch.cat(
            [thres_t, torch.tensor([self.max_val], dtype=thres_t.dtype)]
        )
        low = boundaries[:-1]
        high = boundaries[1:]
        mids = torch.where(low < high, (low + high) / 2, low)
        values = (mids.clamp(min=0) * scale).to(torch.int32)

        lo, hi = _lut_value_range(self.output_signed)
        values.clamp_(lo, hi)

        return thres_t.round().to(torch.int32), values


class LutReLUSymmetric(LutActivation):
    """
    Symmetric version of LutReLU.
    Unlike standard ReLU which maps negative values to 0 and shrinks the negative input bins,
    this class allocates uniform bins across the entire [min_val, max_val] range to preserve
    the negative input space (even though outputs for negative inputs are still 0).
    It can output signed 8-bit integers when configured with
    ``output_signed=True``.
    """

    def generate_lut(self) -> tuple[Tensor, Tensor]:
        # Uniformly allocate LUT bins across [min_val, max_val].
        thres_t = torch.linspace(
            self.min_val, self.max_val, LUT_TABLE_SIZE, dtype=torch.float64
        )

        if self.is_float:
            return thres_t, torch.relu(thres_t)

        # Map max_val to the positive output limit.
        scale = (
            (_INT8_MAX if self.output_signed else _UINT8_MAX) / self.max_val
            if self.max_val > 0
            else 0
        )

        # Compute LUT values using the midpoint of each bin.
        boundaries = torch.cat(
            [thres_t, torch.tensor([self.max_val], dtype=thres_t.dtype)]
        )
        low = boundaries[:-1]
        high = boundaries[1:]
        mids = torch.where(low < high, (low + high) / 2, low)
        values = (mids.clamp(min=0) * scale).to(torch.int32)

        lo, hi = _lut_value_range(self.output_signed)
        values.clamp_(lo, hi)

        return thres_t.round().to(torch.int32), values


class LutLinear(LutActivation):
    """LUT-based linear mapping activation.

    Maps ``[min_val, max_val]`` to ``[-128, 127]``.
    """

    def __init__(
        self,
        min_val: float = _INT32_MIN,
        max_val: float = _INT32_MAX,
        output_signed: bool = True,
        is_float: bool = False,
    ) -> None:
        super().__init__(min_val, max_val, output_signed, is_float)

    def generate_lut(self) -> tuple[Tensor, Tensor]:
        step = (self.max_val - self.min_val) / LUT_TABLE_SIZE
        thres_t = torch.linspace(
            self.min_val, self.max_val, LUT_TABLE_SIZE, dtype=torch.float64
        )

        if self.is_float:
            return thres_t, thres_t.clone()

        # slope = (out_max - out_min) / input_range
        slope = (
            _UINT8_MAX / (self.max_val - self.min_val)
            if self.max_val > self.min_val
            else 0
        )
        midpoints = (
            self.min_val
            + (torch.arange(LUT_TABLE_SIZE, dtype=thres_t.dtype) + 0.5) * step
        )
        values = (
            (_INT8_MIN + (midpoints - self.min_val) * slope).round().to(torch.int32)
        )
        values.clamp_(_INT8_MIN, _INT8_MAX)

        return thres_t.round().to(torch.int32), values


class LutAdaptiveActivation(LutActivation):
    """Base class for activations that use non-uniform binning.

    Uses the inverse function to place thresholds where the activation
    changes rapidly, providing higher resolution in those regions.

    Subclasses must implement ``get_math_range``, ``get_forward_func``,
    ``get_inverse_func``, and ``get_output_scale_func``.
    """

    def __init__(
        self,
        min_val: float = _INT32_MIN,
        max_val: float = _INT32_MAX,
        output_signed: bool = False,
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

        super().__init__(min_val, max_val, output_signed, is_float)

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
        For each LUT entry, the inverse function maps *y* back to the
        normalised input *x*, placing thresholds densely where the activation
        curve is steep.
        """
        thres = []
        for i in range(LUT_TABLE_SIZE):
            y_frac = y_min + (i / LUT_TABLE_SIZE) * (y_max - y_min)
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
        for i in range(LUT_TABLE_SIZE):
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
            return (
                torch.full((LUT_TABLE_SIZE,), self.min_val),
                torch.zeros(LUT_TABLE_SIZE),
            )

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

        values_dtype = torch.float32 if self.is_float else torch.int32
        thresholds_out = (
            thresholds.float() if self.is_float else thresholds.round().to(torch.int32)
        )
        return thresholds_out, torch.as_tensor(values, dtype=values_dtype)


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
        return lambda x: x * float(_UINT8_MAX)


class LutTanh(LutAdaptiveActivation):
    """LUT-based Tanh activation."""

    def __init__(
        self,
        min_val: float = _INT32_MIN,
        max_val: float = _INT32_MAX,
        output_signed: bool = True,
        act_range: float = 10.0,
        is_float: bool = False,
    ) -> None:
        super().__init__(min_val, max_val, output_signed, act_range, is_float)

    def get_math_range(self) -> tuple[float, float]:
        return -1.0, 1.0

    def get_forward_func(self) -> Callable[[float], float]:
        return math.tanh

    def get_inverse_func(self) -> Callable[[float], float]:
        return math.atanh

    def get_output_scale_func(self) -> Callable[[float], float]:
        # tanh [-1, 1] -> signed output range
        return lambda x: x * float(_INT8_MAX)


class LutSoftsign(LutAdaptiveActivation):
    """LUT-based Softsign activation."""

    def __init__(
        self,
        min_val: float = _INT32_MIN,
        max_val: float = _INT32_MAX,
        output_signed: bool = True,
        act_range: float = 10.0,
        is_float: bool = False,
    ) -> None:
        super().__init__(min_val, max_val, output_signed, act_range, is_float)

    def get_math_range(self) -> tuple[float, float]:
        return -1.0, 1.0

    def get_forward_func(self) -> Callable[[float], float]:
        return lambda x: x / (1 + abs(x))

    def get_inverse_func(self) -> Callable[[float], float]:
        # y = x / (1 + |x|)  =>  x = y / (1 - |y|)
        return lambda y: y / (1 - abs(y))

    def get_output_scale_func(self) -> Callable[[float], float]:
        # softsign [-1, 1] -> signed output range
        return lambda x: x * float(_INT8_MAX)


class LutCustom(LutActivation):
    """LUT from user-provided threshold and value arrays.

    The caller supplies pre-computed threshold and value tensors directly.
    Dtype conversion is handled by the base class.

    Args:
        thresholds: Fixed-size threshold tensor.
        values: Fixed-size output value tensor.
        output_signed: Whether integer LUT outputs use signed int8 codes.
            ``None`` infers signedness from whether rounded values contain
            negatives.
        is_float: Use float32 thresholds and bfloat16 values when True.
    """

    def __init__(
        self,
        thresholds: Tensor,
        values: Tensor,
        output_signed: bool | None = None,
        is_float: bool = False,
    ) -> None:
        if thresholds.numel() != LUT_TABLE_SIZE:
            raise ValueError(
                f"Expected {LUT_TABLE_SIZE} thresholds, got {thresholds.numel()}"
            )
        if values.numel() != LUT_TABLE_SIZE:
            raise ValueError(f"Expected {LUT_TABLE_SIZE} values, got {values.numel()}")
        if is_float:
            init_thresholds = _flat_floating_tensor("float LUT thresholds", thresholds)
            init_values = _flat_floating_tensor("float LUT values", values)
            resolved_output_signed = False if output_signed is None else output_signed
        else:
            init_thresholds = _flat_integer_tensor("integer LUT thresholds", thresholds)
            init_values = _flat_integer_tensor("integer LUT values", values)
            _validate_int32_range("integer LUT thresholds", init_thresholds)
            _validate_int32_range("integer LUT values", init_values)
            init_thresholds = init_thresholds.to(torch.int32)
            init_values = init_values.to(torch.int32)
            resolved_output_signed = (
                bool(init_values.numel() and init_values.min().item() < 0)
                if output_signed is None
                else output_signed
            )

        # Store before super().__init__ which calls generate_lut()
        self._init_thresholds = init_thresholds
        self._init_values = init_values
        # min/max derived from the flattened threshold range
        min_val = float(init_thresholds[0].item())
        max_val = float(init_thresholds[-1].item())
        super().__init__(min_val, max_val, resolved_output_signed, is_float)
        # Clean up temporary storage
        del self._init_thresholds, self._init_values

    @classmethod
    def from_intervals(
        cls,
        starts: Sequence[int] | Tensor,
        values: Sequence[int] | Tensor,
        *,
        output_signed: bool | None = None,
    ) -> "LutCustom":
        """Build a custom integer LUT from interval starts and values.

        This is the recommended constructor for low-interval integer LUTs.
        It creates a logical LUT with fixed ``bucketize(..., right=True)``
        semantics:

        * ``values[0]`` covers inputs below the first switching boundary
          ``starts[1]``. With one interval, all inputs map to ``values[0]``.
        * ``starts[i] <= x < starts[i + 1]`` maps to ``values[i]``.
        * Inputs at or above ``starts[-1]`` map to ``values[-1]``.

        ``starts`` must be strictly increasing and have the same length as
        ``values``. The generated 256-entry logical table stores the provided
        pairs first, then pads remaining entries by repeating the last start
        and value. This representation keeps the user-facing interval meaning
        simple while allowing the compiler to prove whether a lower-width
        hardware SAR LUT can exactly represent it.
        """
        start_count = starts.numel() if isinstance(starts, Tensor) else len(starts)
        value_count = values.numel() if isinstance(values, Tensor) else len(values)
        if start_count == 0:
            raise ValueError("LUT intervals must not be empty")
        if start_count != value_count:
            raise ValueError("LUT interval starts and values must have the same length")
        if start_count > LUT_TABLE_SIZE:
            raise ValueError(
                f"LUT intervals support at most {LUT_TABLE_SIZE} entries, got "
                f"{start_count}"
            )

        starts_t = _flat_integer_tensor("LUT interval starts", torch.as_tensor(starts))
        values_t = _flat_integer_tensor("LUT interval values", torch.as_tensor(values))

        _validate_int32_range("LUT interval starts", starts_t)
        _validate_int32_range("LUT interval values", values_t)
        starts_i = starts_t.to(torch.int32)
        values_i = values_t.to(torch.int32)
        if bool(torch.any(starts_i[1:] <= starts_i[:-1])):
            raise ValueError("LUT interval starts must be strictly increasing")

        thresholds = starts_i
        lut_values = values_i
        if thresholds.numel() < LUT_TABLE_SIZE:
            pad_count = LUT_TABLE_SIZE - thresholds.numel()
            thresholds = torch.cat((thresholds, thresholds[-1].repeat(pad_count)))
            lut_values = torch.cat((lut_values, lut_values[-1].repeat(pad_count)))

        return cls(thresholds, lut_values, output_signed, is_float=False)

    def generate_lut(self) -> tuple[Tensor, Tensor]:
        return self._init_thresholds, self._init_values
