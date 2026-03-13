"""Offline core computation parameters.

Uses paicorelib enum types to describe the configuration of an offline core.
The IR is version-agnostic; the backend translates these parameters to
chip-specific register values (v2.0 or v2.5).
"""

from dataclasses import dataclass

from paicorelib import (
    RM,
    AddPotentialMode,
    DataSign,
    DataWidth,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    OutputType,
    PoolingMode,
    SNNMode,
    ThresholdNegMode,
    ThresholdPosMode,
    ZeroOutputMode,
)
from torch import Tensor

__all__ = ["LutData", "OfflineCoreParams", "NeuronParams", "OnlineCoreParams"]


@dataclass(frozen=True)
class LutData:
    """LUT lookup table data for ANN mode offline cores.

    Contains the 256-entry threshold and value arrays that define
    the activation function lookup table on chip.
    """

    thresholds: Tensor  # shape (256,), bin boundaries
    values: Tensor  # shape (256,), output values
    is_float: bool = False  # True for float32 thresholds / bfloat16 values

    def _tensor_hash(self, t: Tensor) -> int:
        # detach + cpu 保证可序列化
        t = t.detach().cpu().contiguous()
        return hash((tuple(t.shape), str(t.dtype), t.numpy().tobytes()))

    def __hash__(self) -> int:
        return hash(
            (
                self._tensor_hash(self.thresholds),
                self._tensor_hash(self.values),
                self.is_float,
            )
        )


# Chip register limits
_TICK_START_MAX = (1 << 16) - 1  # 16-bit unsigned
_TICK_DURATION_MAX = (1 << 32) - 1  # 32-bit unsigned
_TICK_INITIAL_MAX = (1 << 16) - 1  # 16-bit unsigned


@dataclass
class OfflineCoreParams:
    """Offline core computation parameters.

    Describes the operating mode, data format, and timing configuration
    of a single offline core.

    Timing parameters:

    - ``tick_start``: Which sync_all to start working at. ``None`` means
      auto-assigned by :func:`~paibox.paiir.passes.assign_tick_params`.
      0 = never start (disabled), N > 0 = start at Nth sync_all.
    - ``tick_duration``: How many sync_all cycles to work. 0 = always
      working, N > 0 = work for N time steps then stop.
    - ``tick_initial``: Auto-reinitialise neuron state every N sync_all
      cycles. 0 = never reinitialise (default).
    """

    snn_mode: SNNMode = SNNMode.SNN
    pooling_mode: PoolingMode = PoolingMode.AVERAGE
    add_potential: AddPotentialMode = AddPotentialMode.NORMAL
    zero_output: ZeroOutputMode = ZeroOutputMode.DISABLE

    input_sign: DataSign = DataSign.SIGNED
    input_width: DataWidth = DataWidth.WIDTH_8BIT
    output_sign: DataSign = DataSign.SIGNED
    output_width: DataWidth = DataWidth.WIDTH_8BIT
    weight_sign: DataSign = DataSign.SIGNED
    weight_width: DataWidth = DataWidth.WIDTH_8BIT

    tick_start: int | None = None
    tick_duration: int = 0
    tick_initial: int = 0

    def set_input_format(self, fmt: tuple[DataSign, DataWidth]) -> None:
        self.input_sign, self.input_width = fmt

    def set_output_format(self, fmt: tuple[DataSign, DataWidth]) -> None:
        self.output_sign, self.output_width = fmt

    def set_weight_format(self, fmt: tuple[DataSign, DataWidth]) -> None:
        self.weight_sign, self.weight_width = fmt

    def validate_tick_params(self) -> None:
        """Validate tick parameters against chip register limits.

        Raises:
            ValueError: If any tick parameter is out of range.
        """
        if self.tick_start is None:
            raise ValueError(
                "tick_start is None (unassigned). Run assign_tick_params() first."
            )
        if not 0 <= self.tick_start <= _TICK_START_MAX:
            raise ValueError(
                f"tick_start must be in [0, {_TICK_START_MAX}], got {self.tick_start}"
            )
        if not 0 <= self.tick_duration <= _TICK_DURATION_MAX:
            raise ValueError(
                f"tick_duration must be in [0, {_TICK_DURATION_MAX}], "
                f"got {self.tick_duration}"
            )
        if not 0 <= self.tick_initial <= _TICK_INITIAL_MAX:
            raise ValueError(
                f"tick_initial must be in [0, {_TICK_INITIAL_MAX}], "
                f"got {self.tick_initial}"
            )


@dataclass
class NeuronParams:
    """Neuron parameter set.

    Full configuration of an offline-core neuron, corresponding 1:1 to
    :class:`~paibox.paiir.core_neuron.CoreNeuronV25` attributes.  The backend maps
    these to chip-specific register layouts.
    """

    reset_mode: RM = RM.MODE_NORMAL
    reset_v: float = 0.0
    thres_neg_mode: ThresholdNegMode = ThresholdNegMode.FLOOR
    thres_pos_mode: ThresholdPosMode = ThresholdPosMode.FIRE
    thres_neg: float = -99999.0
    thres_pos: float = 0.0
    lateral_inhi: LateralInhibitionMode = LateralInhibitionMode.DISABLE
    leak_multi_sequence: LeakMultiComparisonOrder = (
        LeakMultiComparisonOrder.AFTER_COMPARE
    )
    leak_multi_input: LeakMultiInputMode = LeakMultiInputMode.DISABLE
    leak_multi_mode: LeakMultiMode = LeakMultiMode.DISABLE
    leak_add_mode: LeakAddMode = LeakAddMode.FORWARD
    leak_tau: int = 0
    leak_v: float | Tensor = 0.0
    init_v: float = 0.0
    output_type: OutputType = OutputType.VALUE


@dataclass
class OnlineCoreParams:
    """Online (STDP) core parameters.

    Placeholder for online learning cores available on both v2.0 and v2.5.
    The backend handles version-specific configuration.
    """

    pass
