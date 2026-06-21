"""Offline and online core computation parameters.

Uses paicorelib enum types to describe the configuration of an offline core.
The IR is version-agnostic; the backend translates these parameters to
chip-specific register values (v2.0 or v2.5).
"""

from dataclasses import dataclass, field
from enum import Enum

import torch
from paicorelib import (
    LCN_EX,
    RM,
    AddPotentialMode,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    OnlineCoreType,
    OnlineCoreUpdateType,
    OnlineCoreWorkMode,
    OnlineDataWidth,
    OnlineSNNMode,
    OutputType,
    PoolingMode,
    SNNMode,
    ThresholdNegMode,
    ThresholdPosMode,
    ZeroOutputMode,
)
from torch import Tensor

__all__ = [
    "DEFAULT_NEG_THRESHOLD",
    "LutData",
    "OfflineCoreParams",
    "NeuronParams",
    "OnlineCoreSemanticMode",
    "OnlineGradientRole",
    "OnlineUpdateDirection",
    "OnlineCoreParams",
]


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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LutData):
            return NotImplemented
        return (
            self.is_float == other.is_float
            and torch.equal(self.thresholds, other.thresholds)
            and torch.equal(self.values, other.values)
        )


# Chip register limits
_TICK_START_MAX = (1 << 16) - 1  # 16-bit unsigned
_TICK_DURATION_MAX = (1 << 32) - 1  # 32-bit unsigned
_TICK_INITIAL_MAX = (1 << 16) - 1  # 16-bit unsigned

# Default negative threshold sentinel for single-sided spike neurons such as
# SpikingJelly-compatible IF/LIF. Chosen as a nearby power-of-two magnitude.
DEFAULT_NEG_THRESHOLD = -(1 << 17)


@dataclass
class OfflineCoreParams:
    """Offline core computation parameters.

    Describes the operating mode, data format, and timing configuration
    of a single offline core.

    Timing parameters are internal hardware fields. Public compile APIs use
    ``timesteps`` and ``auto_reset`` and map those values onto these fields.

    - ``tick_start``: Which sync_all to start working at. ``None`` means
      auto-assigned by :func:`~paibox.paiir.pipeline.passes.assign_tick_params`.
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
    _input_format_assigned: bool = field(default=False, init=False, repr=False)
    _output_format_assigned: bool = field(default=False, init=False, repr=False)
    _weight_format_assigned: bool = field(default=False, init=False, repr=False)

    def set_input_format(self, fmt: tuple[DataSign, DataWidth]) -> None:
        self.input_sign, self.input_width = fmt
        self._input_format_assigned = True

    def set_output_format(self, fmt: tuple[DataSign, DataWidth]) -> None:
        self.output_sign, self.output_width = fmt
        self._output_format_assigned = True

    def set_weight_format(self, fmt: tuple[DataSign, DataWidth]) -> None:
        self.weight_sign, self.weight_width = fmt
        self._weight_format_assigned = True

    def override_compile_state_from(self, other: "OfflineCoreParams") -> None:
        """Copy non-semantic compile-time state from another params object.

        Semantic mode fields such as ``snn_mode``, ``pooling_mode``, and
        ``add_potential`` are intentionally left untouched so the receiving
        node keeps the values derived from its own operator semantics.
        """
        self.zero_output = other.zero_output
        self.input_sign = other.input_sign
        self.input_width = other.input_width
        self.output_sign = other.output_sign
        self.output_width = other.output_width
        self.weight_sign = other.weight_sign
        self.weight_width = other.weight_width
        self.tick_start = other.tick_start
        self.tick_duration = other.tick_duration
        self.tick_initial = other.tick_initial
        self._input_format_assigned = other._input_format_assigned
        self._output_format_assigned = other._output_format_assigned
        self._weight_format_assigned = other._weight_format_assigned

    def validate_data_formats(self) -> None:
        """Validate that all data-format fields were explicitly assigned."""
        missing: list[str] = []
        if not self._input_format_assigned:
            missing.append("input_format")
        if not self._output_format_assigned:
            missing.append("output_format")
        if not self._weight_format_assigned:
            missing.append("weight_format")

        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"missing propagated data format(s): {missing_str}. "
                "Run propagate_data_format() first."
            )

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
    :class:`~paibox.paiir.ir.core_neuron.CoreNeuronV25` attributes.  The backend maps
    these to chip-specific register layouts.
    """

    reset_mode: RM = RM.MODE_NORMAL
    reset_v: float = 0.0
    thres_neg_mode: ThresholdNegMode = ThresholdNegMode.FLOOR
    thres_pos_mode: ThresholdPosMode = ThresholdPosMode.FIRE
    thres_neg: float = DEFAULT_NEG_THRESHOLD
    thres_pos: float | Tensor = 0.0
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


class OnlineCoreSemanticMode(str, Enum):
    """Semantic online-core stage before hardware work-mode refinement."""

    FORWARD = "forward"
    LOSS = "loss"
    GRADIENT = "gradient"
    POOL_GRADIENT = "pool_gradient"
    UPDATE = "update"


class OnlineGradientRole(str, Enum):
    """Gradient-stage refinement used to select the hardware work mode."""

    OUTPUT = "output"
    HIDDEN = "hidden"


class OnlineUpdateDirection(str, Enum):
    """Update-stage refinement used to select the hardware work mode."""

    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class OnlineCoreParams:
    """Online-core parameters.

    The frontend keeps a compact 5-mode semantic view and only refines into the
    8 hardware ``work_mode`` states when the compile pipeline has enough graph
    context to do so.
    """

    semantic_mode: OnlineCoreSemanticMode = OnlineCoreSemanticMode.FORWARD
    gradient_role: OnlineGradientRole | None = None
    update_direction: OnlineUpdateDirection | None = None
    work_mode: OnlineCoreWorkMode | None = None

    snn_mode: OnlineSNNMode = OnlineSNNMode.ANN_NO_ACT
    pooling_mode: PoolingMode = PoolingMode.AVERAGE
    add_potential: AddPotentialMode = AddPotentialMode.NORMAL
    zero_output: ZeroOutputMode = ZeroOutputMode.DISABLE

    input_core: OnlineCoreType = OnlineCoreType.ONLINE
    input_width: OnlineDataWidth = OnlineDataWidth.TYPE_FP16
    output_core: OnlineCoreType = OnlineCoreType.ONLINE
    output_width: OnlineDataWidth | OnlineCoreUpdateType = OnlineDataWidth.TYPE_FP16

    lcn_at: LCN_EX = LCN_EX.LCN_1X
    lcn_mp: LCN_EX = LCN_EX.LCN_1X
    lcn_lg: LCN_EX = LCN_EX.LCN_1X
    target_lcn_at: LCN_EX = LCN_EX.LCN_1X
    target_lcn_mp: LCN_EX = LCN_EX.LCN_1X
    target_lcn_lg: LCN_EX = LCN_EX.LCN_1X

    axon_skew: int = 0
    neuron_number: int = 0
    update_number: int = 0
    csc_accelerate: CSCAccelerateMode = CSCAccelerateMode.DISABLE

    scale_in: float = 1.0
    bias_in: float = 0.0
    scale_out: float = 1.0
    bias_out: float = 0.0
    learning_rate: float = 0.0

    update_core_xy: int = 0
    update_core_x: int = 0
    update_core_y: int = 0
    test_core_xy: int = 0
    test_core_x: int = 0
    test_core_y: int = 0

    global_send: int = 0
    global_receive: int = 0
    thread_number: int = 0
    busy_cycle: int = 2
    delay_cycle: int = 2
    width_cycle: int = 2

    tick_start: int | None = None
    tick_duration: int = 0
    tick_initial: int = 0

    def resolve_work_mode(self) -> OnlineCoreWorkMode:
        """Resolve the hardware work mode from semantic state."""
        if self.work_mode is not None:
            return self.work_mode

        if self.semantic_mode is OnlineCoreSemanticMode.FORWARD:
            return OnlineCoreWorkMode.FORWARD_INFERENCE
        if self.semantic_mode is OnlineCoreSemanticMode.LOSS:
            return OnlineCoreWorkMode.LOSS_FN
        if self.semantic_mode is OnlineCoreSemanticMode.GRADIENT:
            if self.gradient_role is OnlineGradientRole.OUTPUT:
                return OnlineCoreWorkMode.OUTPUT_LAYER_GRADIENT
            if self.gradient_role is OnlineGradientRole.HIDDEN:
                return OnlineCoreWorkMode.MIDDLE_LAYER_GRADIENT
            raise ValueError("gradient_role is required for semantic_mode='gradient'")
        if self.semantic_mode is OnlineCoreSemanticMode.POOL_GRADIENT:
            if self.pooling_mode is PoolingMode.MAX:
                return OnlineCoreWorkMode.MAX_POOLING_GRADIENT
            return OnlineCoreWorkMode.AVG_POOLING_GRADIENT
        if self.semantic_mode is OnlineCoreSemanticMode.UPDATE:
            if self.update_direction is OnlineUpdateDirection.FORWARD:
                return OnlineCoreWorkMode.FORWARD_WEIGHT_UPDATE
            if self.update_direction is OnlineUpdateDirection.BACKWARD:
                return OnlineCoreWorkMode.BACKWARD_WEIGHT_UPDATE
            raise ValueError("update_direction is required for semantic_mode='update'")

        raise AssertionError(f"unexpected online semantic mode: {self.semantic_mode}")

    def refine_work_mode(self) -> None:
        """Materialize the hardware work mode in-place."""
        self.work_mode = self.resolve_work_mode()

    def validate_tick_params(self) -> None:
        """Validate tick parameters against chip register limits."""
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
