from dataclasses import asdict, dataclass
from typing import Any, TypedDict

from paicorelib import (
    RM,
    AddPotentialMode,
    InputSignMode,
    InputWidthFormat,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    OutputSignMode,
    PoolingMode,
    SNNMode,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightSignMode,
    WeightWidth,
    ZeroOutputMode,
    OutputType,
)
from torch import Tensor

OutputWidthFormat = InputWidthFormat


@dataclass
class ClacParams:
    def make_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NeuV2ClacParams(ClacParams):
    reset_mode: RM = RM.MODE_NORMAL
    reset_v: float = 0.0
    thres_neg_mode: ThresholdNegMode = ThresholdNegMode.FIRE
    thres_pos_mode: ThresholdPosMode = ThresholdPosMode.FIRE
    thres_neg: float = -99999.0  # OfflineNeuRegLimV2.THRES_NEG_MIN
    thres_pos: float = 0.0
    lateral_inhi: LateralInhibitionMode = LateralInhibitionMode.DISABLE
    leak_multi_sequence: LeakMultiComparisonOrder = LeakMultiComparisonOrder.AFTER_COMPARE
    leak_multi_input: LeakMultiInputMode = LeakMultiInputMode.DISABLE
    leak_multi_mode: LeakMultiMode = LeakMultiMode.DISABLE
    leak_add_mode: LeakAddMode = LeakAddMode.FORWARD
    leak_tau: int = 0
    leak_v: float = 0.0
    init_v: float = 0.0
    output_type: OutputType = OutputType.VALUE

    @classmethod
    def default(cls):
        return cls(
            RM.MODE_NORMAL,
            0,
            ThresholdNegMode.FIRE,
            ThresholdPosMode.FIRE,
            -99999,  # OfflineNeuRegLimV2.THRES_NEG_MIN,
            0,
            LateralInhibitionMode.DISABLE,
            LeakMultiComparisonOrder.AFTER_COMPARE,
            LeakMultiInputMode.DISABLE,
            LeakMultiMode.DISABLE,
            LeakAddMode.FORWARD,
            0,
            0,
            0,
            OutputType.VALUE,
        )


class CoreClacParams(ClacParams):
    pass


@dataclass
class OfflineCoreCalcParamsV2(CoreClacParams):
    snn_ann: int | SNNMode = SNNMode.SNN
    max_pooling: int | PoolingMode = PoolingMode.AVERAGE
    add_potential: int | AddPotentialMode = AddPotentialMode.NORMAL
    zero_output: int | ZeroOutputMode = ZeroOutputMode.DISABLE
    input_sign: int | InputSignMode = InputSignMode.SIGNED
    input_width: int | WeightWidth = WeightWidth.WEIGHT_WIDTH_8BIT
    output_sign: int | OutputSignMode = OutputSignMode.SIGNED
    output_width: int | WeightWidth = WeightWidth.WEIGHT_WIDTH_8BIT
    weight_sign: int | WeightSignMode = WeightSignMode.SIGNED
    weight_width: int | WeightWidth = WeightWidth.WEIGHT_WIDTH_8BIT
    tick_start: int = 1
    tick_duration: int = 0
    tick_initial: int = 0


class OnlineCoreV2CalcParamsKwds(TypedDict, total=False):
    # TODO
    snn_ann: SNNMode  # differ
    max_pooling: bool | PoolingMode
    add_potential: AddPotentialMode
    zero_output: bool | ZeroOutputMode
    work_mode: bool  # differ
    input_core: bool  # differ
    input_sign: bool | InputSignMode
    input_width: InputWidthFormat
    output_sign: bool | OutputSignMode
    output_width: OutputWidthFormat
    weight_sign: bool | WeightSignMode
    weight_width: WeightWidth
    tick_start: int
    tick_duration: int
    tick_initial: int
