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
)
from torch import Tensor

OutputWidthFormat = InputWidthFormat


@dataclass
class ClacParams:
    def make_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NeuV2ClacParams(ClacParams):
    reset_mode: RM
    reset_v: float
    thres_neg_mode: ThresholdNegMode
    thres_pos_mode: ThresholdPosMode
    thres_neg: float
    thres_pos: float
    lateral_inhi: LateralInhibitionMode
    leak_multi_sequence: LeakMultiComparisonOrder
    leak_multi_input: LeakMultiInputMode
    leak_multi_mode: LeakMultiMode
    leak_add_mode: LeakAddMode
    leak_tau: int
    leak_v: float | Tensor
    init_v: float

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
        )


class CoreClacParams(ClacParams):
    pass


@dataclass
class OfflineCoreCalcParamsV2(CoreClacParams):
    snn_ann: SNNMode
    max_pooling: PoolingMode
    add_potential: AddPotentialMode
    zero_output: ZeroOutputMode
    input_sign: InputSignMode
    input_width: InputWidthFormat
    output_sign: OutputSignMode
    output_width: OutputWidthFormat
    weight_sign: WeightSignMode
    weight_width: WeightWidth
    tick_start: int
    tick_duration: int
    tick_initial: int

    @classmethod
    def default(cls):
        return OfflineCoreCalcParamsV2(
            SNNMode.SNN,
            PoolingMode.AVERAGE,
            AddPotentialMode.NORMAL,
            ZeroOutputMode.DISABLE,
            InputSignMode.UNSIGNED,
            InputWidthFormat.WIDTH_1BIT,
            OutputSignMode.UNSIGNED,
            OutputWidthFormat.WIDTH_1BIT,
            WeightSignMode.SIGNED,
            WeightWidth.WEIGHT_WIDTH_8BIT,
            1,
            0,
            0,
        )


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
