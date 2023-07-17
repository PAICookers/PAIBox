"""
    This file defines parameters of registers & RAM of offline cores.
"""


from pydantic import BaseModel, Field, field_serializer, model_validator
from params_types import *
from _core_mode import CoreMode


__all__ = ["ParamsReg", "ParamsRAM", "get_core_mode"]


class ParamsReg(BaseModel, extra="ignore", validate_assignment=True):
    """Parameter model of register parameters listed in Section 2.4.1"""

    _WEIGHT_PRECISION_BIT_MAX = 2
    _LCN_EXTENSION_BIT_MAX = 2
    _INPUT_WIDTH_FORMAT_BIT_MAX = 2
    _SPIKE_WIDTH_FORMAT_BIT_MAX = 2
    _NEURON_NUM_BIT_MAX = 13
    _POOL_MAX_EN_BIT_MAX = 1
    _TICK_WAIT_START_BIT_MAX = 15
    _TICK_WAIT_END_BIT_MAX = 15
    _SNN_MODE_EN_BIT_MAX = 1
    _TARGET_LCN_BIT_MAX = 4
    _TEST_CHIP_ADDR_BIT_MAX = 10

    weight_precision: WeightPrecisionType = Field(
        default=WeightPrecisionType.WEIGHT_WIDTH_8BIT,
        lt=WeightPrecisionType.WEIGHT_WIDTH_MAX,
        serialization_alias="weight_width",
        description="Weight precision of crossbar.",
    )

    LCN_extension: LCNExtensionType = Field(
        default=LCNExtensionType.LCN_1X,
        lt=LCNExtensionType.LCN_MAX,
        serialization_alias="LCN",
        description="Scale of Fan-in extension.",
    )

    input_width_format: InputWidthFormatType = Field(
        default=InputWidthFormatType.WIDTH_1BIT,
        serialization_alias="input_width",
        description="Format of input spike.",
    )

    spike_width_format: SpikeWidthFormatType = Field(
        default=SpikeWidthFormatType.WIDTH_1BIT,
        serialization_alias="spike_width",
        description="Format of output spike.",
    )

    neuron_num: int = Field(
        ...,
        ge=0,
        lt=(1 << _NEURON_NUM_BIT_MAX),
        description="Valid number of dendrites.",
    )

    max_pooling_en: MaxPoolingEnableType = Field(
        default=MaxPoolingEnableType.DISABLE,
        serialization_alias="pool_max",
        description="Enable max pooling or not in 8-bit input format.",
    )

    tick_wait_start: int = Field(
        ...,
        ge=0,
        lt=(1 << _TICK_WAIT_START_BIT_MAX),
        description="The core begins to work at #N sync_all. 0 for not starting.",
    )

    tick_wait_end: int = Field(
        ...,
        ge=0,
        lt=(1 << _TICK_WAIT_END_BIT_MAX),
        description="The core keeps working within #N sync_all. 0 for  not stopping.",
    )

    snn_mode_en: SNNModeEnableType = Field(
        default=SNNModeEnableType.DISABLE,
        serialization_alias="snn_en",
        description="Enable SNN mode or not.",
    )

    target_LCN: int = Field(
        ...,
        ge=0,
        lt=(1 << _TARGET_LCN_BIT_MAX),
        serialization_alias="targetLCN",
        description="LCN of destination core.",
    )

    test_chip_addr: int = Field(
        ...,
        ge=0,
        lt=(1 << _TEST_CHIP_ADDR_BIT_MAX),
        description="Destination address of output test frames.",
    )

    """Parameter checks"""

    @model_validator(mode="after")  # type: ignore
    def _neuron_num_range_limit(cls, m: "ParamsReg") -> "ParamsReg":
        if m.input_width_format is InputWidthFormatType.WIDTH_1BIT:
            if m.neuron_num > 512:
                raise ValueError(
                    f"Param neuron_num out of range. When input_width_format is 1-bit, neuron_num should be less than 512."
                )
        else:
            if m.neuron_num > 4096:
                raise ValueError(
                    f"Param neuron_num out of range. When input_width_format is 8-bit, neuron_num should be less than 4096."
                )

        return m

    @model_validator(mode="after")  # type: ignore
    def _max_pooling_en_check(cls, m: "ParamsReg") -> "ParamsReg":
        if (
            m.input_width_format is InputWidthFormatType.WIDTH_1BIT
            and m.max_pooling_en is MaxPoolingEnableType.ENABLE
        ):
            m.max_pooling_en = MaxPoolingEnableType.DISABLE
            print(
                f"[Warning] Param max_pooling_en is set to MaxPoolingEnableType.DISABLE when input_width_format is 1-bit."
            )

        return m

    """Parameter serializers"""

    @field_serializer("weight_precision")
    def _weight_precision(self, weight_precision: WeightPrecisionType) -> int:
        return weight_precision.value

    @field_serializer("LCN_extension")
    def _LCN_extension(self, LCN_extension: LCNExtensionType) -> int:
        return LCN_extension.value

    @field_serializer("input_width_format")
    def _input_width_format(self, input_width_format: InputWidthFormatType) -> int:
        return input_width_format.value

    @field_serializer("spike_width_format")
    def _spike_width_format(self, spike_width_format: SpikeWidthFormatType) -> int:
        return spike_width_format.value

    @field_serializer("max_pooling_en")
    def _max_pooling_en(self, max_pooling_en: MaxPoolingEnableType) -> int:
        return max_pooling_en.value

    @field_serializer("snn_mode_en")
    def _snn_mode_en(self, snn_mode_en: SNNModeEnableType) -> int:
        return snn_mode_en.value


class ParamsRAM(BaseModel, extra="ignore", validate_assignment=True):
    """Parameter model of RAM parameters listed in Section 2.4.2"""

    _TICK_RELATIVE_BIT_MAX = 8
    _ADDR_AXON_BIT_MAX = 11
    _ADDR_CORE_X_BIT_MAX = 5
    _ADDR_CORE_Y_BIT_MAX = 5
    _ADDR_CORE_X_EX_BIT_MAX = 5
    _ADDR_CORE_Y_EX_BIT_MAX = 5
    _ADDR_CHIP_X_BIT_MAX = 5
    _ADDR_CHIP_Y_BIT_MAX = 5
    _RESET_MODE_BIT_MAX = 2
    _RESET_V_BIT_MAX = 30
    _LEAKING_COMPARISON_BIT_MAX = 1
    _THRESHOLD_MASK_CTRL_BIT_MAX = 5
    _NEGATIVE_THRESHOLD_MODE_BIT_MAX = 1
    _NEGATIVE_THRESHOLD_VALUE_BIT_MAX = 29
    _POSITIVE_THRESHOLD_VALUE_BIT_MAX = 29
    _LEAKING_DIRECTION_BIT_MAX = 1
    _LEAKING_MODE_BIT_MAX = 1
    _LEAK_V_BIT_MAX = 30
    _WEIGHT_MODE_BIT_MAX = 1
    _BIT_TRUNCATE_BIT_MAX = 5
    _VJT_PRE_BIT_MAX = 30

    tick_relative: int = Field(
        ...,
        ge=0,
        lt=(1 << _TICK_RELATIVE_BIT_MAX),
        description="Information of relative ticks.",
    )

    addr_axon: int = Field(
        ..., ge=0, lt=(1 << _ADDR_AXON_BIT_MAX), description="Destination axon"
    )

    addr_core_x: int = Field(
        ...,
        ge=0,
        lt=(1 << _ADDR_CORE_X_BIT_MAX),
        description="X address of destination core",
    )

    addr_core_y: int = Field(
        ...,
        ge=0,
        lt=(1 << _ADDR_CORE_Y_BIT_MAX),
        description="Y address of destination core",
    )

    addr_core_x_ex: int = Field(
        ...,
        ge=0,
        lt=(1 << _ADDR_CORE_X_EX_BIT_MAX),
        description="X broadcast address of destination core",
    )

    addr_core_y_ex: int = Field(
        ...,
        ge=0,
        lt=(1 << _ADDR_CORE_Y_EX_BIT_MAX),
        description="Y broadcast address of destination core",
    )

    addr_chip_x: int = Field(
        default=0,
        ge=0,
        lt=(1 << _ADDR_CHIP_X_BIT_MAX),
        description="X address of destination chip",
    )
    addr_chip_y: int = Field(
        default=0,
        ge=0,
        lt=(1 << _ADDR_CHIP_Y_BIT_MAX),
        description="Y address of destination chip",
    )

    reset_mode: ResetModeType = Field(
        default=ResetModeType.MODE_NORMAL, description="Reset modes of cores."
    )

    reset_v: int = Field(
        default=0,
        ge=-(1 << (_RESET_V_BIT_MAX - 1)),
        lt=(1 << (_RESET_V_BIT_MAX - 1)),
        description="Reset signed value of membrane potential.",
    )

    leaking_comparison: LeakingComparisonType = Field(
        default=LeakingComparisonType.LEAK_AFTER_COMP,
        serialization_alias="leak_post",
        description="Leak after comparison or before.",
    )

    thres_mask_ctrl: int = Field(
        default=0,
        ge=0,
        lt=(1 << _THRESHOLD_MASK_CTRL_BIT_MAX),
        serialization_alias="threshold_mask_ctrl",
        description="Signed value of threshold mask.",
    )

    neg_thres_mode: NegativeThresModeType = Field(
        default=NegativeThresModeType.MODE_RESET,
        serialization_alias="threshold_neg_mode",
        description="Modes of negative threshold.",
    )

    neg_thres_value: int = Field(
        default=0,
        ge=0,
        lt=(1 << _NEGATIVE_THRESHOLD_VALUE_BIT_MAX),
        serialization_alias="threshold_neg",
        description="Negative threshold value.",
    )

    pos_thres_value: int = Field(
        default=0,
        ge=0,
        lt=(1 << _POSITIVE_THRESHOLD_VALUE_BIT_MAX),
        serialization_alias="threshold_pos",
        description="Positive threshold value.",
    )

    leaking_direction: LeakingDirectionType = Field(
        default=LeakingDirectionType.FORWARD,
        serialization_alias="leak_reversal_flag",
        description="Direction of leaking, forward or reversal.",
    )

    leaking_mode: LeakingModeType = Field(
        default=LeakingModeType.MODE_DETERMINISTIC,
        serialization_alias="leak_det_stoch",
        description="Modes of leaking, deterministic or stochastic.",
    )

    leak_v: int = Field(
        default=0,
        ge=-(1 << (_LEAK_V_BIT_MAX - 1)),
        lt=(1 << (_LEAK_V_BIT_MAX - 1)),
        description="Value of leaking potential.",
    )

    weight_mode: WeightModeType = Field(
        default=WeightModeType.MODE_DETERMINISTIC,
        serialization_alias="weight_det_stoch",
        description="Modes of weights, deterministic or stochastic.",
    )

    bit_truncate: int = Field(
        default=8,  # TODO consider to judge based on *spike_width*
        ge=0,
        lt=(1 << _BIT_TRUNCATE_BIT_MAX),
        description="Value of bit truncation.",
    )

    vjt_pre: int = Field(
        default=0,
        ge=-(1 << (_VJT_PRE_BIT_MAX - 1)),
        lt=(1 << (_VJT_PRE_BIT_MAX - 1)),
        description="Value of membrane potential of pre-synaptic neuron.",
    )

    """Parameter serializers"""

    @field_serializer("reset_mode")
    def _reset_mode(self, reset_mode: ResetModeType) -> int:
        return reset_mode.value

    @field_serializer("leaking_comparison")
    def _leaking_comparison(self, leaking_comparison: LeakingComparisonType) -> int:
        return leaking_comparison.value

    @field_serializer("neg_thres_mode")
    def _neg_thres_mode(self, neg_thres_mode: NegativeThresModeType) -> int:
        return neg_thres_mode.value

    @field_serializer("leaking_direction")
    def _leaking_direction(self, leaking_direction: LeakingDirectionType) -> int:
        return leaking_direction.value

    @field_serializer("leaking_mode")
    def _leaking_mode(self, leaking_mode: LeakingModeType) -> int:
        return leaking_mode.value

    @field_serializer("weight_mode")
    def _weight_mode(self, weight_mode: WeightModeType) -> int:
        return weight_mode.value


def get_core_mode(
    iwidth_format: InputWidthFormatType,
    swidth_format: SpikeWidthFormatType,
    snn_en: SNNModeEnableType,
) -> CoreMode:
    """Get the working mode of the core.

    Decided by `input_width`, `spike_width` and `SNN_EN` of core parameters registers.

    NOTE: See table below for details.

    Mode            input_width    spike_width    SNN_EN
    BANN                0               0           0
    SNN                 0               0           1
    BANN/SNN to ANN     0               1           0
    BANN/SNN to SNN     0               1           1
    ANN to BANN/SNN     1               0       Don't care
    ANN                 1               1       Don't care
    """
    if iwidth_format is InputWidthFormatType.WIDTH_1BIT:
        if swidth_format is SpikeWidthFormatType.WIDTH_1BIT:
            if snn_en is SNNModeEnableType.DISABLE:
                # 0 / 0 / 0
                return CoreMode.MODE_BANN
            else:
                # 0 / 0 / 1
                return CoreMode.MODE_SNN
        else:
            if snn_en is SNNModeEnableType.DISABLE:
                # 0 / 1 / 0
                return CoreMode.MODE_BANN_OR_SNN_TO_ANN
            else:
                # 0 / 1 / 1
                return CoreMode.MODE_BANN_OR_SNN_TO_SNN
    elif swidth_format is SpikeWidthFormatType.WIDTH_1BIT:
        # 1 / 0 / *
        return CoreMode.MODE_ANN_TO_BANN_OR_SNN
    else:
        # 1 / 1 / *
        return CoreMode.MODE_ANN


if __name__ == "__main__":
    """Usages"""

    model_reg = ParamsReg(
        weight_precision=WeightPrecisionType.WEIGHT_WIDTH_8BIT,
        LCN_extension=LCNExtensionType.LCN_16X,
        input_width_format=InputWidthFormatType.WIDTH_1BIT,
        spike_width_format=SpikeWidthFormatType.WIDTH_1BIT,
        neuron_num=511,
        max_pooling_en=MaxPoolingEnableType.DISABLE,
        tick_wait_start=1,
        tick_wait_end=0,
        snn_mode_en=SNNModeEnableType.DISABLE,
        target_LCN=1,
        test_chip_addr=0,
    )
    a = model_reg.model_dump(by_alias=True)
    print(a)

    model_ram = ParamsRAM(
        tick_relative=1,
        addr_axon=0,
        addr_core_x=0,
        addr_core_y=0,
        addr_core_x_ex=0,
        addr_core_y_ex=0,
        addr_chip_x=0,
        addr_chip_y=0,
        reset_mode=ResetModeType.MODE_NORMAL,
        reset_v=0,
        leaking_comparison=LeakingComparisonType.LEAK_AFTER_COMP,
        thres_mask_ctrl=0,
        neg_thres_mode=NegativeThresModeType.MODE_RESET,
        neg_thres_value=0,
        pos_thres_value=0,
        leaking_direction=LeakingDirectionType.REVERSAL,
        leaking_mode=LeakingModeType.MODE_DETERMINISTIC,
        leak_v=0,
        weight_mode=WeightModeType.MODE_DETERMINISTIC,
        bit_truncate=8,
        vjt_pre=1,
    )
    b = model_ram.model_dump(by_alias=True, warnings=True)
    print(b)
