"""
    This file defines parameters of registers & RAM of offline cores.
"""


from pydantic import BaseModel, Field, field_serializer, model_validator

from ._core_mode import CoreMode
from .params_types import *

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
    """Parameter model of RAM parameters listed in Section 2.4.2

    Example:
        model_ram = ParamsRAM(...)

        params_ram_dict = model_ram.model_dump(by_alias=True)

    Return:
        a dictionary of RAM parameters in which the keys are serialization alias if defined.

    NOTE: The parameters input in the model are declared in `docs/Table-of-Terms.md`.
    """

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
        ..., ge=0, lt=(1 << _ADDR_AXON_BIT_MAX), description="Destination axon address."
    )

    addr_core_x: int = Field(
        ...,
        ge=0,
        lt=(1 << _ADDR_CORE_X_BIT_MAX),
        description="Address X of destination core.",
    )

    addr_core_y: int = Field(
        ...,
        ge=0,
        lt=(1 << _ADDR_CORE_Y_BIT_MAX),
        description="Address Y of destination core.",
    )

    addr_core_x_ex: int = Field(
        ...,
        ge=0,
        lt=(1 << _ADDR_CORE_X_EX_BIT_MAX),
        description="Broadcast address X of destination core.",
    )

    addr_core_y_ex: int = Field(
        ...,
        ge=0,
        lt=(1 << _ADDR_CORE_Y_EX_BIT_MAX),
        description="Broadcast address Y of destination core.",
    )

    addr_chip_x: int = Field(
        default=0,
        ge=0,
        lt=(1 << _ADDR_CHIP_X_BIT_MAX),
        description="Address X of destination chip.",
    )
    addr_chip_y: int = Field(
        default=0,
        ge=0,
        lt=(1 << _ADDR_CHIP_Y_BIT_MAX),
        description="Address Y of destination chip.",
    )

    reset_mode: ResetMode = Field(
        default=ResetMode.MODE_NORMAL,
        description="Reset modes of neurons.",
    )

    reset_v: int = Field(
        default=0,
        ge=-(1 << (_RESET_V_BIT_MAX - 1)),
        lt=(1 << (_RESET_V_BIT_MAX - 1)),
        description="Reset value of membrane potential, 30-bit signed.",
    )

    leaking_comparison: LeakingComparisonMode = Field(
        default=LeakingComparisonMode.LEAK_AFTER_COMP,
        serialization_alias="leak_post",
        description="Leak after comparison or before.",
    )

    threshold_mask_bits: int = Field(
        default=0,
        ge=0,
        lt=(1 << _THRESHOLD_MASK_CTRL_BIT_MAX),
        serialization_alias="threshold_mask_ctrl",
        description="X-bits mask for random threshold.",
    )

    neg_thres_mode: NegativeThresholdMode = Field(
        default=NegativeThresholdMode.MODE_RESET,
        serialization_alias="threshold_neg_mode",
        description="Modes of negative threshold.",
    )

    neg_thres_value: int = Field(
        default=0,
        ge=0,
        lt=(1 << _NEGATIVE_THRESHOLD_VALUE_BIT_MAX),
        serialization_alias="threshold_neg",
        description="Negative threshold, 29-bit unsigned.",
    )

    pos_thres_value: int = Field(
        default=0,
        ge=0,
        lt=(1 << _POSITIVE_THRESHOLD_VALUE_BIT_MAX),
        serialization_alias="threshold_pos",
        description="Positive threshold, 29-bit unsigned.",
    )

    leaking_direction: LeakingDirectionMode = Field(
        default=LeakingDirectionMode.MODE_FORWARD,
        serialization_alias="leak_reversal_flag",
        description="Direction of leaking, forward or reversal.",
    )

    leaking_integration_mode: LeakingIntegrationMode = Field(
        default=LeakingIntegrationMode.MODE_DETERMINISTIC,
        serialization_alias="leak_det_stoch",
        description="Modes of leaking integration, deterministic or stochastic.",
    )

    leak_v: int = Field(
        default=0,
        ge=-(1 << (_LEAK_V_BIT_MAX - 1)),
        lt=(1 << (_LEAK_V_BIT_MAX - 1)),
        description="Leaking potential, 30-bit signed.",
    )

    synaptic_integration_mode: SynapticIntegrationMode = Field(
        default=SynapticIntegrationMode.MODE_DETERMINISTIC,
        serialization_alias="weight_det_stoch",
        description="Modes of synaptic integration, deterministic or stochastic.",
    )

    bit_truncate: int = Field(
        default=8,  # TODO consider to judge based on *spike_width*
        ge=0,
        lt=(1 << _BIT_TRUNCATE_BIT_MAX),
        description="Position of truncation, unsigned int, 5-bits.",
    )

    vjt_pre: int = Field(
        default=0,  # TODO Good for a fixed value?
        ge=-(1 << (_VJT_PRE_BIT_MAX - 1)),
        lt=(1 << (_VJT_PRE_BIT_MAX - 1)),
        description="Membrane potential of neuron at last time step, 30-bit signed. 0 at initialization.",
    )

    """Parameter serializers"""

    @field_serializer("reset_mode")
    def _reset_mode(self, reset_mode: ResetMode) -> int:
        return reset_mode.value

    @field_serializer("leaking_comparison")
    def _leaking_comparison(self, leaking_comparison: LeakingComparisonMode) -> int:
        return leaking_comparison.value

    @field_serializer("neg_thres_mode")
    def _neg_thres_mode(self, neg_thres_mode: NegativeThresholdMode) -> int:
        return neg_thres_mode.value

    @field_serializer("leaking_direction")
    def _leaking_direction(self, leaking_direction: LeakingDirectionMode) -> int:
        return leaking_direction.value

    @field_serializer("leaking_integration_mode")
    def _leaking_integration_mode(
        self, leaking_integration_mode: LeakingIntegrationMode
    ) -> int:
        return leaking_integration_mode.value

    @field_serializer("synaptic_integration_mode")
    def _synaptic_integration_mode(
        self, synaptic_integration_mode: SynapticIntegrationMode
    ) -> int:
        return synaptic_integration_mode.value


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
