from pydantic import BaseModel, Field, field_serializer, model_validator

from .reg_types import *


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
