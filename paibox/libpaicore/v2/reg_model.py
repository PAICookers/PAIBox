from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from .reg_types import *

WEIGHT_PRECISION_BIT_MAX = 2  # Not used
LCN_EXTENSION_BIT_MAX = 2  # Not used
INPUT_WIDTH_FORMAT_BIT_MAX = 2  # Not used
SPIKE_WIDTH_FORMAT_BIT_MAX = 2  # Not used
NEURON_NUM_BIT_MAX = 13
POOL_MAX_EN_BIT_MAX = 1  # Not used
TICK_WAIT_START_BIT_MAX = 15
TICK_WAIT_END_BIT_MAX = 15
SNN_MODE_EN_BIT_MAX = 1  # Not used
TARGET_LCN_BIT_MAX = 4
TEST_CHIP_ADDR_BIT_MAX = 10


class CoreParams(BaseModel, validate_assignment=True):
    """Parameter model of register parameters listed in Section 2.4.1"""

    model_config = ConfigDict(extra="ignore")

    weight_precision: WeightPrecisionType = Field(
        lt=WeightPrecisionType.WEIGHT_WIDTH_MAX,
        serialization_alias="weight_width",
        description="Weight precision of crossbar.",
    )

    lcn_extension: LCNExtensionType = Field(
        lt=LCNExtensionType.LCN_MAX,
        serialization_alias="LCN",
        description="Scale of Fan-in extension.",
    )

    input_width_format: InputWidthFormatType = Field(
        serialization_alias="input_width",
        description="Format of input spike.",
    )

    spike_width_format: SpikeWidthFormatType = Field(
        serialization_alias="spike_width",
        description="Format of output spike.",
    )

    neuron_num: int = Field(
        ge=0,
        lt=(1 << NEURON_NUM_BIT_MAX),
        description="Valid number of dendrites.",
    )

    max_pooling_en: MaxPoolingEnableType = Field(
        serialization_alias="pool_max",
        description="Enable max pooling or not in 8-bit input format.",
    )

    tick_wait_start: int = Field(
        ge=0,
        lt=(1 << TICK_WAIT_START_BIT_MAX),
        description="The core begins to work at #N sync_all. 0 for not starting.",
    )

    tick_wait_end: int = Field(
        ge=0,
        lt=(1 << TICK_WAIT_END_BIT_MAX),
        description="The core keeps working within #N sync_all. 0 for  not stopping.",
    )

    snn_mode_en: SNNModeEnableType = Field(
        serialization_alias="snn_en",
        description="Enable SNN mode or not.",
    )

    target_lcn: LCNExtensionType = Field(
        lt=LCNExtensionType.LCN_MAX,
        serialization_alias="target_LCN",
        description="LCN of the target core.",
    )

    test_chip_addr: int = Field(
        default=0,
        ge=0,
        lt=(1 << TEST_CHIP_ADDR_BIT_MAX),
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

        return m

    """Parameter serializers"""

    @field_serializer("weight_precision")
    def _weight_precision(self, weight_precision: WeightPrecisionType) -> int:
        return weight_precision.value

    @field_serializer("lcn_extension")
    def _lcn_extension(self, lcn_extension: LCNExtensionType) -> int:
        return lcn_extension.value

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

    @field_serializer("target_lcn")
    def _target_lcn(self, target_lcn: LCNExtensionType) -> int:
        return target_lcn.value


ParamsReg = CoreParams
