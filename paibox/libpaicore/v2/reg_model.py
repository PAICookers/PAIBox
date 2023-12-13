from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_serializer,
    model_validator,
)
from typing_extensions import TypedDict  # Use `typing_extensions.TypedDict`.

from .coordinate import Coord
from .hw_defs import HwConfig
from .reg_types import *

__all__ = ["CoreParams", "ParamsReg"]

WEIGHT_PRECISION_BIT_MAX = 2  # Not used
LCN_EXTENSION_BIT_MAX = 2  # Not used
INPUT_WIDTH_FORMAT_BIT_MAX = 2  # Not used
SPIKE_WIDTH_FORMAT_BIT_MAX = 2  # Not used
NUM_DENDRITE_BIT_MAX = 13
POOL_MAX_EN_BIT_MAX = 1  # Not used
TICK_WAIT_START_BIT_MAX = 15
TICK_WAIT_END_BIT_MAX = 15
SNN_MODE_EN_BIT_MAX = 1  # Not used
TARGET_LCN_BIT_MAX = 4
TEST_CHIP_ADDR_BIT_MAX = 10


class CoreParams(BaseModel):
    """Parameter model of register parameters listed in Section 2.4.1.

    NOTE: The parameters input in the model are declared in `docs/Table-of-Terms.md`.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    name: str = Field(description="Name of the physical core.", exclude=True)

    weight_precision: WeightPrecisionType = Field(
        le=WeightPrecisionType.WEIGHT_WIDTH_8BIT,
        serialization_alias="weight_width",
        description="Weight precision of crossbar.",
    )

    lcn_extension: LCNExtensionType = Field(
        le=LCNExtensionType.LCN_64X,
        serialization_alias="LCN",
        description="Scale of fan-in extension.",
    )

    input_width_format: InputWidthFormatType = Field(
        serialization_alias="input_width",
        description="Format of input spike.",
    )

    spike_width_format: SpikeWidthFormatType = Field(
        serialization_alias="spike_width",
        description="Format of output spike.",
    )

    num_dendrite: int = Field(
        ge=0,
        lt=(1 << NUM_DENDRITE_BIT_MAX),
        serialization_alias="neuron_num",
        description="The number of valid dendrites.",
    )

    max_pooling_en: MaxPoolingEnableType = Field(
        default=MaxPoolingEnableType.DISABLE,
        serialization_alias="pool_max",
        description="Enable max pooling or not in 8-bit input format.",
    )

    tick_wait_start: int = Field(
        default=0,
        ge=0,
        lt=(1 << TICK_WAIT_START_BIT_MAX),
        description="The core begins to work at #N sync_all. 0 for not starting.",
    )

    tick_wait_end: int = Field(
        default=0,
        ge=0,
        lt=(1 << TICK_WAIT_END_BIT_MAX),
        description="The core keeps working within #N sync_all. 0 for not stopping.",
    )

    snn_mode_en: SNNModeEnableType = Field(
        default=SNNModeEnableType.ENABLE,
        serialization_alias="snn_en",
        description="Enable SNN mode or not.",
    )

    target_lcn: LCNExtensionType = Field(
        le=LCNExtensionType.LCN_64X,
        serialization_alias="target_LCN",
        description="LCN of the target core.",
    )

    test_chip_addr: Coord = Field(
        default=Coord(0, 0),
        description="Destination address of output test frames.",
    )

    """Parameter checks"""

    @model_validator(mode="after")
    def _neuron_num_range_limit(self):
        if self.input_width_format is InputWidthFormatType.WIDTH_1BIT:
            if self.num_dendrite > HwConfig.N_DENDRITE_MAX_ANN:
                raise ValueError(
                    f"Param 'num_dendrite' out of range. When input width is 1-bit,"
                    f"The #N of dendrites should be <= {HwConfig.N_DENDRITE_MAX_ANN}."
                )
        else:
            if self.num_dendrite > HwConfig.N_DENDRITE_MAX_SNN:
                raise ValueError(
                    f"Param 'num_dendrite' out of range. When input width is 8-bit,"
                    f"The #N of dendrite should be <= {HwConfig.N_DENDRITE_MAX_SNN}."
                )

        return self

    @model_validator(mode="after")
    def _max_pooling_en_check(self):
        if (
            self.input_width_format is InputWidthFormatType.WIDTH_1BIT
            and self.max_pooling_en is MaxPoolingEnableType.ENABLE
        ):
            self.max_pooling_en = MaxPoolingEnableType.DISABLE

        return self

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

    @field_serializer("test_chip_addr")
    def _test_chip_addr(self, test_chip_addr: Coord) -> int:
        return test_chip_addr.address


ParamsReg = CoreParams


class _ParamsRegDict(TypedDict):
    """Typed dictionary of `ParamsReg` for typing check."""

    weight_width: int
    LCN: int
    input_width: int
    spike_width: int
    neuron_num: int
    pool_max: int
    tick_wait_start: int
    tick_wait_end: int
    snn_en: int
    target_LCN: int
    test_chip_addr: int


ParamsRegChecker = TypeAdapter(_ParamsRegDict)
