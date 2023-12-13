from typing import List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import TypedDict  # Use `typing_extensions.TypedDict`.

from .hw_defs import HwConfig
from .hw_types import AxonCoord
from .ram_types import *

__all__ = ["NeuronDestInfo", "NeuronAttrs"]

TICK_RELATIVE_BIT_MAX = 8
ADDR_AXON_BIT_MAX = 11
ADDR_CORE_X_BIT_MAX = 5
ADDR_CORE_Y_BIT_MAX = 5
ADDR_CORE_X_EX_BIT_MAX = 5
ADDR_CORE_Y_EX_BIT_MAX = 5
ADDR_CHIP_X_BIT_MAX = 5
ADDR_CHIP_Y_BIT_MAX = 5


class _BasicNeuronDest(BaseModel):
    """Parameter model of RAM parameters listed in Section 2.4.2.

    NOTE: The parameters input in the model are declared in `docs/Table-of-Terms.md`.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    addr_core_x: int = Field(
        ge=0,
        lt=(1 << ADDR_CORE_X_BIT_MAX),
        description="Address X of destination core.",
    )

    addr_core_y: int = Field(
        ge=0,
        lt=(1 << ADDR_CORE_Y_BIT_MAX),
        description="Address Y of destination core.",
    )

    addr_core_x_ex: int = Field(
        ge=0,
        lt=(1 << ADDR_CORE_X_EX_BIT_MAX),
        description="Broadcast address X of destination core.",
    )

    addr_core_y_ex: int = Field(
        ge=0,
        lt=(1 << ADDR_CORE_Y_EX_BIT_MAX),
        description="Broadcast address Y of destination core.",
    )

    addr_chip_x: int = Field(
        ge=0,
        lt=(1 << ADDR_CHIP_X_BIT_MAX),
        description="Address X of destination chip.",
    )
    addr_chip_y: int = Field(
        ge=0,
        lt=(1 << ADDR_CHIP_Y_BIT_MAX),
        description="Address Y of destination chip.",
    )


class NeuronDestInfo(_BasicNeuronDest):
    tick_relative: List[InstanceOf[int]] = Field(
        description="Information of relative ticks.",
    )

    addr_axon: List[InstanceOf[int]] = Field(description="Destination axon address.")

    @field_validator("tick_relative")
    @classmethod
    def _tick_relative_check(cls, v):
        if any(tr >= (1 << TICK_RELATIVE_BIT_MAX) or tr < 0 for tr in v):
            # DO NOT change the type of exception `ValueError` in the validators below.
            raise ValueError("Parameter 'tick relative' out of range.")

        return v

    @field_validator("addr_axon")
    @classmethod
    def _addr_axon_check(cls, v):
        if any(addr > HwConfig.ADDR_AXON_MAX or addr < 0 for addr in v):
            raise ValueError("Parameter 'addr_axon' out of range.")

        return v

    @model_validator(mode="after")
    def _length_match_check(self):
        if len(self.tick_relative) != len(self.addr_axon):
            raise ValueError(
                "Parameter 'tick relative' & 'addr_axon' must have the same "
                f"length: {len(self.tick_relative)}, {len(self.addr_axon)}."
            )

        return self


class OutpuNeuronDestInfo(_BasicNeuronDest):
    # TODO
    start_axon_coord: AxonCoord = Field(description="Address X of destination chip.")
    end_axon_coord: AxonCoord = Field(description="Address X of destination chip.")


RESET_MODE_BIT_MAX = 2  # Not used
RESET_V_BIT_MAX = 30
LEAKING_COMPARISON_BIT_MAX = 1  # Not used
THRESHOLD_MASK_CTRL_BIT_MAX = 5
NEGATIVE_THRESHOLD_MODE_BIT_MAX = 1  # Not used
NEGATIVE_THRESHOLD_VALUE_BIT_MAX = 29
POSITIVE_THRESHOLD_VALUE_BIT_MAX = 29
LEAKING_DIRECTION_BIT_MAX = 1  # Not used
LEAKING_MODE_BIT_MAX = 1  # Not used
LEAK_V_BIT_MAX = 30
WEIGHT_MODE_BIT_MAX = 1  # Not used
BIT_TRUNCATE_BIT_MAX = 5
VJT_PRE_BIT_MAX = 30


class NeuronAttrs(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    reset_mode: ResetMode = Field(
        description="Reset mode of neuron.",
    )

    reset_v: int = Field(
        ge=-(1 << (RESET_V_BIT_MAX - 1)),
        lt=(1 << (RESET_V_BIT_MAX - 1)),
        description="Reset value of membrane potential, 30-bit signed.",
    )

    leaking_comparison: LeakingComparisonMode = Field(
        serialization_alias="leak_post",
        description="Leaking after threshold comparison or before.",
    )

    threshold_mask_bits: int = Field(
        ge=0,
        lt=(1 << THRESHOLD_MASK_CTRL_BIT_MAX),
        serialization_alias="threshold_mask_ctrl",
        description="X-bits mask for random threshold.",
    )

    neg_thres_mode: NegativeThresholdMode = Field(
        serialization_alias="threshold_neg_mode",
        description="Modes of negative threshold.",
    )

    neg_threshold: int = Field(
        ge=0,
        lt=(1 << NEGATIVE_THRESHOLD_VALUE_BIT_MAX),
        serialization_alias="threshold_neg",
        description="Negative threshold, 29-bit unsigned.",
    )

    pos_threshold: int = Field(
        ge=0,
        lt=(1 << POSITIVE_THRESHOLD_VALUE_BIT_MAX),
        serialization_alias="threshold_pos",
        description="Positive threshold, 29-bit unsigned.",
    )

    leaking_direction: LeakingDirectionMode = Field(
        serialization_alias="leak_reversal_flag",
        description="Direction of leaking, forward or reversal.",
    )

    leaking_integration_mode: LeakingIntegrationMode = Field(
        serialization_alias="leak_det_stoch",
        description="Modes of leaking integration, deterministic or stochastic.",
    )

    leak_v: int = Field(
        ge=-(1 << (LEAK_V_BIT_MAX - 1)),
        lt=(1 << (LEAK_V_BIT_MAX - 1)),
        description="Leaking potential, 30-bit signed.",
    )

    synaptic_integration_mode: SynapticIntegrationMode = Field(
        serialization_alias="weight_det_stoch",
        description="Modes of synaptic integration, deterministic or stochastic.",
    )

    bit_truncate: int = Field(
        ge=0,
        lt=(1 << BIT_TRUNCATE_BIT_MAX),
        description="Position of truncation, unsigned int, 5-bits.",
    )

    vjt_init: int = Field(
        ge=-(1 << (VJT_PRE_BIT_MAX - 1)),
        lt=(1 << (VJT_PRE_BIT_MAX - 1)),
        serialization_alias="vjt_pre",
        description="Membrane potential of neuron at last time step, 30-bit signed. \
            0 at initialization.",
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
    def _lim(self, lim: LeakingIntegrationMode) -> int:
        return lim.value

    @field_serializer("synaptic_integration_mode")
    def _sim(self, sim: SynapticIntegrationMode) -> int:
        return sim.value


class _NeuronAttrsDict(TypedDict):
    """Typed dictionary of `NeuronAttrs` for typing check."""

    reset_mode: int
    reset_v: int
    leak_post: int
    threshold_mask_ctrl: int
    threshold_neg_mode: int
    threshold_neg: int
    threshold_pos: int
    leak_reversal_flag: int
    leak_det_stoch: int
    weight_det_stoch: int
    bit_truncate: int
    vjt_pre: int


class _NeuronDestInfoDict(TypedDict):
    """Typed dictionary of `NeuronDestInfo` for typing check."""

    addr_core_x: int
    addr_core_y: int
    addr_core_x_ex: int
    addr_core_y_ex: int
    addr_chip_x: int
    addr_chip_y: int
    tick_relative: List[int]
    addr_axon: List[int]


NeuronAttrsChecker = TypeAdapter(_NeuronAttrsDict)
NeuronDestInfoChecker = TypeAdapter(_NeuronDestInfoDict)
