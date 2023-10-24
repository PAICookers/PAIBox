from pydantic import BaseModel, Field, field_serializer

from .ram_types import *


class NeuronDestConfig(BaseModel, extra="ignore", validate_assignment=True):
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


class NeuronSelfConfig(BaseModel, extra="ignore", validate_assignment=True):
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

    neg_threshold: int = Field(
        default=0,
        ge=0,
        lt=(1 << _NEGATIVE_THRESHOLD_VALUE_BIT_MAX),
        serialization_alias="threshold_neg",
        description="Negative threshold, 29-bit unsigned.",
    )

    pos_threshold: int = Field(
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

    vjt_init: int = Field(
        default=0,
        ge=-(1 << (_VJT_PRE_BIT_MAX - 1)),
        lt=(1 << (_VJT_PRE_BIT_MAX - 1)),
        serialization_alias="vjt_pre",
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


class NeuronParams(NeuronDestConfig, NeuronSelfConfig):
    pass


ParamsRAM = NeuronParams
