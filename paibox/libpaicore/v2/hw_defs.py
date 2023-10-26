class HwConfig:
    """Basic hardware configuration of PAICORE 2.0."""

    Y_PRIORITY = True
    """Coordinate priority"""

    CHIP_X_MIN = 0
    CHIP_X_MAX = 0
    CHIP_Y_MIN = 0
    CHIP_Y_MAX = 0

    N_BIT_CORE_X = 5
    N_BIT_CORE_Y = 5

    CORE_X_MIN = 0
    CORE_X_MAX = (1 << N_BIT_CORE_X) - 1
    CORE_Y_MIN = 0
    CORE_Y_MAX = (1 << N_BIT_CORE_Y) - 1

    N_CORE_OFFLINE = 1008
    """The #N of offline cores."""
    CORE_X_OFFLINE_MIN = CORE_X_MIN
    CORE_Y_OFFLINE_MIN = CORE_Y_MIN
    CORE_X_OFFLINE_MAX = 28
    CORE_Y_OFFLINE_MAX = 28
    CORE_X_ONLINE_MIN = 29
    CORE_Y_ONLINE_MIN = 29
    CORE_X_ONLINE_MAX = CORE_X_MAX
    CORE_Y_ONLINE_MAX = CORE_Y_MAX

    N_AXON_DEFAULT = 1152
    N_NEURON_ONE_CORE_DEFAULT = 512
    N_NEURON_ONE_CORE_MAX = N_NEURON_ONE_CORE_DEFAULT
    """The maximum #N of neurons in one core."""

    N_TIMESLOT_MAX = 256

    N_SUB_ROUTING_NODE = 4
    """The number of sub routing nodes of a node."""
