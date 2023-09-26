class HwConfig:
    Y_PRIORITY = True

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

    N_AXON_DEFAULT = 1152
    N_NEURON_DEFAULT = 512
    N_TIMESLOT_MAX = 256
