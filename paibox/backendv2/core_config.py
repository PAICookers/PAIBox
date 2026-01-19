from paicorelib import (
    LCN_EX,
    AddPotentialMode,
    CSCAccelerateMode,
    InputSignMode,
    OutputSignMode,
    PoolingMode,
    SNNMode,
    WeightSignMode,
    WeightWidth,
    ZeroOutputMode,
)


# default core configs that not used temporarily, all cores share the same default configs
class Default_Core_Config:
    def __init__(self):
        self.busy_cycle: int = 1
        self.delay_cycle: int = 1
        self.width_cycle: int = 1
        self.thread_number: int = 0
        self.add_potential: AddPotentialMode = AddPotentialMode.NORMAL
        self.csc_accelerate: CSCAccelerateMode = CSCAccelerateMode.DISABLE


# this core configs are automatically set by backend according to routing and allocation result
class Auto_Core_Config:
    def __init__(self):
        self.neuron_number: int = 0
        self.test_core_xy: int = 0
        self.test_core_x: int = 0
        self.test_core_y: int = 0
        self.global_send: int = 0
        self.global_receive: int = 0


# this core configs are inherited from frontend's compute op nodes, backend can not modify them except weight_width
# weigth_width can be modified by backend for optimization, if don't change, set weight_width in Manual_Core_Config to the same value
class Inherited_Core_Config:
    def __init__(self):
        self.snn_ann: SNNMode = SNNMode.SNN
        self.max_pooling: PoolingMode = PoolingMode.MAX
        self.zero_output: ZeroOutputMode = ZeroOutputMode.DISABLE
        self.input_sign: InputSignMode = InputSignMode.SIGNED
        self.input_width: WeightWidth = WeightWidth.WEIGHT_WIDTH_1BIT
        self.output_sign: OutputSignMode = OutputSignMode.SIGNED
        self.output_width: WeightWidth = WeightWidth.WEIGHT_WIDTH_1BIT
        self.weight_sign: WeightSignMode = WeightSignMode.SIGNED
        self.weight_width: WeightWidth = WeightWidth.WEIGHT_WIDTH_1BIT
        self.tick_start: int = 0
        self.tick_duration: int = 0
        self.tick_initial: int = 0


# these core configs can be set for optimization requirements
class Manual_Core_Config:
    def __init__(self):
        self.lcn: LCN_EX = LCN_EX.LCN_1X
        self.target_lcn: LCN_EX = LCN_EX.LCN_1X
        self.weight_width: WeightWidth = WeightWidth.WEIGHT_WIDTH_1BIT
        self.axon_skew: int = 0
