from dataclasses import dataclass

from paicorelib import (
    LCN_EX,
    AddPotentialMode,
    CoordXY,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    OfflineCoreRegV2,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
)

TEST_DEST_CORE = CoordXY(0, 0)


# default core configs that not used temporarily, all cores share the same default configs
@dataclass
class Default_Core_Config:
    busy_cycle: int = 2
    delay_cycle: int = 2
    width_cycle: int = 2
    thread_number: int = 0
    csc_accelerate: CSCAccelerateMode = CSCAccelerateMode.ENABLE


# this core configs are automatically set by backend according to routing and allocation result
@dataclass
class Auto_Core_Config:
    neuron_number: int = 0
    test_core_xy: int = 0
    test_core_x: int = 0
    test_core_y: int = 0
    global_send: int = 0
    global_receive: int = 0


# these core configs can be set for optimization requirements
@dataclass(frozen=True)
class Backend_Core_Config:
    lcn: LCN_EX = LCN_EX.LCN_1X
    target_lcn: LCN_EX = LCN_EX.LCN_1X
    axon_skew: int = 0
    add_potential: AddPotentialMode = AddPotentialMode.NORMAL


@dataclass(frozen=True)
class Frontend_Core_Config:
    snn_ann: SNNMode = SNNMode.SNN
    max_pooling: PoolingMode = PoolingMode.AVERAGE
    zero_output: ZeroOutputMode = ZeroOutputMode.DISABLE
    input_sign: DataSign = DataSign.SIGNED
    input_width: DataWidth = DataWidth.WIDTH_8BIT
    output_sign: DataSign = DataSign.SIGNED
    output_width: DataWidth = DataWidth.WIDTH_8BIT
    weight_sign: DataSign = DataSign.SIGNED
    weight_width: DataWidth = DataWidth.WIDTH_8BIT
    tick_start: int = 1
    tick_duration: int = 0
    tick_initial: int = 0


def to_core_reg(
    default_conf: Default_Core_Config,
    auto_conf: Auto_Core_Config,
    backend_conf: Backend_Core_Config,
    frontend_conf: Frontend_Core_Config,
    coord: CoordXY,
) -> OfflineCoreRegV2:

    core_reg = OfflineCoreRegV2(
        name=f"core_reg_at_({coord.x},{coord.y})",
        snn_ann=frontend_conf.snn_ann,
        max_pooling=frontend_conf.max_pooling,
        add_potential=backend_conf.add_potential,
        zero_output=frontend_conf.zero_output,
        input_sign=frontend_conf.input_sign,
        input_width=frontend_conf.input_width,
        output_sign=frontend_conf.output_sign,
        output_width=frontend_conf.output_width,
        weight_sign=frontend_conf.weight_sign,
        weight_width=frontend_conf.weight_width,
        lcn=backend_conf.lcn,
        target_lcn=backend_conf.target_lcn,
        axon_skew=backend_conf.axon_skew,
        neuron_number=auto_conf.neuron_number,
        test_core_xy=auto_conf.test_core_xy,
        test_core_x=auto_conf.test_core_x,
        test_core_y=auto_conf.test_core_y,
        global_send=auto_conf.global_send,
        csc_accelerate=default_conf.csc_accelerate,
        global_receive=auto_conf.global_receive,
        thread_number=default_conf.thread_number,
        busy_cycle=default_conf.busy_cycle,
        delay_cycle=default_conf.delay_cycle,
        width_cycle=default_conf.width_cycle,
        tick_start=frontend_conf.tick_start,
        tick_duration=frontend_conf.tick_duration,
        tick_initial=frontend_conf.tick_initial,
    )
    return core_reg
