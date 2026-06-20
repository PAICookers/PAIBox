from dataclasses import dataclass

from paicorelib import (
    LCN_EX,
    AddPotentialMode,
    CoordXY,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    OfflineCoreRegV2,
    OnlineCoreRegV2,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
)

from paibox.paiir.ir.calc_params import LutData, OnlineCoreParams

TEST_DEST_CORE = CoordXY(0, 0)


# default core configs that not used temporarily, all cores share the same default configs
@dataclass
class Default_Core_Config:
    # Hardware timing margin for complete-frame emission after busy is low.
    # Route selection should still align DATA and control-frame CPU ingress sides;
    # do not rely on this margin as the primary ordering fix.
    busy_cycle: int = 20
    delay_cycle: int = 20
    width_cycle: int = 10
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


@dataclass(frozen=True)
class Frontend_Core_Config:
    add_potential: AddPotentialMode = AddPotentialMode.NORMAL
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
    lut_data: LutData | None = None


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
        add_potential=frontend_conf.add_potential,
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


def to_online_core_reg(
    core_params: OnlineCoreParams,
    coord: CoordXY,
    auto_conf: Auto_Core_Config | None = None,
) -> OnlineCoreRegV2:
    if core_params.tick_start is None:
        raise ValueError(
            "online backend bridge requires tick_start to be assigned before "
            "mapping to OnlineCoreRegV2"
        )

    work_mode = core_params.work_mode
    if work_mode is None:
        work_mode = core_params.resolve_work_mode()

    if (
        auto_conf is not None
        and core_params.test_core_xy == 0
        and core_params.test_core_x == 0
        and core_params.test_core_y == 0
    ):
        test_core_xy = auto_conf.test_core_xy
        test_core_x = auto_conf.test_core_x
        test_core_y = auto_conf.test_core_y
    else:
        test_core_xy = core_params.test_core_xy
        test_core_x = core_params.test_core_x
        test_core_y = core_params.test_core_y

    return OnlineCoreRegV2(
        name=f"online_core_reg_at_({coord.x},{coord.y})",
        snn_ann=core_params.snn_mode,
        max_pooling=core_params.pooling_mode,
        add_potential=core_params.add_potential,
        zero_output=core_params.zero_output,
        work_mode=work_mode,
        input_core=core_params.input_core,
        input_width=core_params.input_width,
        output_core=core_params.output_core,
        output_width=core_params.output_width,
        lcn_at=core_params.lcn_at,
        lcn_mp=core_params.lcn_mp,
        lcn_lg=core_params.lcn_lg,
        target_lcn_at=core_params.target_lcn_at,
        target_lcn_mp=core_params.target_lcn_mp,
        target_lcn_lg=core_params.target_lcn_lg,
        axon_skew=core_params.axon_skew,
        neuron_number=core_params.neuron_number,
        update_number=core_params.update_number,
        csc_accelerate=core_params.csc_accelerate,
        scale_in=core_params.scale_in,
        bias_in=core_params.bias_in,
        scale_out=core_params.scale_out,
        bias_out=core_params.bias_out,
        learning_rate=core_params.learning_rate,
        update_core_xy=core_params.update_core_xy,
        update_core_x=core_params.update_core_x,
        update_core_y=core_params.update_core_y,
        test_core_xy=test_core_xy,
        test_core_x=test_core_x,
        test_core_y=test_core_y,
        global_send=core_params.global_send,
        global_receive=core_params.global_receive,
        thread_number=core_params.thread_number,
        busy_cycle=core_params.busy_cycle,
        delay_cycle=core_params.delay_cycle,
        width_cycle=core_params.width_cycle,
        tick_start=core_params.tick_start,
        tick_duration=core_params.tick_duration,
        tick_initial=core_params.tick_initial,
    )
