from core.core_defs import ParamsReg
from core.params_types import *
from frame.frame_gen import *
from frame.frame_params import *

model = ParamsReg(
    weight_precision=WeightPrecisionType.WEIGHT_WIDTH_4BIT,
    LCN_extension=LCNExtensionType.LCN_16X,
    input_width_format=InputWidthFormatType.INPUT_WIDTH_1BIT,
    spike_width_format=SpikeWidthFormatType.SPIKE_WIDTH_8BIT,
    neuron_num=511,
    pool_max_en=PoolMaxEnableType.POOL_MAX_DISABLE,
    tick_wait_start=0b100000000000011,
    tick_wait_end=0,
    snn_mode_en=SNNModeEnableType.SNN_MODE_DISABLE,
    target_LCN=1,
    test_chip_addr=0,
)
parameter_reg = model.model_dump(by_alias=True)

x = FrameGen.GenConfigFrame(header=FrameHead.CONFIG_TYPE2, chip_coord=Coord(1, 2), core_coord=Coord(3, 4),
                            core_ex_coord=Coord(5, 6), parameter_reg=parameter_reg)
for xx in x:
    print(bin(xx))
# weight_width = parameter_reg["weight_width"]
# LCN = parameter_reg["LCN"]
#
# ConfigFrameGroup = []
#
# print(bin(weight_width))
# tick_wait_start_high8, tick_wait_start_low7 = config_frame2_split(parameter_reg["tick_wait_start"], 7)
# test_chip_addr_high3, test_chip_addr_low7 = config_frame2_split(parameter_reg["test_chip_addr"], 7)
# reg_frame1 = parameter_reg["weight_width"] << ConfigFrame2Mask.WEIGHT_WIDTH_OFFSET
# reg_frame1 = parameter_reg["LCN"] << ConfigFrame2Mask.LCN_OFFSET | reg_frame1
# reg_frame1 = parameter_reg["input_width"] << ConfigFrame2Mask.INPUT_WIDTH_OFFSET | reg_frame1
# reg_frame1 = parameter_reg["spike_width"] << ConfigFrame2Mask.SPIKE_WIDTH_OFFSET | reg_frame1
# reg_frame1 = parameter_reg["neuron_num"] << ConfigFrame2Mask.NEURON_NUM_OFFSET | reg_frame1
# reg_frame1 = parameter_reg["pool_max"] << ConfigFrame2Mask.POOL_MAX_OFFSET | reg_frame1
# reg_frame1 = tick_wait_start_high8 << ConfigFrame2Mask.TICK_WAIT_START_HIGH8_OFFSET | reg_frame1
#
