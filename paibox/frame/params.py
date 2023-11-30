from enum import Enum, unique

import numpy as np


# 帧类型标志位
@unique
class FrameType(Enum):
    """Types of Frames"""

    FRAME_CONFIG = 0
    FRAME_TEST = 0x1
    FRAME_WORK = 0x2
    FRAME_UNKNOWN = 0x3


# 帧头
@unique
class FrameHead(Enum):
    "配置帧帧头"
    CONFIG_TYPE1 = np.uint64(0b0000)  # 配置帧 1型（RANDOM_SEED）
    CONFIG_TYPE2 = np.uint64(0b0001)  # 配置帧 2型（PARAMETER_REG）
    CONFIG_TYPE3 = np.uint64(0b0010)  # 配置帧 3型（Neuron RAM）
    CONFIG_TYPE4 = np.uint64(0b0011)  # 配置帧 4型（Weight RAM）

    "测试帧帧头"
    TEST_TYPE1 = np.uint64(0b0100)  # 测试输入帧 1型（RANDOM_SEED_REG）
    TEST_TYPE2 = np.uint64(0b0101)  # 测试输入帧 2型（PARAMETER_REG）
    TEST_TYPE3 = np.uint64(0b0110)  # 测试输入帧 3型（Neuron RAM）
    TEST_TYPE4 = np.uint64(0b0111)  # 测试输入帧 4型（Weight RAM）

    "工作帧帧头"
    WORK_TYPE1 = np.uint64(0b1000)  # 工作帧 1型（Spike，脉冲帧）
    WORK_TYPE2 = np.uint64(0b1001)  # 工作帧 2型（同步帧）
    WORK_TYPE3 = np.uint64(0b1010)  # 工作帧 3型（清除帧）
    WORK_TYPE4 = np.uint64(0b1011)  # 工作帧 4型（初始化帧）


class FrameFormat:
    """FrameMask 通用数据帧掩码 和 数据包格式起始帧掩码"""

    GENERAL_MASK = np.uint64((1 << 64) - 1)
    "通用"
    # Header
    GENERAL_HEADER_OFFSET = np.uint64(60)
    GENERAL_HEADER_MASK = np.uint64((1 << 4) - 1)

    GENERAL_FRAME_TYPE_OFFSET = GENERAL_HEADER_OFFSET
    GENERAL_FRAME_TYPE_MASK = GENERAL_HEADER_MASK

    # Chip address
    GENERAL_CHIP_ADDR_OFFSET = np.uint64(50)
    GENERAL_CHIP_ADDR_MASK = np.uint64((1 << 10) - 1)
    # Chip X/Y address
    GENERAL_CHIP_X_ADDR_OFFSET = np.uint64(55)
    GENERAL_CHIP_X_ADDR_MASK = np.uint64((1 << 5) - 1)
    GENERAL_CHIP_Y_ADDR_OFFSET = GENERAL_CHIP_ADDR_OFFSET
    GENERAL_CHIP_Y_ADDR_MASK = np.uint64((1 << 5) - 1)

    # Core address
    GENERAL_CORE_ADDR_OFFSET = np.uint64(40)
    GENERAL_CORE_ADDR_MASK = np.uint64((1 << 10) - 1)
    # Core X/Y address
    GENERAL_CORE_X_ADDR_OFFSET = np.uint64(45)
    GENERAL_CORE_X_ADDR_MASK = np.uint64((1 << 5) - 1)
    GENERAL_CORE_Y_ADDR_OFFSET = GENERAL_CORE_ADDR_OFFSET
    GENERAL_CORE_Y_ADDR_MASK = np.uint64((1 << 5) - 1)

    # Core* address
    GENERAL_CORE_EX_ADDR_OFFSET = np.uint64(30)
    GENERAL_CORE_EX_ADDR_MASK = np.uint64((1 << 10) - 1)
    # Core* X/Y address
    GENERAL_CORE_X_EX_ADDR_OFFSET = np.uint64(35)
    GENERAL_CORE_X_EX_ADDR_MASK = np.uint64((1 << 5) - 1)
    GENERAL_CORE_Y_EX_ADDR_OFFSET = GENERAL_CORE_EX_ADDR_OFFSET
    GENERAL_CORE_Y_EX_ADDR_MASK = np.uint64((1 << 5) - 1)

    # Global core = Chip address + core address
    GENERAL_CORE_GLOBAL_ADDR_OFFSET = GENERAL_CORE_ADDR_OFFSET
    GENERAL_CORE_GLOBAL_ADDR_MASK = np.uint64((1 << 20) - 1)

    # Frame 前34位
    GENERAL_FRAME_PRE_OFFSET = np.uint64(30)
    GENERAL_FRAME_PRE_MASK = np.uint64((1 << 34) - 1)

    # 通用数据帧LOAD掩码
    GENERAL_PAYLOAD_OFFSET = np.uint64(0)
    GENERAL_PAYLOAD_MASK = np.uint64((1 << 30) - 1)

    # 通用数据包LOAD掩码
    DATA_PACKAGE_SRAM_NEURON_OFFSET = np.uint64(20)
    DATA_PACKAGE_SRAM_NEURON_MASK = np.uint64((1 << 10) - 1)

    DATA_PACKAGE_TYPE_OFFSET = np.uint64(19)
    DATA_PACKAGE_TYPE_MASK = np.uint64(1)

    DATA_PACKAGE_NUM_OFFSET = np.uint64(0)
    DATA_PACKAGE_NUM_MASK = np.uint64((1 << 19) - 1)


"""配置帧"""


class ConfigFrame1Format(FrameFormat):
    pass


class ConfigFrame2Format(FrameFormat):
    pass


class ConfigFrame3Format(FrameFormat):
    pass


class ConfigFrame4Format(FrameFormat):
    pass


class ParameterRegFormat(FrameFormat):
    """配置帧2型"""

    """Frame #1"""

    WEIGHT_WIDTH_OFFSET = np.uint64(28)
    WEIGHT_WIDTH_MASK = np.uint64((1 << 2) - 1)

    LCN_OFFSET = np.uint64(24)
    LCN_MASK = np.uint64((1 << 4) - 1)

    INPUT_WIDTH_OFFSET = np.uint64(23)
    INPUT_WIDTH_MASK = np.uint64(1)

    SPIKE_WIDTH_OFFSET = np.uint64(22)
    SPIKE_WIDTH_MASK = np.uint64(1)

    NEURON_NUM_OFFSET = np.uint64(9)
    NEURON_NUM_MASK = np.uint64((1 << 13) - 1)

    POOL_MAX_OFFSET = np.uint64(8)
    POOL_MAX_MASK = np.uint64(1)

    TICK_WAIT_START_HIGH8_OFFSET = np.uint64(0)
    TICK_WAIT_START_COMBINATION_OFFSET = np.uint64(7)
    TICK_WAIT_START_HIGH8_MASK = np.uint64((1 << 8) - 1)

    """Frame #2"""
    TICK_WAIT_START_LOW7_OFFSET = np.uint64(23)
    TICK_WAIT_START_LOW7_MASK = np.uint64((1 << 7) - 1)

    TICK_WAIT_END_OFFSET = np.uint64(8)
    TICK_WAIT_END_MASK = np.uint64((1 << 15) - 1)

    SNN_EN_OFFSET = np.uint64(7)
    SNN_EN_MASK = np.uint64(1)

    TARGET_LCN_OFFSET = np.uint64(3)
    TARGET_LCN_MASK = np.uint64((1 << 4) - 1)

    # 用于配置帧2型test_chip_addr
    TEST_CHIP_ADDR_HIGH3_OFFSET = np.uint64(0)
    TEST_CHIP_ADDR_COMBINATION_OFFSET = np.uint64(7)
    TEST_CHIP_ADDR_HIGH3_MASK = np.uint64((1 << 3) - 1)

    """Frame #3"""
    TEST_CHIP_ADDR_LOW7_OFFSET = np.uint64(23)
    TEST_CHIP_ADDR_LOW7_MASK = np.uint64((1 << 7) - 1)


class ParameterRAMFormat(FrameFormat):
    """配置帧3型（Neuron RAM）"""

    # 1
    VJT_PRE_OFFSET = np.uint64(0)
    VJT_PRE_MASK = np.uint64((1 << 30) - 1)

    BIT_TRUNCATE_OFFSET = np.uint64(30)
    BIT_TRUNCATE_MASK = np.uint64((1 << 5) - 1)

    WEIGHT_DET_STOCH_OFFSET = np.uint64(35)
    WEIGHT_DET_STOCH_MASK = np.uint64((1 << 1) - 1)

    LEAK_V_LOW28_OFFSET = np.uint64(36)
    LEAK_V_LOW28_MASK = np.uint64((1 << 28) - 1)

    # 2
    LEAK_V_HIGH2_OFFSET = np.uint64(0)
    LEAK_V_HIGH2_MASK = np.uint64((1 << 2) - 1)

    LEAK_DET_STOCH_OFFSET = np.uint64(2)
    LEAK_DET_STOCH_MASK = np.uint64((1 << 1) - 1)

    LEAK_REVERSAL_FLAG_OFFSET = np.uint64(3)
    LEAK_REVERSAL_FLAG_MASK = np.uint64((1 << 1) - 1)

    THRESHOLD_POS_OFFSET = np.uint64(4)
    THRESHOLD_POS_MASK = np.uint64((1 << 29) - 1)

    THRESHOLD_NEG_OFFSET = np.uint64(33)
    THRESHOLD_NEG_MASK = np.uint64((1 << 29) - 1)

    THRESHOLD_NEG_MODE_OFFSET = np.uint64(62)
    THRESHOLD_NEG_MODE_MASK = np.uint64((1 << 1) - 1)

    THRESHOLD_MASK_CTRL_LOW1_OFFSET = np.uint64(63)
    THRESHOLD_MASK_CTRL_LOW1_MASK = np.uint64((1 << 1) - 1)

    # 3
    THRESHOLD_MASK_CTRL_HIGH4_OFFSET = np.uint64(0)
    THRESHOLD_MASK_CTRL_HIGH4_MASK = np.uint64((1 << 4) - 1)

    LEAK_POST_OFFSET = np.uint64(4)
    LEAK_POST_MASK = np.uint64((1 << 1) - 1)

    RESET_V_OFFSET = np.uint64(5)
    RESET_V_MASK = np.uint64((1 << 30) - 1)

    RESET_MODE_OFFSET = np.uint64(35)
    RESET_MODE_MASK = np.uint64((1 << 2) - 1)

    ADDR_CHIP_Y_OFFSET = np.uint64(37)
    ADDR_CHIP_Y_MASK = np.uint64((1 << 5) - 1)

    ADDR_CHIP_X_OFFSET = np.uint64(42)
    ADDR_CHIP_X_MASK = np.uint64((1 << 5) - 1)

    ADDR_CORE_Y_EX_OFFSET = np.uint64(47)
    ADDR_CORE_Y_EX_MASK = np.uint64((1 << 5) - 1)

    ADDR_CORE_X_EX_OFFSET = np.uint64(52)
    ADDR_CORE_X_EX_MASK = np.uint64((1 << 5) - 1)

    ADDR_CORE_Y_OFFSET = np.uint64(57)
    ADDR_CORE_Y_MASK = np.uint64((1 << 5) - 1)

    ADDR_CORE_X_LOW2_OFFSET = np.uint64(62)
    ADDR_CORE_X_LOW2_MASK = np.uint64((1 << 2) - 1)

    # 4
    ADDR_CORE_X_HIGH3_OFFSET = np.uint64(0)
    ADDR_CORE_X_HIGH3_MASK = np.uint64((1 << 3) - 1)

    ADDR_AXON_OFFSET = np.uint64(3)
    ADDR_AXON_MASK = np.uint64((1 << 11) - 1)

    TICK_RELATIVE_OFFSET = np.uint64(14)
    TICK_RELATIVE_MASK = np.uint64((1 << 8) - 1)


# TODO: 测试帧


"""工作帧"""


class WorkFrame1Format(FrameFormat):
    "工作帧 1 型（Spike，脉冲帧）"
    RESERVED_OFFSET = np.uint64(27)
    RESERVED_MASK = np.uint64((1 << 3) - 1)

    AXON_OFFSET = np.uint64(16)
    AXON_MASK = np.uint64((1 << 11) - 1)

    TIME_SLOT_OFFSET = np.uint64(8)
    TIME_SLOT_MASK = np.uint64((1 << 8) - 1)

    DATA_OFFSET = np.uint64(0)
    DATA_MASK = np.uint64((1 << 8) - 1)


class WorkFrame2Format(FrameFormat):
    "工作帧 2 型（同步帧）"
    RESERVED_OFFSET = np.uint64(30)
    RESERVED_MASK = np.uint64((1 << 20) - 1)

    TIME_OFFSET = np.uint64(0)
    TIME_MASK = np.uint64((1 << 30) - 1)


class WorkFrame3Format(FrameFormat):
    "工作帧 3 型（清除帧）"
    RESERVED_OFFSET = np.uint64(0)
    RESERVED_MASK = np.uint64((1 << 50) - 1)


class WorkFrame4Format(FrameFormat):
    "工作帧4 型（初始化帧）"
    RESERVED_OFFSET = np.uint64(0)
    RESERVED_MASK = np.uint64((1 << 50) - 1)


# TODO: 在线学习处理核数据帧格式
