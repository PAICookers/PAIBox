from enum import Enum, unique


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
    CONFIG_TYPE1 = 0b0000  # 配置帧 1型（RANDOM_SEED）
    CONFIG_TYPE2 = 0b0001  # 配置帧 2型（PARAMETER_REG）
    CONFIG_TYPE3 = 0b0010  # 配置帧 3型（Neuron RAM）
    CONFIG_TYPE4 = 0b0011  # 配置帧 4型（Weight RAM）

    "测试帧帧头"
    TEST_TYPE1 = 0b0100  # 测试输入帧 1型（RANDOM_SEED_REG）
    TEST_TYPE2 = 0b0101  # 测试输入帧 2型（PARAMETER_REG）
    TEST_TYPE3 = 0b0110  # 测试输入帧 3型（Neuron RAM）
    TEST_TYPE4 = 0b0111  # 测试输入帧 4型（Weight RAM）

    "工作帧帧头"
    WORK_TYPE1 = 0b1000  # 工作帧 1型（Spike，脉冲帧）
    WORK_TYPE2 = 0b1001  # 工作帧 2型（同步帧）
    WORK_TYPE3 = 0b1010  # 工作帧 3型（清除帧）
    WORK_TYPE4 = 0b1011  # 工作帧 4型（初始化帧）


class FrameMask:
    """FrameMask 通用数据帧掩码 和 数据包格式起始帧掩码"""

    GENERAL_MASK = (1 << 64) - 1
    "通用"
    # Header
    GENERAL_HEADER_OFFSET = 60
    GENERAL_HEADER_MASK = (1 << 4) - 1

    GENERAL_FRAME_TYPE_OFFSET = GENERAL_HEADER_OFFSET
    GENERAL_FRAME_TYPE_MASK = GENERAL_HEADER_MASK

    # Chip address
    GENERAL_CHIP_ADDR_OFFSET = 50
    GENERAL_CHIP_ADDR_MASK = (1 << 10) - 1
    # Chip X/Y address
    GENERAL_CHIP_ADDR_X_OFFSET = 55
    GENERAL_CHIP_ADDR_X_MASK = (1 << 5) - 1
    GENERAL_CHIP_ADDR_Y_OFFSET = GENERAL_CHIP_ADDR_OFFSET
    GENERAL_CHIP_ADDR_Y_MASK = (1 << 5) - 1

    # Core address
    GENERAL_CORE_ADDR_OFFSET = 40
    GENERAL_CORE_ADDR_MASK = (1 << 10) - 1
    # Core X/Y address
    GENERAL_CORE_ADDR_X_OFFSET = 45
    GENERAL_CORE_ADDR_X_MASK = (1 << 5) - 1
    GENERAL_CORE_ADDR_Y_OFFSET = GENERAL_CORE_ADDR_OFFSET
    GENERAL_CORE_ADDR_Y_MASK = (1 << 5) - 1

    # Core* address
    GENERAL_CORE_EX_ADDR_OFFSET = 30
    GENERAL_CORE_EX_ADDR_MASK = (1 << 10) - 1
    # Core* X/Y address
    GENERAL_CORE_EX_ADDR_X_OFFSET = 35
    GENERAL_CORE_EX_ADDR_X_MASK = (1 << 5) - 1
    GENERAL_CORE_EX_ADDR_Y_OFFSET = GENERAL_CORE_EX_ADDR_OFFSET
    GENERAL_CORE_EX_ADDR_Y_MASK = (1 << 5) - 1

    # Global core = Chip address + core address
    GENERAL_CORE_GLOBAL_ADDR_OFFSET = GENERAL_CORE_ADDR_OFFSET
    GENERAL_CORE_GLOBAL_ADDR_MASK = (1 << 20) - 1

    """通用数据帧LOAD掩码"""
    GENERAL_PAYLOAD_OFFSET = 0
    GENERAL_PAYLOAD_MASK = (1 << 30) - 1
    GENERAL_PAYLOAD_FILLED_MASK = (1 << 4) - 1

    """数据包格式起始帧LOAD掩码"""
    DATA_PACKAGE_OFFSET = 0
    DATA_PACKAGE_MASK = (1 << 30) - 1

    DATA_PACKAGE_SRAM_NEURON_ADDR_OFFSET = 20
    DATA_PACKAGE_SRAM_NEURON_ADDR_MASK = (1 << 10) - 1

    DATA_PACKAGE_TYPE_OFFSET = 19
    DATA_PACKAGE_TYPE_MASK = 0x1

    DATA_PACKAGE_COUNT_OFFSET = DATA_PACKAGE_OFFSET
    DATA_PACKAGE_COUNT_MASK = (1 << 19) - 1


"""配置帧"""
# 配置帧 1型
ConfigFrame1Mask = FrameMask


# 配置帧 2型
class ConfigFrame2Mask(FrameMask):
    """Frame #1"""

    WEIGHT_WIDTH_OFFSET = 28
    WEIGHT_WIDTH_MASK = (1 << 2) - 1

    LCN_OFFSET = 24
    LCN_MASK = (1 << 4) - 1

    INPUT_WIDTH_OFFSET = 23
    INPUT_WIDTH_MASK = 1

    SPIKE_WIDTH_OFFSET = 22
    SPIKE_WIDTH_MASK = 1

    NEURON_NUM_OFFSET = 9
    NEURON_NUM_MASK = (1 << 13) - 1

    POOL_MAX_OFFSET = 8
    POOL_MAX_MASK = 1

    TICK_WAIT_START_HIGH8_OFFSET = 0
    TICK_WAIT_START_COMBINATION_OFFSET = 7
    TICK_WAIT_START_HIGH8_MASK = (1 << 8) - 1

    """Frame #2"""
    TICK_WAIT_START_LOW7_OFFSET = 23
    TICK_WAIT_START_LOW7_MASK = (1 << 7) - 1

    TICK_WAIT_END_OFFSET = 8
    TICK_WAIT_END_MASK = (1 << 15) - 1

    SNN_EN_OFFSET = 7
    SNN_EN_MASK = 1

    TARGET_LCN_OFFSET = 3
    TARGET_LCN_MASK = (1 << 4) - 1
    
    #用于配置帧2 型test_chip_addr
    TEST_CHIP_ADDR_HIGH3_OFFSET = 0
    TEST_CHIP_ADDR_COMBINATION_OFFSET = 7
    TEST_CHIP_ADDR_HIGH3_MASK = (1 << 3) - 1

    """Frame #3"""
    TEST_CHIP_ADDR_LOW7_OFFSET = 23
    TEST_CHIP_ADDR_LOW7_MASK = (1 << 7) - 1


# 配置帧3 型（Neuron RAM） 数据包起始帧
ConfigFrame3StartMask = FrameMask

# 配置帧4 型（Weight RAM） 数据包起始帧
ConfigFrame4StartMask = FrameMask

"""
测试帧使用 FrameMask
"""

"""工作帧"""


# 工作帧1 型（Spike，脉冲帧）
class WorkFrame1Mask(FrameMask):
    RESERVED_OFFSET = 27
    RESERVED_MASK = (1 << 3) - 1

    AXON_OFFSET = 16
    AXON_MASK = (1 << 11) - 1

    TIME_SLOT_OFFSET = 8
    TIME_SLOT_MASK = (1 << 8) - 1

    DATA_OFFSET = 0
    DATA_MASK = (1 << 8) - 1


# 工作帧2 型（同步帧）
WorkFrame2Mask = FrameMask

# 工作帧3 型（清除帧）
WorkFrame3Mask = FrameMask

# 工作帧4 型（初始化帧）
WorkFrame4Mask = FrameMask

# TODO: 在线学习处理核数据帧格式
