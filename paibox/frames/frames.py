from enum import unique

@unique
class FrameHead:
    
    "配置帧帧头"
    CONFIG_TYPE1 = 0b0000 # 配置帧 1型
    CONFIG_TYPE2 = 0b0001 # 配置帧 2型
    CONFIG_TYPE3 = 0b0010 # 配置帧 3型
    CONFIG_TYPE4 = 0b0011 # 配置帧 4型
    
    "测试帧帧头"
    TEST_TYPE1 = 0b0100 # 测试输入帧 1型
    TEST_TYPE2 = 0b0101 # 测试输入帧 2型
    TEST_TYPE3 = 0b0110 # 测试输入帧 3型
    TEST_TYPE4 = 0b0111 # 测试输入帧 4型
    
    "工作帧帧头"
    WORK_TYPE1 = 0b1000 # 工作帧 1型
    WORK_TYPE2 = 0b1001 # 工作帧 2型
    WORK_TYPE3 = 0b1010 # 工作帧 3型
    WORK_TYPE4 = 0b1011 # 工作帧 4型
    
    