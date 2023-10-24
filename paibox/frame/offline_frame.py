from .base_frame import *
from .params import *
from .params import ParameterRAMFormat as RAMMask
from .params import ParameterRegFormat as RegMask
from .util import *
from paibox.libpaicore.v2 import *


class OfflineConfigFrame1(Frame):
    def __init__(self, chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, random_seed: int):
        header = FrameHead.CONFIG_TYPE1
        super().__init__(header, chip_coord, core_coord, core_e_coord, random_seed)

class OfflineConfigFrame2(FrameGroup):
    def __init__(self, chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, parameter_reg: dict) -> None:
        header = FrameHead.CONFIG_TYPE2

        tick_wait_start_high8, tick_wait_start_low7 = bin_split(parameter_reg["tick_wait_start"], 8, 7)
        test_chip_addr_high3, test_chip_addr_low7 = bin_split(parameter_reg["test_chip_addr"], 3, 7)
        reg_frame1 = int(
            ((parameter_reg["weight_width"] & RegMask.WEIGHT_WIDTH_MASK) << RegMask.WEIGHT_WIDTH_OFFSET)
            | ((parameter_reg["LCN"] & RegMask.LCN_MASK) << RegMask.LCN_OFFSET)
            | ((parameter_reg["input_width"] & RegMask.INPUT_WIDTH_MASK) << RegMask.INPUT_WIDTH_OFFSET)
            | ((parameter_reg["spike_width"] & RegMask.SPIKE_WIDTH_MASK) << RegMask.SPIKE_WIDTH_OFFSET)
            | ((parameter_reg["neuron_num"] & RegMask.NEURON_NUM_MASK) << RegMask.NEURON_NUM_OFFSET)
            | ((parameter_reg["pool_max"] & RegMask.POOL_MAX_MASK) << RegMask.POOL_MAX_OFFSET)
            | ((tick_wait_start_high8 & RegMask.TICK_WAIT_START_HIGH8_MASK) << RegMask.TICK_WAIT_START_HIGH8_OFFSET)
        )

        reg_frame2 = int(
            ((tick_wait_start_low7 & RegMask.TICK_WAIT_START_LOW7_MASK) << RegMask.TICK_WAIT_START_LOW7_OFFSET)
            | ((parameter_reg["tick_wait_end"] & RegMask.TICK_WAIT_END_MASK) << RegMask.TICK_WAIT_END_OFFSET)
            | ((parameter_reg["snn_en"] & RegMask.SNN_EN_MASK) << RegMask.SNN_EN_OFFSET)
            | ((parameter_reg["targetLCN"] & RegMask.TARGET_LCN_MASK) << RegMask.TARGET_LCN_OFFSET)
            | ((test_chip_addr_high3 & RegMask.TEST_CHIP_ADDR_HIGH3_MASK) << RegMask.TEST_CHIP_ADDR_HIGH3_OFFSET)
        )

        reg_frame3 = int((test_chip_addr_low7 & RegMask.TEST_CHIP_ADDR_LOW7_MASK) << RegMask.TEST_CHIP_ADDR_LOW7_OFFSET)

        self.reg_list = [reg_frame1, reg_frame2, reg_frame3]

        super().__init__(header, chip_coord, core_coord, core_e_coord, self.reg_list)

    @property
    def parameter_reg(self):
        return self.reg_list

class OfflineConfigFrame3(FramePackage):
    def __init__(
        self, chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, sram_start_addr: int, data_package_num: int, neuron_ram: dict
    ):
        header = FrameHead.CONFIG_TYPE3
        neuron_ram_load = (
            ((sram_start_addr & ConfigFrame3Format.DATA_PACKAGE_SRAM_NEURON_MASK) << ConfigFrame3Format.DATA_PACKAGE_SRAM_NEURON_OFFSET)
            | ((0b0 & ConfigFrame3Format.DATA_PACKAGE_TYPE_MASK) << ConfigFrame3Format.DATA_PACKAGE_TYPE_OFFSET)
            | ((data_package_num & ConfigFrame3Format.DATA_PACKAGE_NUM_MASK) << ConfigFrame3Format.DATA_PACKAGE_NUM_OFFSET)
        )

        leak_v_high2, leak_v_low28 = bin_split(neuron_ram["leak_v"], 2, 28)
        threshold_mask_ctrl_high4, threshold_mask_ctrl_low1 = bin_split(neuron_ram["threshold_mask_ctrl"], 4, 1)
        addr_core_x_high3, addr_core_x_low2 = bin_split(neuron_ram["addr_core_x"], 3, 2)
        # 1
        ram_frame1 = int(
            ((neuron_ram["vjt_pre"] & RAMMask.VJT_PRE_MASK) << RAMMask.VJT_PRE_OFFSET)
            | ((neuron_ram["bit_truncate"] & RAMMask.BIT_TRUNCATE_MASK) << RAMMask.BIT_TRUNCATE_OFFSET)
            | ((neuron_ram["weight_det_stoch"] & RAMMask.WEIGHT_DET_STOCH_MASK) << RAMMask.WEIGHT_DET_STOCH_OFFSET)
            | ((leak_v_low28 & RAMMask.LEAK_V_LOW28_MASK) << RAMMask.LEAK_V_LOW28_OFFSET)
        )

        # 2
        ram_frame2 = int(
            ((leak_v_high2 & RAMMask.LEAK_V_HIGH2_MASK) << RAMMask.LEAK_V_HIGH2_OFFSET)
            | ((neuron_ram["leak_det_stoch"] & RAMMask.LEAK_DET_STOCH_MASK) << RAMMask.LEAK_DET_STOCH_OFFSET)
            | ((neuron_ram["leak_reversal_flag"] & RAMMask.LEAK_REVERSAL_FLAG_MASK) << RAMMask.LEAK_REVERSAL_FLAG_OFFSET)
            | ((neuron_ram["threshold_pos"] & RAMMask.THRESHOLD_POS_MASK) << RAMMask.THRESHOLD_POS_OFFSET)
            | ((neuron_ram["threshold_neg"] & RAMMask.THRESHOLD_NEG_MASK) << RAMMask.THRESHOLD_NEG_OFFSET)
            | ((neuron_ram["threshold_neg_mode"] & RAMMask.THRESHOLD_NEG_MODE_MASK) << RAMMask.THRESHOLD_NEG_MODE_OFFSET)
            | ((threshold_mask_ctrl_low1 & RAMMask.THRESHOLD_MASK_CTRL_LOW1_MASK) << RAMMask.THRESHOLD_MASK_CTRL_LOW1_OFFSET)
        )

        # 3
        ram_frame3 = int(
            ((threshold_mask_ctrl_high4 & RAMMask.THRESHOLD_MASK_CTRL_HIGH4_MASK) << RAMMask.THRESHOLD_MASK_CTRL_HIGH4_OFFSET)
            | ((neuron_ram["leak_post"] & RAMMask.LEAK_POST_MASK) << RAMMask.LEAK_POST_OFFSET)
            | ((neuron_ram["reset_v"] & RAMMask.RESET_V_MASK) << RAMMask.RESET_V_OFFSET)
            | ((neuron_ram["reset_mode"] & RAMMask.RESET_MODE_MASK) << RAMMask.RESET_MODE_OFFSET)
            | ((neuron_ram["addr_chip_y"] & RAMMask.ADDR_CHIP_Y_MASK) << RAMMask.ADDR_CHIP_Y_OFFSET)
            | ((neuron_ram["addr_chip_x"] & RAMMask.ADDR_CHIP_X_MASK) << RAMMask.ADDR_CHIP_X_OFFSET)
            | ((neuron_ram["addr_core_y_ex"] & RAMMask.ADDR_CORE_Y_EX_MASK) << RAMMask.ADDR_CORE_Y_EX_OFFSET)
            | ((neuron_ram["addr_core_x_ex"] & RAMMask.ADDR_CORE_X_EX_MASK) << RAMMask.ADDR_CORE_X_EX_OFFSET)
            | ((neuron_ram["addr_core_y"] & RAMMask.ADDR_CORE_Y_MASK) << RAMMask.ADDR_CORE_Y_OFFSET)
            | ((addr_core_x_low2 & RAMMask.ADDR_CORE_X_LOW2_MASK) << RAMMask.ADDR_CORE_X_LOW2_OFFSET)
        )

        # 4
        ram_frame4 = int(
            ((addr_core_x_high3 & RAMMask.ADDR_CORE_X_HIGH3_MASK) << RAMMask.ADDR_CORE_X_HIGH3_OFFSET)
            | ((neuron_ram["addr_axon"] & RAMMask.ADDR_AXON_MASK) << RAMMask.ADDR_AXON_OFFSET)
            | ((neuron_ram["tick_relative"] & RAMMask.TICK_RELATIVE_MASK) << RAMMask.TICK_RELATIVE_OFFSET)
        )

        super().__init__(header, chip_coord, core_coord, core_e_coord, neuron_ram_load, [ram_frame1, ram_frame2, ram_frame3, ram_frame4])

class OfflineConfigFrame4(FramePackage):
    def __init__(
        self, chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, sram_start_addr: int, data_package_num: int, weight_ram: List
    ):
        header = FrameHead.CONFIG_TYPE4
        weight_ram_load = (
            ((sram_start_addr & ConfigFrame4Format.DATA_PACKAGE_SRAM_NEURON_MASK) << ConfigFrame4Format.DATA_PACKAGE_SRAM_NEURON_OFFSET)
            | ((0b0 & ConfigFrame4Format.DATA_PACKAGE_TYPE_MASK) << ConfigFrame4Format.DATA_PACKAGE_TYPE_OFFSET)
            | ((data_package_num & ConfigFrame4Format.DATA_PACKAGE_NUM_MASK) << ConfigFrame4Format.DATA_PACKAGE_NUM_OFFSET)
        )

        super().__init__(header, chip_coord, core_coord, core_e_coord, weight_ram_load, weight_ram)


# TODO:test frame


class OfflineWorkFrame1(Frame):
    def __init__(self, chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, axon: int, time_slot: int, data: int):
        header = FrameHead.WORK_TYPE1
        self.axon = axon
        self.time_slot = time_slot
        self.data = data
        payload =(((0x00 & WorkFrame1Format.RESERVED_MASK) << WorkFrame1Format.RESERVED_OFFSET)
            | ((axon & WorkFrame1Format.AXON_MASK) << WorkFrame1Format.AXON_OFFSET)
            | ((time_slot & WorkFrame1Format.TIME_SLOT_MASK) << WorkFrame1Format.TIME_SLOT_OFFSET)
            | ((data & WorkFrame1Format.DATA_MASK) << WorkFrame1Format.DATA_OFFSET))
        
        super().__init__(header, chip_coord, core_coord, core_e_coord, payload)
        

class OfflineWorkFrame1Muti:
    def __init__(
        self,
        chip_coord_list : List[Coord],
        core_coord_list : List[Coord],
        core_e_coord_list : List[ReplicationId],
        axon_list: List[int],
        time_slot_list: List[int],
        *args
    ):
        self.chip_coord_list = chip_coord_list
        self.core_coord_list = core_coord_list
        self.core_e_coord_list = core_e_coord_list
        self.axon_list = axon_list
        self.time_slot_list = time_slot_list
        
        chip_address_array = np.array([coord.address for coord in chip_coord_list])
        core_address_array = np.array([coord.address for coord in core_coord_list])
        core_e_address_array = np.array([coord.address for coord in core_e_coord_list])
        
        header_array = np.tile(FrameHead.WORK_TYPE1.value,len(chip_address_array))
        axon_array = np.array(axon_list)
        time_slot_array = np.array(time_slot_list)
        
        temp_header_array = header_array & np.uint64(FrameFormat.GENERAL_HEADER_MASK)
        temp_chip_coord_array = chip_address_array & np.uint64(FrameFormat.GENERAL_CHIP_ADDR_MASK)
        temp_core_address_array = core_address_array & np.uint64(FrameFormat.GENERAL_CORE_ADDR_MASK)
        temp_core_e_address_array = core_e_address_array & np.uint64(FrameFormat.GENERAL_CORE_E_ADDR_MASK)
        temp_axon_array = axon_array & np.uint64(WorkFrame1Format.AXON_MASK)
        temp_time_slot_array = time_slot_array & np.uint64(WorkFrame1Format.TIME_SLOT_MASK)
        
        self.frameinfo = (
            (temp_header_array << np.uint64(FrameFormat.GENERAL_HEADER_OFFSET))
            | (temp_chip_coord_array <<  np.uint64(FrameFormat.GENERAL_CHIP_ADDR_OFFSET))
            | (temp_core_address_array << np.uint64(FrameFormat.GENERAL_CORE_ADDR_OFFSET))
            | (temp_core_e_address_array << np.uint64(FrameFormat.GENERAL_CORE_E_ADDR_OFFSET))
            | (temp_axon_array << np.uint64(WorkFrame1Format.AXON_OFFSET))
            | (temp_time_slot_array << np.uint64(WorkFrame1Format.TIME_SLOT_OFFSET))
        )
        
        if args:
            self.add_data_remove_zero(args[0])

    @classmethod
    def init_frameinfo(
        cls,
        chip_coord_list : List[Coord],
        core_coord_list : List[Coord],
        core_e_coord_list : List[ReplicationId],
        axon_list: List[int],
        time_slot_list: List[int]
    ):
        return cls(chip_coord_list,core_coord_list,core_e_coord_list,axon_list,time_slot_list)
    
    def add_data_remove_zero(
        self,
        origin_data : np.ndarray
    ):
        self.origin_data = origin_data
        data = origin_data.reshape(-1)
        indexes = np.nonzero(data)
        spike_frame_info = self.frameinfo[indexes]
        data = data[indexes]
        self.data = data
        self.frames = spike_frame_info << np.uint64(8) | data
        
    def __repr__(self) -> str:
        return (
            f"Frame info:\n"
            f"Head:             {FrameHead.WORK_TYPE1}\n"
            f"Chip address:     {self.chip_coord_list}\n"
            f"Core address:     {self.core_coord_list}\n"
            f"Core_E address:   {self.core_e_coord_list}\n"
            f"axon:             {self.axon_list}\n"
            f"time_slot:        {self.time_slot_list}\n"
            f"origin data:      {self.origin_data}\n"
        )


class OfflineWorkFrame2(Frame):
    def __init__(self, chip_coord: Coord, time: int):
        header = FrameHead.WORK_TYPE2
        core_coord = Coord(0, 0)
        core_e_coord = ReplicationId(0, 0)

        super().__init__(header, chip_coord, core_coord, core_e_coord, time)
        

class OfflineWorkFrame3(Frame):
    def __init__(self, chip_coord: Coord):
        header = FrameHead.WORK_TYPE3
        core_coord = Coord(0, 0)
        core_e_coord = ReplicationId(0, 0)
        payload = 0
        super().__init__(header, chip_coord, core_coord, core_e_coord, payload)


class OfflineWorkFrame4(Frame):
    def __init__(self, chip_coord: Coord):
        header = FrameHead.WORK_TYPE4
        core_coord = Coord(0, 0)
        core_e_coord = ReplicationId(0, 0)
        payload = 0
        super().__init__(header, chip_coord, core_coord, core_e_coord, payload)
