from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Union, overload

import numpy as np

from paibox.backend.config_template import CorePlacementConfig, NeuronConfig
from paibox.libpaicore import LCN_EX, Coord, ParamsReg
from paibox.libpaicore import ReplicationId as RId
from paibox.libpaicore import WeightPrecision as WP

from .libframe._types import *
from .libframe.frames import *
from .libframe.utils import *

__all__ = ["OfflineFrameGen"]


class OfflineFrameGen:
    """Offline frame generator."""

    @staticmethod
    def gen_config_frame1(
        chip_coord: Coord, core_coord: Coord, rid: RId, /, random_seed: DataType
    ) -> OfflineConfigFrame1:
        return OfflineConfigFrame1(chip_coord, core_coord, rid, random_seed)

    @overload
    @staticmethod
    def gen_config_frame2(
        chip_coord: Coord, core_coord: Coord, rid: RId, /, params_reg: ParamsReg
    ) -> OfflineConfigFrame2:
        ...

    @overload
    @staticmethod
    def gen_config_frame2(
        chip_coord: Coord, core_coord: Coord, rid: RId, /, params_reg: Dict[str, Any]
    ) -> OfflineConfigFrame2:
        ...

    @staticmethod
    def gen_config_frame2(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        params_reg: Union[ParamsReg, Dict[str, Any]],
    ) -> OfflineConfigFrame2:
        if isinstance(params_reg, ParamsReg):
            pr = params_reg.model_dump(by_alias=True)
        else:
            pr = params_reg

        return OfflineConfigFrame2(chip_coord, core_coord, rid, pr)

    @staticmethod
    def gen_config_frame3(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        lcn_ex: LCN_EX,
        weight_precision: WP,
        neuron_configs: List[NeuronConfig],
    ) -> List[OfflineConfigFrame3]:
        frames = []

        # Iterate every neuron group
        for neuron_config in neuron_configs:
            attrs = neuron_config.neuron_attrs.model_dump(by_alias=True)
            dest_info = neuron_config.neuron_dest_info.model_dump(by_alias=True)

            frames.append(
                OfflineConfigFrame3(
                    chip_coord,
                    core_coord,
                    rid,
                    neuron_config.addr_offset,
                    neuron_config.n_neuron,
                    attrs,
                    dest_info,
                    neuron_config.tick_relative,
                    neuron_config.addr_axon,
                    repeat=(1 << lcn_ex) * (1 << weight_precision),
                )
            )

        return frames

    @staticmethod
    def gen_config_frame4(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        data_package_num: int,
        weight_ram: FrameArrayType,
    ) -> OfflineConfigFrame4:
        return OfflineConfigFrame4(
            chip_coord,
            core_coord,
            rid,
            sram_start_addr,
            data_package_num,
            weight_ram,
        )

    @staticmethod
    def gen_config_frames(
        target_chip_coord: Coord,
        config_dict: Dict[Coord, CorePlacementConfig],
        write_to_file: bool,
        fp: Path,
        format: Literal["txt", "bin", "npy"],
    ) -> Dict[Coord, FrameArrayType]:
        """Generate all configuration frames.

        Args:
            - target_chip_coord: the local chip to configurate.
            - config_dict: a dictionary of configurations.
        """
        _debug_dict: Dict[Coord, Dict[str, Any]] = defaultdict()
        frame_arrays_on_core: Dict[Coord, FrameArrayType] = dict()

        for core_coord, v in config_dict.items():
            # 1. Only one config frame type I for each physical core.
            config_frame_type1 = OfflineFrameGen.gen_config_frame1(
                target_chip_coord,
                core_coord,
                RId(0, 0),
                v.random_seed,
            )

            # 2. Only one config frame type II for each physical core.
            config_frame_type2 = OfflineFrameGen.gen_config_frame2(
                target_chip_coord,
                core_coord,
                RId(0, 0),
                v.params_reg,
            )

            # Iterate all the neuron segments in the function inside
            config_frame_type3 = OfflineFrameGen.gen_config_frame3(
                target_chip_coord,
                core_coord,
                RId(0, 0),
                v.params_reg.lcn_extension,
                v.params_reg.weight_precision,
                list(v.neuron_configs.values()),
            )

            # 4. Only one config frame type IV for each physical core.
            n_addr_write = v.params_reg.num_dendrite  # The number of address to write
            config_frame_type4 = OfflineFrameGen.gen_config_frame4(
                target_chip_coord,
                core_coord,
                RId(0, 0),
                0,
                18 * n_addr_write,
                v.weight_ram[:n_addr_write],
            )

            _debug_dict[core_coord] = {
                "config1": config_frame_type1,
                "config2": config_frame_type2,
                "config3": config_frame_type3,
                "config4": config_frame_type4,
            }

            frame3 = np.concatenate(
                [f.value for f in config_frame_type3], dtype=FRAME_DTYPE
            )

            frame_arrays_on_core[core_coord] = np.concatenate(
                [
                    config_frame_type1.value,
                    config_frame_type2.value,
                    frame3,
                    config_frame_type4.value,
                ],
                dtype=FRAME_DTYPE,
            )

        if write_to_file:
            for core_coord, f in frame_arrays_on_core.items():
                addr = core_coord.address
                fn = f"config_core{addr}.{format}"

                if format == "npy":
                    np2npy(fp / fn, f)
                elif format == "bin":
                    np2bin(fp / fn, f)
                else:
                    np2txt(fp / fn, f)

        return frame_arrays_on_core

    # @staticmethod
    # def gen_reset_frame(chip_coord, core_coord, core_ex_coord=RId(0, 0)):
    #     """每次推理或配置前先发送复位帧，再进行配置"""
    #     frame_array = np.array([]).astype(FRAME_DTYPE)
    #     frame1 = Frame(
    #         header=FrameHeader.CONFIG_TYPE1,
    #         chip_coord=chip_coord,
    #         core_coord=core_coord,
    #         core_ex_coord=core_ex_coord,
    #         payload=0,
    #     )
    #     frame2 = Frame(
    #         header=FrameHeader.CONFIG_TYPE1,
    #         chip_coord=chip_coord,
    #         core_coord=core_coord,
    #         core_ex_coord=core_ex_coord,
    #         payload=0,
    #     )
    #     frame3 = OfflineFrameGen.gen_work_frame4(chip_coord)
    #     frame4 = Frame(
    #         header=FrameHeader.CONFIG_TYPE1,
    #         chip_coord=chip_coord,
    #         core_coord=core_coord,
    #         core_ex_coord=core_ex_coord,
    #         payload=0,
    #     )
    #     frame5 = OfflineWorkFrame1(
    #         chip_coord=chip_coord,
    #         core_coord=core_coord,
    #         core_ex_coord=core_ex_coord,
    #         axon=0,
    #         time_slot=0,
    #         data=np.array([0]),
    #     )
    #     for frame in [frame1, frame2, frame3, frame4, frame5]:
    #         frame_array = np.append(frame_array, frame.value)
    #     return frame_array

    @staticmethod
    def gen_testin_frame1(
        chip_coord: Coord, core_coord: Coord, rid: RId, /
    ) -> OfflineTestInFrame1:
        return OfflineTestInFrame1(chip_coord, core_coord, rid)

    @staticmethod
    def gen_testout_frame1(
        test_chip_coord: Coord, core_coord: Coord, rid: RId, /, random_seed: DataType
    ) -> OfflineTestOutFrame1:
        return OfflineTestOutFrame1(test_chip_coord, core_coord, rid, random_seed)

    @staticmethod
    def gen_testin_frame2(
        chip_coord: Coord, core_coord: Coord, rid: RId, /
    ) -> OfflineTestInFrame2:
        return OfflineTestInFrame2(chip_coord, core_coord, rid)

    @overload
    @staticmethod
    def gen_testout_frame2(
        test_chip_coord: Coord, core_coord: Coord, rid: RId, /, params_reg: ParamsReg
    ) -> OfflineTestOutFrame2:
        ...

    @overload
    @staticmethod
    def gen_testout_frame2(
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        params_reg: Dict[str, Any],
    ) -> OfflineTestOutFrame2:
        ...

    @staticmethod
    def gen_testout_frame2(
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        params_reg: Union[ParamsReg, Dict[str, Any]],
    ) -> OfflineTestOutFrame2:
        if isinstance(params_reg, ParamsReg):
            pr = params_reg.model_dump(by_alias=True)
        else:
            pr = params_reg

        return OfflineTestOutFrame2(test_chip_coord, core_coord, rid, pr)

    @staticmethod
    def gen_testin_frame3(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        sram_start_addr: DataType,
        data_package_num: DataType,
    ) -> OfflineTestInFrame3:
        return OfflineTestInFrame3(
            chip_coord, core_coord, rid, int(sram_start_addr), int(data_package_num)
        )

    @staticmethod
    def gen_testin_frame4(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        sram_start_addr: DataType,
        data_package_num: DataType,
    ) -> OfflineTestInFrame4:
        return OfflineTestInFrame4(
            chip_coord, core_coord, rid, int(sram_start_addr), int(data_package_num)
        )

    @staticmethod
    def gen_work_frame1(
        input_proj_info: Dict[str, Any],
        data: DataArrayType,
    ) -> FrameArrayType:
        common_frame_dest = OfflineWorkFrame1._frame_dest_reorganized(input_proj_info)
        _data = np.asarray(data, dtype=FRAME_DTYPE)

        return OfflineFrameGen.gen_work_frame1_fast(common_frame_dest, _data)

    @staticmethod
    def gen_work_frame1_fast(
        frame_dest_info: FrameArrayType, data: np.ndarray
    ) -> np.ndarray:
        if any(d > np.iinfo(np.int8).max for d in data):
            # TODO
            raise ValueError

        return OfflineWorkFrame1._gen_frame_fast(frame_dest_info, data)

    #     @staticmethod
    #     def gen_frameinfo(
    #         chip_coord: Union[List[Coord], Coord],
    #         core_coord: Union[List[Coord], Coord],
    #         core_ex_coord: Union[List[RId], RId],
    #         axon: Union[List[int], int],
    #         time_slot: Union[List[int], int],
    #         save_path: Optional[str] = None,
    #     ) -> np.ndarray:
    #         header = [FrameHeader.WORK_TYPE1]
    #         if not isinstance(chip_coord, list):
    #             chip_coord = [chip_coord]
    #         if not isinstance(core_coord, list):
    #             core_coord = [core_coord]
    #         if not isinstance(core_ex_coord, list):
    #             core_ex_coord = [core_ex_coord]
    #         if not isinstance(axon, list):
    #             axon = [axon]
    #         if not isinstance(time_slot, list):
    #             time_slot = [time_slot]

    #         header_value = np.array([head.value for head in header]).astype(FRAME_DTYPE)
    #         chip_address = np.array([coord.address for coord in chip_coord]).astype(
    #             FRAME_DTYPE
    #         )
    #         core_address = np.array([coord.address for coord in core_coord]).astype(
    #             FRAME_DTYPE
    #         )
    #         core_ex_address = np.array([coord.address for coord in core_ex_coord]).astype(
    #             FRAME_DTYPE
    #         )
    #         axon_array = np.array(axon, dtype=FRAME_DTYPE)
    #         time_slot_array = np.array(time_slot, dtype=FRAME_DTYPE)

    #         temp_header = header_value & FrameFormat.GENERAL_HEADER_MASK
    #         temp_chip_address = chip_address & FrameFormat.GENERAL_CHIP_ADDR_MASK
    #         temp_core_address = core_address & FrameFormat.GENERAL_CORE_ADDR_MASK
    #         temp_core_ex_address = core_ex_address & FrameFormat.GENERAL_CORE_EX_ADDR_MASK
    #         temp_reserve = 0x00 & WorkFrame1Format.RESERVED_MASK
    #         temp_axon_array = axon_array & WorkFrame1Format.AXON_MASK
    #         temp_time_slot_array = time_slot_array & WorkFrame1Format.TIME_SLOT_MASK

    #         frameinfo = (
    #             (temp_header << FrameFormat.GENERAL_HEADER_OFFSET)
    #             | (temp_chip_address << FrameFormat.GENERAL_CHIP_ADDR_OFFSET)
    #             | (temp_core_address << FrameFormat.GENERAL_CORE_ADDR_OFFSET)
    #             | (temp_core_ex_address << FrameFormat.GENERAL_CORE_EX_ADDR_OFFSET)
    #             | (temp_reserve << WorkFrame1Format.RESERVED_OFFSET)
    #             | (temp_axon_array << WorkFrame1Format.AXON_OFFSET)
    #             | (temp_time_slot_array << WorkFrame1Format.TIME_SLOT_OFFSET)
    #         )

    #         if save_path is not None:
    #             np.save(save_path, frameinfo)

    #         return frameinfo

    @staticmethod
    def gen_work_frame2(chip_coord: Coord, n_sync: IntScalarType) -> OfflineWorkFrame2:
        return OfflineWorkFrame2(chip_coord, int(n_sync))

    @staticmethod
    def gen_work_frame3(chip_coord: Coord) -> OfflineWorkFrame3:
        return OfflineWorkFrame3(chip_coord)

    @staticmethod
    def gen_work_frame4(chip_coord: Coord) -> OfflineWorkFrame4:
        return OfflineWorkFrame4(chip_coord)


# class OfflineFrameParser:
#     @staticmethod
#     def parse(value):
#         header = OfflineFrameParser.get_header(value)

#         if header == FrameHeader.WORK_TYPE1:
#             pass
#         if header == FrameHeader.TEST_TYPE1:
#             return OfflineTestOutFrame1(value=value)
#         elif header == FrameHeader.TEST_TYPE2:
#             return OfflineTestOutFrame2(value=value)
#         elif header == FrameHeader.TEST_TYPE3:
#             return OfflineTestOutFrame3(value=value)
#         elif header == FrameHeader.TEST_TYPE4:
#             pass

#         else:
#             raise ValueError("The header of the frame is not supported.")

#     @staticmethod
#     def get_header(value):
#         return FrameHeader(
#             (value[0] >> FrameFormat.GENERAL_HEADER_OFFSET)
#             & FrameFormat.GENERAL_HEADER_MASK
#         )
