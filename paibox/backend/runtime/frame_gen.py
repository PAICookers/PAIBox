from typing import Any, Dict, Union, overload

import numpy as np
from numpy.typing import NDArray

from paibox.libpaicore import LCN_EX, Coord, NeuronAttrs, NeuronDestInfo, ParamsReg
from paibox.libpaicore import ReplicationId as RId
from paibox.libpaicore import WeightPrecision as WP

from .libframe._types import DataArrayType, DataType, FrameArrayType, IntScalarType
from .libframe.frames import *

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
            _params_reg = params_reg.model_dump(by_alias=True)
        else:
            _params_reg = params_reg

        return OfflineConfigFrame2(chip_coord, core_coord, rid, _params_reg)

    @overload
    @staticmethod
    def gen_config_frame3(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        n_neuron: int,
        attrs: NeuronAttrs,
        dest_info: NeuronDestInfo,
        *,
        lcn_ex: LCN_EX = LCN_EX.LCN_1X,
        weight_precision: WP = WP.WEIGHT_WIDTH_1BIT,
    ) -> OfflineConfigFrame3:
        ...

    @overload
    @staticmethod
    def gen_config_frame3(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        n_neuron: int,
        attrs: Dict[str, Any],
        dest_info: Dict[str, Any],
        *,
        lcn_ex: LCN_EX = LCN_EX.LCN_1X,
        weight_precision: WP = WP.WEIGHT_WIDTH_1BIT,
    ) -> OfflineConfigFrame3:
        ...

    @staticmethod
    def gen_config_frame3(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        n_neuron: int,
        attrs: Union[NeuronAttrs, Dict[str, Any]],
        dest_info: Union[NeuronDestInfo, Dict[str, Any]],
        *,
        lcn_ex: LCN_EX = LCN_EX.LCN_1X,
        weight_precision: WP = WP.WEIGHT_WIDTH_1BIT,
    ) -> OfflineConfigFrame3:
        if isinstance(attrs, NeuronAttrs):
            _attrs = attrs.model_dump(by_alias=True)
        else:
            _attrs = attrs

        if isinstance(dest_info, NeuronDestInfo):
            _dest_info = dest_info.model_dump(by_alias=True)
        else:
            _dest_info = dest_info

        return OfflineConfigFrame3(
            chip_coord,
            core_coord,
            rid,
            sram_start_addr,
            n_neuron,
            _attrs,
            _dest_info,
            repeat=(1 << lcn_ex) * (1 << weight_precision),
        )

    @staticmethod
    def gen_config_frame4(
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        n_data_package: int,
        weight_ram: FrameArrayType,
    ) -> OfflineConfigFrame4:
        return OfflineConfigFrame4(
            chip_coord,
            core_coord,
            rid,
            sram_start_addr,
            n_data_package,
            weight_ram,
        )

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
        one_input_proj: Dict[str, Any],
        data: DataArrayType,
    ) -> FrameArrayType:
        """Generate the common part of the input spike frames by given the      \
            dictionary of input projections.

        Args:
            - one_input_proj: the dictionary of one input projection exported   \
                from `paibox.Mapper`.
        """
        common_frame_dest = OfflineWorkFrame1._frame_dest_reorganized(one_input_proj)
        _data = np.asarray(data, dtype=np.uint8)

        return OfflineFrameGen.gen_work_frame1_fast(common_frame_dest, _data)

    @staticmethod
    def gen_work_frame1_fast(
        frame_dest_info: FrameArrayType, data: NDArray[np.uint8]
    ) -> FrameArrayType:
        _max, _min = np.max(data, axis=None), np.min(data, axis=None)

        if _min < np.iinfo(np.uint8).min or _max > np.iinfo(np.uint8).max:
            raise ValueError(f"Data out of range int8 ({_min}, {_max})")

        if frame_dest_info.size != data.size:
            raise ValueError(
                "The size of frame dest info & data are not equal ({frame_dest_info.size}, {data.size})"
            )

        return OfflineWorkFrame1._gen_frame_fast(frame_dest_info, data.flatten())

    @staticmethod
    def gen_work_frame2(chip_coord: Coord, n_sync: IntScalarType) -> OfflineWorkFrame2:
        return OfflineWorkFrame2(chip_coord, int(n_sync))

    @staticmethod
    def gen_work_frame3(chip_coord: Coord) -> OfflineWorkFrame3:
        return OfflineWorkFrame3(chip_coord)

    @staticmethod
    def gen_work_frame4(chip_coord: Coord) -> OfflineWorkFrame4:
        return OfflineWorkFrame4(chip_coord)
