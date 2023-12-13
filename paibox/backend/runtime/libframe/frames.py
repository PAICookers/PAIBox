import warnings
from typing import Any, ClassVar, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from paibox.libpaicore import Coord, CoordLike
from paibox.libpaicore import FrameFormat as FF
from paibox.libpaicore import FrameHeader as FH
from paibox.libpaicore import ParameterRAMFormat as RAMF
from paibox.libpaicore import ParameterRegFormat as RegF
from paibox.libpaicore import ReplicationId as RId
from paibox.libpaicore import RIdLike
from paibox.libpaicore import SpikeFrameFormat as WF1F
from paibox.libpaicore import WeightRAMFormat as WRF
from paibox.libpaicore import to_coord, to_rid
from paibox.libpaicore.v2.ram_model import NeuronAttrsChecker, NeuronDestInfoChecker
from paibox.libpaicore.v2.reg_model import ParamsRegChecker

from ._types import FRAME_DTYPE, ArrayType, DataType, FrameArrayType, IntScalarType
from .base import Frame, FramePackage
from .utils import bin_split, params_check, params_check2

__all__ = [
    "OfflineConfigFrame1",
    "OfflineConfigFrame2",
    "OfflineConfigFrame3",
    "OfflineConfigFrame4",
    "OfflineTestInFrame1",
    "OfflineTestInFrame2",
    "OfflineTestInFrame3",
    "OfflineTestInFrame4",
    "OfflineTestOutFrame1",
    "OfflineTestOutFrame2",
    "OfflineTestOutFrame3",
    "OfflineTestOutFrame4",
    "OfflineWorkFrame1",
    "OfflineWorkFrame2",
    "OfflineWorkFrame3",
    "OfflineWorkFrame4",
]


class _RandomSeedFrame(Frame):
    """For reusing the method of splitting the random seed."""

    def __init__(
        self,
        header: FH,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        random_seed: DataType,
    ) -> None:
        _rs = int(random_seed)
        # TODO Transfer the warning to the previous phase.
        if _rs > FF.GENERAL_MASK:
            warnings.warn(
                f"Random seed {_rs} is too large, truncated into 64 bits!",
                UserWarning,
            )

        self._random_seed = _rs & FF.GENERAL_MASK
        payload = self._random_seed_split()

        super().__init__(header, chip_coord, core_coord, rid, payload)

    def _random_seed_split(self) -> FrameArrayType:
        return np.asarray(
            [
                (self._random_seed >> 34) & FF.GENERAL_PAYLOAD_MASK,
                (self._random_seed >> 4) & FF.GENERAL_PAYLOAD_MASK,
                (self._random_seed & ((1 << 4) - 1)) << 26,
            ],
            dtype=FRAME_DTYPE,
        )

    @property
    def random_seed(self) -> FRAME_DTYPE:
        return FRAME_DTYPE(self._random_seed)


class _ParamRAMFrame(Frame):
    def __init__(
        self,
        header: FH,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        params_reg_dict: Dict[str, Any],
    ) -> None:
        payload = self._payload_reorganized(params_reg_dict)

        super().__init__(header, chip_coord, core_coord, rid, payload)

    @staticmethod
    @params_check(ParamsRegChecker)
    def _payload_reorganized(reg_dict: Dict[str, Any]) -> FrameArrayType:
        # High 8 bits & low 7 bits of tick_wait_start
        tws_high8, tws_low7 = bin_split(reg_dict["tick_wait_start"], 7, 8)
        # High 3 bits & low 7 bits of test_chip_addrs
        tca_high3, tca_low7 = bin_split(reg_dict["test_chip_addr"], 7, 3)

        reg_frame1 = (
            (reg_dict["weight_width"] & RegF.WEIGHT_WIDTH_MASK)
            << RegF.WEIGHT_WIDTH_OFFSET
            | ((reg_dict["LCN"] & RegF.LCN_MASK) << RegF.LCN_OFFSET)
            | (
                (reg_dict["input_width"] & RegF.INPUT_WIDTH_MASK)
                << RegF.INPUT_WIDTH_OFFSET
            )
            | (
                (reg_dict["spike_width"] & RegF.SPIKE_WIDTH_MASK)
                << RegF.SPIKE_WIDTH_OFFSET
            )
            | (
                (reg_dict["neuron_num"] & RegF.NEURON_NUM_MASK)
                << RegF.NEURON_NUM_OFFSET
            )
            | ((reg_dict["pool_max"] & RegF.POOL_MAX_MASK) << RegF.POOL_MAX_OFFSET)
            | (
                (tws_high8 & RegF.TICK_WAIT_START_HIGH8_MASK)
                << RegF.TICK_WAIT_START_HIGH8_OFFSET
            )
        )

        reg_frame2 = (
            (
                (tws_low7 & RegF.TICK_WAIT_START_LOW7_MASK)
                << RegF.TICK_WAIT_START_LOW7_OFFSET
            )
            | (
                (reg_dict["tick_wait_end"] & RegF.TICK_WAIT_END_MASK)
                << RegF.TICK_WAIT_END_OFFSET
            )
            | ((reg_dict["snn_en"] & RegF.SNN_EN_MASK) << RegF.SNN_EN_OFFSET)
            | (
                (reg_dict["target_LCN"] & RegF.TARGET_LCN_MASK)
                << RegF.TARGET_LCN_OFFSET
            )
            | (
                (tca_high3 & RegF.TEST_CHIP_ADDR_HIGH3_MASK)
                << RegF.TEST_CHIP_ADDR_HIGH3_OFFSET
            )
        )

        reg_frame3 = (
            tca_low7 & RegF.TEST_CHIP_ADDR_LOW7_MASK
        ) << RegF.TEST_CHIP_ADDR_LOW7_OFFSET

        return np.asarray([reg_frame1, reg_frame2, reg_frame3], dtype=FRAME_DTYPE)

    @property
    def params_reg(self) -> FrameArrayType:
        return self.payload


class _NeuronRAMFrame(FramePackage):
    def __init__(
        self,
        header: FH,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        sram_start_addr: int,
        neuron_num: int,
        neuron_attrs: Dict[str, Any],
        neuron_dest_info: Dict[str, Any],
        repeat: int,
    ) -> None:
        n_package = 4 * neuron_num * repeat
        payload = np.uint64(
            (
                (sram_start_addr & RAMF.DATA_PACKAGE_SRAM_NEURON_MASK)
                << RAMF.DATA_PACKAGE_SRAM_NEURON_OFFSET
            )
            | ((0 & RAMF.DATA_PACKAGE_TYPE_MASK) << RAMF.DATA_PACKAGE_TYPE_OFFSET)
            | ((n_package & RAMF.DATA_PACKAGE_NUM_MASK) << RAMF.DATA_PACKAGE_NUM_OFFSET)
        )
        packages = self._packages_reorganized(
            neuron_attrs,
            neuron_dest_info,
            neuron_num,
            repeat,
        )

        super().__init__(header, chip_coord, core_coord, rid, payload, packages)

    @staticmethod
    @params_check2(NeuronAttrsChecker, NeuronDestInfoChecker)
    def _packages_reorganized(
        attrs: Dict[str, Any],
        dest_info: Dict[str, Any],
        neuron_num: int,
        repeat: int,
    ) -> FrameArrayType:
        tick_relative = dest_info["tick_relative"]
        addr_axon = dest_info["addr_axon"]

        assert len(tick_relative) == len(addr_axon)

        _packages = np.zeros((neuron_num, 4), dtype=FRAME_DTYPE)

        leak_v_high2, leak_v_low28 = bin_split(attrs["leak_post"], 28, 2)
        threshold_mask_ctrl_high4, threshold_mask_ctrl_low1 = bin_split(
            attrs["threshold_mask_ctrl"], 1, 4
        )
        addr_core_x_high3, addr_core_x_low2 = bin_split(dest_info["addr_core_x"], 2, 3)

        # LSB: [63:0], [127:64], [191:128], [213:192]
        # Package #1, [63:0]
        ram_frame1 = (
            ((attrs["vjt_pre"] & RAMF.VJT_PRE_MASK) << RAMF.VJT_PRE_OFFSET)
            | (
                (attrs["bit_truncate"] & RAMF.BIT_TRUNCATE_MASK)
                << RAMF.BIT_TRUNCATE_OFFSET
            )
            | (
                (attrs["weight_det_stoch"] & RAMF.WEIGHT_DET_STOCH_MASK)
                << RAMF.WEIGHT_DET_STOCH_OFFSET
            )
            | ((leak_v_low28 & RAMF.LEAK_V_LOW28_MASK) << RAMF.LEAK_V_LOW28_OFFSET)
        )

        # Package #2, [127:64]
        ram_frame2 = (
            ((leak_v_high2 & RAMF.LEAK_V_HIGH2_MASK) << RAMF.LEAK_V_HIGH2_OFFSET)
            | (
                (attrs["leak_det_stoch"] & RAMF.LEAK_DET_STOCH_MASK)
                << RAMF.LEAK_DET_STOCH_OFFSET
            )
            | (
                (attrs["leak_reversal_flag"] & RAMF.LEAK_REVERSAL_FLAG_MASK)
                << RAMF.LEAK_REVERSAL_FLAG_OFFSET
            )
            | (
                (attrs["threshold_pos"] & RAMF.THRESHOLD_POS_MASK)
                << RAMF.THRESHOLD_POS_OFFSET
            )
            | (
                (attrs["threshold_neg"] & RAMF.THRESHOLD_NEG_MASK)
                << RAMF.THRESHOLD_NEG_OFFSET
            )
            | (
                (attrs["threshold_neg_mode"] & RAMF.THRESHOLD_NEG_MODE_MASK)
                << RAMF.THRESHOLD_NEG_MODE_OFFSET
            )
            | (
                (threshold_mask_ctrl_low1 & RAMF.THRESHOLD_MASK_CTRL_LOW1_MASK)
                << RAMF.THRESHOLD_MASK_CTRL_LOW1_OFFSET
            )
        )

        # Package #3, [191:128]
        ram_frame3 = (
            (
                (threshold_mask_ctrl_high4 & RAMF.THRESHOLD_MASK_CTRL_HIGH4_MASK)
                << RAMF.THRESHOLD_MASK_CTRL_HIGH4_OFFSET
            )
            | ((attrs["leak_post"] & RAMF.LEAK_POST_MASK) << RAMF.LEAK_POST_OFFSET)
            | ((attrs["reset_v"] & RAMF.RESET_V_MASK) << RAMF.RESET_V_OFFSET)
            | ((attrs["reset_mode"] & RAMF.RESET_MODE_MASK) << RAMF.RESET_MODE_OFFSET)
            | (
                (dest_info["addr_chip_y"] & RAMF.ADDR_CHIP_Y_MASK)
                << RAMF.ADDR_CHIP_Y_OFFSET
            )
            | (
                (dest_info["addr_chip_x"] & RAMF.ADDR_CHIP_X_MASK)
                << RAMF.ADDR_CHIP_X_OFFSET
            )
            | (
                (dest_info["addr_core_y_ex"] & RAMF.ADDR_CORE_Y_EX_MASK)
                << RAMF.ADDR_CORE_Y_EX_OFFSET
            )
            | (
                (dest_info["addr_core_x_ex"] & RAMF.ADDR_CORE_X_EX_MASK)
                << RAMF.ADDR_CORE_X_EX_OFFSET
            )
            | (
                (dest_info["addr_core_y"] & RAMF.ADDR_CORE_Y_MASK)
                << RAMF.ADDR_CORE_Y_OFFSET
            )
            | (
                (addr_core_x_low2 & RAMF.ADDR_CORE_X_LOW2_MASK)
                << RAMF.ADDR_CORE_X_LOW2_OFFSET
            )
        )
        _package_common = np.array(
            [ram_frame1, ram_frame2, ram_frame3], dtype=FRAME_DTYPE
        )

        # Repeat the common part of packages.
        _packages[:, :3] = np.tile(_package_common, (neuron_num, 1))

        # Iterate destination infomation of every neuron
        for i in range(neuron_num):
            # Package #4, [213:192]
            ram_frame4 = (
                (
                    (addr_core_x_high3 & RAMF.ADDR_CORE_X_HIGH3_MASK)
                    << RAMF.ADDR_CORE_X_HIGH3_OFFSET
                )
                | ((tick_relative[i] & RAMF.ADDR_AXON_MASK) << RAMF.ADDR_AXON_OFFSET)
                | (
                    (addr_axon[i] & RAMF.TICK_RELATIVE_MASK)
                    << RAMF.TICK_RELATIVE_OFFSET
                )
            )
            _packages[i][-1] = ram_frame4

        # Repeat every neuron `repeat` times & flatten
        # (neuron_num, 4) -> (neuron_num * repeat * 4,)
        packages_repeated = np.repeat(_packages, repeat)

        return packages_repeated


class _WeightRAMFrame(FramePackage):
    def __init__(
        self,
        header: FH,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: IntScalarType,
        data_package_num: IntScalarType,
        weight_ram: FrameArrayType,
    ) -> None:
        payload = np.uint64(
            (
                (int(sram_start_addr) & WRF.DATA_PACKAGE_SRAM_NEURON_MASK)
                << WRF.DATA_PACKAGE_SRAM_NEURON_OFFSET
            )
            | ((0 & WRF.DATA_PACKAGE_TYPE_MASK) << WRF.DATA_PACKAGE_TYPE_OFFSET)
            | (
                (int(data_package_num) & WRF.DATA_PACKAGE_NUM_MASK)
                << WRF.DATA_PACKAGE_NUM_OFFSET
            )
        )
        _weight_ram = weight_ram.flatten()

        super().__init__(header, chip_coord, core_coord, rid, payload, _weight_ram)


class OfflineConfigFrame1(_RandomSeedFrame):
    """Offline config frame type I"""

    header: ClassVar[FH] = FH.CONFIG_TYPE1

    def __init__(
        self,
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        random_seed: DataType,
    ) -> None:
        super().__init__(self.header, test_chip_coord, core_coord, rid, random_seed)


class OfflineConfigFrame2(_ParamRAMFrame):
    header: ClassVar[FH] = FH.CONFIG_TYPE2

    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        params_reg_dict: Dict[str, Any],
    ) -> None:
        super().__init__(self.header, chip_coord, core_coord, rid, params_reg_dict)


class OfflineConfigFrame3(_NeuronRAMFrame):
    header: ClassVar[FH] = FH.CONFIG_TYPE3

    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        neuron_num: int,
        neuron_attrs: Dict[str, Any],
        neuron_dest_info: Dict[str, Any],
        *,
        repeat: int = 1,
    ) -> None:
        super().__init__(
            self.header,
            chip_coord,
            core_coord,
            rid,
            sram_start_addr,
            neuron_num,
            neuron_attrs,
            neuron_dest_info,
            repeat,
        )


class OfflineConfigFrame4(_WeightRAMFrame):
    header: ClassVar[FH] = FH.CONFIG_TYPE4

    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: IntScalarType,
        data_package_num: IntScalarType,
        weight_ram: FrameArrayType,
    ) -> None:
        super().__init__(
            self.header,
            chip_coord,
            core_coord,
            rid,
            sram_start_addr,
            data_package_num,
            weight_ram,
        )


class OfflineTestInFrame1(Frame):
    header: ClassVar[FH] = FH.TEST_TYPE1

    def __init__(self, chip_coord: Coord, core_coord: Coord, rid: RId, /) -> None:
        super().__init__(
            self.header, chip_coord, core_coord, rid, np.asarray(0, dtype=FRAME_DTYPE)
        )


class OfflineTestOutFrame1(_RandomSeedFrame):
    header: ClassVar[FH] = FH.TEST_TYPE1

    def __init__(
        self,
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        random_seed: DataType,
    ) -> None:
        super().__init__(self.header, test_chip_coord, core_coord, rid, random_seed)


class OfflineTestInFrame2(Frame):
    header: ClassVar[FH] = FH.TEST_TYPE2

    def __init__(self, chip_coord: Coord, core_coord: Coord, rid: RId, /) -> None:
        super().__init__(
            self.header, chip_coord, core_coord, rid, np.asarray(0, dtype=FRAME_DTYPE)
        )


class OfflineTestOutFrame2(_ParamRAMFrame):
    header: ClassVar[FH] = FH.TEST_TYPE2

    def __init__(
        self,
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        params_reg_dict: Dict[str, Any],
    ) -> None:
        super().__init__(self.header, test_chip_coord, core_coord, rid, params_reg_dict)


class OfflineTestInFrame3(Frame):
    header: ClassVar[FH] = FH.TEST_TYPE3

    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        data_package_num: int,
    ) -> None:
        payload = np.asarray(
            (
                (sram_start_addr & FF.DATA_PACKAGE_SRAM_NEURON_MASK)
                << FF.DATA_PACKAGE_SRAM_NEURON_OFFSET
            )
            | ((1 & FF.DATA_PACKAGE_TYPE_MASK) << FF.DATA_PACKAGE_TYPE_OFFSET)
            | (
                (data_package_num & FF.DATA_PACKAGE_NUM_MASK)
                << FF.DATA_PACKAGE_NUM_OFFSET
            ),
            dtype=FRAME_DTYPE,
        )

        super().__init__(self.header, chip_coord, core_coord, rid, payload)


class OfflineTestOutFrame3(_NeuronRAMFrame):
    header: ClassVar[FH] = FH.CONFIG_TYPE4

    def __init__(
        self,
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        neuron_num: int,
        neuron_attrs: Dict[str, Any],
        neuron_dest_info: Dict[str, Any],
        *,
        repeat: int = 1,
    ) -> None:
        super().__init__(
            self.header,
            test_chip_coord,
            core_coord,
            rid,
            sram_start_addr,
            neuron_num,
            neuron_attrs,
            neuron_dest_info,
            repeat,
        )


class OfflineTestInFrame4(Frame):
    header: ClassVar[FH] = FH.TEST_TYPE4

    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: int,
        data_package_num: int,
    ):
        payload = np.asarray(
            (
                (sram_start_addr & FF.DATA_PACKAGE_SRAM_NEURON_MASK)
                << FF.DATA_PACKAGE_SRAM_NEURON_OFFSET
            )
            | ((1 & FF.DATA_PACKAGE_TYPE_MASK) << FF.DATA_PACKAGE_TYPE_OFFSET)
            | (
                (data_package_num & FF.DATA_PACKAGE_NUM_MASK)
                << FF.DATA_PACKAGE_NUM_OFFSET
            ),
            dtype=FRAME_DTYPE,
        )

        super().__init__(self.header, chip_coord, core_coord, rid, payload)


class OfflineTestOutFrame4(_WeightRAMFrame):
    header: ClassVar[FH] = FH.TEST_TYPE4

    def __init__(
        self,
        test_chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        sram_start_addr: IntScalarType,
        data_package_num: IntScalarType,
        weight_ram: FrameArrayType,
    ) -> None:
        super().__init__(
            self.header,
            test_chip_coord,
            core_coord,
            rid,
            sram_start_addr,
            data_package_num,
            weight_ram,
        )


class OfflineWorkFrame1(Frame):
    header: ClassVar[FH] = FH.WORK_TYPE1

    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        /,
        timeslot: int,
        axon: int,
        _data: DataType,
    ) -> None:
        if isinstance(_data, np.ndarray) and _data.size != 1:
            raise ValueError("Size of data must be 1.")

        if _data < np.iinfo(np.uint8).min or _data > np.iinfo(np.uint8).max:
            raise ValueError(f"Data out of range int8.")

        self.data = np.uint8(_data)
        self._axon = int(axon)
        self._timeslot = int(timeslot)

        payload = np.asarray(
            ((self._axon & WF1F.AXON_MASK) << WF1F.AXON_OFFSET)
            | ((self._timeslot & WF1F.TIMESLOT_MASK) << WF1F.TIMESLOT_OFFSET)
            | ((self.data & WF1F.DATA_MASK) << WF1F.DATA_OFFSET),
            dtype=FRAME_DTYPE,
        )

        super().__init__(self.header, chip_coord, core_coord, rid, payload)

    @property
    def target_timeslot(self) -> int:
        return self._timeslot

    @property
    def target_axon(self) -> int:
        return self._axon

    @staticmethod
    @params_check(NeuronDestInfoChecker)
    def _frame_dest_reorganized(dest_info: Dict[str, Any]) -> FrameArrayType:
        return OfflineWorkFrame1.concat_frame_dest(
            (dest_info["addr_chip_x"], dest_info["addr_chip_y"]),
            (dest_info["addr_core_x"], dest_info["addr_core_y"]),
            (dest_info["addr_core_x_ex"], dest_info["addr_core_y_ex"]),
            dest_info["addr_axon"],
            dest_info["tick_relative"],
        )

    @staticmethod
    def _gen_frame_fast(
        frame_dest_info: FrameArrayType, data: NDArray[np.uint8]
    ) -> FrameArrayType:
        """DO NOT call `OfflineWorkFrame1._gen_frame_fast()` directly."""
        indexes = np.nonzero(data)

        return (frame_dest_info[indexes] + data[indexes]).astype(FRAME_DTYPE)

    @classmethod
    def concat_frame_dest(
        cls,
        chip_coord: CoordLike,
        core_coord: CoordLike,
        rid: RIdLike,
        /,
        axons: ArrayType,
        timeslots: Optional[ArrayType] = None,
    ) -> FrameArrayType:
        _axons = np.asarray(axons, dtype=FRAME_DTYPE).flatten()

        if timeslots is not None:
            _timeslots = np.asarray(timeslots, dtype=FRAME_DTYPE).flatten()
        else:
            _timeslots = np.zeros_like(_axons)

        if _axons.size != _timeslots.size:
            raise ValueError(
                f"The size of axons & timeslots are not equal ({_axons.size}, {_timeslots.size})"
            )

        _chip_coord = to_coord(chip_coord)
        _core_coord = to_coord(core_coord)
        _rid = to_rid(rid)

        header = cls.header.value & FF.GENERAL_HEADER_MASK
        chip_addr = _chip_coord.address & FF.GENERAL_CHIP_ADDR_MASK
        core_addr = _core_coord.address & FF.GENERAL_CORE_ADDR_MASK
        rid_addr = _rid.address & FF.GENERAL_CORE_EX_ADDR_MASK

        common_head = (
            (header << FF.GENERAL_HEADER_OFFSET)
            + (chip_addr << FF.GENERAL_CHIP_ADDR_OFFSET)
            + (core_addr << FF.GENERAL_CORE_ADDR_OFFSET)
            + (rid_addr << FF.GENERAL_CORE_EX_ADDR_OFFSET)
        )

        common_payload = ((_axons & WF1F.AXON_MASK) << WF1F.AXON_OFFSET) | (
            (_timeslots & WF1F.TIMESLOT_MASK) << WF1F.TIMESLOT_OFFSET
        )

        return (common_head + common_payload).astype(FRAME_DTYPE)


class OfflineWorkFrame2(Frame):
    header: ClassVar[FH] = FH.WORK_TYPE2

    def __init__(self, chip_coord: Coord, /, n_sync: int) -> None:
        super().__init__(
            self.header,
            chip_coord,
            Coord(0, 0),
            RId(0, 0),
            np.asarray(n_sync, dtype=FRAME_DTYPE),
        )


class OfflineWorkFrame3(Frame):
    header: ClassVar[FH] = FH.WORK_TYPE3

    def __init__(self, chip_coord: Coord) -> None:
        super().__init__(
            self.header,
            chip_coord,
            Coord(0, 0),
            RId(0, 0),
            np.asarray(0, dtype=FRAME_DTYPE),
        )


class OfflineWorkFrame4(Frame):
    header: ClassVar[FH] = FH.WORK_TYPE4

    def __init__(self, chip_coord: Coord) -> None:
        super().__init__(
            self.header,
            chip_coord,
            Coord(0, 0),
            RId(0, 0),
            np.asarray(0, dtype=FRAME_DTYPE),
        )
