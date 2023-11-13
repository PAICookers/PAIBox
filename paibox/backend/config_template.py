from dataclasses import dataclass
from typing import ClassVar, List, NamedTuple

import numpy as np

from paibox.base import NeuDyn
from paibox.frame.frame_params import FrameType
from paibox.libpaicore import (
    LCN_EX,
    AxonCoord,
    Coord,
    CoordLike,
    InputWidthFormat,
    HwConfig,
    MaxPoolingEnable,
    NeuronAttrs,
    NeuronDestInfo,
    ParamsRAM,
    ParamsReg,
    SNNModeEnable,
    SpikeWidthFormat,
    WeightPrecision,
    get_replication_id,
    to_coord,
)


class GlobalConfig:
    TEST_CHIP_ADDR = Coord(0, 0)

    @property
    def test_chip_addr(self) -> Coord:
        return self.TEST_CHIP_ADDR

    @test_chip_addr.setter
    def test_chip_addr(self, addr: CoordLike) -> None:
        self.TEST_CHIP_ADDR = to_coord(addr)


class CoreConfigDict(NamedTuple):
    random_seed: np.uint64
    weight_ram: np.ndarray
    weight_precision: WeightPrecision
    lcn_extension: LCN_EX
    input_width_format: InputWidthFormat
    spike_width_format: SpikeWidthFormat
    neuron_num: int
    max_pooling_en: MaxPoolingEnable
    tick_wait_start: int
    tick_wait_end: int
    snn_mode_en: SNNModeEnable
    target_lcn: LCN_EX
    test_chip_addr: int


class ConfigTemplate:
    """A configuration template."""

    frame_type: ClassVar[FrameType] = FrameType.FRAME_CONFIG


@dataclass(eq=False)
class NeuronConfig(ConfigTemplate):
    addr_ram: List[int]
    addr_offset: int
    params_ram: ParamsRAM

    @classmethod
    def build(
        cls,
        neuron: NeuDyn,
        addr_ram: List[int],
        addr_offset: int,
        axon_coords: List[AxonCoord],
        dest_coords: List[Coord],
    ):
        """Build the `NeuronConfig`.

        Args:
            - neuron: the target `NeuDyn`.
            - addr_ram: the assigned RAM address of the target neuron.
            - addr_offset: the offset of the RAM address.
            - axon_segs: the destination axon segments.
            - dest_coords: the coordinates of destination axons.
        """
        attrs = NeuronAttrs.model_validate(neuron.export_params(), strict=True)
        axon_rid = get_replication_id(dest_coords)

        dest_info_dict = {
            "tick_relative": [coord.tick_relative for coord in axon_coords],
            "addr_axon": [coord.addr_axon for coord in axon_coords],
            "addr_core_x": dest_coords[0].x,
            "addr_core_y": dest_coords[0].y,
            "addr_core_x_ex": axon_rid.x,
            "addr_core_y_ex": axon_rid.y,
            "addr_chip_x": 0,
            "addr_chip_y": 0,
        }

        neuron_dest_info = NeuronDestInfo.model_validate(dest_info_dict, strict=True)

        return cls(
            addr_ram,
            addr_offset,
            ParamsRAM(attrs=attrs, dest_info=neuron_dest_info),
        )

    def export_params(self):
        dict_ = {"addr_ram": self.addr_ram, "addr_offset": self.addr_offset}
        dict_ |= self.params_ram.model_dump(by_alias=True)

        return dict_


@dataclass(eq=False)
class CoreConfig(ConfigTemplate):
    coord: Coord
    random_seed: np.uint64
    weight_ram: np.ndarray
    params_reg: ParamsReg
    neuron_ram: NeuronConfig

    @classmethod
    def from_dict(
        cls,
        coord: Coord,
        core_config: CoreConfigDict,
        neuron_ram: NeuronConfig,
    ):
        return cls(
            coord,
            core_config.random_seed,
            core_config.weight_ram,
            ParamsReg.model_validate(**core_config._asdict(), strict=True),
            neuron_ram,
        )

    def export_type1(self):
        return self.random_seed

    def export_type2(self):
        return self.params_reg.model_dump(by_alias=True)

    def export_type3(self):
        raise NotImplementedError

    def export_type4(self):
        return self.weight_ram
