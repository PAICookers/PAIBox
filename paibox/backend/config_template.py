import numpy as np
from dataclasses import dataclass

from typing import ClassVar, List, NamedTuple


from paibox.base import NeuDyn
from paibox.frame.frame_params import FrameType
from paibox.libpaicore import (
    AxonCoord,
    LCN_EX,
    Coord,
    WeightPrecision,
    InputWidthFormat,
    SpikeWidthFormat,
    MaxPoolingEnable,
    NeuronAttrs,
    NeuronDestInfo,
    SNNModeEnable,
    get_replication_id,
    ParamsReg,
)
from paibox.libpaicore.v2.coordinate import CoordLike, to_coord

try:
    import ujson as json
except ModuleNotFoundError:
    import json


class GlobalConfig:
    TEST_CHIP_ADDR = Coord(0, 0)

    @property
    def test_chip_addr(self) -> Coord:
        return self.TEST_CHIP_ADDR

    @test_chip_addr.setter
    def test_chip_addr(self, addr: CoordLike) -> None:
        self.TEST_CHIP_ADDR = to_coord(addr)


class ConfigTemplate:
    """A configuration template."""

    frame_type: ClassVar[FrameType] = FrameType.FRAME_CONFIG


@dataclass(eq=False)
class NeuronConfig(ConfigTemplate):
    attr: NeuronAttrs
    addr_ram: slice
    dest_info: NeuronDestInfo

    @classmethod
    def build(
        cls,
        neuron: NeuDyn,
        addr_ram: slice,
        axon_coords: List[AxonCoord],
        dest_coords: List[Coord],
    ):
        """Build the `NeuronConfig`.

        Args:
            - neuron: the target `NeuDyn`.
            - addr_ram: the assigned RAM address of the target neuron.
            - axon_segs: the destination axon segments.
            - dest_coords: the coordinates of destination axons.
        """
        attr = NeuronAttrs(**neuron.export_params())
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
        return dest_info_dict
        # return cls(attr, addr_ram, NeuronDestInfo(**dest_info_dict))


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
            ParamsReg(**core_config._asdict()),
            neuron_ram,
        )

    def export_type1(self):
        return self.random_seed

    def export_type2(self):
        model = ParamsReg.model_validate(self.params_reg, strict=True)
        return model.model_dump(by_alias=True)

    def export_type4(self):
        return self.weight_ram
