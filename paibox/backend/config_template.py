from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, NamedTuple

import numpy as np

from paibox.base import NeuDyn
from paibox.libpaicore import (
    LCN_EX,
    AxonCoord,
    Coord,
    FrameType,
    InputWidthFormat,
    MaxPoolingEnable,
    NeuronAttrs,
    NeuronDestInfo,
    ParamsRAM,
    ParamsReg,
    SNNModeEnable,
    SpikeWidthFormat,
    WeightPrecision,
    get_replication_id,
)


class CoreConfigDict(NamedTuple):
    """Configurations of core."""

    weight_precision: WeightPrecision
    lcn_extension: LCN_EX
    input_width_format: InputWidthFormat
    spike_width_format: SpikeWidthFormat
    num_dentrite: int
    max_pooling_en: MaxPoolingEnable
    tick_wait_start: int
    tick_wait_end: int
    snn_mode_en: SNNModeEnable
    target_lcn: LCN_EX
    test_chip_addr: Coord

    def export(self) -> Dict[str, Any]:
        return ParamsReg.model_validate(self._asdict(), strict=True).model_dump(
            by_alias=True
        )


class ConfigTemplate:
    """A configuration template."""

    frame_type: ClassVar[FrameType] = FrameType.FRAME_CONFIG


@dataclass(eq=False)
class NeuronConfig(ConfigTemplate):
    addr_ram: List[int]
    addr_offset: int
    params_ram: ParamsRAM

    @classmethod
    def encapsulate(
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

    def export(self) -> Dict[str, Any]:
        dict_ = {"addr_ram": self.addr_ram, "addr_offset": self.addr_offset}
        dict_ |= self.params_ram.model_dump(by_alias=True)

        return dict_

    def config_dump(self) -> Dict[str, Any]:
        """Dump the configs for debugging."""
        dict_ = {"addr_offset": self.addr_offset}
        dict_ |= self.params_ram.model_dump(
            by_alias=True,
            exclude={"dest_info": self.params_ram.dest_info._exclude_vars},
        )

        return dict_


@dataclass(eq=False)
class CorePlacementConfig(ConfigTemplate):
    coord: Coord
    random_seed: np.uint64
    weight_ram: np.ndarray
    params_reg: ParamsReg
    neuron_ram: Dict[NeuDyn, NeuronConfig]

    @classmethod
    def encapsulate(
        cls,
        coord: Coord,
        random_seed: np.uint64,
        weight_ram: np.ndarray,
        core_config: CoreConfigDict,
        neuron_ram: Dict[NeuDyn, NeuronConfig],
    ):
        return cls(
            coord,
            random_seed,
            weight_ram,
            ParamsReg.model_validate(core_config._asdict(), strict=True),
            neuron_ram,
        )
