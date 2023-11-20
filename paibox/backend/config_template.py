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

from .context import _BACKEND_CONTEXT


class CoreConfig(NamedTuple):
    """Configurations of core."""

    _extra_params = ("name",)
    """Extra parameters for debugging."""

    name: str
    weight_precision: WeightPrecision
    lcn_extension: LCN_EX
    input_width_format: InputWidthFormat
    spike_width_format: SpikeWidthFormat
    num_dendrite: int
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

    def config_dump(self) -> Dict[str, Any]:
        """Dump the configs for debugging."""
        dict_ = self.export()

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_


class NeuronDest(NamedTuple):
    """Information of neuron destination(Axon address information)."""

    dest_coords: List[Coord]
    tick_relative: List[int]
    addr_axon: List[int]
    addr_core_x: int
    addr_core_y: int
    addr_core_x_ex: int
    addr_core_y_ex: int
    addr_chip_x: int
    addr_chip_y: int

    def config_dump(self) -> Dict[str, Any]:
        dest_info = NeuronDestInfo.model_validate(self._asdict(), strict=True)

        return dest_info.model_dump(by_alias=True, exclude={*dest_info._exclude_vars})


class ConfigTemplate:
    """A configuration template."""

    frame_type: ClassVar[FrameType] = FrameType.FRAME_CONFIG


@dataclass(eq=False)
class NeuronConfig(ConfigTemplate):
    _extra_params = ("addr_offset",)
    """Extra parameters for debugging."""

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
        dest_core_coords: List[Coord],
        dest_chip_coord: Coord = _BACKEND_CONTEXT["output_chip_addr"],
    ):
        """Build the `NeuronConfig`.

        Args:
            - neuron: the target `NeuDyn`.
            - addr_ram: assigned RAM address of the target neuron.
            - addr_offset: offset of the RAM address.
            - axon_segs: the destination axon segments.
            - dest_core_coords: coordinates of the core of the destination axons.
            - dest_chip_coord: coordinate of the chip of the destination axons. \
                The default is `output_chip_addr` in the backend context.
        """
        attrs = NeuronAttrs.model_validate(neuron.export_params(), strict=True)
        dest_rid = get_replication_id(dest_core_coords)

        dest_info = NeuronDest(
            dest_core_coords,
            [coord.tick_relative for coord in axon_coords],
            [coord.addr_axon for coord in axon_coords],
            dest_core_coords[0].x,
            dest_core_coords[0].y,
            dest_rid.x,
            dest_rid.y,
            dest_chip_coord.x,
            dest_chip_coord.y,
        )

        neuron_dest_info = NeuronDestInfo.model_validate(
            dest_info._asdict(), strict=True
        )

        return cls(
            addr_ram,
            addr_offset,
            ParamsRAM(attrs=attrs, dest_info=neuron_dest_info),
        )

    def export(self) -> Dict[str, Any]:
        dict_ = {
            "addr_ram": self.addr_ram,
            "addr_offset": self.addr_offset,
            **self.params_ram.model_dump(by_alias=True),
        }

        return dict_

    def config_dump(self) -> Dict[str, Any]:
        """Dump the configs for debugging."""
        dict_ = self.params_ram.model_dump(
            by_alias=True,
            exclude={"dest_info": self.params_ram.dest_info._exclude_vars},
        )

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

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
        core_config: CoreConfig,
        neuron_ram: Dict[NeuDyn, NeuronConfig],
    ):
        return cls(
            coord,
            random_seed,
            weight_ram,
            ParamsReg.model_validate(core_config._asdict(), strict=True),
            neuron_ram,
        )

    def config_dump(self) -> Dict[str, Any]:
        dict_ = {
            "coord": self.coord.address,
            "random_seed": int(self.random_seed),
            "neuron_ram": dict(),
            **self.params_reg.model_dump(by_alias=True),
        }

        for neu, neu_config in self.neuron_ram.items():
            dict_["neuron_ram"][neu.name] = neu_config.config_dump()

        return dict_
