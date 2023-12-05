from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple

import numpy as np
from numpy.typing import NDArray

from paibox.base import NeuDyn
from paibox.libpaicore import (
    LCN_EX,
    AxonCoord,
    Coord,
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

    def __json__(self) -> Dict[str, Any]:
        """Dump the configs into json for debugging."""
        dict_ = self.export()

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_


class NeuronDest(NamedTuple):
    """Information of neuron destination(Axon address information)."""

    _extra_params = ("dest_coords", "tick_relative", "addr_axon")
    """Extra parameters for debugging."""

    dest_coords: List[Coord]
    tick_relative: List[int]
    addr_axon: List[int]
    addr_core_x: int
    addr_core_y: int
    addr_core_x_ex: int
    addr_core_y_ex: int
    addr_chip_x: int
    addr_chip_y: int

    def export(self) -> Dict[str, Any]:
        dest_info = NeuronDestInfo.model_validate(self._asdict(), strict=True)
        dict_ = dest_info.model_dump(by_alias=True)

        return dict_

    def __json__(self) -> Dict[str, Any]:
        """Dump the configs into json for debugging."""
        dest_info = NeuronDestInfo.model_validate(self._asdict(), strict=True)
        dict_ = dest_info.model_dump(by_alias=True)

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_


class ConfigTemplate:
    """A configuration template."""

    pass


@dataclass(eq=False)
class NeuronConfig(ConfigTemplate):
    _extra_params = ("n_neuron", "addr_ram", "addr_offset")
    """Extra parameters for debugging."""

    n_neuron: int
    addr_ram: List[int]
    """RAM Address of neurons"""
    addr_offset: int
    "RAM starting address(offset)"
    neuron_attrs: NeuronAttrs
    neuron_dest_info: NeuronDestInfo

    tick_relative: List[int]
    addr_axon: List[int]

    @classmethod
    def encapsulate(
        cls,
        neuron: NeuDyn,
        n_neuron: int,
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
            n_neuron,
            addr_ram,
            addr_offset,
            attrs,
            neuron_dest_info,
            dest_info.tick_relative,
            dest_info.addr_axon,
        )

    def __json__(self) -> Dict[str, Any]:
        """Dump the configs into json for debugging."""
        dict_ = self.neuron_attrs.model_dump(
            by_alias=True,
            # exclude={"dest_info": self.params_ram.dest_info._exclude_vars},
        )

        dict_ |= self.neuron_dest_info.model_dump(by_alias=True)

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_


@dataclass(eq=False)
class CorePlacementConfig(ConfigTemplate):
    _extra_params = ()
    """Extra parameters for debugging."""

    random_seed: int
    weight_ram: NDArray[np.uint64]
    params_reg: ParamsReg
    neuron_configs: Dict[NeuDyn, NeuronConfig]

    @classmethod
    def encapsulate(
        cls,
        random_seed: int,
        weight_ram: NDArray[np.uint64],
        core_config: CoreConfig,
        neuron_configs: Dict[NeuDyn, NeuronConfig],
    ):
        return cls(
            random_seed,
            weight_ram,
            ParamsReg.model_validate(core_config._asdict(), strict=True),
            neuron_configs,
        )

    def __json__(self) -> Dict[str, Any]:
        """Dump the configs into json for debugging."""
        dict_ = {
            "name": self.params_reg.name,
            "random_seed": self.random_seed,
            "neuron_rams": dict(),
            **self.params_reg.model_dump(by_alias=True),
        }

        for neu, neu_config in self.neuron_configs.items():
            dict_["neuron_rams"][neu.name] = neu_config.__json__()

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_
