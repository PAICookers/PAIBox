import sys
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, NamedTuple, TypedDict, Union

import numpy as np
from numpy.typing import NDArray
from paicorelib import (
    LCN_EX,
    ChipCoord,
    Coord,
    InputWidthFormat,
    MaxPoolingEnable,
    NeuronAttrs,
    NeuronConf,
    NeuronDestInfo,
    ParamsReg,
    SNNModeEnable,
    SpikeWidthFormat,
    WeightWidth,
    get_replication_id,
)

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

from paibox.base import DataFlowFormat
from paibox.components import Neuron

from .types import AxonCoord, NeuSegment, NeuSegRAMAddr, NodeName, WRAMPackedType

try:
    import orjson

    _USE_ORJSON = True

    def PAIConfigJsonDefault(o: Any) -> Any:
        if isinstance(o, Coord):
            return str(o)
        elif isinstance(o, NeuronAttrs):
            return o.model_dump(by_alias=True)
        elif isinstance(o, NeuronDestInfo):
            return o.model_dump(by_alias=True)

        raise TypeError(f"type {type(o)} not defined in custom Json encoder.")

except ModuleNotFoundError:
    import json

    _USE_ORJSON = False

    class PAIConfigJsonEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, Coord):
                return str(o)
            elif is_dataclass(o):
                return asdict(o)  # type: ignore
            elif isinstance(o, Enum):
                return o.value
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, NeuronAttrs):
                return o.model_dump_json(indent=2, by_alias=True)
            elif isinstance(o, NeuronDestInfo):
                return o.model_dump(by_alias=True)

            return super().default(o)


# Prevent import errors caused by changes in type definitions in paicorelib.
from paicorelib import framelib

if hasattr(framelib.types, "FRAME_DTYPE"):
    FRAME_DTYPE = framelib.types.FRAME_DTYPE
else:
    FRAME_DTYPE = np.uint64

if hasattr(framelib.types, "FrameArrayType"):
    FrameArrayType = framelib.types.FrameArrayType
else:
    FrameArrayType = NDArray[FRAME_DTYPE]


class CoreConfig(NamedTuple):
    """Configurations of core."""

    _extra_params = ("name",)
    """Extra parameters for debugging."""

    name: str
    weight_width: WeightWidth
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

    def export(self) -> ParamsReg:
        return ParamsReg.model_validate(self._asdict(), strict=True)

    def to_json(self) -> dict[str, Any]:
        """Dump the configs into json for debugging."""
        dict_ = self.export().model_dump(by_alias=True)

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_


@dataclass(frozen=True)
class NeuronDest:
    """Information of neuron destination (axon address information)."""

    _extra_params = ("tick_relative", "addr_axon")
    """Extra parameters for debugging."""

    tick_relative: list[int]
    addr_axon: list[int]
    addr_core_x: int
    addr_core_y: int
    addr_core_x_ex: int
    addr_core_y_ex: int
    addr_chip_x: int
    addr_chip_y: int

    def export(self) -> NeuronDestInfo:
        return NeuronDestInfo.model_validate(asdict(self), strict=True)

    def to_json(self) -> dict[str, Any]:
        """Dump the configs into json for debugging."""
        dict_ = self.export().model_dump(by_alias=True)

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_


@dataclass(frozen=True)
class InputNeuronDest(NeuronDest):
    lcn: int

    def to_json(self) -> dict[str, Any]:
        dict_ = super().to_json()
        dict_ |= {"lcn": self.lcn}

        return dict_


class OutputNeuronDest(NamedTuple):
    # TODO
    addr_core_x: int
    addr_core_y: int
    addr_core_x_ex: int
    addr_core_y_ex: int
    addr_chip_x: int
    addr_chip_y: int
    start: AxonCoord
    end: AxonCoord


@dataclass(frozen=True)
class NeuronConfig:
    """Extra parameters for debugging."""

    neu_seg: NeuSegment
    """Neuron segment."""
    axon_coords: list[AxonCoord]
    """The destination axon segments."""
    dest_core_coords: list[Coord]
    """Coordinates of the core of the destination axons."""
    dest_chip_coord: Coord
    """Coordinate of the chip of the destination axons."""

    def __getitem__(self, s: slice) -> "NeuronConfig":
        return NeuronConfig(
            self.neu_seg[s],
            self.axon_coords[s],
            self.dest_core_coords,
            self.dest_chip_coord,
        )

    def export(self) -> NeuronConf:
        return NeuronConf(attrs=self.neuron_attrs, dest_info=self.neuron_dest_info)

    def to_json(self) -> Union[str, bytes]:
        """Dump the configs into json for debugging."""
        dict_ = {
            "n_neuron": self.neu_seg.n_neuron,
            "addr_offset": self.neu_seg.offset,
            "addr_ram": self.neu_seg.addr_ram,
        }
        dict_ |= self.export().model_dump(by_alias=True)

        if _USE_ORJSON:
            return orjson.dumps(
                dict_, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
            )
        else:
            return json.dumps(dict_, indent=2, cls=PAIConfigJsonEncoder)

    @property
    def neuron_attrs(self) -> NeuronAttrs:
        return NeuronAttrs.model_validate(self.neu_seg.attrs, strict=True)

    @property
    def neuron_dest_info(self) -> NeuronDestInfo:
        base_coord, dest_rid = get_replication_id(self.dest_core_coords)
        dest_info = NeuronDest(
            [coord.tick_relative for coord in self.axon_coords],
            [coord.addr_axon for coord in self.axon_coords],
            base_coord.x,
            base_coord.y,
            dest_rid.x,
            dest_rid.y,
            self.dest_chip_coord.x,
            self.dest_chip_coord.y,
        )
        return NeuronDestInfo.model_validate(asdict(dest_info), strict=True)


class CorePlmConfig(NamedTuple):
    _extra_params = ()
    """Extra parameters for debugging."""

    random_seed: int
    weight_ram: WRAMPackedType
    params_reg: ParamsReg
    neuron_configs: dict[Neuron, NeuronConfig]

    @classmethod
    def encapsulate(
        cls,
        random_seed: int,
        weight_ram: WRAMPackedType,
        core_cfg: CoreConfig,
        neuron_cfg: dict[Neuron, NeuronConfig],
    ):
        return cls(
            random_seed,
            weight_ram,
            ParamsReg.model_validate(core_cfg._asdict(), strict=True),
            neuron_cfg,
        )

    def export(self) -> dict[str, Any]:
        dict_ = {
            "name": self.params_reg.name,
            "random_seed": self.random_seed,
            "neuron_rams": dict(),
            **self.params_reg.model_dump(by_alias=True),
        }

        for neu, neu_cfg in self.neuron_configs.items():
            if _USE_ORJSON:
                dict_["neuron_rams"][neu.name] = orjson.loads(neu_cfg.to_json())
            else:
                dict_["neuron_rams"][neu.name] = json.loads(neu_cfg.to_json())

        return dict_

    def to_json(self) -> dict[str, Any]:
        """Dump the configs into json for debugging."""
        dict_ = self.export()

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_


InputNodeConf: TypeAlias = dict[NodeName, InputNeuronDest]
OutputDestConf: TypeAlias = dict[NodeName, dict[Coord, NeuronDestInfo]]
CorePlmConfInChip: TypeAlias = dict[Coord, CorePlmConfig]
CorePlmConf: TypeAlias = dict[ChipCoord, CorePlmConfInChip]
CoreConfInChip: TypeAlias = dict[Coord, CoreConfig]
CoreConf: TypeAlias = dict[ChipCoord, CoreConfInChip]

# Only one segment of a neuron is placed on a core
NeuPhyLocChipLoc: TypeAlias = dict[Coord, NeuSegRAMAddr]
NeuPhyLoc: TypeAlias = dict[ChipCoord, NeuPhyLocChipLoc]
NeuPhyLocMap: TypeAlias = dict[NodeName, NeuPhyLoc]


class _ExportedGraphInfo(TypedDict):
    name: str
    """Name of the graph."""
    inherent_timestep: int  # TODO this attibute will be deprecated.
    output_flow_format: dict[NodeName, DataFlowFormat]
    n_core_required: int
    """The actual used cores."""
    n_core_occupied: int
    """The occupied cores, including used & wasted."""
    misc: NotRequired[dict[str, Any]]
    """Miscellaneous information. Not required."""


class GraphInfo(_ExportedGraphInfo):
    """Information of graph after compilation."""

    input: InputNodeConf
    output: OutputDestConf
    members: CorePlmConf


def _gh_info2exported_gh_info(gh_info: GraphInfo) -> _ExportedGraphInfo:
    return _ExportedGraphInfo(
        name=gh_info["name"],
        inherent_timestep=gh_info["inherent_timestep"],
        output_flow_format=gh_info["output_flow_format"],
        n_core_required=gh_info["n_core_required"],
        n_core_occupied=gh_info["n_core_occupied"],
    )
