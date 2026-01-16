import sys
from abc import abstractmethod
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray
from paicorelib import (
    LCN_EX,
    ChipCoord,
    Coord,
    CoreReg,
    DecayRandomEnable,
    InputWidthFormat,
    LeakOrder,
    MaxPoolingEnable,
    NeuAttrs,
    NeuDestInfo,
    OfflineCoreReg,
    OfflineNeuAttrs,
    OfflineNeuConf,
    OfflineNeuDestInfo,
    OnlineCoreReg,
    OnlineModeEnable,
    OnlineNeuAttrs,
    OnlineNeuConf,
    OnlineNeuDestInfo,
    SNNModeEnable,
    SpikeWidthFormat,
    WeightWidth,
)
from paicorelib import ReplicationId as RId
from paicorelib.framelib.types import LUTDataType

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

from paibox.base import DataFlowFormat
from paibox.components import Neuron

from .types import AxonCoord, DendriteSegment, NeuSegAddr, NodeName, WRAMPackedType

try:
    import orjson

    _USE_ORJSON = True

    def PAIConfigJsonDefault(o: Any) -> Any:
        if isinstance(o, Coord):
            return str(o)
        elif isinstance(o, OfflineNeuAttrs):
            return o.model_dump(by_alias=True)
        elif isinstance(o, OfflineNeuDestInfo):
            return o.model_dump(by_alias=True)
        elif isinstance(o, OnlineNeuAttrs):
            return o.model_dump(by_alias=True)
        elif isinstance(o, OnlineNeuDestInfo):
            return o.model_dump(by_alias=True)

        raise TypeError(f"type {type(o).__name__} not defined in custom Json encoder.")

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
            elif isinstance(o, OfflineNeuAttrs):
                return o.model_dump_json(indent=2, by_alias=True)
            elif isinstance(o, OfflineNeuDestInfo):
                return o.model_dump(by_alias=True)

            return super().default(o)


# Prevent import errors caused by changes in type definitions in paicorelib.
from paicorelib import framelib

FRAME_DTYPE = framelib.types.FRAME_DTYPE
FrameArrayType = NDArray[FRAME_DTYPE]


def asdict_shallow(obj):
    if not is_dataclass(obj):
        raise TypeError("asdict_shallow() should be called on dataclass instances")
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


@dataclass(frozen=True)
class CoreConfig:
    """Configurations of core."""

    _extra_params = ("name",)
    """Extra parameters for debugging."""

    name: str

    def _asdict(self) -> dict[str, Any]:
        return asdict_shallow(self)

    def export(self) -> CoreReg:
        raise NotImplementedError("Subclasses must implement export method.")

    def to_json(self) -> dict[str, Any]:
        """Dump the configs into json for debugging."""
        dict_ = self.export().model_dump(by_alias=True)

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_


@dataclass(frozen=True)
class OfflineCoreConfig(CoreConfig):
    """Configurations of offline core."""

    weight_width: WeightWidth
    lcn: LCN_EX
    input_width: InputWidthFormat
    spike_width: SpikeWidthFormat
    num_dendrite: int
    max_pooling_en: MaxPoolingEnable
    tick_wait_start: int
    tick_wait_end: int
    snn_en: SNNModeEnable
    target_lcn: LCN_EX
    test_chip_addr: ChipCoord

    def export(self) -> OfflineCoreReg:
        return OfflineCoreReg.model_validate(self._asdict(), strict=True)


@dataclass(frozen=True)
class OnlineCoreConfig(CoreConfig):
    """Configurations of online core."""

    weight_width: WeightWidth
    lcn: LCN_EX
    lateral_inhi_value: int
    weight_decay_value: int
    upper_weight: int
    lower_weight: int
    neuron_start: int
    neuron_end: int
    inhi_core_x_ex: int
    inhi_core_y_ex: int
    tick_wait_start: int
    tick_wait_end: int
    lut_random_en: NDArray[np.uint8]
    decay_random_en: DecayRandomEnable
    leak_order: LeakOrder
    online_mode_en: OnlineModeEnable
    test_chip_addr: ChipCoord
    random_seed: int

    def export(self) -> OnlineCoreReg:
        return OnlineCoreReg.model_validate(self._asdict(), strict=True)


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

    def export(self) -> NeuDestInfo:
        raise NotImplementedError("Subclasses must implement export method.")

    def to_json(self) -> dict[str, Any]:
        """Dump the configs into json for debugging."""
        dict_ = self.export().model_dump(by_alias=True)

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_


@dataclass(frozen=True)
class OfflineNeuronDest(NeuronDest):
    def export(self) -> OfflineNeuDestInfo:
        return OfflineNeuDestInfo.model_validate(asdict(self), strict=True)


# Online version neuron destination
@dataclass(frozen=True)
class OnlineNeuronDest(NeuronDest):
    def export(self) -> OnlineNeuDestInfo:
        return OnlineNeuDestInfo.model_validate(asdict(self), strict=True)


@dataclass(frozen=True)
class InputNeuronDest(NeuronDest):
    lcn: int

    def to_json(self) -> dict[str, Any]:
        dict_ = super().to_json()
        dict_ |= {"lcn": self.lcn}

        return dict_

    def export(self) -> OfflineNeuDestInfo:
        return OfflineNeuDestInfo.model_validate(asdict(self), strict=True)


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
class NeuConfig:
    """Configuration of neuron."""

    neu_seg: DendriteSegment
    """Neuron segment."""
    axon_coords: list[AxonCoord]
    """The destination axon segments."""
    base_coord: Coord
    """Coordinates of the core of the destination axons."""
    dest_rid: RId
    """Replication ID of the core of the destination axons."""
    dest_chip_coord: Coord
    """Coordinate of the chip of the destination axons."""

    @abstractmethod
    def __getitem__(self, s: slice) -> "NeuConfig":
        pass

    @abstractmethod
    def export(self) -> OfflineNeuConf | OnlineNeuConf:
        """Export the neuron configuration."""
        pass

    @abstractmethod
    def to_json(self) -> str | bytes:
        """Dump the configs into json for debugging."""
        pass

    @property
    @abstractmethod
    def neuron_attrs(self) -> NeuAttrs:
        """Return the neuron attributes."""
        pass

    @property
    @abstractmethod
    def neuron_dest_info(self) -> NeuDestInfo:
        """Return the neuron destination information."""
        pass


@dataclass(frozen=True)
class OfflineNeuConfig(NeuConfig):
    def __getitem__(self, s: slice) -> "OfflineNeuConfig":
        return OfflineNeuConfig(
            self.neu_seg[s],
            self.axon_coords[s],
            self.base_coord,
            self.dest_rid,
            self.dest_chip_coord,
        )

    def export(self) -> OfflineNeuConf:
        return OfflineNeuConf(attrs=self.neuron_attrs, dest_info=self.neuron_dest_info)

    def to_json(self) -> str | bytes:
        """Dump the configs into json for debugging."""
        dict_ = {
            "n_neuron": self.neu_seg.n_neuron,
            "addr_offset": self.neu_seg.offset,
            "addr_occupied": self.neu_seg.occupied_addr,
        }
        dict_ |= self.export().model_dump(by_alias=True)

        if _USE_ORJSON:
            return orjson.dumps(
                dict_, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
            )
        else:
            return json.dumps(dict_, cls=PAIConfigJsonEncoder, indent=2)

    @property
    def neuron_attrs(self) -> OfflineNeuAttrs:
        return OfflineNeuAttrs.model_validate(self.neu_seg.attrs, strict=True)

    @property
    def neuron_dest_info(self) -> OfflineNeuDestInfo:
        dest_info = OfflineNeuronDest(
            [coord.tick_relative for coord in self.axon_coords],
            [coord.addr_axon for coord in self.axon_coords],
            self.base_coord.x,
            self.base_coord.y,
            self.dest_rid.x,
            self.dest_rid.y,
            self.dest_chip_coord.x,
            self.dest_chip_coord.y,
        )
        return OfflineNeuDestInfo.model_validate(asdict(dest_info), strict=True)


@dataclass(frozen=True)
class OnlineNeuConfig(NeuConfig):
    weight_width: WeightWidth

    def __getitem__(self, s: slice) -> "OnlineNeuConfig":
        return OnlineNeuConfig(
            self.neu_seg[s],
            self.axon_coords[s],
            self.base_coord,
            self.dest_rid,
            self.dest_chip_coord,
            self.weight_width,
        )

    def export(self) -> OnlineNeuConf:
        return OnlineNeuConf(attrs=self.neuron_attrs, dest_info=self.neuron_dest_info)

    def to_json(self) -> str | bytes:
        """Dump the configs into json for debugging."""
        dict_ = {
            "n_neuron": self.neu_seg.n_neuron,
            "addr_offset": self.neu_seg.offset,
            "addr_occupied": self.neu_seg.occupied_addr,
        }
        dict_ |= self.export().model_dump(by_alias=True)

        if _USE_ORJSON:
            return orjson.dumps(
                dict_, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
            )
        else:
            return json.dumps(dict_, cls=PAIConfigJsonEncoder, indent=2)

    @property
    def neuron_attrs(self) -> OnlineNeuAttrs:
        return OnlineNeuAttrs.model_validate(
            self.neu_seg.attrs, strict=True, context={"weight_width": self.weight_width}
        )

    @property
    def neuron_dest_info(self) -> OnlineNeuDestInfo:
        dest_info = OnlineNeuronDest(
            [coord.tick_relative for coord in self.axon_coords],
            [coord.addr_axon for coord in self.axon_coords],
            self.base_coord.x,
            self.base_coord.y,
            self.dest_rid.x,
            self.dest_rid.y,
            self.dest_chip_coord.x,
            self.dest_chip_coord.y,
        )
        return OnlineNeuDestInfo.model_validate(asdict(dest_info), strict=True)


@dataclass(frozen=True)
class CorePlmConfig:
    _extra_params = ()
    """Extra parameters for debugging."""

    def export(self) -> dict[str, Any]:
        """Export the core PLM configuration."""
        raise NotImplementedError("Subclasses must implement export method.")

    def to_json(self) -> dict[str, Any]:
        """Dump the configs into json for debugging."""
        dict_ = self.export()

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_


@dataclass(frozen=True)
class OfflineCorePlmConfig(CorePlmConfig):
    random_seed: int
    weight_ram: WRAMPackedType
    params_reg: OfflineCoreReg
    neuron_configs: dict[Neuron, OfflineNeuConfig]

    @classmethod
    def encapsulate(
        cls,
        random_seed: int,
        weight_ram: WRAMPackedType,
        core_cfg: OfflineCoreConfig,
        neuron_cfg: dict[Neuron, OfflineNeuConfig],
    ):
        return cls(
            random_seed,
            weight_ram,
            OfflineCoreReg.model_validate(core_cfg._asdict(), strict=True),
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


@dataclass(frozen=True)
class OnlineCorePlmConfig(CorePlmConfig):
    weight_ram: WRAMPackedType
    core_params: OnlineCoreReg
    lut: LUTDataType
    neuron_configs: dict[Neuron, OnlineNeuConfig]

    @classmethod
    def encapsulate(
        cls,
        weight_ram: WRAMPackedType,
        core_cfg: OnlineCoreConfig,
        lut: LUTDataType,
        neuron_cfg: dict[Neuron, OnlineNeuConfig],
    ):
        return cls(
            weight_ram,
            OnlineCoreReg.model_validate(core_cfg._asdict(), strict=True),
            lut,
            neuron_cfg,
        )

    def export(self) -> dict[str, Any]:
        dict_ = {
            "name": self.core_params.name,
            "neuron_rams": dict(),
            **self.core_params.model_dump(by_alias=True),
        }

        for neu, neu_cfg in self.neuron_configs.items():
            if _USE_ORJSON:
                dict_["neuron_rams"][neu.name] = orjson.loads(neu_cfg.to_json())
            else:
                dict_["neuron_rams"][neu.name] = json.loads(neu_cfg.to_json())

        return dict_


InputNodeConf = dict[NodeName, InputNeuronDest]
OutputDestConf = dict[NodeName, dict[Coord, NeuDestInfo]]
CorePlmConfInChip = dict[Coord, CorePlmConfig]
CorePlmConf = dict[ChipCoord, CorePlmConfInChip]
CoreConfInChip = dict[Coord, CoreConfig]
CoreConf = dict[ChipCoord, CoreConfInChip]

# Only one segment of a neuron is placed on a core
NeuPhyLocChipLoc = dict[Coord, NeuSegAddr]
NeuPhyLoc = dict[ChipCoord, NeuPhyLocChipLoc]
NeuPhyLocMap = dict[NodeName, NeuPhyLoc]


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
