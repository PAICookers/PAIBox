import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray
from paicorelib import (
    LCN_EX,
    AxonCoord,
    Coord,
    InputWidthFormat,
    MaxPoolingEnable,
    NeuronAttrs,
    NeuronDestInfo,
    ParamsReg,
)
from paicorelib import ReplicationId as RId
from paicorelib import (
    SNNModeEnable,
    SpikeWidthFormat,
    WeightPrecision,
    get_replication_id,
)
from paicorelib.framelib._types import FRAME_DTYPE, FrameArrayType
from paicorelib.framelib.frame_gen import OfflineFrameGen
from paicorelib.framelib.utils import np2bin, np2npy, np2txt
from typing_extensions import NotRequired, TypeAlias

from paibox.base import NeuDyn

from .context import _BACKEND_CONTEXT
from .graphs_types import NodeName


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
    """Information of neuron destination (axon address information)."""

    _extra_params = ("tick_relative", "addr_axon")
    """Extra parameters for debugging."""

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


class ConfigTemplate:
    """A configuration template."""

    pass


@dataclass(eq=False)
class NeuronConfig(ConfigTemplate):
    _extra_params = (
        "n_neuron",
        "addr_ram",
        "addr_offset",
    )
    """Extra parameters for debugging."""

    n_neuron: int
    addr_ram: List[int]
    """RAM Address of neurons"""
    addr_offset: int
    "RAM starting address(offset)"
    neuron_attrs: NeuronAttrs
    neuron_dest_info: NeuronDestInfo

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
        )

    def __json__(self) -> Dict[str, Any]:
        """Dump the configs into json for debugging."""
        dict_ = self.neuron_attrs.model_dump(
            by_alias=True,
            # exclude={"dest_info": self.params_ram.dest_info._exclude_vars},
        )

        dict_.update(
            self.neuron_dest_info.model_dump(by_alias=True)
        )  # compatible for py3.8

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


InputNodeInfo: TypeAlias = Dict[NodeName, NeuronDest]
OutputDestInfo: TypeAlias = Dict[NodeName, Dict[int, NeuronDestInfo]]
CorePlacementInfo: TypeAlias = Dict[Coord, CorePlacementConfig]


class GraphInfo(TypedDict):
    """Information of compiled graph.

    TODO Optimize the data structure
    """

    input: InputNodeInfo
    output: OutputDestInfo
    members: CorePlacementInfo
    inherent_timestep: int
    n_core_required: int
    """The actual used cores."""
    n_core_occupied: int
    """The occupied cores, including used & wasted."""
    extras: NotRequired[Dict[str, Any]]


def gen_config_frames_by_coreconf(
    config_dict: Dict[Coord, CorePlacementConfig],
    target_chip_coord: Coord,
    write_to_file: bool,
    fp: Path,
    split_by_coord: bool,
    format: Literal["txt", "bin", "npy"] = "bin",
) -> Dict[Coord, FrameArrayType]:
    """Generate configuration frames by given the `CorePlacementConfig`.

    Args:
        - config_dict: the dictionary of configurations.
        - target_chip_coord: local chip coordinate.
        - write_to_file: whether to write frames to file.
        - fp: If `write_to_file` is `True`, specify the path.
        - split_by_coord: whether to split the generated frames file by the core coordinates.
        - format: it can be `txt`, `bin`, or `npy`. `bin` & `npy` are recommended.
    """

    def _write_to_f(name: str, array: np.ndarray) -> None:
        nonlocal fp, format

        _fp = fp / (name + f".{format}")
        if format == "npy":
            np2npy(_fp, array)
        elif format == "bin":
            np2bin(_fp, array)
        else:
            np2txt(_fp, array)

    _default_rid = RId(0, 0)
    _debug_dict: Dict[Coord, Dict[str, Any]] = dict()
    frame_arrays_on_core: Dict[Coord, FrameArrayType] = dict()

    for core_coord, v in config_dict.items():
        # 1. Only one config frame type I for each physical core.
        config_frame_type1 = OfflineFrameGen.gen_config_frame1(
            target_chip_coord,
            core_coord,
            _default_rid,
            v.random_seed,
        )

        # 2. Only one config frame type II for each physical core.
        config_frame_type2 = OfflineFrameGen.gen_config_frame2(
            target_chip_coord,
            core_coord,
            _default_rid,
            v.params_reg,
        )

        # 3. Iterate all the neuron segments in the function inside
        config_frame_type3 = []
        for neu_conf in v.neuron_configs.values():
            config_frame_type3.append(
                OfflineFrameGen.gen_config_frame3(
                    target_chip_coord,
                    core_coord,
                    _default_rid,
                    neu_conf.addr_offset,
                    neu_conf.n_neuron,
                    neu_conf.neuron_attrs,
                    neu_conf.neuron_dest_info,
                    lcn_ex=v.params_reg.lcn_extension,
                    weight_precision=v.params_reg.weight_precision,
                )
            )

        frame3 = np.concatenate(
            [f.value for f in config_frame_type3], dtype=FRAME_DTYPE
        )

        # 4. Only one config frame type IV for each physical core.
        n_addr_write = v.params_reg.num_dendrite  # The number of address to write
        config_frame_type4 = OfflineFrameGen.gen_config_frame4(
            target_chip_coord,
            core_coord,
            _default_rid,
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
        if split_by_coord:
            for core_coord, f in frame_arrays_on_core.items():
                addr = core_coord.address
                _write_to_f(f"config_core{addr}", f)
        else:
            _f = np.concatenate(list(frame_arrays_on_core.values()), dtype=FRAME_DTYPE)
            _write_to_f(f"config_cores_all", _f)

    return frame_arrays_on_core


class PAIConfigJsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Coord):
            return o.address
        elif isinstance(o, Enum):
            return o.value
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, NeuronDestInfo):
            return o.model_dump(by_alias=True)
        else:
            return super().default(o)


DEFAULT_CORE_PARAMS_CONF_JSON = "core_params.json"
DEFAULT_INPUT_NODES_CONF_JSON = "input_proj_info.json"
DEFAULT_OUTPUT_DESTS_CONF_JSON = "output_dest_info.json"


def export_core_params_json(core_conf: Dict[Coord, CoreConfig], fp: Path) -> None:
    _valid_conf = {k.address: v.__json__() for k, v in core_conf.items()}

    with open(fp / DEFAULT_CORE_PARAMS_CONF_JSON, "w") as f:
        json.dump(_valid_conf, f, ensure_ascii=True, indent=4, cls=PAIConfigJsonEncoder)


def export_inp_nodes_conf_json(inp_nodes_info: InputNodeInfo, fp: Path) -> None:
    _valid_conf = {k: v.__json__() for k, v in inp_nodes_info.items()}

    with open(fp / DEFAULT_INPUT_NODES_CONF_JSON, "w") as f:
        json.dump(_valid_conf, f, ensure_ascii=True, indent=4, cls=PAIConfigJsonEncoder)


def export_outp_dests_conf_json(outp_dests_info: OutputDestInfo, fp: Path) -> None:
    with open(fp / DEFAULT_OUTPUT_DESTS_CONF_JSON, "w") as f:
        json.dump(
            outp_dests_info, f, ensure_ascii=True, indent=4, cls=PAIConfigJsonEncoder
        )
