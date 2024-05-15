from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray
from paicorelib import (
    LCN_EX,
    AxonCoord,
    ChipCoord,
    Coord,
    CoordAddr,
    HwConfig,
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
from paicorelib.framelib import types
from paicorelib.framelib.frame_gen import OfflineFrameGen
from paicorelib.framelib.utils import np2bin, np2npy, np2txt
from typing_extensions import NotRequired, TypeAlias

from paibox.base import NeuDyn

from .context import _BACKEND_CONTEXT
from .graphs_types import NodeName

try:
    import orjson  # type: ignore

    _use_orjson = True

    def PAIConfigJsonDefault(o: Any):
        if isinstance(o, Coord):
            return o.address
        elif isinstance(o, NeuronDestInfo):
            return o.model_dump(by_alias=True)

        raise TypeError

except ModuleNotFoundError:
    import json

    _use_orjson = False

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


# Prevent import errors caused by changes in type definitions in paicorelib.
if hasattr(types, "FRAME_DTYPE"):
    FRAME_DTYPE = types.FRAME_DTYPE
else:
    FRAME_DTYPE = np.uint64

if hasattr(types, "FrameArrayType"):
    FrameArrayType = types.FrameArrayType
else:
    FrameArrayType = NDArray[FRAME_DTYPE]


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
        dict_ = self.export()

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


class NeuronConfig(NamedTuple):
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
        dest_chip_coord: Coord,
    ):
        """Build the `NeuronConfig`.

        Args:
            - neuron: the target `NeuDyn`.
            - addr_ram: assigned RAM address of the target neuron.
            - addr_offset: offset of the RAM address.
            - axon_segs: the destination axon segments.
            - dest_core_coords: coordinates of the core of the destination axons.
            - dest_chip_coord: coordinate of the chip of the destination axons.
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

    def export(self) -> Dict[str, Any]:
        dict_ = self.neuron_attrs.model_dump(
            by_alias=True,
            # exclude={"dest_info": self.params_ram.dest_info._exclude_vars},
        )
        dict_.update(self.neuron_dest_info.model_dump(by_alias=True))

        return dict_

    def __json__(self) -> Dict[str, Any]:
        """Dump the configs into json for debugging."""
        dict_ = self.export()

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_


class CorePlmConfig(NamedTuple):
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

    def export(self) -> Dict[str, Any]:
        dict_ = {
            "name": self.params_reg.name,
            "random_seed": self.random_seed,
            "neuron_rams": dict(),
            **self.params_reg.model_dump(by_alias=True),
        }

        for neu, neu_config in self.neuron_configs.items():
            dict_["neuron_rams"][neu.name] = neu_config.export()

        return dict_

    def __json__(self) -> Dict[str, Any]:
        """Dump the configs into json for debugging."""
        dict_ = self.export()

        for var in self._extra_params:
            dict_[var] = getattr(self, var)

        return dict_


class EmptyCorePlmConfig(CorePlmConfig):
    _default_seed: ClassVar[int] = 0
    _default_zero_wram: ClassVar[NDArray[np.uint64]] = np.zeros(
        (HwConfig.ADDR_RAM_MAX, 18), dtype=np.uint64
    )
    _default_neuron_conf = {}  # don't care

    @classmethod
    def encapsulate(cls, core_config: CoreConfig):
        return cls(
            cls._default_seed,
            cls._default_zero_wram,
            ParamsReg.model_validate(core_config._asdict(), strict=True),
            cls._default_neuron_conf,
        )


InputNodeConf: TypeAlias = Dict[NodeName, NeuronDest]
OutputDestConf: TypeAlias = Dict[NodeName, Dict[CoordAddr, NeuronDestInfo]]
CorePlmConfInChip: TypeAlias = Dict[Coord, CorePlmConfig]
CorePlmConf: TypeAlias = Dict[ChipCoord, CorePlmConfInChip]


class GraphInfo(TypedDict):
    """Information of compiled graph.

    TODO Optimize the data structure
    """

    input: InputNodeConf
    output: OutputDestConf
    members: CorePlmConf
    inherent_timestep: int
    n_core_required: int
    """The actual used cores."""
    n_core_occupied: int
    """The occupied cores, including used & wasted."""
    extras: NotRequired[Dict[str, Any]]


def gen_config_frames_by_coreconf(
    config_dict: CorePlmConf,
    write_to_file: bool,
    fp: Path,
    split_by_coord: bool,
    format: Literal["txt", "bin", "npy"],
) -> Dict[Coord, FrameArrayType]:
    """Generate configuration frames by given the `CorePlmConfig`.

    Args:
        - config_dict: the dictionary of configurations.
        - write_to_file: whether to write frames to file.
        - fp: If `write_to_file` is `True`, specify the path.
        - split_by_coord: whether to split the generated frames file by the core coordinates.
        - format: `txt`, `bin`, or `npy`.
    """

    def _write_to_f(name: str, array: FrameArrayType) -> None:
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

    for chip_coord, conf_in_chip in config_dict.items():
        for core_coord, v in conf_in_chip.items():
            # 1. Only one config frame type I for each physical core.
            config_frame_type1 = OfflineFrameGen.gen_config_frame1(
                chip_coord, core_coord, _default_rid, v.random_seed
            )

            # 2. Only one config frame type II for each physical core.
            config_frame_type2 = OfflineFrameGen.gen_config_frame2(
                chip_coord, core_coord, _default_rid, v.params_reg
            )

            # 3. Iterate all the neuron segments inside the physical core.
            config_frame_type3 = []
            for neu_conf in v.neuron_configs.values():
                config_frame_type3.append(
                    OfflineFrameGen.gen_config_frame3(
                        chip_coord,
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

            if config_frame_type3:
                frame3 = np.concatenate(
                    [f.value for f in config_frame_type3],
                    dtype=FRAME_DTYPE,
                    casting="no",
                )
            else:
                frame3 = np.asarray([], dtype=FRAME_DTYPE)

            # 4. Only one config frame type IV for each physical core.
            n_addr_write = v.params_reg.num_dendrite  # The number of address to write
            if n_addr_write > 0:
                config_frame_type4 = OfflineFrameGen.gen_config_frame4(
                    chip_coord,
                    core_coord,
                    _default_rid,
                    0,
                    18 * n_addr_write,
                    v.weight_ram[:n_addr_write],
                )
            else:
                config_frame_type4 = None

            _debug_dict[core_coord] = {
                "config1": config_frame_type1,
                "config2": config_frame_type2,
                "config3": config_frame_type3,
                "config4": config_frame_type4,
            }

            if config_frame_type4:
                frame_arrays_on_core[core_coord] = np.concatenate(
                    [
                        config_frame_type1.value,
                        config_frame_type2.value,
                        frame3,
                        config_frame_type4.value,
                    ],
                    dtype=FRAME_DTYPE,
                    casting="no",
                )
            else:
                frame_arrays_on_core[core_coord] = np.concatenate(
                    [config_frame_type1.value, config_frame_type2.value, frame3],
                    dtype=FRAME_DTYPE,
                    casting="no",
                )

    if write_to_file:
        if split_by_coord:
            for core_coord, f in frame_arrays_on_core.items():
                addr = core_coord.address
                _write_to_f(f"config_core{addr}", f)
        else:
            _f = np.concatenate(
                list(frame_arrays_on_core.values()), dtype=FRAME_DTYPE, casting="no"
            )
            _write_to_f(f"config_cores_all", _f)

    return frame_arrays_on_core


def export_core_params_json(core_conf: Dict[Coord, CoreConfig], fp: Path) -> None:
    _valid_conf = {str(k): v.export() for k, v in core_conf.items()}

    if _use_orjson:
        with open(fp / _BACKEND_CONTEXT["core_conf_json"], "wb") as f:
            f.write(
                orjson.dumps(
                    _valid_conf,
                    option=orjson.OPT_NON_STR_KEYS | orjson.OPT_INDENT_2,
                )
            )
    else:
        with open(fp / _BACKEND_CONTEXT["core_conf_json"], "w") as f:
            json.dump(
                _valid_conf, f, ensure_ascii=True, indent=2, cls=PAIConfigJsonEncoder
            )


def export_input_conf_json(input_conf_info: InputNodeConf, fp: Path) -> None:
    _valid_conf = {k: v.export() for k, v in input_conf_info.items()}

    if _use_orjson:
        with open(fp / _BACKEND_CONTEXT["input_conf_json"], "wb") as f:
            f.write(orjson.dumps(_valid_conf, option=orjson.OPT_INDENT_2))
    else:
        with open(fp / _BACKEND_CONTEXT["input_conf_json"], "w") as f:
            json.dump(_valid_conf, f, indent=2, cls=PAIConfigJsonEncoder)


def export_output_conf_json(output_conf_info: OutputDestConf, fp: Path) -> None:
    if _use_orjson:
        with open(fp / _BACKEND_CONTEXT["output_conf_json"], "wb") as f:
            f.write(
                orjson.dumps(
                    output_conf_info,
                    default=PAIConfigJsonDefault,
                    option=orjson.OPT_NON_STR_KEYS | orjson.OPT_INDENT_2,
                )
            )
    else:
        with open(fp / _BACKEND_CONTEXT["output_conf_json"], "w") as f:
            json.dump(output_conf_info, f, indent=2, cls=PAIConfigJsonEncoder)


def export_neuconf_json(neuron_conf: Dict[NeuDyn, NeuronConfig], full_fp: Path) -> None:
    _valid_conf = {k.name: v.export() for k, v in neuron_conf.items()}

    if _use_orjson:
        with open(full_fp, "wb") as f:
            f.write(orjson.dumps(_valid_conf, option=orjson.OPT_INDENT_2))
    else:
        with open(full_fp, "w") as f:
            json.dump(_valid_conf, f, indent=2, cls=PAIConfigJsonEncoder)


def export_core_plm_conf_json(core_plm_conf: CorePlmConf, full_fp: Path) -> None:
    _valid_conf = {}

    for chip_coord, cconf in core_plm_conf.items():
        _valid_conf[str(chip_coord)] = {}
        for core_coord, conf in cconf.items():
            _valid_conf[str(chip_coord)][str(core_coord)] = conf.export()

    if _use_orjson:
        with open(full_fp, "wb") as f:
            f.write(
                orjson.dumps(
                    _valid_conf, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_INDENT_2
                )
            )
    else:
        with open(full_fp, "w") as f:
            json.dump(_valid_conf, f, indent=2, cls=PAIConfigJsonEncoder)
