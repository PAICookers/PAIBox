import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TypedDict, Union

import numpy as np
from numpy.typing import NDArray
from paicorelib import (
    LCN_EX,
    ChipCoord,
    Coord,
    CoordAddr,
    HwConfig,
    InputWidthFormat,
    MaxPoolingEnable,
    NeuronAttrs,
    NeuronConf,
    NeuronDestInfo,
    ParamsReg,
)
from paicorelib import ReplicationId as RId
from paicorelib import (
    RoutingCoord,
    SNNModeEnable,
    SpikeWidthFormat,
    WeightWidth,
    get_replication_id,
)
from paicorelib.framelib import OfflineFrameGen
from paicorelib.framelib import types as flib_types
from paicorelib.framelib.utils import _mask, np2bin, np2npy, np2txt

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from paibox.components import Neuron
from paibox.utils import reverse_8bit

from .context import _BACKEND_CONTEXT
from .types import AxonCoord, NeuSegment, NodeName, WRAMPackedType

try:
    import orjson

    _USE_ORJSON = True

    def PAIConfigJsonDefault(o: Any):
        if isinstance(o, Coord):
            return o.address
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
                return o.address
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
if hasattr(flib_types, "FRAME_DTYPE"):
    FRAME_DTYPE = flib_types.FRAME_DTYPE
else:
    FRAME_DTYPE = np.uint64

if hasattr(flib_types, "FrameArrayType"):
    FrameArrayType = flib_types.FrameArrayType
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
        dict_ = super().export().model_dump(by_alias=True)
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


class NeuronConfig(NamedTuple):
    _extra_params = (
        "n_neuron",
        "addr_ram",
        "addr_offset",
    )
    """Extra parameters for debugging."""

    n_neuron: int
    addr_ram: list[int]
    """RAM Address of neurons"""
    addr_offset: int
    "RAM starting address(offset)"
    neuron_attrs: NeuronAttrs
    neuron_dest_info: NeuronDestInfo

    @classmethod
    def encapsulate(
        cls,
        neu_seg: NeuSegment,
        axon_coords: list[AxonCoord],
        dest_core_coords: list[Coord],
        dest_chip_coord: Coord,
    ):
        """Build the `NeuronConfig`.

        Args:
            - neu_seg: neuron segment.
            - axon_segs: the destination axon segments.
            - dest_core_coords: coordinates of the core of the destination axons.
            - dest_chip_coord: coordinate of the chip of the destination axons.
        """
        attrs = NeuronAttrs.model_validate(
            neu_seg.target._slice_attrs(neu_seg.index), strict=True
        )
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

        neuron_dest_info = NeuronDestInfo.model_validate(asdict(dest_info), strict=True)

        return cls(
            neu_seg.n_neuron, neu_seg.addr_ram, neu_seg.offset, attrs, neuron_dest_info
        )

    def export(self) -> NeuronConf:
        return NeuronConf(attrs=self.neuron_attrs, dest_info=self.neuron_dest_info)

    def to_json(self) -> Union[str, bytes]:
        """Dump the configs into json for debugging."""
        dict_ = {var: getattr(self, var) for var in self._extra_params}
        dict_ |= self.export().model_dump(by_alias=True)

        if _USE_ORJSON:
            return orjson.dumps(
                dict_, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
            )
        else:
            return json.dumps(dict_, indent=2, cls=PAIConfigJsonEncoder)


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
OutputDestConf: TypeAlias = dict[NodeName, dict[CoordAddr, NeuronDestInfo]]
CorePlmConfInChip: TypeAlias = dict[Coord, CorePlmConfig]
CorePlmConf: TypeAlias = dict[ChipCoord, CorePlmConfInChip]
CoreConfInChip: TypeAlias = dict[Coord, CoreConfig]
CoreConf: TypeAlias = dict[ChipCoord, CoreConfInChip]


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
    misc: dict[str, Any]
    """Miscellaneous information."""


_RID_UNSET = RId(0, 0)


def gen_config_frames_by_coreconf(
    config_dict: CorePlmConf,
    write_to_file: bool,
    fp: Path,
    split_by_chip: bool,
    formats: Sequence[str],
) -> dict[ChipCoord, list[FrameArrayType]]:
    """Generate configuration frames by given the `CorePlmConfig`."""

    def _write_to_f(name: str, array: FrameArrayType) -> None:
        for format in formats:
            _fp = (fp / name).with_suffix("." + format)  # don't forget "."
            if format == "npy":
                np2npy(_fp, array)
            elif format == "bin":
                np2bin(_fp, array)
            else:
                np2txt(_fp, array)

    frame_arrays_total: dict[ChipCoord, list[FrameArrayType]] = defaultdict(list)

    for chip_coord, conf_inchip in config_dict.items():
        for core_coord, v in conf_inchip.items():
            # 1. Only one config frame type I for each physical core.
            config_frame_type1 = OfflineFrameGen.gen_config_frame1(
                chip_coord, core_coord, _RID_UNSET, v.random_seed
            )

            # 2. Only one config frame type II for each physical core.
            config_frame_type2 = OfflineFrameGen.gen_config_frame2(
                chip_coord, core_coord, _RID_UNSET, v.params_reg
            )

            # 3. Iterate all the neuron segments inside the physical core.
            # FIXME Unfortunately, at present, only the corresponding NRAM can be written based on
            # the neuron configurations, and it cannot handle the case where the NRAM address is >= 512,
            # that is, some neurons need to occupy the NRAM, which is inconsistent with the current logic.
            # Additional neuron configurations has been written to the NRAM within the CorePlacement.
            # NOTE The meaning of 'n_neuron' in function 'gen_config_frame3' is the number of neurons in
            # the NRAM. See notes of function '_weight_ram_mapping' of `CorePlacement` in file
            # backend/placement.py for details.
            config_frame_type3 = []
            for neu_conf in v.neuron_configs.values():
                # The actual number of neurons placed in NRAM.
                _n_neuron_nram = (
                    HwConfig.ADDR_RAM_MAX + 1
                    if neu_conf.n_neuron > HwConfig.ADDR_RAM_MAX + 1
                    else neu_conf.n_neuron
                )

                config_frame_type3.append(
                    OfflineFrameGen.gen_config_frame3(
                        chip_coord,
                        core_coord,
                        _RID_UNSET,
                        neu_conf.addr_offset,
                        _n_neuron_nram,
                        neu_conf.neuron_attrs,
                        neu_conf.neuron_dest_info,
                        v.params_reg.n_repeat_nram,
                    )
                )

            if config_frame_type3:
                frame3 = np.concatenate(
                    [f.value for f in config_frame_type3],
                    dtype=FRAME_DTYPE,
                    casting="no",
                )
            else:
                frame3 = np.array([], dtype=FRAME_DTYPE)

            # 4. Only one config frame type IV for each physical core.
            # NOTE To avoid logical complications, write the entire weights to the WRAM, rather than just the
            # valid partial weights, because there are still some neurons configurations in the WRAM.
            if v.params_reg.num_dendrite > 0:
                config_frame_type4 = OfflineFrameGen.gen_config_frame4(
                    chip_coord,
                    core_coord,
                    _RID_UNSET,
                    0,
                    18 * (HwConfig.ADDR_RAM_MAX + 1),
                    v.weight_ram[: HwConfig.ADDR_RAM_MAX + 1],
                )
            else:  # empty core placement
                config_frame_type4 = None

            if config_frame_type4:
                frame_arrays_total[chip_coord].append(
                    np.concatenate(
                        [
                            config_frame_type1.value,
                            config_frame_type2.value,
                            frame3,
                            config_frame_type4.value,
                        ],
                        dtype=FRAME_DTYPE,
                        casting="no",
                    )
                )
            else:
                frame_arrays_total[chip_coord].append(
                    np.concatenate(
                        [config_frame_type1.value, config_frame_type2.value, frame3],
                        dtype=FRAME_DTYPE,
                        casting="no",
                    )
                )

    if write_to_file:
        if split_by_chip:
            for chip, frame_arrays_onchip in frame_arrays_total.items():
                f = np.concatenate(frame_arrays_onchip, dtype=FRAME_DTYPE, casting="no")
                _write_to_f(f"config_chip{chip.address}_cores_all", f)
        else:
            _fa = []
            for f in frame_arrays_total.values():
                _fa.extend(f)

            f = np.concatenate(_fa, dtype=FRAME_DTYPE, casting="no")
            _write_to_f("config_all", f)

    return frame_arrays_total


def _with_suffix_json(fp: Path, fname: str) -> Path:
    return (fp / fname).with_suffix(".json")


def export_core_params_json(core_conf: CoreConf, fp: Path) -> None:
    _full_fp = _with_suffix_json(fp, _BACKEND_CONTEXT["core_conf_json"])
    _valid_conf = {}

    for chip_coord, cconf in core_conf.items():
        _valid_conf[str(chip_coord)] = {}
        for core_coord, conf in cconf.items():
            _valid_conf[str(chip_coord)][str(core_coord)] = conf.to_json()

    if _USE_ORJSON:
        with open(_full_fp, "wb") as f:
            f.write(orjson.dumps(_valid_conf, option=orjson.OPT_INDENT_2))
    else:
        with open(_full_fp, "w") as f:
            json.dump(_valid_conf, f, indent=2)


def export_input_conf_json(input_conf_info: InputNodeConf, fp: Path) -> None:
    _full_fp = _with_suffix_json(fp, _BACKEND_CONTEXT["input_conf_json"])
    _valid_conf = {k: v.to_json() for k, v in input_conf_info.items()}

    if _USE_ORJSON:
        with open(_full_fp, "wb") as f:
            f.write(orjson.dumps(_valid_conf, option=orjson.OPT_INDENT_2))
    else:
        with open(_full_fp, "w") as f:
            json.dump(_valid_conf, f, indent=2)


def export_output_conf_json(output_conf_info: OutputDestConf, fp: Path) -> None:
    _full_fp = _with_suffix_json(fp, _BACKEND_CONTEXT["output_conf_json"])
    if _USE_ORJSON:
        with open(_full_fp, "wb") as f:
            f.write(
                orjson.dumps(
                    output_conf_info,
                    default=PAIConfigJsonDefault,
                    option=orjson.OPT_NON_STR_KEYS | orjson.OPT_INDENT_2,
                )
            )
    else:
        with open(_full_fp, "w") as f:
            json.dump(output_conf_info, f, indent=2, cls=PAIConfigJsonEncoder)


if _USE_ORJSON:

    def export_neuconf_json(
        neuron_conf: dict[Neuron, NeuronConfig], fp: Path, fname: str = "neu_conf"
    ) -> None:
        _full_fp = _with_suffix_json(fp, fname)
        _valid_conf = {
            k.name: orjson.loads(v.to_json()) for k, v in neuron_conf.items()
        }

        with open(_full_fp, "wb") as f:
            f.write(orjson.dumps(_valid_conf, option=orjson.OPT_INDENT_2))

else:

    def export_neuconf_json(
        neuron_conf: dict[Neuron, NeuronConfig], fp: Path, fname: str = "neu_conf"
    ) -> None:
        _full_fp = _with_suffix_json(fp, fname)
        _valid_conf = {k.name: json.loads(v.to_json()) for k, v in neuron_conf.items()}

        with open(_full_fp, "w") as f:
            json.dump(_valid_conf, f, indent=2)


def export_core_plm_conf_json(
    core_plm_conf: CorePlmConf, fp: Path, fname: str = "core_plm"
) -> None:
    _full_fp = _with_suffix_json(fp, fname)
    _valid_conf = {}

    for chip_coord, cconf in core_plm_conf.items():
        _valid_conf[str(chip_coord)] = {}
        for core_coord, conf in cconf.items():
            _valid_conf[str(chip_coord)][str(core_coord)] = conf.to_json()

    if _USE_ORJSON:
        with open(_full_fp, "wb") as f:
            f.write(orjson.dumps(_valid_conf, option=orjson.OPT_INDENT_2))
    else:
        with open(_full_fp, "w") as f:
            json.dump(_valid_conf, f, indent=2)


def export_used_L2_clusters(
    clk_en_L2_dict: dict[ChipCoord, list[int]], fp: Path, fname: str = "used_L2"
) -> None:
    _full_fp = _with_suffix_json(fp, fname)
    _valid_conf = {str(k): v for k, v in clk_en_L2_dict.items()}

    if _USE_ORJSON:
        with open(_full_fp, "wb") as f:
            f.write(orjson.dumps(_valid_conf, option=orjson.OPT_INDENT_2))
    else:
        with open(_full_fp, "w") as f:
            json.dump(_valid_conf, f, indent=2)


def _get_clk_en_L2_dict(
    chip_list: list[ChipCoord], used_L2: list[list[RoutingCoord]]
) -> dict[ChipCoord, list[int]]:
    """Generate serial port data for controlling the L2 cluster clocks of the chip.

    Args:
        - chip_list: the available chip list.
        - used_L2: the routing coordinates of used L2 clusters in each chip.

    Returns:
        A dictionary of chip address & the corresponding L2 cluster clocks enable uint8 data.

    NOTE: Serial port data for L2 cluster clocks enable:
        #1 [7:0] L2 clk en #0~#7 (x=0b000, y=0b000) ~ (x=0b000, y=0b111)
        #2 [7:0] L2 clk en #8~#15(x=0b001, y=0b000) ~ (x=0b001, y=0b111)
        ...
        #8 [7:0] L2 clk en #8~#15(x=0b111, y=0b000) ~ (x=0b111, y=0b111)
    """

    def L2_to_idx(L2: RoutingCoord) -> int:
        x = sum(L2[i].value[0] << (2 - i) for i in range(3))
        y = sum(L2[i].value[1] << (2 - i) for i in range(3))

        return (x << 3) + y

    def to_clk_en_L2_u8(L2_inchip: list[RoutingCoord]) -> list[int]:
        clk_en = []
        # L2_inchip is out of order
        bitmap = sum(1 << L2_to_idx(l2) for l2 in L2_inchip)

        for _ in range(8):
            u8 = bitmap & _mask(8)
            bitmap >>= 8
            clk_en.append(reverse_8bit(u8))

        return clk_en

    if sys.version_info >= (3, 10):
        iterator = zip(chip_list, used_L2, strict=True)
    else:
        if len(chip_list) != len(used_L2):
            raise ValueError(
                "the length of chip list & used L2 clusters must be equal, "
                f"but {len(chip_list)} != {len(used_L2)}."
            )

        iterator = zip(chip_list, used_L2)

    clk_en_L2_dict = dict()
    for chip_addr, used_L2_inchip in iterator:
        clk_en_L2_dict[chip_addr] = to_clk_en_L2_u8(used_L2_inchip)

    return clk_en_L2_dict
