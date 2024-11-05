import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from paicorelib import ChipCoord, HwConfig, RoutingCoord
from paicorelib.framelib import OfflineFrameGen
from paicorelib.framelib.utils import _mask, np2bin, np2npy, np2txt

from paibox.components import Neuron
from paibox.utils import reverse_8bit

from .conf_types import (
    _USE_ORJSON,
    FRAME_DTYPE,
    CoreConf,
    CorePlmConf,
    FrameArrayType,
    GraphInfo,
    InputNodeConf,
    NeuronConfig,
    OutputDestConf,
)
from .context import _BACKEND_CONTEXT
from .placement import CorePlacement
from .types import _RID_UNSET

if _USE_ORJSON:
    import orjson

    from .conf_types import PAIConfigJsonDefault
else:
    import json

    from .conf_types import PAIConfigJsonEncoder


__all__ = [
    "gen_config_frames_by_coreconf",
    "export_core_params_json",
    "export_input_conf_json",
    "export_output_conf_json",
    "export_neuconf_json",
    "export_core_plm_conf_json",
    "export_graph_info",
    "export_used_L2_clusters",
    "get_clk_en_L2_dict",
]


def gen_config_frames_by_coreconf(
    config_dict: CorePlmConf,
    write_to_file: bool,
    fp: Path,
    split_by_chip: bool,
    formats: Sequence[str],
) -> dict[ChipCoord, list[FrameArrayType]]:
    """Generate configuration frames by given the `CorePlmConf`."""

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
            # The meaning of 'n_neuron' in function 'gen_config_frame3' is the number of neurons in the NRAM.
            config_frame_type3 = []
            neu_conf_on_wram: list[NeuronConfig] = []

            for neu_conf in v.neuron_configs.values():
                if (
                    neu_conf.neu_seg.offset + neu_conf.neu_seg.n_neuron
                    <= HwConfig.ADDR_RAM_MAX + 1
                ):
                    # Place in the NRAM
                    config_frame_type3.append(
                        OfflineFrameGen.gen_config_frame3(
                            chip_coord,
                            core_coord,
                            _RID_UNSET,
                            neu_conf.neu_seg.offset,
                            neu_conf.neu_seg.n_neuron,
                            neu_conf.neuron_attrs,
                            neu_conf.neuron_dest_info,
                            neu_conf.neu_seg.repeat,
                        )
                    )
                else:
                    # Only happens in ANN mode, where the repeat=1
                    assert neu_conf.neu_seg.repeat == 1

                    if (
                        n_on_nram := HwConfig.ADDR_RAM_MAX + 1 - neu_conf.neu_seg.offset
                    ) > 0:
                        # Place in the NRAM partially
                        neu_on_nram_conf = neu_conf[:n_on_nram]
                        config_frame_type3.append(
                            OfflineFrameGen.gen_config_frame3(
                                chip_coord,
                                core_coord,
                                _RID_UNSET,
                                neu_on_nram_conf.neu_seg.offset,
                                neu_on_nram_conf.neu_seg.n_neuron,
                                neu_on_nram_conf.neuron_attrs,
                                neu_on_nram_conf.neuron_dest_info,
                                neu_on_nram_conf.neu_seg.repeat,
                            )
                        )
                        # Place the rest in the WRAM
                        neu_conf_on_wram.append(neu_conf[n_on_nram:])
                    else:
                        # Place in the WRAM totally
                        neu_conf_on_wram.append(neu_conf)

            if config_frame_type3:
                frame3 = np.concatenate(
                    [f.value for f in config_frame_type3],
                    dtype=FRAME_DTYPE,
                    casting="no",
                )
            else:
                frame3 = np.array([], dtype=FRAME_DTYPE)

            _concat_frames = [
                config_frame_type1.value,
                config_frame_type2.value,
                frame3,
            ]
            # 4. Only one config frame type IV for each physical core.
            if v.params_reg.num_dendrite > 0:
                # Weight part
                config_frame_type4_w = OfflineFrameGen.gen_config_frame4(
                    chip_coord,
                    core_coord,
                    _RID_UNSET,
                    0,
                    v.weight_ram.size,
                    v.weight_ram,
                )

                _concat_frames.append(config_frame_type4_w.value)

            # Extra neurons part
            if neu_conf_on_wram:
                # Only the part that is mapped to the neuron parameters is returned.
                neu_on_wram = CorePlacement.neu_params_mapping(neu_conf_on_wram)
                assert (
                    v.weight_ram.shape[0] + neu_on_wram.shape[0]
                    <= CorePlacement.WRAM_BASE_SHAPE[1]
                )

                config_frame_type4_n = OfflineFrameGen.gen_config_frame4(
                    chip_coord,
                    core_coord,
                    _RID_UNSET,
                    # `v.weigh_ram` already contains the mapped & unallocated parts for weight mapping,
                    # so `neu_on_wram` can be placed next to it.
                    v.weight_ram.shape[0],
                    neu_on_wram.size,
                    neu_on_wram,
                )

                _concat_frames.append(config_frame_type4_n.value)

            frame_arrays_total[chip_coord].append(
                np.hstack(_concat_frames, casting="no")
            )

    if write_to_file:
        if split_by_chip:
            for chip, frame_arrays_onchip in frame_arrays_total.items():
                f = np.hstack(frame_arrays_onchip, casting="no")
                _write_to_f(f"config_chip{chip.address}_cores_all", f)
        else:
            _fa_list = []
            for f in frame_arrays_total.values():
                _fa_list.extend(f)

            f = np.hstack(_fa_list, casting="no")
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
    _valid_conf = {}

    for dest, dest_info in output_conf_info.items():
        _valid_conf[dest] = {}
        for k, v in dest_info.items():
            _valid_conf[dest][str(k)] = v

    if _USE_ORJSON:
        with open(_full_fp, "wb") as f:
            f.write(
                orjson.dumps(
                    _valid_conf,
                    default=PAIConfigJsonDefault,
                    option=orjson.OPT_INDENT_2,
                )
            )
    else:
        with open(_full_fp, "w") as f:
            json.dump(_valid_conf, f, indent=2, cls=PAIConfigJsonEncoder)


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


def export_aux_gh_info(gh_info: GraphInfo, fp: Path, export_clk_en_L2: bool) -> None:
    _full_fp = _with_suffix_json(fp, _BACKEND_CONTEXT["graph_info_json"])
    aux_gh_info_dict = {
        "name": gh_info["name"],
        "n_core_required": gh_info["n_core_required"],
        "n_core_occupied": gh_info["n_core_occupied"],
        "layer_num": gh_info["inherent_timestep"],
    }

    if misc := gh_info.get("misc"):
        aux_gh_info_dict["misc"] = {}
        # Export the serial port data of the L2 cluster clocks
        if export_clk_en_L2 and (clk_en_L2_dict := misc.get("clk_en_L2")):
            # dict[ChipCoord, list[int]]
            aux_gh_info_dict["misc"]["clk_en_L2"] = clk_en_L2_dict
        if lst := misc.get("target_chip_list"):  # list of ChipCoord
            aux_gh_info_dict["misc"]["target_chip_list"] = lst

    if _USE_ORJSON:
        with open(_full_fp, "wb") as f:
            f.write(
                orjson.dumps(
                    aux_gh_info_dict,
                    default=PAIConfigJsonDefault,
                    option=orjson.OPT_INDENT_2,
                )
            )
    else:
        with open(_full_fp, "w") as f:
            json.dump(aux_gh_info_dict, f, indent=2, cls=PAIConfigJsonEncoder)


def export_graph_info(
    gh_info: GraphInfo,
    fp: Path,
    export_clk_en_L2: bool,
    export_core_placements: bool = False,
) -> None:
    # Export the configurations of input nodes
    export_input_conf_json(gh_info["input"], fp)
    # Export the configurations of output destinations
    export_output_conf_json(gh_info["output"], fp)
    export_aux_gh_info(gh_info, fp, export_clk_en_L2)

    if export_core_placements:
        export_core_plm_conf_json(gh_info["members"], fp)


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


def get_clk_en_L2_dict(
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
