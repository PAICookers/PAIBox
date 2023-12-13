from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np

from paibox.libpaicore import Coord
from paibox.libpaicore import ReplicationId as RId

from .conf_template import CorePlacementConfig
from .runtime import OfflineFrameGen
from .runtime.libframe._types import FRAME_DTYPE, FrameArrayType
from .runtime.libframe.utils import np2bin, np2npy, np2txt

# This file is mainly a compatible layer of PAIBox runtime library.


def gen_config_frames_by_coreconf(
    target_chip_coord: Coord,
    config_dict: Dict[Coord, CorePlacementConfig],
    write_to_file: bool,
    fp: Path,
    format: Literal["txt", "bin", "npy"],
) -> Dict[Coord, FrameArrayType]:
    """Generate all configuration frames by given the `CorePlacementConfig`.

    Args:
        - target_chip_coord: the local chip to configurate.
        - config_dict: a dictionary of configurations.
        - write_to_file: whether to write frames into file.
        - fp: If `write_to_file` is `True`, specify the path.
        - format: it can be `txt`, `bin`, or `npy`. `bin` & `npy` are recommended.
    """
    _default_rid = RId(0, 0)
    _debug_dict: Dict[Coord, Dict[str, Any]] = defaultdict()
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
        for core_coord, f in frame_arrays_on_core.items():
            addr = core_coord.address
            fn = f"config_core{addr}.{format}"
            if format == "npy":
                np2npy(fp / fn, f)
            elif format == "bin":
                np2bin(fp / fn, f)
            else:
                np2txt(fp / fn, f)

    return frame_arrays_on_core
