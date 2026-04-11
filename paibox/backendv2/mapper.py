from __future__ import annotations

import os
from typing import TextIO

from paicorelib import FrameArrayType

from paibox.paiir import PAIIRGraph

from .group_tile import tile_groups
from .op_node import AllNode, InputElem, Neuron, RemapElem, build_nodes
from .rg_build import build_groups
from .route_solver import route_solve
from .routing import (
    InputGroup,
    OutputGroup,
    RemapGroup,
    RoutingGroup,
    toposort_for_rg,
)


def export_single_framearray(
    frame_array: FrameArrayType, file: TextIO, prefix: str = ""
) -> None:
    for frame in frame_array:
        hex_str = f"{frame:016x}"
        hex_str = "_".join(hex_str[i : i + 4] for i in range(0, 16, 4))
        file.write(f"{prefix}{hex_str}\n")


def export_framearray_to_bit(
    frame_array: FrameArrayType,
    file: TextIO,
    prefix: str = "",
    base: str = "bin",  # 新增参数："bin" 或 "hex"
) -> None:
    for frame in frame_array:
        # mask 取高32位和低32位
        high32 = (frame >> 32) & 0xFFFFFFFF
        low32 = frame & 0xFFFFFFFF

        if base == "bin":
            high_str = f"0b{high32:032b}"
            low_str = f"0b{low32:032b}"
        elif base == "hex":
            high_str = f"0x{high32:08X}"
            low_str = f"0x{low32:08X}"
        else:
            raise ValueError("base must be 'bin' or 'hex'")

        file.write(f"{prefix}{high_str},{low_str},\n")


class Mapper:
    def __init__(self):
        self.groups: list[RoutingGroup | RemapGroup] = []
        self.routing_groups: list[RoutingGroup] = []
        self.nodes: list[AllNode] = []
        self.output_groups: list[OutputGroup] = []
        self.input_groups: list[InputGroup] = []

    def generate_routing_groups(self, pai_graph: PAIIRGraph):
        self.nodes = build_nodes(pai_graph)
        self.groups, self.input_groups, self.output_groups = build_groups(self.nodes)

    def set_rough_dest(self):
        # determine which routing group each neuron sends to
        source_groups: list[InputGroup | RemapGroup | RoutingGroup] = []
        source_groups.extend(self.input_groups)
        source_groups.extend(self.groups)

        dest_groups: list[RemapGroup | RoutingGroup | OutputGroup] = []
        dest_groups.extend(self.groups)
        dest_groups.extend(self.output_groups)

        for grp in self.groups:
            if isinstance(grp, RoutingGroup):
                grp.set_lcn()

        for src_grp in source_groups:
            for elem in src_grp.raw_elems:
                dest_found = False
                # print(f"\nSetting rough dest for neuron {neu} in group {group.name}:")
                for dest_grp in dest_groups:
                    # print(f"\tChecking if neuron {neu} sends to group {dest_grp.name}")
                    # print(f"Group {dest_grp.name} has input set: {dest_grp.input_set}")
                    # use set to accelerate lookup
                    if elem in dest_grp.input_set:
                        # print(f"\tDest Found: Neuron {neu} sends to group {dest_grp.name}")
                        dest_found = True
                        if isinstance(elem, RemapElem):
                            assert isinstance(src_grp, RemapGroup)
                            src_grp.dests[elem] = dest_grp
                        elif isinstance(elem, InputElem):
                            assert isinstance(src_grp, InputGroup)
                            src_grp.dests[elem] = dest_grp
                        elif isinstance(elem, Neuron):
                            assert isinstance(src_grp, RoutingGroup)
                            src_grp.dests[elem] = dest_grp
                        else:
                            raise TypeError(
                                f"Unsupported element type: {type(elem)} in group {src_grp.name}"
                            )
                        break
                if not dest_found:
                    raise ValueError(
                        f"Dest not found for neuron {elem} in group {src_grp.name}"
                    )

    def routing(self):
        self.routing_groups, next_rg_group = toposort_for_rg(self.groups)

        areas = [rg.n_core_required for rg in self.routing_groups]
        copy_configs, coords = route_solve(
            areas=areas,
            next_area_id=next_rg_group,
            io_target=(0, 0),
            input_area_ids=[],
            output_area_ids=[],
        )
        print("Routing result:")
        print("Copy Configs:", copy_configs)
        print("Coords:", coords)

        for rg, copy_config, rg_coords in zip(
            self.routing_groups, copy_configs, coords
        ):
            rg.assign_coord(rg_coords, copy_config)

    def set_detail_dest(self):
        for rg in self.routing_groups:
            rg.set_detail_dest()
        for in_grp in self.input_groups:
            in_grp.set_detail_dest()

    def set_auto_core_config(self):
        for rg in self.routing_groups:
            rg.set_auto_core_config()

    def export_cheader_file(self, output_path: str, base: str = "bin"):
        os.makedirs(output_path, exist_ok=True)
        frame1_path = output_path + "/frame_type1.h"
        frame2_path = output_path + "/frame_type2.h"
        frame3_path = output_path + "/frame_type3.h"
        with (
            open(frame1_path, "w") as frame1_file,
            open(frame2_path, "w") as frame2_file,
            open(frame3_path, "w") as frame3_file,
        ):
            frame1_file.write(
                'volatile unsigned int config_frame1[] __attribute__((section(".large_const_data"))) ={\n'
            )
            frame2_file.write(
                'volatile unsigned int config_frame2[] __attribute__((section(".large_const_data"))) ={\n'
            )
            frame3_file.write(
                'volatile unsigned int config_frame3[] __attribute__((section(".large_const_data"))) ={\n'
            )
            for rg in self.routing_groups:
                for core_placement in rg.core_placements:
                    core_frame_type1, core_frame_type2, core_frame_type3 = (
                        core_placement.to_frame()
                    )
                    # export core_frame_type1 and core_frame_type3 to output_path
                    export_framearray_to_bit(
                        core_frame_type1, frame1_file, "\t", base=base
                    )
                    if core_frame_type2 is not None:
                        export_framearray_to_bit(
                            core_frame_type2, frame2_file, "\t", base=base
                        )
                    export_framearray_to_bit(
                        core_frame_type3, frame3_file, "\t", base=base
                    )

            frame1_file.write("};\n")
            frame2_file.write("};\n")
            frame3_file.write("};\n")

    def export(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        frame1_path = output_path + "/frame_type1.txt"
        frame2_path = output_path + "/frame_type2.txt"
        frame3_path = output_path + "/frame_type3.txt"
        with (
            open(frame1_path, "w") as frame1_file,
            open(frame2_path, "w") as frame2_file,
            open(frame3_path, "w") as frame3_file,
        ):
            for rg in self.routing_groups:
                for core_placement in rg.core_placements:
                    frame1_file.write(
                        f"# Core at coord ({core_placement.coord.x}, {core_placement.coord.y}):\n"
                    )
                    frame2_file.write(
                        f"# Core at coord ({core_placement.coord.x}, {core_placement.coord.y}):\n"
                    )
                    frame3_file.write(
                        f"# Core at coord ({core_placement.coord.x}, {core_placement.coord.y}):\n"
                    )

                    core_frame_type1, core_frame_type2, core_frame_type3 = (
                        core_placement.to_frame()
                    )
                    # export core_frame_type1 and core_frame_type3 to output_path
                    # framearray is np.ndarray of np.uint64 with shape (n_frames, )
                    # print each frame with 16 hex digits each line
                    export_single_framearray(
                        core_frame_type1, frame1_file, prefix="\t0x"
                    )
                    if core_frame_type2 is not None:
                        export_single_framearray(
                            core_frame_type2, frame2_file, prefix="\t0x"
                        )
                    export_single_framearray(
                        core_frame_type3, frame3_file, prefix="\t0x"
                    )

    def compile(
        self,
        pai_graph: PAIIRGraph,
        base: str = "bin",  # 新增参数，指定导出格式
        output_path: str = "./output",
    ) -> None:
        # determine raw_neus in routing groups, other properties remain unset
        self.generate_routing_groups(pai_graph)

        all_groups: list[RoutingGroup | InputGroup | OutputGroup | RemapGroup] = []
        all_groups.extend(self.input_groups)
        all_groups.extend(self.groups)
        all_groups.extend(self.output_groups)
        for grp in all_groups:
            print(grp)

        # determine which rg each neuron sends to
        # dests and input_list set
        # other properties remain unset
        all_groups = tile_groups(all_groups)
        for grp in all_groups:
            print(grp)

        self.input_groups = []
        self.groups = []
        self.output_groups = []
        for grp in all_groups:
            if isinstance(grp, InputGroup):
                self.input_groups.append(grp)
            elif isinstance(grp, OutputGroup):
                self.output_groups.append(grp)
            elif isinstance(grp, RoutingGroup):
                self.groups.append(grp)
            elif isinstance(grp, RemapGroup):
                self.groups.append(grp)
            else:
                raise TypeError(f"Unsupported group type: {type(grp)}")
        # raise NotImplementedError("Conv tiling is not implemented yet.")

        self.set_rough_dest()

        for grp in all_groups:
            print(grp.info())

        self.routing_groups = [
            grp for grp in self.groups if isinstance(grp, RoutingGroup)
        ]

        for rg in self.routing_groups:
            rg.allocate_neurons()

        for rg in all_groups:
            print(rg.info())

        # set core placements' coord, and generate detailed dest info for each neuron
        self.routing()

        print("\nAfter routing:")
        for grp in all_groups:
            print(grp.info())

        self.set_detail_dest()
        for rg in all_groups:
            print(rg.routing_summary())

        self.set_auto_core_config()

        # export to hardware executable format
        self.export(output_path=output_path)
        self.export_cheader_file(output_path=output_path, base=base)

        # for in_grp in self.input_groups:
        #     for elem, dest in in_grp.dest_infos.items():
        #         print(
        #             f"Input element {elem}({elem.output_bit_num} bits) sends to dest \n\t{dest}"
        #         )

        # for out_grp in self.output_groups:
        #     for coord, bit_map in out_grp.axon_bit_map.items():
        #         print(f"Output from coord {coord} receives bits:")
        #         for bit_count, elem in bit_map:
        #             print(f"\t[{bit_count}]{elem}({elem.output_bit_num} bits)")
