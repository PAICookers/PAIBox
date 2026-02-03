from __future__ import annotations

import os
from typing import TextIO

from paicorelib import LCN_EX, AERPacketZXYCopy, CoordXY, FrameArrayType
from torch.fx import GraphModule

from .op_node import CoreOpNode, InputNode, build_nodes
from .rg_build import build_routing_groups
from .route_solver import route_solve
from .routing import RoutingGroup, toposort_for_rg


def export_single_framearray(frame_array: FrameArrayType, file: TextIO) -> None:
    for frame in frame_array:
        file.write(f"{frame:016x}\n")


class Mapper:
    def __init__(self):
        self.routing_groups: list[RoutingGroup] = []
        self.nodes: list[CoreOpNode | InputNode] = []
        self.output_routing_group: RoutingGroup = RoutingGroup([], [])
        self.output_routing_group._base_coord = CoordXY(0, 0)
        self.output_routing_group._multicast_config = AERPacketZXYCopy(z=0, x=0, y=0)

    def generate_routing_groups(self, torch_graph: GraphModule):
        self.nodes = build_nodes(torch_graph)
        self.routing_groups = build_routing_groups(self.nodes)

    def set_rough_dest(self):
        # determine which routing group each neuron sends to
        for rg in self.routing_groups:
            rg.set_lcn()
            for neu in rg.raw_neus:
                for dest_rg in self.routing_groups:
                    # use set to accelerate lookup
                    if neu in dest_rg.input_set:
                        rg.dests[neu] = dest_rg
                        break
                rg.dests[neu] = self.output_routing_group
                self.output_routing_group.input_list.append(neu)

        self.output_routing_group.input_set = set(self.output_routing_group.input_list)
        self.output_routing_group.lcn = LCN_EX.LCN_128X

    def routing(self):
        self.routing_groups, next_rg_group = toposort_for_rg(self.routing_groups)

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

    def set_auto_core_config(self):
        for rg in self.routing_groups:
            rg.set_auto_core_config()

    def export(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        frame1_path = output_path + "/frame_type1.txt"
        frame3_path = output_path + "/frame_type3.txt"
        with (
            open(frame1_path, "w") as frame1_file,
            open(frame3_path, "w") as frame3_file,
        ):
            for rg in self.routing_groups:
                for core_placement in rg.core_placements:
                    core_frame_type1, core_frame_type3 = core_placement.to_frame()
                    # export core_frame_type1 and core_frame_type3 to output_path
                    # framearray is np.ndarray of np.uint64 with shape (n_frames, )
                    # print each frame with 16 hex digits each line
                    export_single_framearray(core_frame_type1, frame1_file)
                    export_single_framearray(core_frame_type3, frame3_file)

    def compile(self, torch_graph: GraphModule):
        # determine raw_neus in routing groups, other properties remain unset
        self.generate_routing_groups(torch_graph)

        print(self.routing_groups)

        # determine which rg each neuron sends to
        # dests and input_list set
        # other properties remain unset
        self.set_rough_dest()

        for rg in self.routing_groups:
            rg.allocate_neurons()

        for rg in self.routing_groups:
            print(rg.info())

        # set core placements' coord, and generate detailed dest info for each neuron
        self.routing()

        for rg in self.routing_groups:
            print(rg.info())

        self.set_detail_dest()
        for rg in self.routing_groups:
            print(rg.routing_summary())

        self.set_auto_core_config()

        # export to hardware executable format
        self.export(output_path="./output")
