from __future__ import annotations

from paicorelib import FRAME_DTYPE, FrameArrayType
from route_solver import route_solve
from routing import RoutingGroup, toposort_for_rg


def export_single_framearray(frame_array: FrameArrayType, file_path: str) -> None:
    with open(file_path, "a") as f:
        for frame in frame_array:
            f.write(f"{frame:016x}\n")


class Mapper:
    def __init__(self):
        self.routing_groups: list[RoutingGroup] = []

    def generate_routing_groups(self) -> list[RoutingGroup]:
        raise NotImplementedError(
            "generate_routing_groups method is not implemented yet."
        )

    def set_rough_dest(self):
        # determine which routing group each neuron sends to
        for rg in self.routing_groups:
            for neu in rg.raw_neus:
                for dest_rg in self.routing_groups:
                    if neu in dest_rg.input_list:
                        rg.dests[neu] = dest_rg
                        break

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
        for rg, copy_config, rg_coords in zip(
            self.routing_groups, copy_configs, coords
        ):
            rg.assign_coord(rg_coords, copy_config)

    def set_detail_dest(self):
        for rg in self.routing_groups:
            rg.set_detail_dest()

    def export(self, output_path: str):
        frame1_path = output_path + "/frame_type1.txt"
        frame3_path = output_path + "/frame_type3.txt"
        for rg in self.routing_groups:
            for core_placement in rg.core_placements:
                core_frame_type1, core_frame_type3 = core_placement.to_frame()
                # export core_frame_type1 and core_frame_type3 to output_path
                # framearray is np.ndarray of np.uint64 with shape (n_frames, )
                # print each frame with 16 hex digits each line
                export_single_framearray(core_frame_type1, frame1_path)
                export_single_framearray(core_frame_type3, frame3_path)

    def compile(self):
        # determine raw_neus in routing groups, other properties remain unset
        self.routing_groups = self.generate_routing_groups()

        # determine which rg each neuron sends to
        # dests and input_list set
        # other properties remain unset
        self.set_rough_dest()

        for rg in self.routing_groups:
            rg.allocate_neurons()

        # set core placements' coord, and generate detailed dest info for each neuron
        self.routing()

        self.set_detail_dest()

        # export to hardware executable format
        self.export(output_path="./output")
