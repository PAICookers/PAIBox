from __future__ import annotations

from route_solver import route_solve
from routing import RoutingGroup, toposort_for_rg


class Mapper:
    def __init__(self):
        self.routing_groups: list[RoutingGroup] = []

    def generate_routing_groups(self) -> list[RoutingGroup]:
        raise NotImplementedError(
            "generate_routing_groups method is not implemented yet."
        )

    def set_rough_dest(self):
        raise NotImplementedError("set_rough_dest method is not implemented yet.")

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

    def export(self):
        raise NotImplementedError("export method is not implemented yet.")

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
        self.export()
