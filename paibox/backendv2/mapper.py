import os
from pathlib import Path

from paicorelib import CoordXY, CoordZXYOffset

from paibox.paiir import PAIIRGraph
from paibox.paiir.ir import OfflineCoreOp

from .coreplacement import CorePlacement
from .export.cheader import export_cheader_files, export_cheader_merged
from .export.npy import export_frame_npy
from .export.proto import export_compile_artifacts
from .export.text import export_debug_txt_files, export_debug_txt_merged
from .export.utils import (
    LiteralFormat,
    TargetPlatform,
    WordOrder,
    make_frame_records,
    resolve_platform_exports,
)
from .global_signal import set_global_signal
from .group_tile import tile_groups
from .op_node import AllNode, InputElem, Neuron, RemapElem, SourceElem, build_nodes
from .output_completion import (
    OutputCompletionPlan,
    OutputProducer,
    select_output_completion_plan,
)
from .output_routes import OutputCpuIngressPlan
from .rg_build import build_groups
from .route_solver import HIVE, route_solve
from .routing import InputGroup, OutputGroup, RemapGroup, RoutingGroup, toposort_for_rg


class Mapper:
    def __init__(self) -> None:
        self.groups: list[RoutingGroup | RemapGroup] = []
        self.routing_groups: list[RoutingGroup] = []
        self.nodes: list[AllNode] = []
        self.output_groups: list[OutputGroup] = []
        self.input_groups: list[InputGroup] = []
        self.coreplacements: list[CorePlacement] = []
        self.global_starts: dict[int, CoordZXYOffset] = {}
        self.output_completion_plan: OutputCompletionPlan | None = None
        self.timesteps: int = 1

    def _resolve_timesteps(self, pai_graph: PAIIRGraph, timesteps: int | None) -> int:
        """Resolve the application runtime length used by output metadata.

        Auto-reset graphs export ``tick_duration=0`` and carry the public
        runtime length in ``tick_initial``, so inference must check
        ``tick_initial`` before falling back to finite ``tick_duration``.
        """

        def _collect_output_timesteps() -> set[int]:
            inferred: set[int] = set()
            visited: set[str] = set()
            pending = [
                pred_name
                for output_node in pai_graph.output_nodes()
                for pred_name in pai_graph.predecessors(output_node.name)
            ]

            while pending:
                name = pending.pop()
                if name in visited:
                    continue
                visited.add(name)

                node = pai_graph.nodes[name]
                if isinstance(node, OfflineCoreOp):
                    cp = node.core_params
                    if cp.tick_initial > 0:
                        inferred.add(cp.tick_initial)
                    elif cp.tick_duration > 0:
                        inferred.add(cp.tick_duration)
                    continue

                pending.extend(pai_graph.predecessors(name))

            return inferred

        if timesteps is not None:
            resolved = int(timesteps)
        else:
            output_timesteps = _collect_output_timesteps()
            resolved = next(iter(output_timesteps)) if len(output_timesteps) == 1 else 1

        if resolved <= 0:
            raise ValueError(f"'timesteps' must be positive, got {resolved}.")

        for name, node in pai_graph.nodes.items():
            if not isinstance(node, OfflineCoreOp):
                continue

            tick_duration = node.core_params.tick_duration
            if tick_duration != 0 and tick_duration < resolved:
                raise ValueError(
                    f"'timesteps' ({resolved}) exceeds finite tick_duration "
                    f"({tick_duration}) of core node '{name}'."
                )

        return resolved

    def generate_routing_groups(self, pai_graph: PAIIRGraph) -> None:
        self.nodes = build_nodes(pai_graph)
        self.groups, self.input_groups, self.output_groups = build_groups(self.nodes)

    def set_rough_dest(self) -> None:
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
            useless_elems: list[SourceElem] = []
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
                            src_grp.used_elems.append(elem)
                        elif isinstance(elem, InputElem):
                            assert isinstance(src_grp, InputGroup)
                            src_grp.dests[elem] = dest_grp
                            src_grp.used_elems.append(elem)
                        elif isinstance(elem, Neuron):
                            assert isinstance(src_grp, RoutingGroup)
                            src_grp.dests[elem] = dest_grp
                            src_grp.used_elems.append(elem)
                        else:
                            raise TypeError(
                                f"Unsupported element type: {type(elem)} in group {src_grp.name}"
                            )
                        break
                if not dest_found:
                    useless_elems.append(elem)
            src_grp.update_raw_elems()

    def routing(self) -> None:
        self.routing_groups, next_rg_group = toposort_for_rg(self.groups)
        areas = [rg.n_core_required for rg in self.routing_groups]
        print(f"\ttotal cores needed: {sum(areas)}")
        if sum(areas) > 63:
            raise ValueError(
                f"Total cores needed {sum(areas)} exceeds the limit of 63."
            )
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

    def set_detail_dest(self, output_route_plan: OutputCpuIngressPlan) -> None:
        # Output collection still targets the CPU. The plan only overrides the
        # Z/X/Y decomposition so DATA and completion use the same CPU port.
        output_route_offsets = output_route_plan.output_route_offsets()
        for rg in self.routing_groups:
            rg.set_detail_dest(output_route_offsets)
        for in_grp in self.input_groups:
            in_grp.set_detail_dest()

    def set_auto_core_config(
        self, control_root_coord: CoordXY, control_offset: CoordZXYOffset
    ) -> None:
        seen_cp_ids: set[int] = set()
        for cp in self.coreplacements:
            if id(cp) in seen_cp_ids:
                continue
            seen_cp_ids.add(id(cp))

            if cp.coord == control_root_coord:
                cp.set_auto_core_config(control_offset)
            else:
                cp.set_auto_core_config()

    def _collect_output_producers(self) -> list[OutputProducer]:
        producer_weights: dict[tuple[CoordXY, CoordXY], int] = {}
        for rg in self.routing_groups:
            for cp in rg.core_placements:
                for neu_placement in cp.neus:
                    dest_group = rg.get_dest(neu_placement.raw_neus[0])
                    if not isinstance(dest_group, OutputGroup):
                        continue
                    key = (cp.coord, dest_group.base_coord)
                    producer_weights[key] = producer_weights.get(key, 0) + len(
                        neu_placement.raw_neus
                    )

        return [
            OutputProducer(coord, target_coord, weight)
            for (coord, target_coord), weight in sorted(
                producer_weights.items(),
                key=lambda item: (
                    item[0][0].x,
                    item[0][0].y,
                    item[0][1].x,
                    item[0][1].y,
                ),
            )
        ]

    def build_output_completion_plan(self) -> OutputCompletionPlan:
        """Select DATA routes and a global signal root before dest encoding."""
        used_core_coords = {cp.coord for cp in self.coreplacements}
        empty_offline_coords = {
            CoordXY(x, y) for x, y in HIVE if CoordXY(x, y) not in used_core_coords
        }
        return select_output_completion_plan(
            self._collect_output_producers(),
            used_core_coords,
            empty_offline_coords,
            set(),
            allow_empty_online=False,
            empty_online_frame_supported=False,
        )

    def export_artifacts(
        self,
        output_path: str | Path,
        literal_format: LiteralFormat = "bin",
        target_platform: TargetPlatform = "riscv",
        word_order: WordOrder = "high_first",
        export_merged_frames: bool = True,
        export_proto_python: bool = True,
        debug: bool = False,
    ) -> None:
        """Export all requested backend artifacts to the output directory."""
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        export_x86, export_riscv = resolve_platform_exports(target_platform, debug)
        frame_records = make_frame_records(self.coreplacements)

        if debug:
            export_debug_txt_files(out, frame_records)
            if export_merged_frames:
                export_debug_txt_merged(out, frame_records)

        if export_x86:
            export_frame_npy(out, frame_records, export_merged_frames)

        if export_riscv:
            export_cheader_files(out, frame_records, literal_format)
            if export_merged_frames:
                export_cheader_merged(out, frame_records, literal_format)

        export_compile_artifacts(
            out,
            target_platform,
            word_order,
            export_proto_python,
            debug,
            self.timesteps,
            self.groups,
            self.input_groups,
            self.output_groups,
            self.coreplacements,
            self.global_starts,
            frame_records,
        )

    def compile(
        self,
        pai_graph: PAIIRGraph,
        output_path: str | Path | None = None,
        literal_format: LiteralFormat = "bin",
        *,
        timesteps: int | None = None,
        target_platform: TargetPlatform = "all",
        word_order: WordOrder = "high_first",
        export_merged_frames: bool = True,
        export_proto_python: bool = True,
        debug: bool = False,
    ) -> None:
        """Compile a PAIIR graph and export backendv2 deployment artifacts.

        Args:
            pai_graph: The compiled PAIIR graph to place, route, and export.
            output_path: Root directory for exported artifacts. When ``None``,
                ``$PAIBOX_OUTPUT_PATH/frame_out`` is used if the environment
                variable is set; otherwise defaults to ``./output`` under the
                current working directory.
            literal_format: Numeric radix used by generated C headers. ``"bin"``
                writes 32-bit words as binary literals, while ``"hex"``
                writes hexadecimal literals.
            timesteps: Application-side inference sequence length. When
                ``None``, a finite and consistent output ``tick_initial`` from
                automatic-reset graphs is used first; otherwise a finite and
                consistent output ``tick_duration`` is used. If neither is
                available, it defaults to ``1``.
            target_platform: Platform-specific artifact set to emit.
                ``"x86"`` exports ``.npy`` frame arrays, ``"riscv"`` exports
                C headers, and ``"all"`` exports both. When ``debug=True``,
                platform-specific exports are always emitted for both targets.
            word_order: Order used when splitting each 64-bit config frame
                into 32-bit words for protobuf export. This affects
                ``proto/config.pb`` and ``proto/config.json`` only; it does not
                change the logical 64-bit frame sequence.
            export_merged_frames: Whether to also emit merged frame artifacts
                that concatenate frame types 1, 2, and 3 into one file per
                platform or debug view.
            debug: Whether to emit extra debug artifacts such as
                human-readable ``cfg_frame*.txt`` / ``cfg_frames.txt`` and
                ``proto/config.json``. When enabled, x86 and riscv platform
                artifacts are both exported regardless of ``target_platform``.
            export_proto_python: Whether to copy ``compile_artifacts_pb2.py``
                and ``compile_artifacts_pb2.pyi`` into the exported ``proto/``
                directory when x86 artifacts are part of the export set.
        """
        self.timesteps = self._resolve_timesteps(pai_graph, timesteps)

        # determine raw_neus in routing groups, other properties remain unset
        self.generate_routing_groups(pai_graph)

        all_groups: list[RoutingGroup | InputGroup | OutputGroup | RemapGroup] = []
        all_groups.extend(self.input_groups)
        all_groups.extend(self.groups)
        all_groups.extend(self.output_groups)

        # determine which rg each neuron sends to
        # dests and input_list set
        # other properties remain unset
        all_groups = tile_groups(all_groups)

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

        self.set_rough_dest()

        for out_grp in self.output_groups:
            out_grp.set_lcn(self.timesteps)

        self.routing_groups = [
            grp for grp in self.groups if isinstance(grp, RoutingGroup)
        ]

        for rg in self.routing_groups:
            rg.allocate_neurons()

        # set core placements' coord, and generate detailed dest info for each neuron
        self.routing()

        for rg in self.routing_groups:
            self.coreplacements.extend(rg.core_placements)

        self.output_completion_plan = self.build_output_completion_plan()

        self.coreplacements, self.global_starts = set_global_signal(
            self.coreplacements,
            self.output_completion_plan.global_signal_root,
            self.output_completion_plan.relay_core_kinds(),
        )

        output_route_plan = self.output_completion_plan.to_cpu_ingress_plan()

        # The global-signal root uses the selected control offset. Other cores
        # keep their own local route to the same fixed CPU destination.
        self.set_detail_dest(output_route_plan)
        self.set_auto_core_config(
            self.output_completion_plan.global_signal_root,
            output_route_plan.control_offset,
        )

        # export to hardware executable format
        if output_path is None:
            env_output_path = os.environ.get("PAIBOX_OUTPUT_PATH")
            if env_output_path is not None:
                output_path = Path(env_output_path) / "frame_out"
            else:
                output_path = Path.cwd() / "output"

        self.export_artifacts(
            output_path,
            literal_format,
            target_platform,
            word_order,
            export_merged_frames,
            export_proto_python,
            debug,
        )
