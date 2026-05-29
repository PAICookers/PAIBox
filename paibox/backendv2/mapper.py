import os
from pathlib import Path

from paicorelib import CoordZXYOffset

from paibox.paiir import PAIIRGraph

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
from .op_node import AllNode, InputElem, Neuron, RemapElem, build_nodes
from .rg_build import build_groups
from .route_solver import route_solve
from .routing import (
    InputGroup,
    OutputGroup,
    RemapGroup,
    RoutingGroup,
    SourceElem,
    toposort_for_rg,
)


class Mapper:
    def __init__(self) -> None:
        self.groups: list[RoutingGroup | RemapGroup] = []
        self.routing_groups: list[RoutingGroup] = []
        self.nodes: list[AllNode] = []
        self.output_groups: list[OutputGroup] = []
        self.input_groups: list[InputGroup] = []
        self.coreplacements: list[CorePlacement] = []
        self.global_starts: dict[int, CoordZXYOffset] = {}

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
            dest_strs: list[str] = []
            if len(useless_elems) > 6:
                print_elems = useless_elems[:3] + useless_elems[-3:]
            else:
                print_elems = useless_elems
            for elem in print_elems:
                dest_strs.append(str(elem))
            if len(useless_elems) > 6:
                dest_strs = dest_strs[:3] + ["..."] + dest_strs[-3:]
            if len(useless_elems) > 0:
                print(
                    f"\nfound {len(useless_elems)} elements not used in group {src_grp.name}:"
                )
                print("    " + "\n    ".join(dest_strs))

            src_grp.update_raw_elems()

    def routing(self) -> None:
        self.routing_groups, next_rg_group = toposort_for_rg(self.groups)
        print("\nTrying to solve routing")
        for rg in self.routing_groups:
            print(f"\tRouting Group {rg.name} requires {rg.n_core_required} cores.")

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

        print("\nRouting result:")
        for rg, copy_config, rg_coords in zip(
            self.routing_groups, copy_configs, coords
        ):
            print(f"\t{rg.name}({rg.n_core_required} cores):")
            print(f"\t\tcopy: {copy_config}")
            print(f"\t\tcoord: {rg_coords}")

        for rg, copy_config, rg_coords in zip(
            self.routing_groups, copy_configs, coords
        ):
            rg.assign_coord(rg_coords, copy_config)

    def set_detail_dest(self) -> None:
        for rg in self.routing_groups:
            rg.set_detail_dest()
        for in_grp in self.input_groups:
            in_grp.set_detail_dest()

    def set_auto_core_config(self) -> None:
        for rg in self.routing_groups:
            rg.set_auto_core_config()

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
        time_steps: int = 1,
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

        print("\nAll groups after tiling:")
        for grp in all_groups:
            print(grp.info("   "))

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

        for out_grp in self.output_groups:
            out_grp.set_lcn(required_steps=time_steps)

        self.routing_groups = [
            grp for grp in self.groups if isinstance(grp, RoutingGroup)
        ]

        for rg in self.routing_groups:
            rg.allocate_neurons()

        print("\nAll groups after neuron allocation:")
        for rg in all_groups:
            print(rg.routing_summary())

        # set core placements' coord, and generate detailed dest info for each neuron
        self.routing()

        print("\nAfter routing:")
        # for grp in all_groups:
        #     print(grp.info())

        for rg in all_groups:
            print(rg.routing_summary(prefix="    "))

        self.set_detail_dest()

        for rg in self.routing_groups:
            self.coreplacements.extend(rg.core_placements)

        self.set_auto_core_config()

        self.coreplacements, self.global_starts = set_global_signal(self.coreplacements)
        print("Global signal relative offset:")
        for thread_id, offset in self.global_starts.items():
            print(f"    Thread {thread_id}: {offset}")

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
