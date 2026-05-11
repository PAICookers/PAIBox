import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
from google.protobuf.json_format import MessageToJson
from paicorelib import CoordZXYOffset, FrameArrayType

from paibox.paiir import PAIIRGraph

from .coreplacement import CorePlacement
from .frame_cheader import write_c_array_close, write_c_array_decl
from .global_signal import set_global_signal
from .group_tile import tile_groups
from .op_node import AllNode, InputElem, Neuron, RemapElem, build_nodes
from .proto import PROTO_SCHEMA_VERSION
from .proto.config_pb2 import (
    CompileArtifacts,
    ConfigFrames,
    InputTensorMapping,
    OutputTensorMapping,
)
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

SuffixType = Literal["bin", "hex"]
WordOrder = Literal["high_first", "low_first"]
TargetPlatform = Literal["x86", "riscv"]


def export_single_framearray(
    frame_array: FrameArrayType, file: TextIO, prefix: str = ""
) -> None:
    lines = []
    for frame in frame_array:
        h = f"{frame:016x}"
        lines.append(f"{prefix}{'_'.join(h[i : i + 4] for i in range(0, 16, 4))}")
    if lines:
        file.write("\n".join(lines) + "\n")


def export_framearray_to_bit(
    frame_array: FrameArrayType,
    file: TextIO,
    prefix: str = "",
    suffix: SuffixType = "bin",
) -> None:
    if suffix == "bin":
        h_fmt = "0b{:032b}"
        l_fmt = "0b{:032b}"
    elif suffix == "hex":
        h_fmt = "0x{:08X}"
        l_fmt = "0x{:08X}"
    else:
        raise ValueError("suffix must be 'bin' or 'hex'")

    lines = [
        f"{prefix}{h_fmt.format((f >> 32) & 0xFFFFFFFF)},{l_fmt.format(f & 0xFFFFFFFF)},"
        for f in frame_array
    ]
    if lines:
        file.write("\n".join(lines) + "\n")


def export_framearray_to_int32(
    frame_array: FrameArrayType, word_order: WordOrder = "high_first"
) -> list[int]:
    n = len(frame_array)
    result = [0] * (n * 2)
    for i, f in enumerate(frame_array):
        lo = f & 0xFFFFFFFF
        hi = (f >> 32) & 0xFFFFFFFF
        if word_order == "high_first":
            result[i * 2] = hi
            result[i * 2 + 1] = lo
        else:
            result[i * 2] = lo
            result[i * 2 + 1] = hi
    return result


class Mapper:
    def __init__(self) -> None:
        self.groups: list[RoutingGroup | RemapGroup] = []
        self.routing_groups: list[RoutingGroup] = []
        self.nodes: list[AllNode] = []
        self.output_groups: list[OutputGroup] = []
        self.input_groups: list[InputGroup] = []
        self.coreplacements: list[CorePlacement] = []
        self.global_starts: dict[int, CoordZXYOffset] = {}

    def _iter_core_frames(self):
        for cp in self.coreplacements:
            yield cp.to_frame()

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

    def export_cheader_file(
        self, output_path: str | Path, suffix: SuffixType = "bin"
    ) -> None:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        paths = [
            (out / "frame_type1.h", "config_frame1"),
            (out / "frame_type2.h", "config_frame2"),
            (out / "frame_type3.h", "config_frame3"),
        ]
        with ExitStack() as stack:
            files = [stack.enter_context(p.open("w")) for p, _ in paths]
            for f, (_, name) in zip(files, paths):
                write_c_array_decl(f, name)
            for f1, f2, f3 in self._iter_core_frames():
                export_framearray_to_bit(f1, files[0], "\t", suffix)
                if f2 is not None:
                    export_framearray_to_bit(f2, files[1], "\t", suffix)
                if f3 is not None:
                    export_framearray_to_bit(f3, files[2], "\t", suffix)
            for f in files:
                write_c_array_close(f)

    def export_cheader_merge(
        self, output_path: str | Path, suffix: SuffixType = "bin"
    ) -> None:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        frame_path = out / "frame_type.h"
        with frame_path.open("w") as frame_file:
            write_c_array_decl(frame_file, "config_frame")
            for f1, f2, f3 in self._iter_core_frames():
                export_framearray_to_bit(f1, frame_file, "\t", suffix)
                if f2 is not None:
                    export_framearray_to_bit(f2, frame_file, "\t", suffix)
                if f3 is not None:
                    export_framearray_to_bit(f3, frame_file, "\t", suffix)
            write_c_array_close(frame_file)

    def export_merge(self, output_path: str | Path) -> None:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        frame_path = out / "frame_type.txt"
        with frame_path.open("w") as frame_file:
            for cp, (f1, f2, f3) in zip(self.coreplacements, self._iter_core_frames()):
                frame_file.write(f"# Core at coord ({cp.coord.x}, {cp.coord.y}):\n")
                frame_file.write("\ttype1:\n")
                export_single_framearray(f1, frame_file, prefix="\t\t0x")
                frame_file.write("\ttype2:\n")
                if f2 is not None:
                    export_single_framearray(f2, frame_file, prefix="\t\t0x")
                frame_file.write("\ttype3:\n")
                if f3 is not None:
                    export_single_framearray(f3, frame_file, prefix="\t\t0x")

    def export(self, output_path: str | Path) -> None:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        frame1_path = out / "frame_type1.txt"
        frame2_path = out / "frame_type2.txt"
        frame3_path = out / "frame_type3.txt"
        with (
            frame1_path.open("w") as frame1_file,
            frame2_path.open("w") as frame2_file,
            frame3_path.open("w") as frame3_file,
        ):
            for cp, (f1, f2, f3) in zip(self.coreplacements, self._iter_core_frames()):
                coord_line = f"# Core at coord ({cp.coord.x}, {cp.coord.y}):\n"
                frame1_file.write(coord_line)
                frame2_file.write(coord_line)
                frame3_file.write(coord_line)

                export_single_framearray(f1, frame1_file, prefix="\t0x")
                if f2 is not None:
                    export_single_framearray(f2, frame2_file, prefix="\t0x")
                if f3 is not None:
                    export_single_framearray(f3, frame3_file, prefix="\t0x")

    def export_proto(
        self,
        output_path: str | Path,
        target_platform: TargetPlatform = "riscv",
        export_python: bool = True,
        debug: bool = True,
        word_order: WordOrder = "high_first",
    ) -> Path:
        if target_platform not in ("x86", "riscv"):
            raise ValueError("target_platform must be 'x86' or 'riscv'")

        proto_dir = Path(__file__).parent / "proto"
        proto_out_dir = Path(output_path) / "proto"
        proto_out_dir.mkdir(parents=True, exist_ok=True)

        pb_path = proto_out_dir / "config.pb"
        pb_text_path = proto_out_dir / "config.json"

        proto_files = ["config.proto"]
        if export_python and target_platform == "x86":
            proto_files.extend(["config_pb2.py", "config_pb2.pyi"])

        for file_name in proto_files:
            src_file = proto_dir / file_name
            if not src_file.exists():
                raise FileNotFoundError(src_file)
            shutil.copy2(src_file, proto_out_dir / file_name)

        artifacts = CompileArtifacts()
        artifacts.schema_version = PROTO_SCHEMA_VERSION
        io_mapping = artifacts.io_mapping

        for thread_id, global_start in self.global_starts.items():
            thread_mapping = io_mapping.threads.add()
            thread_mapping.thread_id = thread_id
            thread_mapping.root_core_offset.xy = global_start.z
            thread_mapping.root_core_offset.x = global_start.x
            thread_mapping.root_core_offset.y = global_start.y

            input_mappings_by_name: dict[str, InputTensorMapping] = {}

            for in_grp in self.input_groups:
                if in_grp.thread_id != thread_id:
                    continue
                for elem, dest in in_grp.dest_infos.items():
                    input_name = elem.target.raw_node.name
                    if input_name not in input_mappings_by_name:
                        input_mapping = thread_mapping.input_mappings.items.add()
                        input_mapping.name = input_name
                        input_mapping.shape.size.extend(list(elem.target.shape))
                        input_mappings_by_name[input_name] = input_mapping

                    input_mapping = input_mappings_by_name[input_name]
                    input_entry = input_mapping.entries.add()
                    input_entry.elem_idx = elem.index.idx
                    input_entry.copy_id = elem.index.copy_id
                    input_entry.bit_width = elem.output_bit_num
                    input_entry.tick_relative = dest.tick_relative
                    input_entry.addr_axon = dest.addr_axon
                    input_entry.core_offset.xy = dest.addr_core_xy
                    input_entry.core_offset.x = dest.addr_core_x
                    input_entry.core_offset.y = dest.addr_core_y
                    input_entry.copy_count.xy = dest.addr_copy_xy
                    input_entry.copy_count.x = dest.addr_copy_x
                    input_entry.copy_count.y = dest.addr_copy_y
                    input_entry.target_lcn = in_grp.dest_lcn[elem]

            output_mappings_by_name: dict[str, OutputTensorMapping] = {}
            for out_grp in self.output_groups:
                if out_grp.thread_id != thread_id:
                    continue
                for axon_bit_idx, elem in sorted(
                    out_grp.axon_bit_allocator.axon_infos, key=lambda item: item[0]
                ):
                    output_name = elem.target.raw_node.name
                    if output_name not in output_mappings_by_name:
                        output_mapping = thread_mapping.output_mappings.items.add()
                        output_mapping.name = output_name
                        output_mapping.shape.size.extend(list(elem.target.shape))
                        output_mappings_by_name[output_name] = output_mapping

                    output_mapping = output_mappings_by_name[output_name]
                    output_entry = output_mapping.entries.add()
                    output_entry.elem_idx = elem.index.idx
                    output_entry.copy_id = elem.index.copy_id
                    output_entry.bit_width = elem.output_bit_num
                    output_entry.axon_bit_idx = axon_bit_idx

        for f1, f2, f3 in self._iter_core_frames():
            artifacts.config_frames.words.extend(
                export_framearray_to_int32(f1, word_order)
            )
            if f2 is not None:
                artifacts.config_frames.words.extend(
                    export_framearray_to_int32(f2, word_order)
                )
            if f3 is not None:
                artifacts.config_frames.words.extend(
                    export_framearray_to_int32(f3, word_order)
                )

        artifacts.config_frames.word_order = (
            ConfigFrames.HIGH_FIRST
            if word_order == "high_first"
            else ConfigFrames.LOW_FIRST
        )

        pb_path.write_bytes(artifacts.SerializeToString())
        if debug:
            pb_text_path.write_text(
                MessageToJson(artifacts, always_print_fields_with_no_presence=True)
            )

        return pb_path

    def export_frame_numpy(
        self, output_path: str | Path, export_merged_frames: bool = True
    ) -> None:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        typed_parts: list[list[FrameArrayType]] = [[], [], []]
        merged_parts: list[FrameArrayType] = []
        for f1, f2, f3 in self._iter_core_frames():
            typed_parts[0].append(f1)
            merged_parts.append(f1)
            if f2 is not None:
                typed_parts[1].append(f2)
                merged_parts.append(f2)
            if f3 is not None:
                typed_parts[2].append(f3)
                merged_parts.append(f3)

        for idx, parts in enumerate(typed_parts, start=1):
            if parts:
                np.save(out / f"frame_type{idx}.npy", np.concatenate(parts, axis=0))

        if export_merged_frames and merged_parts:
            np.save(out / "frames.npy", np.concatenate(merged_parts, axis=0))

    def export_artifacts(
        self,
        output_path: str | Path,
        suffix: SuffixType = "bin",
        target_platform: TargetPlatform = "riscv",
        word_order: WordOrder = "high_first",
        export_merged_frames: bool = True,
        debug: bool = True,
        export_proto_python: bool = True,
    ) -> None:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        if debug:
            self.export(out)
            if export_merged_frames:
                self.export_merge(out)

        if target_platform == "x86":
            self.export_frame_numpy(out, export_merged_frames)
        elif target_platform == "riscv":
            self.export_cheader_file(out, suffix)
            if export_merged_frames:
                self.export_cheader_merge(out, suffix)
        else:
            raise ValueError("target_platform must be 'x86' or 'riscv'")

        self.export_proto(
            out,
            target_platform,
            export_python=export_proto_python,
            debug=debug,
            word_order=word_order,
        )

    def compile(
        self,
        pai_graph: PAIIRGraph,
        output_path: str | Path | None = None,
        suffix: SuffixType = "bin",
        *,
        target_platform: TargetPlatform = "riscv",
        word_order: WordOrder = "high_first",
        export_merged_frames: bool = True,
        debug: bool = True,
        export_proto_python: bool = True,
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
            suffix,
            target_platform,
            word_order,
            export_merged_frames,
            debug,
            export_proto_python,
        )
