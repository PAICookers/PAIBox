import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
from google.protobuf.json_format import MessageToJson
from paicorelib import CoordZXYOffset, FrameArrayType

from paibox.paiir import PAIIRGraph
from paibox.paiir.ir.signal_domain import SignalDomain

from .coreplacement import CorePlacement
from .frame_cheader import write_c_array_close, write_c_array_decl
from .global_signal import set_global_signal
from .group_tile import tile_groups
from .op_node import AllNode, InputElem, Neuron, RemapElem, build_nodes
from .proto import PROTO_SCHEMA_VERSION
from .proto.compile_artifacts_pb2 import (
    CompileArtifacts,
    ConfigFrames,
    InputTensorMapping,
    OutputEntry,
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

LiteralFormat = Literal["bin", "hex"]
WordOrder = Literal["high_first", "low_first"]
TargetPlatform = Literal["x86", "riscv", "all"]


def _set_output_entry_kind(output_entry: OutputEntry, elem: SourceElem) -> None:
    domain = elem.target.raw_node.signal_semantics.output_domain
    if domain is SignalDomain.VALUE:
        kind = OutputEntry.DATA
    else:
        kind = OutputEntry.VOLTAGE

    if kind == OutputEntry.DATA:
        if elem.output_bit_num > 8:
            raise ValueError(
                f"DATA output {elem} has unsupported bit width "
                f"{elem.output_bit_num}; expected <= 8."
            )
    else:
        if elem.output_bit_num != 32:
            raise ValueError(
                f"VOLTAGE output {elem} has bit width "
                f"{elem.output_bit_num}; expected 32."
            )

    output_entry.kind = kind


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
    literal_format: LiteralFormat = "bin",
) -> None:
    if literal_format == "bin":
        h_fmt = "0b{:032b}"
        l_fmt = "0b{:032b}"
    elif literal_format == "hex":
        h_fmt = "0x{:08X}"
        l_fmt = "0x{:08X}"
    else:
        raise ValueError("literal_format must be 'bin' or 'hex'")

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


def _for_each_frame_type(
    frames: tuple[FrameArrayType, FrameArrayType | None, FrameArrayType | None],
    callback,
) -> None:
    for idx, frame_array in enumerate(frames, start=1):
        if frame_array is not None:
            callback(idx, frame_array)


def _resolve_platform_exports(
    target_platform: TargetPlatform, debug: bool
) -> tuple[bool, bool]:
    if target_platform == "x86":
        export_x86 = True
        export_riscv = False
    elif target_platform == "riscv":
        export_x86 = False
        export_riscv = True
    elif target_platform == "all":
        export_x86 = True
        export_riscv = True
    else:
        raise ValueError("target_platform must be 'x86', 'riscv', or 'all'")

    if debug:
        export_x86 = True
        export_riscv = True

    return export_x86, export_riscv


class Mapper:
    def __init__(self) -> None:
        self.groups: list[RoutingGroup | RemapGroup] = []
        self.routing_groups: list[RoutingGroup] = []
        self.nodes: list[AllNode] = []
        self.output_groups: list[OutputGroup] = []
        self.input_groups: list[InputGroup] = []
        self.coreplacements: list[CorePlacement] = []
        self.global_starts: dict[int, CoordZXYOffset] = {}

    def _iter_frame_triplets(self):
        for cp in self.coreplacements:
            yield cp, cp.to_frame()

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
        self, output_path: str | Path, literal_format: LiteralFormat = "bin"
    ) -> None:
        """Export per-type config frames as C header arrays."""
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        paths = [
            (out / "cfg_frame1.h", "config_frame1"),
            (out / "cfg_frame2.h", "config_frame2"),
            (out / "cfg_frame3.h", "config_frame3"),
        ]
        with ExitStack() as stack:
            files = [stack.enter_context(p.open("w")) for p, _ in paths]
            for f, (_, name) in zip(files, paths):
                write_c_array_decl(f, name)
            for _, frames in self._iter_frame_triplets():
                _for_each_frame_type(
                    frames,
                    lambda idx, frame_array: export_framearray_to_bit(
                        frame_array, files[idx - 1], "\t", literal_format
                    ),
                )
            for f in files:
                write_c_array_close(f)

    def export_cheader_merge(
        self, output_path: str | Path, literal_format: LiteralFormat = "bin"
    ) -> None:
        """Export all config frame types into one merged C header array."""
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        frame_path = out / "cfg_frames.h"
        with frame_path.open("w") as frame_file:
            write_c_array_decl(frame_file, "config_frame")
            for _, frames in self._iter_frame_triplets():
                _for_each_frame_type(
                    frames,
                    lambda _, frame_array: export_framearray_to_bit(
                        frame_array, frame_file, "\t", literal_format
                    ),
                )
            write_c_array_close(frame_file)

    def export_merge(self, output_path: str | Path) -> None:
        """Export all config frame types into one human-readable text file."""
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        frame_path = out / "cfg_frames.txt"
        with frame_path.open("w") as frame_file:
            for cp, frames in self._iter_frame_triplets():
                frame_file.write(
                    f"# Core at coord (X,Y)=({cp.coord.x},{cp.coord.y}):\n"
                )
                _for_each_frame_type(
                    frames,
                    lambda idx, frame_array: (
                        frame_file.write(f"\ttype{idx}:\n"),
                        export_single_framearray(
                            frame_array, frame_file, prefix="\t\t0x"
                        ),
                    ),
                )

    def export_txt(self, output_path: str | Path) -> None:
        """Export per-type config frames as human-readable text files."""
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        frame1_path = out / "cfg_frame1.txt"
        frame2_path = out / "cfg_frame2.txt"
        frame3_path = out / "cfg_frame3.txt"
        with (
            frame1_path.open("w") as frame1_file,
            frame2_path.open("w") as frame2_file,
            frame3_path.open("w") as frame3_file,
        ):
            files = [frame1_file, frame2_file, frame3_file]
            for cp, frames in self._iter_frame_triplets():
                coord_line = f"# Core at coord (X,Y)=({cp.coord.x},{cp.coord.y}):\n"
                frame1_file.write(coord_line)
                frame2_file.write(coord_line)
                frame3_file.write(coord_line)
                _for_each_frame_type(
                    frames,
                    lambda idx, frame_array: export_single_framearray(
                        frame_array, files[idx - 1], prefix="\t0x"
                    ),
                )

    def export_proto(
        self,
        output_path: str | Path,
        target_platform: TargetPlatform = "riscv",
        word_order: WordOrder = "high_first",
        export_python: bool = True,
        debug: bool = False,
    ) -> Path:
        """Export protobuf artifacts for config frames and I/O mappings."""
        export_x86, _ = _resolve_platform_exports(target_platform, debug)

        proto_dir = Path(__file__).parent / "proto"
        proto_out_dir = Path(output_path) / "proto"
        proto_out_dir.mkdir(parents=True, exist_ok=True)

        pb_path = proto_out_dir / "config.pb"
        pb_text_path = proto_out_dir / "config.json"

        proto_files = ["compile_artifacts.proto"]
        if export_python and export_x86:
            proto_files.extend(
                ["compile_artifacts_pb2.py", "compile_artifacts_pb2.pyi"]
            )

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
                thread_mapping.output_mappings.target_lcn = out_grp.lcn
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
                    _set_output_entry_kind(output_entry, elem)

        config_words: list[int] = []
        for _, frames in self._iter_frame_triplets():
            _for_each_frame_type(
                frames,
                lambda _, frame_array: config_words.extend(
                    export_framearray_to_int32(frame_array, word_order)
                ),
            )
        artifacts.config_frames.words.extend(config_words)

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

    def export_frame_npy(
        self, output_path: str | Path, export_merged_frames: bool = True
    ) -> None:
        """Export frame arrays as explicit little-endian uint64 NumPy files."""
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        typed_parts: list[list[FrameArrayType]] = [[], [], []]
        typed_counts = [0, 0, 0]
        merged_parts: list[FrameArrayType] | None = [] if export_merged_frames else None
        merged_count = 0
        for _, frames in self._iter_frame_triplets():
            _for_each_frame_type(
                frames,
                lambda idx, frame_array: (
                    typed_parts[idx - 1].append(frame_array),
                    typed_counts.__setitem__(
                        idx - 1, typed_counts[idx - 1] + len(frame_array)
                    ),
                    (
                        merged_parts.append(frame_array)
                        if merged_parts is not None
                        else None
                    ),
                ),
            )
            if merged_parts is not None:
                merged_count += sum(
                    len(frame_array)
                    for frame_array in frames
                    if frame_array is not None
                )

        for idx, parts in enumerate(typed_parts, start=1):
            if parts:
                frame_array = np.zeros(typed_counts[idx - 1], dtype="<u8")
                cursor = 0
                for part in parts:
                    n = len(part)
                    frame_array[cursor : cursor + n] = part
                    cursor += n
                np.save(out / f"cfg_frame{idx}.npy", frame_array)

        if merged_parts:
            frame_array = np.zeros(merged_count, dtype="<u8")
            cursor = 0
            for part in merged_parts:
                n = len(part)
                frame_array[cursor : cursor + n] = part
                cursor += n
            np.save(out / "cfg_frames.npy", frame_array)

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
        export_x86, export_riscv = _resolve_platform_exports(target_platform, debug)

        if debug:
            self.export_txt(out)
            if export_merged_frames:
                self.export_merge(out)

        if export_x86:
            self.export_frame_npy(out, export_merged_frames)

        if export_riscv:
            self.export_cheader_file(out, literal_format)
            if export_merged_frames:
                self.export_cheader_merge(out, literal_format)

        self.export_proto(out, target_platform, word_order, export_proto_python, debug)

    def compile(
        self,
        pai_graph: PAIIRGraph,
        output_path: str | Path | None = None,
        literal_format: LiteralFormat = "bin",
        *,
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
            debug: Whether to emit human-readable debug artifacts such as
                ``cfg_frame*.txt``, merged text frames, and ``proto/config.json``.
                When enabled, x86 and riscv platform artifacts are both
                exported regardless of ``target_platform``.
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
