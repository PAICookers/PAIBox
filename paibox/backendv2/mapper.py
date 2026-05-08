from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import TextIO

import numpy as np
import torch
from google.protobuf import text_format
from paicorelib import CoordZXYOffset, FrameArrayType, find_coordxy_shortest_path

from paibox.paiir import PAIIRGraph

from .config_pb2 import Config, InputInfoWithEntry, OutputInfoWithEntry
from .coreplacement import CorePlacement
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


def export_framearray_to_int32(
    frame_array: FrameArrayType,
) -> list[int]:
    result = []
    for frame in frame_array:
        low32 = frame & 0xFFFFFFFF
        high32 = (frame >> 32) & 0xFFFFFFFF
        result.append(low32)
        result.append(high32)
    return result


class Mapper:
    def __init__(self):
        self.groups: list[RoutingGroup | RemapGroup] = []
        self.routing_groups: list[RoutingGroup] = []
        self.nodes: list[AllNode] = []
        self.output_groups: list[OutputGroup] = []
        self.input_groups: list[InputGroup] = []
        self.coreplacements: list[CorePlacement] = []
        self.global_starts: dict[int, CoordZXYOffset] = {}

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

    def routing(self):
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
            for core_placement in self.coreplacements:
                core_frame_type1, core_frame_type2, core_frame_type3 = (
                    core_placement.to_frame()
                )
                # export core_frame_type1 and core_frame_type3 to output_path
                export_framearray_to_bit(core_frame_type1, frame1_file, "\t", base=base)
                if core_frame_type2 is not None:
                    export_framearray_to_bit(
                        core_frame_type2, frame2_file, "\t", base=base
                    )
                if core_frame_type3 is not None:
                    export_framearray_to_bit(
                        core_frame_type3, frame3_file, "\t", base=base
                    )

            frame1_file.write("};\n")
            frame2_file.write("};\n")
            frame3_file.write("};\n")

    def export_cheader_merge(self, output_path: str, base: str = "bin"):
        os.makedirs(output_path, exist_ok=True)
        frame_path = output_path + "/frame_type.h"
        with (open(frame_path, "w") as frame_file,):
            frame_file.write(
                'volatile unsigned int config_frame[] __attribute__((section(".large_const_data"))) ={\n'
            )
            for core_placement in self.coreplacements:
                core_frame_type1, core_frame_type2, core_frame_type3 = (
                    core_placement.to_frame()
                )
                # export core_frame_type1 and core_frame_type3 to output_path
                export_framearray_to_bit(core_frame_type1, frame_file, "\t", base=base)
                if core_frame_type2 is not None:
                    export_framearray_to_bit(
                        core_frame_type2, frame_file, "\t", base=base
                    )
                if core_frame_type3 is not None:
                    export_framearray_to_bit(
                        core_frame_type3, frame_file, "\t", base=base
                    )

            frame_file.write("};\n")

    def export_merge(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        frame_path = output_path + "/frame_type.txt"
        with (open(frame_path, "w") as frame_file,):
            for core_placement in self.coreplacements:
                frame_file.write(
                    f"# Core at coord ({core_placement.coord.x}, {core_placement.coord.y}):\n"
                )

                core_frame_type1, core_frame_type2, core_frame_type3 = (
                    core_placement.to_frame()
                )
                # export core_frame_type1 and core_frame_type3 to output_path
                # framearray is np.ndarray of np.uint64 with shape (n_frames, )
                # print each frame with 16 hex digits each line
                frame_file.write("\ttype1:\n")
                export_single_framearray(core_frame_type1, frame_file, prefix="\t\t0x")
                frame_file.write("\ttype2:\n")
                if core_frame_type2 is not None:
                    export_single_framearray(
                        core_frame_type2, frame_file, prefix="\t\t0x"
                    )
                frame_file.write("\ttype3:\n")
                if core_frame_type3 is not None:
                    export_single_framearray(
                        core_frame_type3, frame_file, prefix="\t\t0x"
                    )

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
            for core_placement in self.coreplacements:
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
                export_single_framearray(core_frame_type1, frame1_file, prefix="\t0x")
                if core_frame_type2 is not None:
                    export_single_framearray(
                        core_frame_type2, frame2_file, prefix="\t0x"
                    )
                if core_frame_type3 is not None:
                    export_single_framearray(
                        core_frame_type3, frame3_file, prefix="\t0x"
                    )

    def export_meta_info(self, output_path: str, info: dict):
        os.makedirs(output_path, exist_ok=True)
        meta_info_path = output_path + "/meta_info.txt"
        with open(meta_info_path, "w") as meta_file:
            for key, value in info.items():
                meta_file.write(f"{key}: {value}\n")

    def _load_pb2_module(self, proto_path: str):
        proto_dir = os.path.dirname(os.path.abspath(proto_path))
        proto_name = os.path.splitext(os.path.basename(proto_path))[0]
        out_dir = tempfile.mkdtemp(prefix="pb_gen_")

        print(f"Compiling proto file {proto_path} to Python module...")
        print(
            f"Proto dir: {proto_dir}, Proto name: {proto_name}, Output dir: {out_dir}"
        )

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "grpc_tools.protoc",
                    f"--proto_path={proto_dir}",
                    f"--python_out={out_dir}",
                    proto_path,
                ],
                check=True,
            )
        except (FileNotFoundError, ModuleNotFoundError) as e:
            raise RuntimeError("需要 grpcio-tools: pip install grpcio-tools") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"protoc 编译失败: {e}") from e

        pb2_file = os.path.join(out_dir, f"{proto_name}_pb2.py")
        spec = importlib.util.spec_from_file_location(f"{proto_name}_pb2", pb2_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def export_proto(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)

        pb_path = os.path.join(output_path, "config.pb")
        pb_text_path = os.path.join(output_path, "config.txt")

        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)

        # 4. 需要复制的文件列表
        files_to_copy = ["config.proto", "config_pb2.pyi", "config_pb2.py"]

        # 5. 执行复制
        for file_name in files_to_copy:
            src_file = os.path.join(current_dir, file_name)
            dest_file = os.path.join(output_path, file_name)

            if os.path.exists(src_file):
                shutil.copy2(src_file, dest_file)
                print(f"已复制: {file_name} -> {output_path}")
            else:
                print(f"跳过: 未找到文件 {src_file}")

        # proto_path = os.path.join(current_dir, "config.proto")
        # pb2 = self._load_pb2_module(proto_path)
        # cfg = pb2.Config()

        cfg = Config()

        cfg.version = 1

        io_config = cfg.io_config

        for thread_id, global_start in self.global_starts.items():
            single_thread_io_config = io_config.thread_io.add()
            single_thread_io_config.thread_id = 0
            single_thread_io_config.root_core.xy = global_start.z
            single_thread_io_config.root_core.x = global_start.x
            single_thread_io_config.root_core.y = global_start.y

            input_info_with_entry_dict: dict[str, InputInfoWithEntry] = {}

            for in_grp in self.input_groups:
                if in_grp.thread_id != thread_id:
                    continue
                for elem, dest in in_grp.dest_infos.items():
                    input_name = elem.target.raw_node.name
                    if input_name not in input_info_with_entry_dict:
                        info_with_entry = (
                            single_thread_io_config.input_info_list.input_info.add()
                        )
                        info_with_entry.name = input_name
                        info_with_entry.shape.size.extend(list(elem.target.shape))
                        input_info_with_entry_dict[input_name] = info_with_entry

                    info_with_entry = input_info_with_entry_dict[input_name]
                    entry_list = info_with_entry.input_entry_list
                    entry = entry_list.entry.add()
                    entry.idx = elem.index.idx
                    entry.copy_id = elem.index.copy_id
                    entry.bit_num = elem.output_bit_num
                    entry.tick_relative = dest.tick_relative
                    entry.addr_axon = dest.addr_axon
                    entry.addr_core_xy = dest.addr_core_xy
                    entry.addr_core_x = dest.addr_core_x
                    entry.addr_core_y = dest.addr_core_y
                    entry.addr_copy_xy = dest.addr_copy_xy
                    entry.addr_copy_x = dest.addr_copy_x
                    entry.addr_copy_y = dest.addr_copy_y
                    entry.target_lcn = in_grp.dest_lcn[elem]

            output_info_with_entry_dict: dict[str, OutputInfoWithEntry] = {}
            for out_grp in self.output_groups:
                if out_grp.thread_id != thread_id:
                    continue
                for bit_count, elem in sorted(
                    out_grp.axon_bit_allocator.axon_infos, key=lambda item: item[0]
                ):
                    output_name = elem.target.raw_node.name
                    if output_name not in output_info_with_entry_dict:
                        info_with_entry = (
                            single_thread_io_config.output_info_list.output_info.add()
                        )
                        info_with_entry.name = output_name
                        info_with_entry.shape.size.extend(list(elem.target.shape))
                        output_info_with_entry_dict[output_name] = info_with_entry

                    info_with_entry = output_info_with_entry_dict[output_name]
                    entry_list = info_with_entry.output_entry_list
                    entry = entry_list.entry.add()
                    entry.idx = elem.index.idx
                    entry.copy_id = elem.index.copy_id
                    entry.bit_num = elem.output_bit_num
                    entry.axon_addr = bit_count

        for core_placement in self.coreplacements:
            core_frame_type1, core_frame_type2, core_frame_type3 = (
                core_placement.to_frame()
            )
            cfg.frame_list.frame.extend(export_framearray_to_int32(core_frame_type1))
            if core_frame_type2 is not None:
                cfg.frame_list.frame.extend(
                    export_framearray_to_int32(core_frame_type2)
                )
            if core_frame_type3 is not None:
                cfg.frame_list.frame.extend(
                    export_framearray_to_int32(core_frame_type3)
                )

        # 4) 序列化二进制
        with open(pb_path, "wb") as f:
            f.write(cfg.SerializeToString())

        # 5) 序列化文本
        with open(pb_text_path, "w") as f:
            f.write(text_format.MessageToString(cfg))

        return pb_path

    def export_frame_numpy(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        frames = []
        for idx, core_placement in enumerate(self.coreplacements):
            core_frame_type1, core_frame_type2, core_frame_type3 = (
                core_placement.to_frame()
            )
            frames.append(core_frame_type1)
            if core_frame_type2 is not None:
                frames.append(core_frame_type2)
            if core_frame_type3 is not None:
                frames.append(core_frame_type3)
        all_frames = np.concatenate(frames, axis=0)
        np.save(os.path.join(output_path, "frames.npy"), all_frames)

    def compile(
        self,
        pai_graph: PAIIRGraph,
        base: str = "bin",  # 新增参数，指定导出格式
        output_path: str | None = None,
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
        print(f"Global signal relative offset:")
        for thread_id, offset in self.global_starts.items():
            print(f"    Thread {thread_id}: {offset}")

        # export to hardware executable format
        if output_path is None:
            env_output_path = os.environ.get("PAIBOX_OUTPUT_PATH")
            if env_output_path is not None:
                output_path = os.path.join(env_output_path, "frame_out")
            else:
                output_path = "./output"
        self.export(output_path=output_path)
        self.export_merge(output_path=output_path)
        self.export_cheader_file(output_path=output_path, base=base)
        self.export_cheader_merge(output_path=output_path, base=base)
        self.export_frame_numpy(output_path=output_path)
        # self.export_io(output_path=output_path)

        self.export_proto(output_path=output_path)
