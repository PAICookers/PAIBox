import shutil
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from paicorelib import (
    LCN_EX,
    AERPacketZXYCopy,
    CoordXY,
    CoordZXYOffset,
    OnlineCoreUpdateType,
    OnlineCoreWorkMode,
    OnlineFrameGenV2,
    find_coordxy_shortest_path,
)
from paicorelib.framelib.base import FramePackageHeaderV2
from paicorelib.framelib.frame_defs import FrameHeader, FramePackageType
from torch import Tensor, nn

from paibox.backendv2.core_config import TEST_DEST_CORE
from paibox.backendv2.coreplacement import OnlineCorePlacementV2
from paibox.backendv2.export.utils import export_framearray_to_int32
from paibox.backendv2.mapper import Mapper
from paibox.backendv2.output_route_offsets import route_coord_path, terminal_route_side
from paibox.backendv2.proto.compile_artifacts_pb2 import DataType, RuntimeParams
from paibox.backendv2.proto.runtime import (
    build_output_mapping_tables,
    decode_online_boundary_output_frames,
    decode_online_data_output_frames,
    encode_online_boundary_output_frames,
    encode_online_data_output_frames,
    encode_online_input_frames,
    iter_config_frame_u64,
    load_compile_artifacts,
    package_online_data_output_frames,
    parse_online_data_output_package_header,
    scatter_data_output,
    strip_online_data_output_package_header,
)
from paibox.paiir import compile_to_paiir, mark_online
from paibox.paiir.exceptions import GraphValidationError
from paibox.paiir.ir.calc_params import (
    OnlineCoreSemanticMode,
    OnlineCoreType,
    OnlineUpdateDirection,
)
from paibox.paiir.ir.op_node import OnlineCoreOp
from tests.paiir.conftest import make_img_1ch_28x28, make_vec_8d
from tests.paiir.online_test_utils import (
    MNISTFlattenOnlineLinear,
    OnlineLinear,
    TwoLayerOnlineLinear,
    WideOnlineLinear,
    export_online_graph,
    single_online_node,
    uniform_online_lcn_kwargs,
)

DEBUG_EXPORT_ROOT = Path(__file__).with_name("debug") / "online_mapper_export"


def _coord_from_offset_tuple(offset: tuple[int, int, int]) -> CoordXY:
    return CoordXY(offset[0] + offset[1], offset[0] + offset[2])


def _expected_core_major_words(mapper: Mapper, word_order: str) -> list[int]:
    words: list[int] = []
    for core_placement in mapper.coreplacements:
        for frame_array in core_placement.to_frame():
            if frame_array is not None:
                words.extend(export_framearray_to_int32(frame_array, word_order))
    return words


def _expected_core_major_frames(mapper: Mapper) -> np.ndarray:
    frame_parts: list[np.ndarray] = []
    for core_placement in mapper.coreplacements:
        for frame_array in core_placement.to_frame():
            if frame_array is not None:
                frame_parts.append(frame_array.astype(np.uint64, copy=False))

    if not frame_parts:
        return np.array([], dtype=np.uint64)

    return np.concatenate(frame_parts).astype(np.uint64, copy=False)


def _online_output_source_names(graph) -> tuple[set[str], set[str]]:
    forward_outputs: set[str] = set()
    update_outputs: set[str] = set()
    for output_node in graph.output_nodes():
        preds = graph.predecessors(output_node.name)
        assert len(preds) == 1
        pred = graph.nodes[preds[0]]
        assert isinstance(pred, OnlineCoreOp)
        if pred.core_params.semantic_mode is OnlineCoreSemanticMode.FORWARD:
            forward_outputs.add(pred.name)
        elif pred.core_params.semantic_mode is OnlineCoreSemanticMode.UPDATE:
            update_outputs.add(pred.name)
    return forward_outputs, update_outputs


def test_mapper_compile_exports_single_layer_online_frames_and_core_ticks():
    export_dir, graph, mapper = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, update_outputs = _online_output_source_names(graph)

    assert all(
        isinstance(core_placement, OnlineCorePlacementV2)
        for core_placement in mapper.coreplacements
    )
    assert len(mapper.coreplacements) == 4
    assert (export_dir / "cfg_frame1.npy").exists()
    assert (export_dir / "cfg_frame2.npy").exists()
    assert not (export_dir / "cfg_frame3.npy").exists()
    assert np.load(export_dir / "cfg_frame1.npy").dtype == np.dtype("<u8")
    assert np.load(export_dir / "cfg_frame2.npy").dtype == np.dtype("<u8")
    assert list(artifacts.config_frames.words) == _expected_core_major_words(
        mapper, "high_first"
    )
    assert len(artifacts.io_mapping.threads) == 1

    thread = artifacts.io_mapping.threads[0]
    assert len(thread.input_mappings.items) == 1
    input_mapping = thread.input_mappings.items[0]
    assert int(np.prod(input_mapping.shape.size)) == 8
    assert input_mapping.bit_width == 16
    assert len(input_mapping.entries) == 8
    assert {entry.dtype for entry in input_mapping.entries} == {DataType.FLOAT16}
    assert {entry.tick_relative for entry in input_mapping.entries} == {0}
    assert [entry.addr_axon for entry in input_mapping.entries] == list(range(8))

    assert len(thread.output_mappings.items) == 1
    output_mapping = thread.output_mappings.items[0]
    assert {item.name for item in thread.output_mappings.items} == forward_outputs
    assert output_mapping.name not in update_outputs
    assert output_mapping.bit_width == 16
    assert len(output_mapping.entries) == 4
    assert {entry.dtype for entry in output_mapping.entries} == {DataType.FLOAT16}
    assert [entry.axon_bit_idx for entry in output_mapping.entries] == list(range(4))
    assert len(thread.core_ticks) == 4
    assert {tuple(core_tick.nodes) for core_tick in thread.core_ticks} == {
        core_placement.node_names for core_placement in mapper.coreplacements
    }
    assert int(thread.runtime.timesteps) == 1
    assert int(thread.runtime.tick_depth) == 1
    assert int(thread.runtime.sync_steps) == 1
    assert int(thread.runtime.decode_mode) == int(RuntimeParams.STREAM)
    assert (
        int(input_mapping.tick.tick_start),
        int(input_mapping.tick.tick_duration),
        int(input_mapping.tick.tick_initial),
    ) == (1, 0, 1)
    assert (
        int(output_mapping.tick.tick_start),
        int(output_mapping.tick.tick_duration),
        int(output_mapping.tick.tick_initial),
    ) == (1, 0, 1)
    assert (
        mapper.coreplacements[-1].core_config.output_width
        == OnlineCoreUpdateType.WEIGHT
    )


def test_mapper_import_does_not_pull_offline_routing_stack():
    repo_root = Path(__file__).resolve().parents[2]
    blocked_modules = (
        "paibox.backendv2.global_signal",
        "paibox.backendv2.group_tile",
        "paibox.backendv2.output_completion",
        "paibox.backendv2.rg_build",
        "paibox.backendv2.route_solver",
        "paibox.backendv2.routing",
    )
    script = textwrap.dedent(f"""
        import importlib.abc
        import sys

        BLOCKED = {blocked_modules!r}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in BLOCKED:
                    raise ImportError(f"blocked import: {{fullname}}")
                return None

        sys.meta_path.insert(0, Blocker())
        from paibox.backendv2.mapper import Mapper
        assert Mapper.__name__ == "Mapper"
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_mapper_compile_exports_online_runtime_params_and_control_frames_from_public_timing():
    export_dir, _, mapper = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_runtime_timing",
        OnlineLinear(),
        compile_kwargs={"timesteps": 7, "auto_reset": False},
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    thread = artifacts.io_mapping.threads[0]

    assert mapper.timesteps == 7
    assert {core_placement.n_timestep for core_placement in mapper.coreplacements} == {
        7
    }
    assert {
        core_placement.core_params.tick_start
        for core_placement in mapper.coreplacements
    } == {1}
    assert {
        core_placement.core_params.tick_duration
        for core_placement in mapper.coreplacements
    } == {7}
    assert {
        core_placement.core_params.tick_initial
        for core_placement in mapper.coreplacements
    } == {0}
    assert int(thread.runtime.timesteps) == 7
    assert int(thread.runtime.tick_depth) == 1
    assert int(thread.runtime.sync_steps) == 7
    assert int(thread.runtime.decode_mode) == int(RuntimeParams.STREAM)
    assert {
        (
            int(core_tick.tick.tick_start),
            int(core_tick.tick.tick_duration),
            int(core_tick.tick.tick_initial),
        )
        for core_tick in thread.core_ticks
    } == {(1, 7, 0)}


def test_mapper_compile_rejects_mismatched_online_auto_reset_timesteps_override():
    graph = compile_to_paiir(
        mark_online(OnlineLinear()),
        make_vec_8d(),
        timesteps=7,
        auto_reset=True,
    )
    mapper = Mapper()

    with pytest.raises(ValueError, match="conflicts with auto-reset tick_initial"):
        mapper.compile(
            graph, DEBUG_EXPORT_ROOT / "mismatched_online_timesteps", timesteps=10
        )


def test_mapper_compile_rejects_conflicting_online_output_tick_inference():
    graph = compile_to_paiir(
        mark_online(OnlineLinear()),
        make_vec_8d(),
        timesteps=7,
        auto_reset=False,
    )
    update = single_online_node(graph, OnlineCoreSemanticMode.UPDATE)
    update.core_params.tick_duration = 9

    mapper = Mapper()
    with pytest.raises(
        ValueError,
        match="could not infer one runtime timesteps value from outputs",
    ):
        mapper.compile(
            graph,
            DEBUG_EXPORT_ROOT / "conflicting_online_output_timesteps",
            target_platform="x86",
            debug=True,
        )


def test_load_compile_artifacts_accepts_export_root_and_proto_dir():
    export_dir, _, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_load_compile_artifacts_path_forms",
        OnlineLinear(),
    )

    from_export_dir = load_compile_artifacts(export_dir)
    from_proto_dir = load_compile_artifacts(export_dir / "proto")
    from_pb_path = load_compile_artifacts(export_dir / "proto" / "config.pb")

    assert from_export_dir.SerializeToString() == from_pb_path.SerializeToString()
    assert from_proto_dir.SerializeToString() == from_pb_path.SerializeToString()


def test_iter_config_frame_u64_restores_exported_core_major_frames():
    export_dir, _, mapper = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_iter_config_frames",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir)

    config_frames = np.fromiter(
        iter_config_frame_u64(artifacts.config_frames),
        dtype=np.uint64,
    )

    np.testing.assert_array_equal(config_frames, _expected_core_major_frames(mapper))


def test_mapper_compile_preserves_online_work_mode_order_for_two_layers():
    _, _, mapper = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "two_layer",
        TwoLayerOnlineLinear(),
    )

    core_configs = [
        core_placement.core_config for core_placement in mapper.coreplacements
    ]

    assert [core_config.work_mode for core_config in core_configs] == [
        OnlineCoreWorkMode.FORWARD_INFERENCE,
        OnlineCoreWorkMode.FORWARD_INFERENCE,
        OnlineCoreWorkMode.LOSS_FN,
        OnlineCoreWorkMode.OUTPUT_LAYER_GRADIENT,
        OnlineCoreWorkMode.MIDDLE_LAYER_GRADIENT,
        OnlineCoreWorkMode.FORWARD_WEIGHT_UPDATE,
        OnlineCoreWorkMode.FORWARD_WEIGHT_UPDATE,
    ]

    for core_config, core_placement in zip(core_configs, mapper.coreplacements):
        assert core_config.update_core_xy == 0
        assert core_config.update_core_x == 0
        assert core_config.update_core_y == 0
        assert core_config.global_send == 0
        assert core_config.global_receive == 0
        test_offset, _ = find_coordxy_shortest_path(
            TEST_DEST_CORE, core_placement.coord
        )
        assert (
            core_config.test_core_xy,
            core_config.test_core_x,
            core_config.test_core_y,
        ) == (test_offset.z, test_offset.x, test_offset.y)


def test_mapper_compile_preserves_explicit_online_test_core_route():
    export_dir = DEBUG_EXPORT_ROOT / "single_layer_explicit_test_route"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(
        mark_online(OnlineLinear(), test_core_xy=1, test_core_x=-1, test_core_y=0),
        make_vec_8d(),
    )
    mapper = Mapper()
    mapper.compile(graph, export_dir, target_platform="x86", debug=True)

    for core_placement in mapper.coreplacements:
        core_config = core_placement.core_config
        assert (
            core_config.test_core_xy,
            core_config.test_core_x,
            core_config.test_core_y,
        ) == (
            1,
            -1,
            0,
        )


def test_mapper_compile_wraps_online_core_coords_after_first_row():
    class DeepOnlineLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(2, 2, bias=False) for _ in range(11)]
            )

        def forward(self, x: Tensor) -> Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

    _, _, mapper = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "deep_online_coords",
        DeepOnlineLinear(),
        sample_input=torch.randn(1, 2),
    )

    assert len(mapper.coreplacements) == 34
    assert mapper.coreplacements[31].coord == CoordXY(31, 0)
    assert mapper.coreplacements[32].coord == CoordXY(0, 1)
    assert mapper.coreplacements[33].coord == CoordXY(1, 1)


def test_mapper_compile_preserves_offline_online_graph_entry_on_forward_only():
    export_dir, graph, mapper = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "offline_graph_entry",
        OnlineLinear(),
        input_core=OnlineCoreType.OFFLINE,
    )
    assert (export_dir / "proto" / "config.pb").exists()

    placements_by_name = {
        core_placement.node_names[0]: core_placement
        for core_placement in mapper.coreplacements
    }
    forward = single_online_node(graph, OnlineCoreSemanticMode.FORWARD)

    for node in graph.nodes.values():
        if not isinstance(node, OnlineCoreOp):
            continue
        core_config = placements_by_name[node.name].core_config
        expected_input_core = (
            OnlineCoreType.OFFLINE
            if node.name == forward.name
            else OnlineCoreType.ONLINE
        )
        assert core_config.input_core == expected_input_core
        assert core_config.output_core == OnlineCoreType.ONLINE


def test_mapper_compile_preserves_online_offline_output_boundary_on_forward_only():
    export_dir, graph, mapper = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "offline_output_boundary",
        OnlineLinear(),
        output_core=OnlineCoreType.OFFLINE,
    )
    assert (export_dir / "proto" / "config.pb").exists()

    placements_by_name = {
        core_placement.node_names[0]: core_placement
        for core_placement in mapper.coreplacements
    }
    forward = single_online_node(graph, OnlineCoreSemanticMode.FORWARD)

    for node in graph.nodes.values():
        if not isinstance(node, OnlineCoreOp):
            continue
        core_config = placements_by_name[node.name].core_config
        expected_output_core = (
            OnlineCoreType.OFFLINE
            if node.name == forward.name
            else OnlineCoreType.ONLINE
        )
        assert core_config.input_core == OnlineCoreType.ONLINE
        assert core_config.output_core == expected_output_core


def test_mapper_compile_exports_unified_non_default_online_lcn_input_mapping():
    sample_input = torch.randn(1, 2048)
    export_dir, _, mapper = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "wide_lcn_2x",
        WideOnlineLinear(),
        sample_input=sample_input,
        **uniform_online_lcn_kwargs(LCN_EX.LCN_2X),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")

    core_configs = [
        core_placement.core_config for core_placement in mapper.coreplacements
    ]
    assert {core_config.lcn_at for core_config in core_configs} == {LCN_EX.LCN_2X}
    assert {core_config.target_lcn_at for core_config in core_configs} == {
        LCN_EX.LCN_2X
    }

    thread = artifacts.io_mapping.threads[0]
    input_mapping = thread.input_mappings.items[0]
    assert input_mapping.bit_width == 16
    assert {entry.target_lcn for entry in input_mapping.entries} == {int(LCN_EX.LCN_2X)}
    assert int(thread.output_mappings.target_lcn) == int(LCN_EX.LCN_2X)

    _, axon_width = OnlineFrameGenV2.LCN_TO_TS_AXON_WIDTHS[int(LCN_EX.LCN_2X)]
    axon_capacity = 1 << axon_width
    first_entry = input_mapping.entries[0]
    boundary_entry = input_mapping.entries[axon_capacity - 1]
    rollover_entry = input_mapping.entries[axon_capacity]
    last_entry = input_mapping.entries[2047]
    assert (first_entry.tick_relative, first_entry.addr_axon) == (0, 0)
    assert (boundary_entry.tick_relative, boundary_entry.addr_axon) == (
        0,
        axon_capacity - 1,
    )
    assert (rollover_entry.tick_relative, rollover_entry.addr_axon) == (1, 0)
    assert (last_entry.tick_relative, last_entry.addr_axon) == (
        2047 // axon_capacity,
        2047 % axon_capacity,
    )

    frames = encode_online_input_frames(
        artifacts,
        input_mapping.name,
        sample_input.numpy().astype(np.float16, copy=False),
        thread_id=int(thread.thread_id),
    )
    ts_axon_words = ((frames >> 8) & 0xFFFF).tolist()
    assert 0 in ts_axon_words
    assert axon_capacity - 1 in ts_axon_words
    assert axon_capacity in ts_axon_words
    assert 2047 in ts_axon_words


def test_mapper_compile_revalidates_phase1_online_update_contract():
    export_dir = DEBUG_EXPORT_ROOT / "single_layer_revalidate_backward_update"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(mark_online(OnlineLinear()), make_vec_8d())
    update = single_online_node(graph, OnlineCoreSemanticMode.UPDATE)
    update.core_params.update_direction = OnlineUpdateDirection.BACKWARD
    update.core_params.work_mode = None

    mapper = Mapper()
    with pytest.raises(
        GraphValidationError,
        match="BACKWARD_WEIGHT_UPDATE remains a later phase",
    ):
        mapper.compile(graph, export_dir, target_platform="x86", debug=True)


def test_mapper_compile_revalidates_phase1_online_update_routes():
    export_dir = DEBUG_EXPORT_ROOT / "single_layer_revalidate_update_route"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(mark_online(OnlineLinear()), make_vec_8d())
    update = single_online_node(graph, OnlineCoreSemanticMode.UPDATE)
    update.core_params.update_core_x = 1

    mapper = Mapper()
    with pytest.raises(
        GraphValidationError,
        match="update_core_x remains a later phase because",
    ):
        mapper.compile(graph, export_dir, target_platform="x86", debug=True)


def test_mapper_compile_exports_transform_aware_online_input_mapping():
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(8, 2, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x.permute(0, 2, 3, 1).flatten(1))

    export_dir = DEBUG_EXPORT_ROOT / "permute_flatten"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(mark_online(Model()), torch.randn(1, 2, 2, 2))
    mapper = Mapper()
    mapper.compile(graph, export_dir, target_platform="x86", debug=True)
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")

    input_mapping = artifacts.io_mapping.threads[0].input_mappings.items[0]
    addr_by_elem = {entry.elem_idx: entry.addr_axon for entry in input_mapping.entries}
    assert addr_by_elem == {0: 0, 1: 2, 2: 4, 3: 6, 4: 1, 5: 3, 6: 5, 7: 7}


def test_build_output_mapping_tables_exports_float16_forward_output():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_output_tables",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    thread = artifacts.io_mapping.threads[0]
    forward_outputs, update_outputs = _online_output_source_names(graph)

    output_tables = build_output_mapping_tables(
        artifacts,
        thread_id=int(thread.thread_id),
    )

    assert set(output_tables) == forward_outputs
    assert set(output_tables).isdisjoint(update_outputs)
    table = output_tables[next(iter(forward_outputs))]
    assert table.kind == thread.output_mappings.items[0].DATA
    assert table.target_lcn == int(thread.output_mappings.target_lcn)
    assert table.shape == tuple(thread.output_mappings.items[0].shape.size)
    assert table.bit_width == int(thread.output_mappings.items[0].bit_width)
    assert {
        axon_bit_idx: (
            entry.elem_idx,
            entry.copy_id,
            entry.bit_width,
            entry.dtype,
        )
        for axon_bit_idx, entry in table.entries_by_axon.items()
    } == {
        0: (0, 0, 16, DataType.FLOAT16),
        1: (1, 0, 16, DataType.FLOAT16),
        2: (2, 0, 16, DataType.FLOAT16),
        3: (3, 0, 16, DataType.FLOAT16),
    }


def test_build_output_mapping_tables_attach_online_boundary_metadata():
    export_dir, graph, mapper = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_output_boundary_metadata",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    thread = artifacts.io_mapping.threads[0]
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)
    table = output_tables[output_name]
    placements_by_name = {
        core_placement.node_names[0]: core_placement
        for core_placement in mapper.coreplacements
    }
    core_ticks_by_name = {
        node_name: core_tick
        for core_tick in thread.core_ticks
        for node_name in core_tick.nodes
    }

    assert table.boundary is not None
    assert table.boundary.producer_core_offset == (
        int(core_ticks_by_name[output_name].core_offset.xy),
        int(core_ticks_by_name[output_name].core_offset.x),
        int(core_ticks_by_name[output_name].core_offset.y),
    )
    assert table.boundary.test_core_offset == (
        placements_by_name[output_name].core_config.test_core_xy,
        placements_by_name[output_name].core_config.test_core_x,
        placements_by_name[output_name].core_config.test_core_y,
    )
    assert table.boundary.work_mode == int(OnlineCoreWorkMode.FORWARD_INFERENCE)
    assert table.boundary.output_core == int(OnlineCoreType.ONLINE)
    assert table.boundary.package_type == int(FramePackageType.CONF_TESTOUT)


def test_build_output_mapping_tables_preserve_explicit_boundary_test_route():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_output_boundary_explicit_test_route",
        OnlineLinear(),
        test_core_xy=1,
        test_core_x=-1,
        test_core_y=0,
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)

    assert output_tables[output_name].boundary is not None
    assert output_tables[output_name].boundary.test_core_offset == (1, -1, 0)


def test_build_output_mapping_tables_attach_output_route_plan_for_offline_boundary():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_offline_output_boundary_route_plan",
        OnlineLinear(),
        output_core=OnlineCoreType.OFFLINE,
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)
    boundary = output_tables[output_name].boundary

    assert boundary is not None
    producer_coord = _coord_from_offset_tuple(boundary.producer_core_offset)
    assert boundary.output_core == int(OnlineCoreType.OFFLINE)
    assert boundary.target_coord == (TEST_DEST_CORE.x, TEST_DEST_CORE.y)
    assert boundary.data_route_offset is not None
    assert boundary.control_ingress_side == boundary.data_ingress_side
    assert boundary.control_ingress_side == terminal_route_side(
        CoordZXYOffset(*boundary.test_core_offset)
    )
    assert boundary.data_ingress_side == terminal_route_side(
        CoordZXYOffset(*boundary.data_route_offset)
    )
    assert (
        route_coord_path(
            producer_coord,
            CoordZXYOffset(*boundary.data_route_offset),
        )[-1]
        == TEST_DEST_CORE
    )


def test_build_output_mapping_tables_preserve_explicit_boundary_target_coord():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "two_layer_explicit_output_boundary_target",
        TwoLayerOnlineLinear(),
        test_core_xy=1,
        test_core_x=-1,
        test_core_y=0,
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)
    boundary = output_tables[output_name].boundary

    assert boundary is not None
    producer_coord = _coord_from_offset_tuple(boundary.producer_core_offset)
    expected_target = CoordXY(
        producer_coord.x + 1 - 1,
        producer_coord.y + 1 + 0,
    )
    assert boundary.test_core_offset == (1, -1, 0)
    assert boundary.target_coord == (expected_target.x, expected_target.y)
    assert boundary.data_route_offset is not None
    assert boundary.control_ingress_side == boundary.data_ingress_side
    assert (
        route_coord_path(
            producer_coord,
            CoordZXYOffset(*boundary.data_route_offset),
        )[-1]
        == expected_target
    )


def test_build_output_mapping_tables_keep_last_forward_output_for_two_layers():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "two_layer_output_tables",
        TwoLayerOnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    thread = artifacts.io_mapping.threads[0]
    forward_outputs, update_outputs = _online_output_source_names(graph)

    output_tables = build_output_mapping_tables(
        artifacts,
        thread_id=int(thread.thread_id),
    )

    assert len(thread.output_mappings.items) == 1
    assert set(output_tables) == forward_outputs
    assert set(output_tables).isdisjoint(update_outputs)
    table = output_tables[next(iter(forward_outputs))]
    assert table.kind == thread.output_mappings.items[0].DATA
    assert table.target_lcn == int(thread.output_mappings.target_lcn)
    assert table.shape == tuple(thread.output_mappings.items[0].shape.size)
    assert table.bit_width == int(thread.output_mappings.items[0].bit_width)
    assert sorted(table.entries_by_axon) == [0, 1, 2, 3]


def test_scatter_data_output_restores_online_forward_tensor_shape():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_scatter_output",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)

    output = scatter_data_output(
        output_tables,
        output_name,
        [
            (3, np.float16(4.0)),
            (1, np.float16(2.0)),
            (0, np.float16(1.0)),
            (2, np.float16(3.0)),
        ],
    )

    assert output.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(
        output,
        np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16),
    )


def test_encode_online_data_output_frames_uses_boundary_core_offset_by_default():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_encode_output_boundary_default",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)
    table = output_tables[output_name]
    payload = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16)

    assert table.boundary is not None

    implicit_frames = encode_online_data_output_frames(
        output_tables,
        output_name,
        payload,
    )
    explicit_frames = encode_online_data_output_frames(
        output_tables,
        output_name,
        payload,
        core_offset=table.boundary.producer_core_offset,
    )
    packaged_frames = encode_online_data_output_frames(
        output_tables,
        output_name,
        payload,
        packaged=True,
    )

    np.testing.assert_array_equal(implicit_frames, explicit_frames)
    np.testing.assert_array_equal(
        packaged_frames,
        package_online_data_output_frames(
            implicit_frames,
            core_offset=table.boundary.producer_core_offset,
        ),
    )


def test_encode_online_boundary_output_frames_use_data_route_offset():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_boundary_output_frames_data_route",
        OnlineLinear(),
        output_core=OnlineCoreType.OFFLINE,
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)
    table = output_tables[output_name]
    payload = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16)

    assert table.boundary is not None
    assert table.boundary.data_route_offset is not None

    boundary_frames = encode_online_boundary_output_frames(
        output_tables,
        output_name,
        payload,
    )
    boundary_packaged_frames = encode_online_boundary_output_frames(
        output_tables,
        output_name,
        payload,
        packaged=True,
    )
    explicit_frames = encode_online_data_output_frames(
        output_tables,
        output_name,
        payload,
        core_offset=table.boundary.data_route_offset,
    )

    np.testing.assert_array_equal(boundary_frames, explicit_frames)
    np.testing.assert_array_equal(
        boundary_packaged_frames,
        package_online_data_output_frames(
            explicit_frames,
            core_offset=table.boundary.data_route_offset,
        ),
    )


def test_encode_online_boundary_output_frames_support_control_route():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_boundary_output_frames_control_route",
        OnlineLinear(),
        test_core_xy=1,
        test_core_x=-1,
        test_core_y=0,
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)
    table = output_tables[output_name]
    payload = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16)

    assert table.boundary is not None

    control_frames = encode_online_boundary_output_frames(
        output_tables,
        output_name,
        payload,
        route_kind="control",
    )
    explicit_frames = encode_online_data_output_frames(
        output_tables,
        output_name,
        payload,
        core_offset=table.boundary.test_core_offset,
    )

    np.testing.assert_array_equal(control_frames, explicit_frames)


def test_encode_online_boundary_output_frames_reject_missing_data_route():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_boundary_output_frames_missing_data_route",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)
    table = output_tables[output_name]

    assert table.boundary is not None
    table.boundary = replace(
        table.boundary, data_route_offset=None, data_ingress_side=None
    )

    with pytest.raises(ValueError, match="has no aligned data-route offset"):
        encode_online_boundary_output_frames(
            output_tables,
            output_name,
            np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16),
        )


def test_decode_online_data_output_frames_and_scatter_restore_two_layer_tensor():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "two_layer_decode_output_frames",
        TwoLayerOnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)

    frames = encode_online_data_output_frames(
        output_tables,
        output_name,
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
    )
    decoded_items = decode_online_data_output_frames(
        output_tables,
        output_name,
        frames,
    )
    output = scatter_data_output(output_tables, output_name, decoded_items)

    assert decoded_items == [
        (0, np.float16(1.0)),
        (1, np.float16(2.0)),
        (2, np.float16(3.0)),
        (3, np.float16(4.0)),
    ]
    np.testing.assert_array_equal(
        output,
        np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16),
    )


def test_decode_online_data_output_frames_and_scatter_restore_forward_tensor():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_decode_output_frames",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)

    frames = encode_online_data_output_frames(
        output_tables,
        output_name,
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
    )
    decoded_items = decode_online_data_output_frames(
        output_tables,
        output_name,
        frames,
    )
    output = scatter_data_output(output_tables, output_name, decoded_items)

    assert decoded_items == [
        (0, np.float16(1.0)),
        (1, np.float16(2.0)),
        (2, np.float16(3.0)),
        (3, np.float16(4.0)),
    ]
    np.testing.assert_array_equal(
        output,
        np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16),
    )


def test_decode_online_data_output_frames_supports_package_header_prefix():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_decode_output_package_frames",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)

    frames = encode_online_data_output_frames(
        output_tables,
        output_name,
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
    )
    packaged_frames = package_online_data_output_frames(frames)

    decoded_items = decode_online_data_output_frames(
        output_tables,
        output_name,
        packaged_frames,
        packaged=True,
    )
    output = scatter_data_output(output_tables, output_name, decoded_items)

    assert decoded_items == [
        (0, np.float16(1.0)),
        (1, np.float16(2.0)),
        (2, np.float16(3.0)),
        (3, np.float16(4.0)),
    ]
    np.testing.assert_array_equal(
        output,
        np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16),
    )


def test_parse_online_data_output_package_header_recovers_route_metadata():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_parse_output_package_header",
        OnlineLinear(),
        output_core=OnlineCoreType.OFFLINE,
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)
    table = output_tables[output_name]

    assert table.boundary is not None
    assert table.boundary.data_route_offset is not None

    packaged_frames = encode_online_boundary_output_frames(
        output_tables,
        output_name,
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
        route_kind="data",
        copy_count=(1, 0, 1),
        packaged=True,
    )
    header = parse_online_data_output_package_header(packaged_frames)

    assert header.frame_header == int(FrameHeader.WORK_TYPE1)
    assert header.core_offset == table.boundary.data_route_offset
    assert header.copy_count == (1, 0, 1)
    assert header.start_addr == 0
    assert header.package_type == int(FramePackageType.CONF_TESTOUT)
    assert header.n_payload_frames == len(packaged_frames) - 1


def test_decode_online_boundary_output_frames_validate_data_route_header():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_decode_boundary_output_frames_data_route",
        OnlineLinear(),
        output_core=OnlineCoreType.OFFLINE,
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)

    packaged_frames = encode_online_boundary_output_frames(
        output_tables,
        output_name,
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
        route_kind="data",
        packaged=True,
    )

    decoded_items = decode_online_boundary_output_frames(
        output_tables,
        output_name,
        packaged_frames,
        route_kind="data",
    )
    output = scatter_data_output(output_tables, output_name, decoded_items)

    assert decoded_items == [
        (0, np.float16(1.0)),
        (1, np.float16(2.0)),
        (2, np.float16(3.0)),
        (3, np.float16(4.0)),
    ]
    np.testing.assert_array_equal(
        output,
        np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16),
    )


def test_decode_online_boundary_output_frames_validate_control_route_header():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_decode_boundary_output_frames_control_route",
        OnlineLinear(),
        test_core_xy=1,
        test_core_x=-1,
        test_core_y=0,
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)

    packaged_frames = encode_online_boundary_output_frames(
        output_tables,
        output_name,
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
        route_kind="control",
        packaged=True,
    )

    decoded_items = decode_online_boundary_output_frames(
        output_tables,
        output_name,
        packaged_frames,
        route_kind="control",
    )

    assert decoded_items == [
        (0, np.float16(1.0)),
        (1, np.float16(2.0)),
        (2, np.float16(3.0)),
        (3, np.float16(4.0)),
    ]


def test_decode_online_boundary_output_frames_reject_route_header_mismatch():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_decode_boundary_output_frames_route_mismatch",
        OnlineLinear(),
        test_core_xy=1,
        test_core_x=-1,
        test_core_y=0,
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)

    packaged_frames = encode_online_data_output_frames(
        output_tables,
        output_name,
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
        packaged=True,
    )

    with pytest.raises(
        ValueError,
        match="expects boundary route 'control' package core_offset",
    ):
        decode_online_boundary_output_frames(
            output_tables,
            output_name,
            packaged_frames,
            route_kind="control",
        )


def test_decode_online_data_output_frames_rejects_package_count_mismatch():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_decode_output_package_mismatch",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)

    frames = encode_online_data_output_frames(
        output_tables,
        output_name,
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
    )
    packaged_frames = package_online_data_output_frames(
        frames,
        n_package=len(frames) - 1,
    )

    with pytest.raises(ValueError, match="expects 7 payload frames, got 8"):
        decode_online_data_output_frames(
            output_tables,
            output_name,
            packaged_frames,
            packaged=True,
        )


def test_strip_online_data_output_package_header_rejects_testin_type():
    frames = np.asarray([1, 2], dtype=np.uint64)
    header = FramePackageHeaderV2.make_pkg_header(
        FrameHeader.WORK_TYPE1,
        CoordZXYOffset(0, 0, 0),
        AERPacketZXYCopy(0, 0, 0),
        0,
        FramePackageType.TESTIN,
        len(frames),
    ).value.astype(np.uint64, copy=False)
    packaged_frames = np.concatenate([header, frames]).astype(np.uint64, copy=False)

    with pytest.raises(ValueError, match="requires CONF_TESTOUT package type"):
        strip_online_data_output_package_header(packaged_frames)


def test_strip_online_data_output_package_header_rejects_non_work_type1_header():
    frames = np.asarray([1, 2], dtype=np.uint64)
    header = FramePackageHeaderV2.make_pkg_header(
        FrameHeader.WORK_TYPE2,
        CoordZXYOffset(0, 0, 0),
        AERPacketZXYCopy(0, 0, 0),
        0,
        FramePackageType.CONF_TESTOUT,
        len(frames),
    ).value.astype(np.uint64, copy=False)
    packaged_frames = np.concatenate([header, frames]).astype(np.uint64, copy=False)

    with pytest.raises(ValueError, match="requires WORK_TYPE1 frame header"):
        strip_online_data_output_package_header(packaged_frames)


def test_decode_online_data_output_frames_rejects_incomplete_float16_lanes():
    export_dir, graph, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_decode_output_frames_incomplete_lanes",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    forward_outputs, _ = _online_output_source_names(graph)
    output_name = next(iter(forward_outputs))
    output_tables = build_output_mapping_tables(artifacts)

    frames = encode_online_data_output_frames(
        output_tables,
        output_name,
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
    )

    with pytest.raises(ValueError, match="incomplete byte lanes"):
        decode_online_data_output_frames(output_tables, output_name, frames[:-1])


def test_scatter_data_output_rejects_missing_output_name():
    export_dir, _, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_scatter_output_missing_name",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    output_tables = build_output_mapping_tables(artifacts)

    with pytest.raises(KeyError, match="output mapping not found"):
        scatter_data_output(output_tables, "missing_output", [])


def _expected_online_input_frames(input_mapping, flat_data: np.ndarray) -> np.ndarray:
    entries = sorted(input_mapping.entries, key=lambda item: item.elem_idx)
    first_entry = entries[0]
    return OnlineFrameGenV2.gen_work_frame1(
        CoordZXYOffset(
            int(first_entry.core_offset.xy),
            int(first_entry.core_offset.x),
            int(first_entry.core_offset.y),
        ),
        AERPacketZXYCopy(
            int(first_entry.copy_count.xy),
            int(first_entry.copy_count.x),
            int(first_entry.copy_count.y),
        ),
        np.asarray([entry.tick_relative for entry in entries], dtype=np.uint64),
        np.asarray([entry.addr_axon for entry in entries], dtype=np.uint64),
        int(first_entry.target_lcn),
        flat_data[[entry.elem_idx for entry in entries]],
    ).astype(np.uint64, copy=False)


def test_encode_online_input_frames_generates_fp16_work_frame1():
    export_dir, _, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_input_frames",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    thread = artifacts.io_mapping.threads[0]
    input_mapping = thread.input_mappings.items[0]
    payload = np.arange(1, 9, dtype=np.float16)

    frames = encode_online_input_frames(
        artifacts,
        input_mapping.name,
        payload,
        thread_id=int(thread.thread_id),
    )

    np.testing.assert_array_equal(
        frames,
        _expected_online_input_frames(input_mapping, payload),
    )


def test_encode_online_input_frames_respects_transform_aware_mapping():
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(8, 2, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x.permute(0, 2, 3, 1).flatten(1))

    export_dir = DEBUG_EXPORT_ROOT / "permute_flatten_input_frames"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(mark_online(Model()), torch.randn(1, 2, 2, 2))
    mapper = Mapper()
    mapper.compile(graph, export_dir, target_platform="x86", debug=True)
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    thread = artifacts.io_mapping.threads[0]
    input_mapping = thread.input_mappings.items[0]
    payload = np.arange(1, 9, dtype=np.float16).reshape(1, 2, 2, 2)

    frames = encode_online_input_frames(
        artifacts,
        input_mapping.name,
        payload,
        thread_id=int(thread.thread_id),
    )

    np.testing.assert_array_equal(
        frames,
        _expected_online_input_frames(input_mapping, payload.reshape(-1)),
    )


def test_encode_online_input_frames_supports_mnist_flatten_linear_mapping():
    export_dir = DEBUG_EXPORT_ROOT / "mnist_flatten_linear_input_frames"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(
        mark_online(MNISTFlattenOnlineLinear()), make_img_1ch_28x28()
    )
    mapper = Mapper()
    mapper.compile(graph, export_dir, target_platform="x86", debug=True)
    artifacts = load_compile_artifacts(export_dir)
    thread = artifacts.io_mapping.threads[0]
    input_mapping = thread.input_mappings.items[0]
    payload = np.arange(1, 28 * 28 + 1, dtype=np.float16).reshape(1, 1, 28, 28)

    frames = encode_online_input_frames(
        artifacts,
        input_mapping.name,
        payload,
        thread_id=int(thread.thread_id),
    )

    assert tuple(input_mapping.shape.size) == (1, 1, 28, 28)
    assert len(input_mapping.entries) == 28 * 28
    np.testing.assert_array_equal(
        frames,
        _expected_online_input_frames(input_mapping, payload.reshape(-1)),
    )


def test_encode_online_input_frames_rejects_shape_mismatch():
    export_dir, _, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_input_frames_shape_mismatch",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    thread = artifacts.io_mapping.threads[0]
    input_mapping = thread.input_mappings.items[0]

    with pytest.raises(ValueError, match="expects 8 elements"):
        encode_online_input_frames(
            artifacts,
            input_mapping.name,
            np.arange(7, dtype=np.float16),
            thread_id=int(thread.thread_id),
        )


def test_encode_online_input_frames_rejects_missing_input_name():
    export_dir, _, _ = export_online_graph(
        DEBUG_EXPORT_ROOT,
        "single_layer_input_frames_missing_name",
        OnlineLinear(),
    )
    artifacts = load_compile_artifacts(export_dir / "proto" / "config.pb")
    thread = artifacts.io_mapping.threads[0]

    with pytest.raises(KeyError, match="online input mapping not found"):
        encode_online_input_frames(
            artifacts,
            "missing_input",
            np.arange(8, dtype=np.float16),
            thread_id=int(thread.thread_id),
        )
