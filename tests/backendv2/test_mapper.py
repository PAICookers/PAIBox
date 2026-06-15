import json
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from paicorelib import (
    LCN_EX,
    CoordXY,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    NeuronType,
    WeightCompressType,
    find_coordxy_shortest_path,
)
from torch import nn

from paibox.backendv2 import routing as routing_mod
from paibox.backendv2.coreplacement import OfflineCorePlacementV2
from paibox.backendv2.export.utils import export_framearray_to_int32
from paibox.backendv2.mapper import Mapper
from paibox.backendv2.op_node import SourceElem
from paibox.backendv2.output_cpu_ingress import OutputRouteEndpoint
from paibox.backendv2.proto import get_schema_version
from paibox.backendv2.proto.compile_artifacts_pb2 import (
    CompileArtifacts,
    ConfigFrames,
    DataType,
    OutputTensorMapping,
    RuntimeParams,
)
from paibox.backendv2.routing import FANIN_BASE, OutputGroup
from paibox.paiir import ANNNodeV25, LutCustom, compile_to_paiir, register_neuron
from tests.paiir.conftest import ANNClassifier, SimpleCNN, SNNTwoLayer, make_img_3ch_8x8
from tests.utils import is_ci_env

DEBUG_EXPORT_ROOT = Path(__file__).with_name("debug") / "mapper_proto_export"


class FloatOutputIdentityLutCustom(LutCustom):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


class RepeatedWeightDifferentBiasLinearLut(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        thresholds = torch.arange(256, dtype=torch.int32)
        values = torch.arange(256, dtype=torch.uint8)
        self.linear1 = nn.Linear(4, 4, bias=True)
        self.lut1 = FloatOutputIdentityLutCustom(
            thresholds, values, output_sign=0, is_float=False
        )
        self.linear2 = nn.Linear(4, 2, bias=True)
        self.lut2 = FloatOutputIdentityLutCustom(
            thresholds, values, output_sign=0, is_float=False
        )
        with torch.no_grad():
            repeated_row = torch.tensor([1, -1, 1, -1], dtype=torch.float32)
            self.linear1.weight.copy_(repeated_row.repeat(4, 1))
            self.linear1.bias.copy_(
                torch.tensor([120, 121, 122, 123], dtype=torch.float32)
            )
            self.linear2.weight.copy_(
                torch.tensor([[1, 1, -1, -1], [-1, 1, -1, 1]], dtype=torch.float32)
            )
            self.linear2.bias.copy_(torch.tensor([128, 130], dtype=torch.float32))

    def forward(self, x):
        x = self.lut1(self.linear1(x))
        return self.lut2(self.linear2(x))


def _register_float_output_identity_lut_custom() -> None:
    try:
        register_neuron(
            FloatOutputIdentityLutCustom,
            converter=lambda lut: ANNNodeV25(
                LutCustom(
                    lut.thresholds.detach().clone(),
                    lut.lut_values.detach().clone(),
                    output_sign=lut.output_sign,
                    is_float=lut.is_float,
                )
            ),
        )
    except ValueError as exc:
        if "already registered" not in str(exc):
            raise


def _assert_debug_frame_text(path: Path) -> None:
    text = path.read_text()
    assert "# Core at coord (X,Y)=" in text
    assert "0x" in text


def _export_simple_cnn_proto(export_root: Path, word_order: str) -> Path:
    export_dir = export_root / word_order
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(SimpleCNN().eval(), make_img_3ch_8x8(), strict=True)
    mapper = Mapper()
    mapper.compile(
        graph,
        export_dir,
        target_platform="x86",
        word_order=word_order,  # type: ignore[arg-type]
        debug=True,
    )

    return export_dir / "proto"


def _export_simple_cnn(
    export_root: Path,
    case_name: str,
    target_platform: str,
    debug: bool,
    export_merged_frames: bool = True,
) -> Path:
    export_dir = export_root / case_name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(SimpleCNN().eval(), make_img_3ch_8x8(), strict=True)
    mapper = Mapper()
    mapper.compile(
        graph,
        export_dir,
        target_platform=target_platform,  # type: ignore[arg-type]
        debug=debug,
        export_merged_frames=export_merged_frames,
    )

    return export_dir


class ConvPotential(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


class DefaultMixedSparseDenseLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(160, 3, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, [16, 40, 80]] = torch.tensor(
                [1, 2, 3], dtype=torch.float32
            )
            self.linear.weight[1, [0, 128]] = torch.tensor([1, 2], dtype=torch.float32)
            self.linear.weight[2, [1, 3, 5, 7, 9]] = 1

    def forward(self, x):
        return self.linear(x)


class DefaultSparseHalfReuseLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(160, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, [16, 40, 80]] = torch.tensor(
                [1, 2, 3], dtype=torch.float32
            )
            self.linear.weight[1, [17, 41, 81]] = torch.tensor(
                [1, 2, 3], dtype=torch.float32
            )

    def forward(self, x):
        return self.linear(x)


class Uint8HighIndexDenseCandidateLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(192, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, [144, 146]] = torch.tensor(
                [1, 1], dtype=torch.float32
            )
            self.linear.weight[1, [160, 162]] = torch.tensor(
                [1, 1], dtype=torch.float32
            )

    def forward(self, x):
        return self.linear(x)


class Uint8LongSpanSparseLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(192, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, [7, 136]] = torch.tensor([1, 1], dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)


class Uint8DifferentSparseBasesLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(192, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, [0, 191]] = torch.tensor([1, 1], dtype=torch.float32)
            self.linear.weight[1, [1, 190]] = torch.tensor([1, 1], dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)


class Uint1HighIndexSparseCscLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(192, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, [64, 191]] = torch.tensor([7, 5], dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)


class _FakeOutputElem:
    def __init__(self, bit_width: int, name: str) -> None:
        self.output_bit_num = bit_width
        self.name = name

    def __repr__(self) -> str:
        return self.name


def _output_elems(bit_width: int, count: int, prefix: str) -> list[SourceElem]:
    return [
        cast(SourceElem, _FakeOutputElem(bit_width, f"{prefix}{i}"))
        for i in range(count)
    ]


def _load_compile_artifacts(pb_path: Path) -> CompileArtifacts:
    artifacts = CompileArtifacts()
    artifacts.ParseFromString(pb_path.read_bytes())
    return artifacts


def _export_graph_proto_with_context(
    export_root: Path,
    case_name: str,
    model,
    sample,
    mapper_timesteps: int | None = None,
    **compile_kwargs,
) -> tuple[Path, set[str], Mapper]:
    export_dir = export_root / case_name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(model.eval(), sample, strict=True, **compile_kwargs)
    output_source_names = {
        pred_name
        for output_node in graph.output_nodes()
        for pred_name in graph.predecessors(output_node.name)
    }
    mapper = Mapper()
    mapper.compile(
        graph, export_dir, target_platform="x86", debug=True, timesteps=mapper_timesteps
    )

    return export_dir / "proto" / "config.pb", output_source_names, mapper


def _export_graph_proto(
    export_root: Path, case_name: str, model, sample, **compile_kwargs
) -> Path:
    pb_path, _, _ = _export_graph_proto_with_context(
        export_root, case_name, model, sample, **compile_kwargs
    )
    return pb_path


def _tick_tuple(tick) -> tuple[int, int, int]:
    return (tick.tick_start, tick.tick_duration, tick.tick_initial)


def _assert_entries_have_dtype(entries, dtype: int) -> None:
    assert entries
    assert all(entry.HasField("dtype") for entry in entries)
    assert {entry.dtype for entry in entries} == {dtype}


def _assert_mapping_bit_width(mapping, expected: int) -> None:
    assert mapping.HasField("bit_width")
    assert mapping.bit_width == expected
    assert all(
        "bit_width" not in entry.DESCRIPTOR.fields_by_name for entry in mapping.entries
    )


def _assert_json_field_order(mapping_json: dict[str, object]) -> None:
    keys = list(mapping_json)
    assert keys.index("bitWidth") < keys.index("tick")
    assert keys.index("bitWidth") < keys.index("entries")


def _expected_core_major_frames(mapper: Mapper) -> np.ndarray:
    parts = []
    for core_placement in mapper.coreplacements:
        for frame_array in core_placement.to_frame():
            if frame_array is not None:
                parts.append(np.asarray(frame_array, dtype="<u8"))
    return np.concatenate(parts) if parts else np.array([], dtype="<u8")


def _expected_core_major_words(mapper: Mapper, word_order: str) -> list[int]:
    words = []
    for core_placement in mapper.coreplacements:
        for frame_array in core_placement.to_frame():
            if frame_array is not None:
                words.extend(export_framearray_to_int32(frame_array, word_order))  # type: ignore
    return words


def _expected_data_lcn(num_outputs: int) -> LCN_EX:
    for lcn_value in range(8):
        if num_outputs <= FANIN_BASE * (1 << lcn_value):
            return LCN_EX(lcn_value)
    raise AssertionError(f"{num_outputs} outputs exceed LCN_128X capacity")


def _expected_lcn_from_max_axon(max_axon_addr: int) -> LCN_EX:
    min_tick_relative_bit = (max_axon_addr // FANIN_BASE).bit_length()
    return LCN_EX(min_tick_relative_bit)


def _assert_output_group_allocator_matches_inputs(out_grp: OutputGroup) -> None:
    assert out_grp.axon_bit_allocator.target_lcn == out_grp.lcn
    assert [elem for _, elem in out_grp.axon_bit_allocator.axon_infos] == (
        out_grp.input_list
    )
    assert out_grp.input_mapping == out_grp.axon_bit_allocator.axon_by_elem


def _shared_sparse_linear_neuron_placements(mapper: Mapper):
    return [
        neu_placement
        for core_placement in mapper.coreplacements
        for neu_placement in core_placement.neus
    ]


def test_mapper_finalizes_output_cpu_ingress_plan_without_full_compile(monkeypatch):
    mapper = Mapper()
    endpoints = [
        OutputRouteEndpoint(CoordXY(0, 2), CoordXY(0, 0)),
        OutputRouteEndpoint(CoordXY(5, 2), CoordXY(0, 0)),
    ]
    applied_targets: list[CoordXY] = []

    monkeypatch.setattr(mapper, "_collect_output_route_endpoints", lambda: endpoints)
    monkeypatch.setattr(mapper, "_global_root_coord", lambda: CoordXY(2, 2))

    def record_output_target(plan) -> None:
        if plan.output_target is not None:
            applied_targets.append(plan.output_target)

    monkeypatch.setattr(mapper, "_apply_output_target", record_output_target)

    plan = mapper.finalize_output_cpu_ingress_plan()

    assert mapper.output_cpu_ingress_plan == plan
    assert plan.output_target == CoordXY(4, 0)
    assert plan.control_target == CoordXY(4, 0)
    assert applied_targets == [CoordXY(4, 0)]


def test_mapper_default_auto_strategy_mixes_sparse_and_dense_csc(tmp_path):
    graph = compile_to_paiir(
        DefaultMixedSparseDenseLinear().eval(),
        torch.zeros(1, 160),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)},
        timesteps=1,
        auto_reset=True,
        strict=True,
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    core = mapper.coreplacements[0]
    placements = _shared_sparse_linear_neuron_placements(mapper)

    assert {c.frontend_core_config.input_width for c in mapper.coreplacements} == {
        DataWidth.WIDTH_1BIT
    }
    assert {c.default_core_config.csc_accelerate for c in mapper.coreplacements} == {
        CSCAccelerateMode.ENABLE
    }
    assert [weight.compress for weight in core.weights] == [True, True, False]
    assert np.flatnonzero(core.weights[0].raw_weights).tolist() == [0, 24, 64]
    assert np.flatnonzero(core.weights[1].raw_weights).tolist() == [0, 128]
    assert core.weights[2].processed_weights == [1, 0, 1, 0, 1, 0, 1, 0, 1]
    assert [placement.neu_attrs_part1.weight_skew for placement in placements] == [
        16,
        0,
        1,
    ]
    assert [placement.neu_attrs_part2 is None for placement in placements] == [
        False,
        False,
        False,
    ]
    assert [placement.neu_attrs_part2.weight_compress for placement in placements] == [
        WeightCompressType.SPARSE,
        WeightCompressType.SPARSE,
        WeightCompressType.DENSE,
    ]
    assert placements[2].neu_attrs_part1.weight_address_start == (
        core.weights[0].n_sram_required
        + core.weights[1].n_sram_required
        + sum(neu.n_sram_required for neu in core.neus)
    )
    assert placements[0].neu_attrs_part2.vjt_initial == (
        placements[0].neu_attrs_part1.weight_address_start
    )
    assert placements[1].neu_attrs_part2.vjt_initial == (
        placements[1].neu_attrs_part1.weight_address_start
    )
    assert placements[2].neu_attrs_part2.vjt_initial == 0


def test_mapper_default_sparse_csc_half_reuse_is_true_sparse_csc(tmp_path):
    graph = compile_to_paiir(
        DefaultSparseHalfReuseLinear().eval(),
        torch.zeros(1, 160),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)},
        timesteps=1,
        auto_reset=True,
        strict=True,
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    core = mapper.coreplacements[0]
    placements = _shared_sparse_linear_neuron_placements(mapper)

    assert {c.frontend_core_config.input_width for c in mapper.coreplacements} == {
        DataWidth.WIDTH_1BIT
    }
    assert {c.default_core_config.csc_accelerate for c in mapper.coreplacements} == {
        CSCAccelerateMode.ENABLE
    }
    assert len(core.weights) == 1
    assert core.weights[0].compress
    assert np.flatnonzero(core.weights[0].raw_weights).tolist() == [0, 24, 64]
    assert [placement.neuron_type for placement in placements] == [
        NeuronType.FULL,
        NeuronType.HALF,
    ]
    assert [placement.neu_attrs_part1.weight_skew for placement in placements] == [
        16,
        17,
    ]
    assert [placement.neu_attrs_part2 is None for placement in placements] == [
        False,
        True,
    ]
    assert placements[0].neu_attrs_part2.weight_compress == WeightCompressType.SPARSE
    assert placements[0].neu_attrs_part2.vjt_initial == (
        placements[0].neu_attrs_part1.weight_address_start
    )
    assert placements[1].neu_attrs_part1.weight_address_start == (
        placements[0].neu_attrs_part1.weight_address_start
    )
    assert placements[1].neu_attrs_part1.weight_address_end == (
        placements[0].neu_attrs_part1.weight_address_end
    )


def test_mapper_does_not_fold_neurons_with_different_part2_attrs(tmp_path):
    _register_float_output_identity_lut_custom()
    graph = compile_to_paiir(
        RepeatedWeightDifferentBiasLinearLut().eval(),
        torch.tensor([[3, 4, 5, 6]], dtype=torch.float32),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
        timesteps=1,
        auto_reset=True,
        strict=True,
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    placements = _shared_sparse_linear_neuron_placements(mapper)
    first_layer = [
        placement
        for placement in placements
        if len(placement.raw_neus) == 1
        and placement.raw_neus[0].target.name == "SequentialOp_0"
    ]
    first_layer = sorted(
        first_layer, key=lambda placement: placement.raw_neus[0].index.idx
    )

    assert len(first_layer) == 4
    assert all(placement.folded_neu_attrs_part1 is None for placement in first_layer)
    assert [placement.neu_attrs_part2.leak_v for placement in first_layer] == [
        120,
        121,
        122,
        123,
    ]


def test_mapper_default_uint8_high_index_shifted_base_tie_uses_dense(tmp_path):
    graph = compile_to_paiir(
        Uint8HighIndexDenseCandidateLinear().eval(),
        torch.zeros(1, 192),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
        timesteps=1,
        auto_reset=True,
        strict=True,
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    core = mapper.coreplacements[0]
    placements = _shared_sparse_linear_neuron_placements(mapper)

    assert core.frontend_core_config.input_width == DataWidth.WIDTH_8BIT
    assert core.default_core_config.csc_accelerate == CSCAccelerateMode.ENABLE
    assert len(core.weights) == 1
    assert not core.weights[0].compress
    assert core.weights[0].processed_weights == [1, 0, 1]
    assert [placement.neuron_type for placement in placements] == [
        NeuronType.FULL,
        NeuronType.HALF,
    ]
    assert [placement.neu_attrs_part1.weight_skew for placement in placements] == [
        144 * 8,
        160 * 8,
    ]
    assert placements[0].neu_attrs_part2.weight_compress == WeightCompressType.DENSE
    assert placements[0].neu_attrs_part2.vjt_initial == 0
    assert placements[1].neu_attrs_part2 is None


def test_mapper_default_uint8_long_span_weight_stays_sparse_when_smaller(tmp_path):
    graph = compile_to_paiir(
        Uint8LongSpanSparseLinear().eval(),
        torch.zeros(1, 192),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
        timesteps=1,
        auto_reset=True,
        strict=True,
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    core = mapper.coreplacements[0]
    placements = _shared_sparse_linear_neuron_placements(mapper)

    assert len(core.weights) == 1
    assert core.weights[0].compress
    assert placements[0].neu_attrs_part2.weight_compress == WeightCompressType.SPARSE
    assert placements[0].neu_attrs_part2.vjt_initial == (
        placements[0].neu_attrs_part1.weight_address_start
    )


def test_mapper_sparse_csc_uses_full_neurons_for_different_weight_addresses(tmp_path):
    graph = compile_to_paiir(
        Uint8DifferentSparseBasesLinear().eval(),
        torch.zeros(1, 192),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
        timesteps=1,
        auto_reset=True,
        strict=True,
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    core = mapper.coreplacements[0]
    placements = _shared_sparse_linear_neuron_placements(mapper)

    assert len(core.weights) == 2
    assert [weight.compress for weight in core.weights] == [True, True]
    assert [placement.neuron_type for placement in placements] == [
        NeuronType.FULL,
        NeuronType.FULL,
    ]
    assert [placement.neu_attrs_part2.weight_compress for placement in placements] == [
        WeightCompressType.SPARSE,
        WeightCompressType.SPARSE,
    ]
    assert placements[0].neu_attrs_part1.weight_address_start != (
        placements[1].neu_attrs_part1.weight_address_start
    )
    assert placements[0].neu_attrs_part2.vjt_initial == (
        placements[0].neu_attrs_part1.weight_address_start
    )
    assert placements[1].neu_attrs_part2.vjt_initial == (
        placements[1].neu_attrs_part1.weight_address_start
    )


def test_mapper_sparse_csc_allows_half_for_different_weight_addresses_without_accel(
    tmp_path, monkeypatch
):
    class NoCscAccelOfflineCorePlacementV2(OfflineCorePlacementV2):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.default_core_config.csc_accelerate = CSCAccelerateMode.DISABLE

    monkeypatch.setattr(
        routing_mod, "OfflineCorePlacementV2", NoCscAccelOfflineCorePlacementV2
    )
    graph = compile_to_paiir(
        Uint8DifferentSparseBasesLinear().eval(),
        torch.zeros(1, 192),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
        timesteps=1,
        auto_reset=True,
        strict=True,
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    core = mapper.coreplacements[0]
    placements = _shared_sparse_linear_neuron_placements(mapper)

    assert core.default_core_config.csc_accelerate == CSCAccelerateMode.DISABLE
    assert len(core.weights) == 2
    assert [weight.compress for weight in core.weights] == [True, True]
    assert [placement.neuron_type for placement in placements] == [
        NeuronType.FULL,
        NeuronType.HALF,
    ]
    assert placements[0].neu_attrs_part2.weight_compress == WeightCompressType.SPARSE
    assert placements[0].neu_attrs_part2.vjt_initial == 0
    assert placements[1].neu_attrs_part2 is None
    assert placements[0].neu_attrs_part1.weight_address_start != (
        placements[1].neu_attrs_part1.weight_address_start
    )


def test_mapper_uint1_high_index_sparse_csc_uses_shifted_base_row(tmp_path):
    graph = compile_to_paiir(
        Uint1HighIndexSparseCscLinear().eval(),
        torch.zeros(1, 192),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)},
        timesteps=1,
        auto_reset=True,
        strict=True,
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    core = mapper.coreplacements[0]
    placements = _shared_sparse_linear_neuron_placements(mapper)

    assert core.frontend_core_config.input_width == DataWidth.WIDTH_1BIT
    assert core.default_core_config.csc_accelerate == CSCAccelerateMode.ENABLE
    assert len(core.weights) == 1
    assert core.weights[0].compress
    assert len(core.weights[0].processed_weights) == 128
    assert core.weights[0].processed_weights[0] == 7
    assert core.weights[0].processed_weights[127] == 5
    assert len(placements) == 1
    assert placements[0].neuron_type == NeuronType.FULL
    assert placements[0].neu_attrs_part1.weight_skew == 64
    assert placements[0].neu_attrs_part2.weight_compress == WeightCompressType.SPARSE
    assert placements[0].neu_attrs_part2.vjt_initial == (
        placements[0].neu_attrs_part1.weight_address_start
    )


CoreTickKey = tuple[int, int, int, tuple[str, ...], tuple[int, int, int]]


def _core_tick_key(core_tick) -> CoreTickKey:
    return (
        core_tick.core_offset.xy,
        core_tick.core_offset.x,
        core_tick.core_offset.y,
        tuple(core_tick.nodes),
        _tick_tuple(core_tick.tick),
    )


def _expected_core_tick_keys(mapper: Mapper, thread_id: int) -> set[CoreTickKey]:
    keys = set()
    for core_placement in mapper.coreplacements:
        if not core_placement.neus:
            continue
        if core_placement.default_core_config.thread_number != thread_id:
            continue
        core_offset, _ = find_coordxy_shortest_path(core_placement.coord)
        nodes = tuple(
            sorted(
                {
                    raw_neu.target.raw_node.name
                    for neu_placement in core_placement.neus
                    for raw_neu in neu_placement.raw_neus
                }
            )
        )
        tick = (
            int(core_placement.frontend_core_config.tick_start),
            int(core_placement.frontend_core_config.tick_duration),
            int(core_placement.frontend_core_config.tick_initial),
        )
        keys.add((core_offset.z, core_offset.x, core_offset.y, nodes, tick))
    return keys


def _assert_core_ticks_match_mapper(thread, mapper: Mapper) -> None:
    core_ticks = list(thread.core_ticks)
    assert core_ticks
    assert all(core_tick.nodes for core_tick in core_ticks)
    assert {_core_tick_key(core_tick) for core_tick in core_ticks} == (
        _expected_core_tick_keys(mapper, thread.thread_id)
    )


def _assert_thread_tick_duration_initial(thread, expected: tuple[int, int]) -> None:
    assert thread.input_mappings.items
    assert thread.output_mappings.items
    assert all(
        _tick_tuple(mapping.tick)[1:] == expected
        for mapping in thread.input_mappings.items
    )
    assert all(
        _tick_tuple(mapping.tick)[1:] == expected
        for mapping in thread.output_mappings.items
    )
    assert all(
        _tick_tuple(core_tick.tick)[1:] == expected for core_tick in thread.core_ticks
    )


@pytest.fixture(scope="module")
def ensure_backendv2_debug_dir(tmp_path_factory):
    if is_ci_env():
        export_root = tmp_path_factory.mktemp("backendv2-debug") / "mapper_proto_export"
    else:
        export_root = DEBUG_EXPORT_ROOT

    if export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    return export_root


@pytest.mark.parametrize(
    ("word_order", "expected_enum", "expected_json_value"),
    [
        ("high_first", ConfigFrames.HIGH_FIRST, "HIGH_FIRST"),
        ("low_first", ConfigFrames.LOW_FIRST, "LOW_FIRST"),
    ],
)
def test_export_proto_real_workflow_keeps_pb_and_json(
    ensure_backendv2_debug_dir, word_order, expected_enum, expected_json_value
):
    proto_dir = _export_simple_cnn_proto(ensure_backendv2_debug_dir, word_order)

    pb_path = proto_dir / "config.pb"
    json_path = proto_dir / "config.json"

    assert pb_path.exists()
    assert json_path.exists()
    assert (proto_dir / "compile_artifacts.proto").exists()
    assert (proto_dir / "compile_artifacts_pb2.py").exists()
    assert (proto_dir / "compile_artifacts_pb2.pyi").exists()

    artifacts = _load_compile_artifacts(pb_path)

    assert artifacts.schema_version == get_schema_version()
    assert len(artifacts.io_mapping.threads) == 1
    assert len(artifacts.config_frames.words) > 0
    assert artifacts.config_frames.word_order == expected_enum

    payload = json.loads(json_path.read_text())
    assert payload["schemaVersion"] == get_schema_version()
    assert payload["configFrames"]["wordOrder"] == expected_json_value
    assert len(payload["configFrames"]["words"]) > 0
    threads = payload["ioMapping"]["threads"]
    assert len(threads) == 1
    for mapping in threads[0]["inputMappings"]["items"]:
        _assert_json_field_order(mapping)
    for mapping in threads[0]["outputMappings"]["items"]:
        _assert_json_field_order(mapping)


def test_export_merged_frames_use_core_major_order(ensure_backendv2_debug_dir):
    export_dir = ensure_backendv2_debug_dir / "core_major_order"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(SimpleCNN().eval(), make_img_3ch_8x8(), strict=True)
    mapper = Mapper()
    mapper.compile(
        graph,
        export_dir,
        target_platform="x86",
        word_order="high_first",
        debug=False,
    )

    merged_frames = np.load(export_dir / "cfg_frames.npy")
    np.testing.assert_array_equal(merged_frames, _expected_core_major_frames(mapper))

    artifacts = _load_compile_artifacts(export_dir / "proto" / "config.pb")
    assert list(artifacts.config_frames.words) == _expected_core_major_words(
        mapper, "high_first"
    )


def test_export_proto_marks_data_outputs_and_target_lcn(
    ensure_backendv2_debug_dir,
):
    pb_path, output_source_names, _ = _export_graph_proto_with_context(
        ensure_backendv2_debug_dir,
        "data_output_kind",
        ANNClassifier(),
        make_img_3ch_8x8(),
    )

    artifacts = _load_compile_artifacts(pb_path)
    thread = artifacts.io_mapping.threads[0]
    output_mappings = thread.output_mappings

    assert thread.runtime.timesteps == 1
    assert thread.runtime.tick_depth >= 1
    assert thread.runtime.sync_steps == thread.runtime.tick_depth
    assert thread.runtime.decode_mode == RuntimeParams.STREAM
    output_count = sum(len(mapping.entries) for mapping in output_mappings.items)
    assert output_mappings.target_lcn == _expected_data_lcn(output_count).value
    assert output_mappings.HasField("target_lcn")
    assert len(output_mappings.items) == 1
    assert {mapping.name for mapping in output_mappings.items} == output_source_names

    output_mapping = output_mappings.items[0]
    assert output_mapping.kind == OutputTensorMapping.DATA
    assert output_mapping.HasField("kind")

    entries = list(output_mapping.entries)
    assert entries
    assert output_mapping.HasField("bit_width")
    assert output_mapping.bit_width <= 8
    assert all("bit_width" not in entry.DESCRIPTOR.fields_by_name for entry in entries)
    assert all(entry.HasField("dtype") for entry in entries)
    assert {entry.dtype for entry in entries}.issubset({DataType.UINT8, DataType.INT8})


def test_export_proto_marks_voltage_outputs_and_base_addresses(
    ensure_backendv2_debug_dir,
):
    model = ConvPotential()
    with torch.no_grad():
        model.conv.weight.fill_(1)

    pb_path = _export_graph_proto(
        ensure_backendv2_debug_dir,
        "voltage_output_kind",
        model,
        torch.ones(1, 1, 3, 3),
    )

    artifacts = _load_compile_artifacts(pb_path)
    thread = artifacts.io_mapping.threads[0]
    output_mappings = thread.output_mappings

    assert thread.runtime.timesteps == 1
    assert thread.runtime.sync_steps == 1
    assert thread.runtime.decode_mode == RuntimeParams.STREAM
    assert len(output_mappings.items) == 1

    output_mapping = output_mappings.items[0]
    assert output_mapping.kind == OutputTensorMapping.VOLTAGE
    assert output_mapping.HasField("kind")

    entries = list(output_mapping.entries)
    assert entries
    _assert_mapping_bit_width(output_mapping, 32)
    assert all(not entry.HasField("dtype") for entry in entries)

    bases = [entry.axon_bit_idx for entry in entries[:10]]
    assert bases == [0, 1, 2, 3, 4, 5, 6, 7, 32]
    assert output_mappings.target_lcn == _expected_lcn_from_max_axon(max(bases)).value


def test_output_lcn_uses_timesteps_not_external_sync_steps(
    ensure_backendv2_debug_dir,
):
    pb_path, _, mapper = _export_graph_proto_with_context(
        ensure_backendv2_debug_dir,
        "output_lcn_uses_timesteps",
        ANNClassifier(),
        make_img_3ch_8x8(),
        timesteps=1,
        auto_reset=False,
    )

    artifacts = _load_compile_artifacts(pb_path)
    thread = artifacts.io_mapping.threads[0]

    assert thread.runtime.timesteps == 1
    assert thread.runtime.tick_depth > 1
    assert thread.runtime.sync_steps == thread.runtime.tick_depth
    output_count = sum(len(mapping.entries) for mapping in thread.output_mappings.items)
    expected_lcn = _expected_data_lcn(output_count)
    assert thread.output_mappings.target_lcn == expected_lcn.value
    assert mapper.output_groups[0].lcn == expected_lcn
    assert not hasattr(mapper.output_groups[0], "runtime_timesteps")


def test_export_proto_exports_ann_io_ticks_and_core_ticks(
    ensure_backendv2_debug_dir,
):
    pb_path, output_source_names, mapper = _export_graph_proto_with_context(
        ensure_backendv2_debug_dir,
        "ann_tick_metadata",
        ANNClassifier(),
        make_img_3ch_8x8(),
    )

    artifacts = _load_compile_artifacts(pb_path)
    thread = artifacts.io_mapping.threads[0]

    assert {mapping.name for mapping in thread.output_mappings.items} == (
        output_source_names
    )
    expected_tick_depth = max(
        mapping.tick.tick_start for mapping in thread.output_mappings.items
    )
    assert thread.runtime.timesteps == 1
    assert thread.runtime.tick_depth == expected_tick_depth
    assert thread.runtime.sync_steps == expected_tick_depth
    assert thread.runtime.decode_mode == RuntimeParams.STREAM
    _assert_core_ticks_match_mapper(thread, mapper)
    _assert_thread_tick_duration_initial(thread, (0, 1))


def test_export_proto_exports_explicit_ann_tick_policy(
    ensure_backendv2_debug_dir,
):
    pb_path, _, mapper = _export_graph_proto_with_context(
        ensure_backendv2_debug_dir,
        "ann_explicit_tick_metadata",
        ANNClassifier(),
        make_img_3ch_8x8(),
        timesteps=6,
        auto_reset=False,
    )

    artifacts = _load_compile_artifacts(pb_path)
    thread = artifacts.io_mapping.threads[0]

    expected_tick_depth = max(
        mapping.tick.tick_start for mapping in thread.output_mappings.items
    )
    assert thread.runtime.timesteps == 6
    assert thread.runtime.tick_depth == expected_tick_depth
    assert thread.runtime.sync_steps == expected_tick_depth + 5
    _assert_core_ticks_match_mapper(thread, mapper)
    _assert_thread_tick_duration_initial(thread, (6, 0))


def test_mapper_rejects_timesteps_exceeding_finite_tick_duration(
    ensure_backendv2_debug_dir,
):
    export_dir = ensure_backendv2_debug_dir / "timesteps_exceed_tick_duration"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    graph = compile_to_paiir(
        ANNClassifier().eval(),
        make_img_3ch_8x8(),
        strict=True,
        timesteps=1,
        auto_reset=False,
    )
    mapper = Mapper()

    with pytest.raises(ValueError, match="exceeds finite tick_duration"):
        mapper.compile(graph, export_dir, target_platform="x86", timesteps=2)


@pytest.mark.parametrize(
    ("required_steps", "expected_lcn"),
    [(1, LCN_EX.LCN_1X), (2, LCN_EX.LCN_1X), (3, LCN_EX.LCN_1X), (4, LCN_EX.LCN_1X)],
)
def test_output_group_lcn_selection_uses_output_capacity(required_steps, expected_lcn):
    out_grp = OutputGroup([])
    out_grp.input_list = _output_elems(8, 1, "out")

    out_grp.set_lcn(required_steps)

    assert out_grp.lcn == expected_lcn
    _assert_output_group_allocator_matches_inputs(out_grp)


def test_output_group_warns_when_runtime_steps_exceed_address_capacity():
    out_grp = OutputGroup([])
    out_grp.input_list = _output_elems(8, 600, "out")

    with pytest.warns(RuntimeWarning, match="STEP mode"):
        out_grp.set_lcn(256)

    assert out_grp.lcn == LCN_EX.LCN_128X
    assert len(out_grp.axon_bit_allocator.axon_infos) == 600
    _assert_output_group_allocator_matches_inputs(out_grp)


def test_output_group_data_capacity_uses_entry_count_not_bit_width_sum():
    out_grp = OutputGroup([])
    out_grp.input_list = _output_elems(1, 1025, "out")

    out_grp.set_lcn(1)

    assert out_grp.lcn == LCN_EX.LCN_4X
    assert len(out_grp.axon_bit_allocator.axon_infos) == 1025
    _assert_output_group_allocator_matches_inputs(out_grp)


def test_output_group_voltage_uses_upstream_address_formula():
    out_grp = OutputGroup([])
    out_grp.input_list = _output_elems(32, 10, "v")

    out_grp.set_lcn(1)

    bases = [axon for axon, _ in out_grp.axon_bit_allocator.axon_infos]
    assert out_grp.lcn == _expected_lcn_from_max_axon(max(bases))
    _assert_output_group_allocator_matches_inputs(out_grp)
    assert bases == [0, 1, 2, 3, 4, 5, 6, 7, 32, 33]

    with pytest.warns(RuntimeWarning, match="STEP mode"):
        out_grp.set_lcn(257)
    assert out_grp.lcn == LCN_EX.LCN_128X
    _assert_output_group_allocator_matches_inputs(out_grp)


def test_output_group_raises_when_max_lcn_capacity_is_too_small():
    out_grp = OutputGroup([])
    out_grp.input_list = _output_elems(8, 66000, "out")

    with pytest.raises(ValueError, match="Output axon space is exhausted"):
        out_grp.set_lcn(1)


def test_export_proto_exports_input_dtype_from_consumer_format(
    ensure_backendv2_debug_dir,
):
    pb_path = _export_graph_proto(
        ensure_backendv2_debug_dir,
        "input_dtype_uint1",
        SNNTwoLayer(),
        make_img_3ch_8x8(),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)},
    )

    artifacts = _load_compile_artifacts(pb_path)
    input_mappings = artifacts.io_mapping.threads[0].input_mappings.items

    assert input_mappings
    for mapping in input_mappings:
        _assert_mapping_bit_width(mapping, 1)
        _assert_entries_have_dtype(list(mapping.entries), DataType.UINT1)


@pytest.mark.parametrize(
    ("timesteps", "auto_reset", "expected_tick"),
    [(7, True, (0, 7)), (7, False, (7, 0))],
    ids=["auto_reset", "manual_reset"],
)
def test_export_proto_exports_snn_tick_policy(
    ensure_backendv2_debug_dir, timesteps, auto_reset, expected_tick
):
    pb_path, output_source_names, mapper = _export_graph_proto_with_context(
        ensure_backendv2_debug_dir,
        f"snn_tick_metadata_{timesteps}_{auto_reset}",
        SNNTwoLayer(),
        make_img_3ch_8x8(),
        timesteps=timesteps,
        auto_reset=auto_reset,
    )

    artifacts = _load_compile_artifacts(pb_path)
    thread = artifacts.io_mapping.threads[0]

    assert {mapping.name for mapping in thread.output_mappings.items} == (
        output_source_names
    )
    expected_tick_depth = max(
        mapping.tick.tick_start for mapping in thread.output_mappings.items
    )
    assert thread.runtime.timesteps == timesteps
    assert thread.runtime.tick_depth == expected_tick_depth
    assert thread.runtime.sync_steps == expected_tick_depth + timesteps - 1
    _assert_core_ticks_match_mapper(thread, mapper)
    _assert_thread_tick_duration_initial(thread, expected_tick)


def test_export_artifacts_all_platforms_when_requested(ensure_backendv2_debug_dir):
    export_dir = _export_simple_cnn(
        ensure_backendv2_debug_dir,
        "all_platforms",
        target_platform="all",
        debug=False,
    )

    assert (export_dir / "cfg_frame1.npy").exists()
    assert (export_dir / "cfg_frame2.npy").exists()
    assert (export_dir / "cfg_frame3.npy").exists()
    assert (export_dir / "cfg_frames.npy").exists()

    assert (export_dir / "cfg_frame1.h").exists()
    assert (export_dir / "cfg_frame2.h").exists()
    assert (export_dir / "cfg_frame3.h").exists()
    assert (export_dir / "cfg_frames.h").exists()

    assert not (export_dir / "cfg_frame1.txt").exists()
    assert not (export_dir / "cfg_frame2.txt").exists()
    assert not (export_dir / "cfg_frame3.txt").exists()
    assert not (export_dir / "cfg_frames.txt").exists()
    assert not (export_dir / "proto" / "config.json").exists()

    cfg_frame1 = np.load(export_dir / "cfg_frame1.npy")
    assert cfg_frame1.dtype == np.dtype("<u8")


def test_export_artifacts_can_skip_merged_frames(ensure_backendv2_debug_dir):
    export_dir = _export_simple_cnn(
        ensure_backendv2_debug_dir,
        "skip_merged_frames",
        target_platform="all",
        debug=True,
        export_merged_frames=False,
    )

    assert not (export_dir / "cfg_frames.txt").exists()
    assert not (export_dir / "cfg_frames.npy").exists()
    assert not (export_dir / "cfg_frames.h").exists()

    assert (export_dir / "cfg_frame1.txt").exists()
    assert (export_dir / "cfg_frame1.npy").exists()
    assert (export_dir / "cfg_frame1.h").exists()
    assert (export_dir / "proto" / "config.pb").exists()
    assert (export_dir / "proto" / "config.json").exists()


def test_export_artifacts_debug_forces_all_platform_outputs(
    ensure_backendv2_debug_dir,
):
    export_dir = _export_simple_cnn(
        ensure_backendv2_debug_dir,
        "debug_forces_all",
        target_platform="riscv",
        debug=True,
    )

    assert (export_dir / "cfg_frame1.txt").exists()
    assert (export_dir / "cfg_frame2.txt").exists()
    assert (export_dir / "cfg_frame3.txt").exists()
    assert (export_dir / "cfg_frames.txt").exists()

    assert (export_dir / "cfg_frame1.npy").exists()
    assert (export_dir / "cfg_frame2.npy").exists()
    assert (export_dir / "cfg_frame3.npy").exists()
    assert (export_dir / "cfg_frames.npy").exists()

    assert (export_dir / "cfg_frame1.h").exists()
    assert (export_dir / "cfg_frame2.h").exists()
    assert (export_dir / "cfg_frame3.h").exists()
    assert (export_dir / "cfg_frames.h").exists()

    assert (export_dir / "proto" / "config.pb").exists()
    assert (export_dir / "proto" / "config.json").exists()
    assert (export_dir / "proto" / "compile_artifacts_pb2.py").exists()
    assert (export_dir / "proto" / "compile_artifacts_pb2.pyi").exists()

    _assert_debug_frame_text(export_dir / "cfg_frame1.txt")
    _assert_debug_frame_text(export_dir / "cfg_frame2.txt")
    _assert_debug_frame_text(export_dir / "cfg_frame3.txt")
    _assert_debug_frame_text(export_dir / "cfg_frames.txt")
