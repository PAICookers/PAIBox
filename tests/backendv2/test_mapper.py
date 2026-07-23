from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from paicorelib import (
    LCN_EX,
    RM,
    CoordXY,
    CoordZXYOffset,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    LeakMultiMode,
    NeuronType,
    OfflineNeuRegLimV2,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressType,
    find_coordxy_shortest_path,
)
from spikingjelly.activation_based import neuron
from torch import nn

from paibox.backendv2 import routing as routing_mod
from paibox.backendv2.coreplacement import (
    EmptyOfflineCorePlacementV2,
    EmptyOnlineCorePlacementV2,
    OfflineCorePlacementV2,
)
from paibox.backendv2.mapper import Mapper
from paibox.backendv2.op_node import SourceElem
from paibox.backendv2.output_completion_planner import (
    EmptyThreadCore,
    OutputCompletionPlan,
    OutputProducer,
)
from paibox.backendv2.pressure_unroll import PressureUnrollConfig, PressureUnroller
from paibox.backendv2.routing import (
    FANIN_BASE,
    InputGroup,
    OutputGroup,
    RemapGroup,
    RoutingGroup,
)
from paibox.paiir import (
    LUT_TABLE_SIZE,
    ANNNodeV25,
    CoreNeuronV25,
    LutCustom,
    LutReLU,
    compile_to_paiir,
    register_neuron,
)
from tests.paiir.conftest import ANNClassifier, make_img_3ch_8x8

DEBUG_EXPORT_ROOT = Path(__file__).with_name("debug") / "mapper_proto_export"


class FloatOutputIdentityLutCustom(LutCustom):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


class RepeatedWeightDifferentBiasLinearLut(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        thresholds = torch.arange(LUT_TABLE_SIZE, dtype=torch.int32)
        values = torch.arange(LUT_TABLE_SIZE, dtype=torch.uint8)
        self.linear1 = nn.Linear(4, 4, bias=True)
        self.lut1 = FloatOutputIdentityLutCustom(
            thresholds, values, output_signed=False, is_float=False
        )
        self.linear2 = nn.Linear(4, 2, bias=True)
        self.lut2 = FloatOutputIdentityLutCustom(
            thresholds, values, output_signed=False, is_float=False
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
                    output_signed=lut.output_signed,
                    is_float=lut.is_float,
                )
            ),
        )
    except ValueError as exc:
        if "already registered" not in str(exc):
            raise


def test_mapper_unroll_pressure_delegates_to_pressure_unroller(monkeypatch):
    mapper = Mapper()
    rg = RoutingGroup([], [])
    mapper.routing_groups = [rg]
    calls = []
    routing_calls = []

    def fake_run(self):
        calls.append(self)

    def fake_routing(**kwargs):
        routing_calls.append(kwargs)

    monkeypatch.setattr(PressureUnroller, "run", fake_run)
    monkeypatch.setattr(mapper, "routing", fake_routing)

    mapper.unroll_pressure(
        max_try=2,
        max_factor=3,
        core_selection="quantile",
        peak_ratio=0.9,
        quantile=0.75,
    )

    assert len(calls) == 1
    assert calls[0].routing_groups is mapper.routing_groups
    assert calls[0].config == PressureUnrollConfig(
        max_try=2,
        max_split_factor=3,
        core_selection="quantile",
        peak_ratio=0.9,
        quantile=0.75,
    )
    calls[0].routing_fn()
    assert routing_calls == [{"feasibility_only": True, "max_time_in_seconds": 30}]


def _capture_mapper_route_call(monkeypatch):
    route_kwargs = {}

    def fake_route_solve(
        routing_groups=None,
        next_area_id=None,
        input_groups=None,
        scope=None,
        **kwargs,
    ):
        route_kwargs.update(
            routing_groups=routing_groups,
            next_area_id=next_area_id,
            input_groups=input_groups,
            scope=scope,
            **kwargs,
        )
        return [], []

    monkeypatch.setattr("paibox.backendv2.mapper.route_solve", fake_route_solve)
    return route_kwargs


def test_mapper_passes_routing_context_to_solver(monkeypatch):
    mapper = Mapper()
    input_a = object()
    input_b = object()
    output_a = object()
    output_b = object()
    input_group = InputGroup([input_a, input_b])
    rg = RoutingGroup([output_a, output_b], [input_a, input_b])
    output_group = OutputGroup([output_a, output_b])

    input_group.dests[input_a] = rg
    input_group.dests[input_b] = rg
    rg.dests[output_a] = output_group
    rg.dests[output_b] = output_group
    rg.core_placements = [OfflineCorePlacementV2()]

    mapper.input_groups = [input_group]
    mapper.output_groups = [output_group]
    mapper.routing_groups = [rg]
    mapper.next_rg_group = {0: []}
    route_kwargs = _capture_mapper_route_call(monkeypatch)

    mapper.routing()

    assert route_kwargs["routing_groups"] == [rg]
    assert route_kwargs["input_groups"] == [input_group]
    assert route_kwargs["scope"] is mapper.route_scope


def test_mapper_passes_remap_input_context_to_solver(monkeypatch):
    mapper = Mapper()
    input_elem = object()
    input_group = InputGroup([input_elem])
    rg = RoutingGroup([], [input_elem])
    remap_group = RemapGroup.__new__(RemapGroup)

    def remap_dest(_elem):
        return rg

    remap_group.remap_dest = remap_dest

    input_group.dests[input_elem] = remap_group
    rg.core_placements = [OfflineCorePlacementV2()]

    mapper.input_groups = [input_group]
    mapper.output_groups = []
    mapper.routing_groups = [rg]
    mapper.next_rg_group = {0: []}
    route_kwargs = _capture_mapper_route_call(monkeypatch)

    mapper.routing()

    assert route_kwargs["routing_groups"] == [rg]
    assert route_kwargs["input_groups"] == [input_group]
    assert route_kwargs["scope"] is mapper.route_scope


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


class ConvIfAvgPoolWithNegativeFoldAxonCandidate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(6, 32, 3, padding=1, bias=True)
        self.ifn = neuron.IFNode(v_threshold=127.0)
        self.avg = nn.AvgPool2d(2)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            for oc in range(32):
                channels = [(oc + i) % 6 for i in range(2 + oc % 5)]
                for idx, ic in enumerate(channels):
                    self.conv.weight[oc, ic, 1, 1] = 1 if idx % 2 == 0 else -1

    def forward(self, x):
        return self.avg(self.ifn(self.conv(x)))


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


@pytest.mark.parametrize("ann", [False, True], ids=["snn", "ann"])
def test_mapper_compiles_flat_vector_neuron_parameters(tmp_path, ann):
    kwargs = {
        "reset_v": torch.tensor([0.0, 1.0, 2.0, 3.0]),
        "thres_neg": torch.tensor([-4.0, -5.0, -6.0, -7.0]),
        "thres_pos": torch.tensor([4.0, 5.0, 6.0, 7.0]),
        "leak_multi_mode": LeakMultiMode.ENABLE,
        "leak_tau_shift": torch.tensor([0, -1, -2, -3]),
        "leak_v": torch.tensor([8.0, 9.0, 10.0, 11.0]),
        "init_v": torch.tensor([12.0, 13.0, 14.0, 15.0]),
    }
    act = ANNNodeV25(LutReLU(), **kwargs) if ann else CoreNeuronV25(**kwargs)
    linear = nn.Linear(3, 4, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.ones_like(linear.weight))
    graph = compile_to_paiir(
        nn.Sequential(linear, act).eval(),
        torch.zeros(1, 3),
        input_formats={"InputNode_0": (DataSign.SIGNED, DataWidth.WIDTH_8BIT)},
        strict=True,
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    placements = sorted(
        _shared_sparse_linear_neuron_placements(mapper),
        key=lambda placement: placement.raw_neus[0].index.idx,
    )

    assert len(placements) == 4
    assert [p.neuron_type for p in placements] == [NeuronType.FULL] * 4
    assert [p.neu_attrs_part2.reset_v for p in placements] == [0, 1, 2, 3]
    assert [p.neu_attrs_part2.threshold_pos for p in placements] == [4, 5, 6, 7]
    assert [p.neu_attrs_part2.leak_tau for p in placements] == [0, -1, -2, -3]
    assert [p.neu_attrs_part2.leak_v for p in placements] == [8, 9, 10, 11]


def test_mapper_applies_auto_reset_before_core_placement(tmp_path):
    linear = nn.Linear(3, 2)
    graph = compile_to_paiir(
        linear.eval(),
        torch.zeros(1, 3),
        input_formats={"InputNode_0": (DataSign.SIGNED, DataWidth.WIDTH_8BIT)},
        timesteps=3,
        auto_reset=True,
    )
    mapper = Mapper()
    mapper.compile(
        graph,
        tmp_path,
        target_platform="x86",
        auto_reset=False,
        debug=False,
    )

    core = mapper.coreplacements[0]

    assert mapper.timesteps == 3
    assert core.frontend_core_config.tick_duration == 3
    assert core.frontend_core_config.tick_initial == 0


def test_mapper_preserves_standalone_potential_reset_config(tmp_path):
    graph = compile_to_paiir(
        nn.Linear(3, 2).eval(),
        torch.zeros(1, 3),
        input_formats={"InputNode_0": (DataSign.SIGNED, DataWidth.WIDTH_8BIT)},
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    attrs = mapper.coreplacements[0].neus[0].neu_attrs_part2

    assert attrs is not None
    assert attrs.reset_mode == RM.MODE_NORMAL.value
    assert attrs.reset_v == 0
    assert attrs.threshold_neg_mode == ThresholdNegMode.FIRE.value
    assert attrs.threshold_pos_mode == ThresholdPosMode.FIRE.value
    assert attrs.threshold_neg == 0
    assert attrs.threshold_pos == 0


def test_mapper_vector_mode_controls_part2_and_half_reuse(tmp_path):
    linear = nn.Linear(3, 4, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1)
    act = CoreNeuronV25(
        leak_multi_mode=torch.tensor([0, 0, 1, 1]),
        leak_tau_shift=torch.tensor([0, 0, 0, 0]),
    )
    graph = compile_to_paiir(
        nn.Sequential(linear, act).eval(),
        torch.zeros(1, 3),
        input_formats={"InputNode_0": (DataSign.SIGNED, DataWidth.WIDTH_8BIT)},
        strict=True,
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    placements = sorted(
        _shared_sparse_linear_neuron_placements(mapper),
        key=lambda placement: placement.raw_neus[0].index.idx,
    )

    assert [placement.neuron_type for placement in placements] == [
        NeuronType.FULL,
        NeuronType.HALF,
        NeuronType.FULL,
        NeuronType.HALF,
    ]
    assert placements[0].neu_attrs_part2.leak_multi_mode == LeakMultiMode.DISABLE
    assert placements[1].neu_attrs_part2 is None
    assert placements[2].neu_attrs_part2.leak_multi_mode == LeakMultiMode.ENABLE
    assert placements[3].neu_attrs_part2 is None


def test_mapper_builds_output_completion_plan_without_full_compile(monkeypatch):
    mapper = Mapper()
    producer_coords = [CoordXY(4, 2), CoordXY(2, 4)]
    mapper.coreplacements = []
    for coord in producer_coords:
        core = OfflineCorePlacementV2()
        core._coord = coord
        mapper.coreplacements.append(core)

    monkeypatch.setattr(
        mapper,
        "_collect_output_producers",
        lambda: [OutputProducer(coord, CoordXY(0, 0), 1) for coord in producer_coords],
    )

    plan = mapper.build_output_completion_plan()

    assert plan.completion_join_point == CoordXY(0, 2)
    assert plan.global_signal_root == CoordXY(0, 2)
    assert {route.target_coord for route in plan.output_routes} == {CoordXY(0, 0)}
    assert plan.output_route_offsets()[
        (CoordXY(4, 2), CoordXY(0, 0))
    ] == CoordZXYOffset(0, -4, -2)
    assert CoordXY(0, 2) in {core.coord for core in plan.completion_thread_cores}
    assert CoordXY(0, 1) not in {core.coord for core in plan.completion_thread_cores}


def test_mapper_adds_selected_empty_output_completion_core(monkeypatch):
    mapper = Mapper()
    producer_coords = [CoordXY(0, 3), CoordXY(2, 2)]
    mapper.coreplacements = []
    for coord in producer_coords:
        core = OfflineCorePlacementV2()
        core._coord = coord
        mapper.coreplacements.append(core)

    monkeypatch.setattr(
        mapper,
        "_collect_output_producers",
        lambda: [OutputProducer(coord, CoordXY(0, 0), 1) for coord in producer_coords],
    )

    plan = mapper.build_output_completion_plan()
    mapper.add_output_completion_thread_cores(plan)

    required_core = next(
        cp for cp in mapper.coreplacements if cp.coord == CoordXY(0, 2)
    )
    assert isinstance(required_core, EmptyOfflineCorePlacementV2)
    assert all(cp.coord != CoordXY(0, 1) for cp in mapper.coreplacements)


def test_mapper_adds_output_prefix_route_cores(monkeypatch):
    mapper = Mapper()
    producer_coords = [
        CoordXY(0, 2),
        CoordXY(1, 3),
        CoordXY(2, 4),
        CoordXY(3, 5),
        CoordXY(4, 6),
        CoordXY(5, 6),
        CoordXY(5, 5),
        CoordXY(5, 4),
        CoordXY(5, 3),
        CoordXY(5, 2),
    ]
    mapper.coreplacements = []
    for coord in producer_coords:
        core = OfflineCorePlacementV2()
        core._coord = coord
        mapper.coreplacements.append(core)

    monkeypatch.setattr(
        mapper,
        "_collect_output_producers",
        lambda: [OutputProducer(coord, CoordXY(0, 0), 1) for coord in producer_coords],
    )

    plan = mapper.build_output_completion_plan()
    mapper.add_output_completion_thread_cores(plan)
    core_by_coord = {cp.coord: cp for cp in mapper.coreplacements}

    assert CoordXY(4, 5) in core_by_coord
    assert isinstance(core_by_coord[CoordXY(4, 5)], EmptyOfflineCorePlacementV2)
    assert CoordXY(0, 1) not in core_by_coord


def test_mapper_adds_required_online_output_completion_route_cores(monkeypatch):
    mapper = Mapper()
    existing_core = OfflineCorePlacementV2()
    existing_core._coord = CoordXY(1, 1)
    mapper.coreplacements = [existing_core]
    plan = OutputCompletionPlan(
        (),
        CoordZXYOffset(0, -1, -1),
        ("y", -1),
        CoordXY(1, 1),
        CoordXY(1, 1),
        (EmptyThreadCore(CoordXY(2, 1), "online"),),
        (),
    )

    mapper.add_output_completion_thread_cores(plan)

    required_core = next(
        cp for cp in mapper.coreplacements if cp.coord == CoordXY(2, 1)
    )
    assert isinstance(required_core, EmptyOnlineCorePlacementV2)


def test_mapper_does_not_broadcast_root_control_offset_to_all_cores():
    mapper = Mapper()
    root_cp = OfflineCorePlacementV2()
    other_cp = OfflineCorePlacementV2()
    root_cp._coord = CoordXY(5, 5)
    other_cp._coord = CoordXY(6, 5)
    mapper.coreplacements = [root_cp, other_cp]

    root_control_offset = CoordZXYOffset(-4, -1, -1)
    mapper.set_auto_core_config(root_cp.coord, root_control_offset)

    other_offset, _ = find_coordxy_shortest_path(CoordXY(0, 0), other_cp.coord)
    assert (
        root_cp.auto_core_config.test_core_xy,
        root_cp.auto_core_config.test_core_x,
        root_cp.auto_core_config.test_core_y,
    ) == root_control_offset.to_tuple()
    assert (
        other_cp.auto_core_config.test_core_xy,
        other_cp.auto_core_config.test_core_x,
        other_cp.auto_core_config.test_core_y,
    ) == other_offset.to_tuple()
    assert other_offset != root_control_offset


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

    compute_cores = [core for core in mapper.coreplacements if core.neus]
    core = compute_cores[0]
    placements = _shared_sparse_linear_neuron_placements(mapper)

    assert {c.frontend_core_config.input_width for c in compute_cores} == {
        DataWidth.WIDTH_1BIT
    }
    assert {c.default_core_config.csc_accelerate for c in compute_cores} == {
        CSCAccelerateMode.ENABLE
    }
    assert [weight.compress for weight in core.weights] == [True, True, False]
    assert np.flatnonzero(core.weights[0].raw_weights).tolist() == [0, 24, 64]
    assert np.flatnonzero(core.weights[1].raw_weights).tolist() == [0, 128]
    assert np.array_equal(
        core.weights[2].processed_weights, [1, 0, 1, 0, 1, 0, 1, 0, 1]
    )
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

    compute_cores = [core for core in mapper.coreplacements if core.neus]
    core = compute_cores[0]
    placements = _shared_sparse_linear_neuron_placements(mapper)

    assert {c.frontend_core_config.input_width for c in compute_cores} == {
        DataWidth.WIDTH_1BIT
    }
    assert {c.default_core_config.csc_accelerate for c in compute_cores} == {
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


def test_mapper_skips_negative_fold_axon_skew_candidate(tmp_path):
    graph = compile_to_paiir(
        ConvIfAvgPoolWithNegativeFoldAxonCandidate().eval(),
        torch.zeros(1, 6, 16, 16),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)},
        timesteps=1,
        auto_reset=True,
        output_approx="sum_approx_if_avgpool",
        strict=True,
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    assert mapper.coreplacements
    placements = _shared_sparse_linear_neuron_placements(mapper)
    folded_attrs = [
        placement.folded_neu_attrs_part1
        for placement in placements
        if placement.folded_neu_attrs_part1 is not None
    ]
    assert folded_attrs
    for attrs in folded_attrs:
        assert 0 <= attrs.fold_axon_y <= OfflineNeuRegLimV2.FOLD_AXON_MAX
        assert 0 <= attrs.fold_axon_x <= OfflineNeuRegLimV2.FOLD_AXON_MAX
        assert 0 <= attrs.fold_axon_xy <= OfflineNeuRegLimV2.FOLD_AXON_MAX
        assert 0 <= attrs.fold_skew_y <= OfflineNeuRegLimV2.FOLD_SKEW_MAX
        assert 0 <= attrs.fold_skew_x <= OfflineNeuRegLimV2.FOLD_SKEW_MAX
        assert 0 <= attrs.fold_skew_xy <= OfflineNeuRegLimV2.FOLD_SKEW_MAX


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


def test_mapper_does_not_fold_neurons_with_different_leak_modes(tmp_path):
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False),
        CoreNeuronV25(
            leak_multi_mode=torch.tensor([0, 1, 0, 1]),
            leak_tau_shift=torch.full((4,), -1),
        ),
        nn.Linear(4, 2, bias=False),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([1, -1, 1, -1]).repeat(4, 1))
        model[2].weight.copy_(torch.tensor([[1, 1, -1, -1], [-1, 1, -1, 1]]))

    graph = compile_to_paiir(
        model.eval(),
        torch.zeros(1, 4),
        input_formats={"InputNode_0": (DataSign.SIGNED, DataWidth.WIDTH_8BIT)},
        timesteps=1,
        auto_reset=True,
        strict=True,
    )
    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)

    placements = _shared_sparse_linear_neuron_placements(mapper)
    first_layer = sorted(
        (
            placement
            for placement in placements
            if len(placement.raw_neus) == 1
            and placement.raw_neus[0].target.name == "SequentialOp_0"
        ),
        key=lambda placement: placement.raw_neus[0].index.idx,
    )

    assert len(first_layer) == 4
    assert all(placement.folded_neu_attrs_part1 is None for placement in first_layer)
    assert [placement.neu_attrs_part2.leak_multi_mode for placement in first_layer] == [
        LeakMultiMode.DISABLE,
        LeakMultiMode.ENABLE,
        LeakMultiMode.DISABLE,
        LeakMultiMode.ENABLE,
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
    np.testing.assert_array_equal(core.weights[0].processed_weights, [1, 0, 1])
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


def test_mapper_rejects_timesteps_exceeding_finite_tick_duration(tmp_path):
    graph = compile_to_paiir(
        ANNClassifier().eval(),
        make_img_3ch_8x8(),
        strict=True,
        timesteps=1,
        auto_reset=False,
    )
    mapper = Mapper()

    with pytest.raises(ValueError, match="exceeds finite tick_duration"):
        mapper.compile(graph, tmp_path, target_platform="x86", timesteps=2)


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
