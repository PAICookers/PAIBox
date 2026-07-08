from collections.abc import Callable
from dataclasses import replace

import pytest
import torch.nn as nn
from paicorelib import (
    CoordXY,
    CSCAccelerateMode,
    DataWidth,
    FoldType,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    NeuronType,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFoldedAttrsV2Part2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
    OutputType,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressType,
)
from paicorelib.neuron_defs import ResetMode

from paibox.backendv2 import pressure_unroll as pressure_unroll_module
from paibox.backendv2.coreplacement import OfflineCorePlacementV2
from paibox.backendv2.neuron import OfflineNeuronPlacement
from paibox.backendv2.op_node import CoreOpNode, CustomIndex, Neuron
from paibox.backendv2.pressure_unroll import (
    PressureUnrollConfig,
    PressureUnroller,
    PressureUnrollStopReason,
)
from paibox.backendv2.route_scope import get_route_scope
from paibox.backendv2.routing import RoutingGroup
from paibox.backendv2.weight import Weight
from paibox.paiir.ir.op_node import StandaloneCompOp

WEIGHT_VALUES_PER_NEURON = 17
PRESSURE_PER_UNIT = 2048


@pytest.fixture
def offline_core_count(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    def set_count(count: int) -> None:
        coords = frozenset(CoordXY(i, 2) for i in range(count))
        scope = replace(get_route_scope("single"), offline_core_coords=coords)
        monkeypatch.setattr(pressure_unroll_module, "get_route_scope", lambda _: scope)

    return set_count


def _attrs_part1() -> OfflineNeuFullAttrsV2Part1:
    return OfflineNeuFullAttrsV2Part1(
        weight_skew=0,
        weight_address_start=0,
        weight_address_end=0,
        fold_type=FoldType.UNFOLDED,
        neuron_type=NeuronType.FULL,
        output_type=OutputType.VALUE,
    )


def _attrs_part2(
    weight_compress: WeightCompressType = WeightCompressType.DENSE,
    vjt_initial: int = 0,
) -> OfflineNeuFullAttrsV2Part2:
    return OfflineNeuFullAttrsV2Part2(
        reset_mode=ResetMode.MODE_NORMAL,
        reset_v=0,
        threshold_neg_mode=ThresholdNegMode.FIRE,
        threshold_pos_mode=ThresholdPosMode.FIRE,
        threshold_neg=0,
        threshold_pos=1,
        lateral_inhibition=LateralInhibitionMode.DISABLE,
        leak_multi_sequence=LeakMultiComparisonOrder.BEFORE_COMPARE,
        leak_multi_input=LeakMultiInputMode.DISABLE,
        leak_multi_mode=LeakMultiMode.DISABLE,
        leak_add_mode=LeakAddMode.FORWARD,
        leak_tau=0,
        leak_v=0,
        weight_compress=weight_compress,
        vjt_initial=vjt_initial,
    )


def _fold_attrs_part1() -> OfflineNeuFoldedAttrsV2Part1:
    return OfflineNeuFoldedAttrsV2Part1(
        fold_range_xy=1,
        fold_range_x=1,
        fold_range_y=1,
        fold_skew_xy=0,
        fold_skew_x=0,
        fold_skew_y=0,
        fold_axon_xy=0,
        fold_axon_x=0,
        fold_axon_y=0,
        fold_number=1,
    )


def _fold_attrs_part2() -> OfflineNeuFoldedAttrsV2Part2:
    return OfflineNeuFoldedAttrsV2Part2(
        fold_vjt_0=1,
        fold_vjt_1=2,
        fold_vjt_2=3,
        fold_vjt_3=4,
    )


def _weight() -> Weight:
    return Weight(
        data=[1] * WEIGHT_VALUES_PER_NEURON,
        compress_type=WeightCompressType.DENSE,
        weight_width=DataWidth.WIDTH_8BIT,
        input_width=DataWidth.WIDTH_8BIT,
    )


def _weight_with_values(values: list[int]) -> Weight:
    return Weight(
        data=values,
        compress_type=WeightCompressType.DENSE,
        weight_width=DataWidth.WIDTH_8BIT,
        input_width=DataWidth.WIDTH_8BIT,
    )


def _sparse_weight() -> Weight:
    return Weight(
        data=[1, 0, 2, 0, 3, 0, 4, 0, 5],
        compress_type=WeightCompressType.SPARSE,
        weight_width=DataWidth.WIDTH_8BIT,
        input_width=DataWidth.WIDTH_8BIT,
    )


def _core_node(name: str, units: int) -> CoreOpNode:
    raw_node = StandaloneCompOp(nn.Identity())
    raw_node.core_params.tick_start = 1
    raw_node.core_params.tick_duration = 0
    raw_node.core_params.tick_initial = 1
    raw_node.output_shape = (units,)
    return CoreOpNode(name, raw_node, raw_node.output_shape)


def _neurons(target: CoreOpNode, units: int) -> list[Neuron]:
    return [Neuron(target, CustomIndex(idx)) for idx in range(units)]


def _make_pressure_core(units: int) -> OfflineCorePlacementV2:
    target = _core_node("core", units)
    core = OfflineCorePlacementV2()
    core.neus = [
        OfflineNeuronPlacement([neu], _attrs_part1(), _attrs_part2())
        for neu in _neurons(target, units)
    ]
    core.weights = [_weight()]
    core.neu_weight_map = {idx: 0 for idx in range(units)}
    return core


def _make_rg(
    units_by_core: list[int], *, target: CoreOpNode | None = None
) -> RoutingGroup:
    rg = RoutingGroup([], [], nodes={target} if target is not None else None)
    rg.core_placements = [_make_pressure_core(units) for units in units_by_core]
    return rg


def _pressure(units: int) -> int:
    return units * PRESSURE_PER_UNIT


def test_unroll_pressure_stops_when_selected_cores_have_no_split_candidate(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(2)

    rg = RoutingGroup([], [])
    original_core = _make_pressure_core(1)
    rg.core_placements = [original_core]

    def fail_if_called():
        raise AssertionError("routing_fn should not run with no split candidate")

    unroller = PressureUnroller([rg], fail_if_called)

    result = unroller.run()

    assert rg.core_placements == [original_core]
    assert len(rg.core_placements) == 1
    assert result.n_try == 0
    assert result.stop_reason is PressureUnrollStopReason.NO_CANDIDATE


def test_unroll_pressure_tries_highest_pressure_group_first(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(6)

    high_rg = _make_rg([100])
    low_rg = _make_rg([80])

    calls = 0

    def routing_fn():
        nonlocal calls
        calls += 1

    unroller = PressureUnroller(
        [low_rg, high_rg], routing_fn, PressureUnrollConfig(max_try=1)
    )

    result = unroller.run()

    assert calls == 1
    assert result.n_try == 1
    assert result.n_commit == 1
    assert result.stop_reason is PressureUnrollStopReason.MAX_TRY
    assert result.peak_drops == [(high_rg.name, _pressure(100), _pressure(50))]
    assert len(high_rg.core_placements) == 2
    assert len(low_rg.core_placements) == 1
    assert all(
        all(neu.neu_attrs_part1.weight_address_end >= 0 for neu in core.neus)
        for core in high_rg.core_placements
    )


def test_unroll_pressure_split_cores_do_not_share_default_config(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(6)

    rg = _make_rg([100])

    result = PressureUnroller([rg], lambda: None, PressureUnrollConfig(1, 2)).run()

    assert result.n_commit == 1
    first_core, second_core = rg.core_placements
    assert first_core.default_core_config is not second_core.default_core_config

    first_core.default_core_config.csc_accelerate = CSCAccelerateMode.DISABLE
    assert second_core.default_core_config.csc_accelerate == CSCAccelerateMode.ENABLE


def test_unroll_pressure_preserves_neuron_weight_mapping_after_split(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(8)

    target = _core_node("layer", 6)
    core = OfflineCorePlacementV2()
    core.neus = [
        OfflineNeuronPlacement([neu], _attrs_part1(), _attrs_part2())
        for neu in _neurons(target, 6)
    ]
    weight_a = _weight_with_values([1] * 17)
    weight_b = _weight_with_values([2] * 33)
    weight_c = _weight_with_values([3] * 49)
    core.weights = [weight_a, weight_b, weight_c]
    core.neu_weight_map = {
        0: 0,
        1: 1,
        2: 0,
        3: 2,
        4: 1,
        5: 2,
    }
    rg = RoutingGroup([], [])
    rg.core_placements = [core]

    result = PressureUnroller(
        [rg], lambda: None, PressureUnrollConfig(max_try=1, max_split_factor=3)
    ).run()

    assert result.n_commit == 1
    split_cores = rg.core_placements
    assert [len(split_core.neus) for split_core in split_cores] == [2, 2, 2]
    assert [
        [neu.raw_neus[0].index.idx for neu in split_core.neus]
        for split_core in split_cores
    ] == [[0, 1], [2, 3], [4, 5]]
    assert [split_cores[0].weights[i] for i in (0, 1)] == [weight_a, weight_b]
    assert [split_cores[1].weights[i] for i in (0, 1)] == [weight_a, weight_c]
    assert [split_cores[2].weights[i] for i in (0, 1)] == [weight_b, weight_c]
    assert [split_core.neu_weight_map for split_core in split_cores] == [
        {0: 0, 1: 1},
        {0: 0, 1: 1},
        {0: 0, 1: 1},
    ]


def test_unroll_pressure_preserves_folded_neurons_after_split(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(4)

    target = _core_node("layer", 4)
    original_fold_attrs = _fold_attrs_part1()
    original_fold_part2s = [_fold_attrs_part2()]
    core = OfflineCorePlacementV2()
    core.neus = [
        OfflineNeuronPlacement(
            [neu],
            _attrs_part1(),
            _attrs_part2(),
            original_fold_attrs,
            original_fold_part2s,
        )
        for neu in _neurons(target, 4)
    ]
    weight_a = _weight_with_values([1])
    weight_b = _weight_with_values([2])
    core.weights = [weight_a, weight_b]
    core.neu_weight_map = {0: 0, 1: 1, 2: 0, 3: 1}
    rg = RoutingGroup([], [])
    rg.core_placements = [core]

    result = PressureUnroller(
        [rg], lambda: None, PressureUnrollConfig(max_try=1, max_split_factor=2)
    ).run()

    assert result.n_commit == 1
    first_core, second_core = rg.core_placements
    assert [len(split_core.neus) for split_core in (first_core, second_core)] == [2, 2]
    assert [first_core.weights[i] for i in (0, 1)] == [weight_a, weight_b]
    assert [second_core.weights[i] for i in (0, 1)] == [weight_a, weight_b]
    assert first_core.neu_weight_map == {0: 0, 1: 1}
    assert second_core.neu_weight_map == {0: 0, 1: 1}

    for split_core in (first_core, second_core):
        for neu in split_core.neus:
            assert neu.folded_neu_attrs_part1 == original_fold_attrs
            assert neu.folded_neu_attrs_part1 is not original_fold_attrs
            assert neu.folded_neu_attrs_part2s[0] is not original_fold_part2s[0]
            assert (
                neu.folded_neu_attrs_part2s[0].fold_vjt_0,
                neu.folded_neu_attrs_part2s[0].fold_vjt_1,
                neu.folded_neu_attrs_part2s[0].fold_vjt_2,
                neu.folded_neu_attrs_part2s[0].fold_vjt_3,
            ) == (0, 0, 0, 0)

    assert original_fold_part2s[0].fold_vjt_0 == 1


def test_unroll_pressure_promotes_split_leading_half_neuron_to_full(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(4)

    target = _core_node("layer", 4)
    source_neus = _neurons(target, 4)
    full_attrs = _attrs_part2()
    full_attrs.threshold_pos = 7
    core = OfflineCorePlacementV2()
    core.neus = [
        OfflineNeuronPlacement([source_neus[0]], _attrs_part1(), full_attrs),
        OfflineNeuronPlacement(
            [source_neus[1]],
            _attrs_part1().model_copy(update={"neuron_type": NeuronType.HALF}),
            full_attrs,
        ),
        OfflineNeuronPlacement(
            [source_neus[2]],
            _attrs_part1().model_copy(update={"neuron_type": NeuronType.HALF}),
            full_attrs,
        ),
        OfflineNeuronPlacement(
            [source_neus[3]],
            _attrs_part1().model_copy(update={"neuron_type": NeuronType.HALF}),
            full_attrs,
        ),
    ]
    core.weights = [_weight_with_values([1]), _weight_with_values([2])]
    core.neu_weight_map = {0: 0, 1: 1, 2: 0, 3: 1}
    rg = RoutingGroup([], [])
    rg.core_placements = [core]

    result = PressureUnroller(
        [rg], lambda: None, PressureUnrollConfig(max_try=1, max_split_factor=2)
    ).run()

    assert result.n_commit == 1
    first_core, second_core = rg.core_placements
    assert [neu.neuron_type for neu in first_core.neus] == [
        NeuronType.FULL,
        NeuronType.HALF,
    ]
    assert [neu.neuron_type for neu in second_core.neus] == [
        NeuronType.FULL,
        NeuronType.HALF,
    ]
    assert second_core.neus[0].neu_attrs_part2 is not None
    assert second_core.neus[0].neu_attrs_part2.threshold_pos == 7
    assert second_core.neus[0].neu_attrs_part2 is not full_attrs


def test_unroll_pressure_preserves_non_divisible_split_boundaries_and_weights(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(4)

    target = _core_node("layer", 10)
    core = OfflineCorePlacementV2()
    core.neus = [
        OfflineNeuronPlacement([neu], _attrs_part1(), _attrs_part2())
        for neu in _neurons(target, 10)
    ]
    weight_a = _weight_with_values([1] * 17)
    weight_b = _weight_with_values([2] * 33)
    weight_c = _weight_with_values([3] * 49)
    core.weights = [weight_a, weight_b, weight_c]
    core.neu_weight_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 0,
        4: 2,
        5: 1,
        6: 2,
        7: 1,
        8: 0,
        9: 1,
    }
    rg = RoutingGroup([], [])
    rg.core_placements = [core]

    result = PressureUnroller(
        [rg], lambda: None, PressureUnrollConfig(max_try=1, max_split_factor=3)
    ).run()

    assert result.n_commit == 1
    split_cores = rg.core_placements
    assert [len(split_core.neus) for split_core in split_cores] == [4, 3, 3]
    assert [
        [neu.raw_neus[0].index.idx for neu in split_core.neus]
        for split_core in split_cores
    ] == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert [split_cores[0].weights[i] for i in (0, 1, 2)] == [
        weight_a,
        weight_b,
        weight_c,
    ]
    assert [split_cores[1].weights[i] for i in (0, 1)] == [weight_c, weight_b]
    assert [split_cores[2].weights[i] for i in (0, 1)] == [weight_b, weight_a]
    assert [split_core.neu_weight_map for split_core in split_cores] == [
        {0: 0, 1: 1, 2: 2, 3: 0},
        {0: 0, 1: 1, 2: 0},
        {0: 0, 1: 1, 2: 0},
    ]


def test_unroll_pressure_clones_sparse_neuron_attrs_before_readdressing(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(6)

    target = _core_node("layer", 4)
    core = OfflineCorePlacementV2()
    core.neus = [
        OfflineNeuronPlacement(
            [neu],
            _attrs_part1(),
            _attrs_part2(WeightCompressType.SPARSE),
        )
        for neu in _neurons(target, 4)
    ]
    core.weights = [_sparse_weight()]
    core.neu_weight_map = {idx: 0 for idx in range(4)}
    core.set_weight_address()
    original_neurons = list(core.neus)

    assert core.default_core_config.csc_accelerate == CSCAccelerateMode.ENABLE
    assert all(neu.neu_attrs_part2.vjt_initial != 0 for neu in original_neurons)

    rg = RoutingGroup([], [])
    rg.core_placements = [core]

    result = PressureUnroller([rg], lambda: None, PressureUnrollConfig(1, 2)).run()

    assert result.n_commit == 1
    first_core, second_core = rg.core_placements
    assert first_core.default_core_config.csc_accelerate == CSCAccelerateMode.ENABLE
    assert second_core.default_core_config.csc_accelerate == CSCAccelerateMode.ENABLE
    assert all(
        neu not in original_neurons for neu in first_core.neus + second_core.neus
    )
    for split_core in (first_core, second_core):
        for idx, neu in enumerate(split_core.neus):
            assert neu.neu_attrs_part2.vjt_initial == (
                neu.neu_attrs_part1.weight_address_start
            )
            assert split_core.neu_weight_map[idx] == 0


def test_unroll_pressure_keeps_real_nonzero_sparse_vjt_initial(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(6)

    target = _core_node("layer", 4)
    core = OfflineCorePlacementV2()
    core.neus = [
        OfflineNeuronPlacement(
            [neu],
            _attrs_part1(),
            _attrs_part2(WeightCompressType.SPARSE, vjt_initial=7),
        )
        for neu in _neurons(target, 4)
    ]
    core.weights = [_sparse_weight()]
    core.neu_weight_map = {idx: 0 for idx in range(4)}
    core.set_weight_address()

    assert core.default_core_config.csc_accelerate == CSCAccelerateMode.DISABLE

    rg = RoutingGroup([], [])
    rg.core_placements = [core]

    result = PressureUnroller([rg], lambda: None, PressureUnrollConfig(1, 2)).run()

    assert result.n_commit == 1
    for split_core in rg.core_placements:
        assert (
            split_core.default_core_config.csc_accelerate == CSCAccelerateMode.DISABLE
        )
        assert all(neu.neu_attrs_part2.vjt_initial == 7 for neu in split_core.neus)


def test_unroll_pressure_prefers_peak_bottleneck_over_efficiency(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(12)

    expensive_peak_rg = _make_rg([100, 100, 100])
    cheap_next_rg = _make_rg([90])

    result = PressureUnroller(
        [expensive_peak_rg, cheap_next_rg],
        lambda: None,
        PressureUnrollConfig(max_try=1, max_split_factor=2),
    ).run()

    assert result.n_try == 1
    assert result.n_commit == 1
    assert result.peak_drops == [
        (expensive_peak_rg.name, _pressure(100), _pressure(50))
    ]
    assert len(expensive_peak_rg.core_placements) == 6
    assert len(cheap_next_rg.core_placements) == 1


def test_unroll_pressure_splits_same_target_groups_as_one_layer(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(8)

    target = _core_node("layer", 1)
    first_rg = _make_rg([100], target=target)
    second_rg = _make_rg([100], target=target)

    calls = 0

    def routing_fn():
        nonlocal calls
        calls += 1

    result = PressureUnroller(
        [first_rg, second_rg], routing_fn, PressureUnrollConfig(1, 2)
    ).run()

    assert calls == 1
    assert result.n_try == 1
    assert result.n_commit == 1
    assert len(first_rg.core_placements) == 2
    assert len(second_rg.core_placements) == 2
    assert result.peak_drops[0][1:] == (_pressure(100), _pressure(50))


def test_unroll_pressure_splits_only_budgeted_selected_cores(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(6)

    target = _core_node("layer", 1)
    rg = _make_rg([100, 90, 80, 70], target=target)

    result = PressureUnroller([rg], lambda: None, PressureUnrollConfig(1, 2)).run()

    assert result.n_try == 1
    assert result.n_commit == 1
    assert result.peak_drops[0][1:] == (_pressure(100), _pressure(80))
    assert [core.get_compute_pressure() for core in rg.core_placements] == [
        _pressure(50),
        _pressure(50),
        _pressure(45),
        _pressure(45),
        _pressure(80),
        _pressure(70),
    ]


def test_unroll_pressure_uses_peak_ratio_by_default(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(10)

    target = _core_node("layer", 1)
    rg = _make_rg([100, 95, 90, 79], target=target)

    result = PressureUnroller([rg], lambda: None, PressureUnrollConfig(1, 2)).run()

    assert result.n_try == 1
    assert result.n_commit == 1
    assert result.peak_drops[0][1:] == (_pressure(100), _pressure(79))
    assert [core.get_compute_pressure() for core in rg.core_placements] == [
        _pressure(50),
        _pressure(50),
        _pressure(48),
        _pressure(47),
        _pressure(45),
        _pressure(45),
        _pressure(79),
    ]


def test_unroll_pressure_quantile_half_matches_median(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(10)

    target = _core_node("layer", 1)
    rg = _make_rg([100, 90, 80, 70], target=target)

    result = PressureUnroller(
        [rg],
        lambda: None,
        PressureUnrollConfig(1, 2, core_selection="quantile", quantile=0.5),
    ).run()

    assert result.n_try == 1
    assert result.n_commit == 1
    assert result.peak_drops[0][1:] == (_pressure(100), _pressure(80))
    assert [core.get_compute_pressure() for core in rg.core_placements] == [
        _pressure(50),
        _pressure(50),
        _pressure(45),
        _pressure(45),
        _pressure(80),
        _pressure(70),
    ]


def test_unroll_pressure_quantile_can_select_higher_tail(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(10)

    target = _core_node("layer", 1)
    rg = _make_rg([100, 90, 80, 70], target=target)

    result = PressureUnroller(
        [rg],
        lambda: None,
        PressureUnrollConfig(1, 2, core_selection="quantile", quantile=0.75),
    ).run()

    assert result.n_try == 1
    assert result.n_commit == 1
    assert result.peak_drops[0][1:] == (_pressure(100), _pressure(90))
    assert [core.get_compute_pressure() for core in rg.core_placements] == [
        _pressure(50),
        _pressure(50),
        _pressure(90),
        _pressure(80),
        _pressure(70),
    ]


def test_unroll_pressure_splits_all_equal_layer_pressures(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(8)

    target = _core_node("layer", 1)
    rg = _make_rg([100, 100, 100], target=target)

    result = PressureUnroller([rg], lambda: None, PressureUnrollConfig(1, 2)).run()

    assert result.n_try == 1
    assert result.n_commit == 1
    assert len(rg.core_placements) == 6
    assert result.peak_drops[0][1:] == (_pressure(100), _pressure(50))


def test_unroll_pressure_recomputes_after_success_until_probe_budget(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(8)

    first_rg = _make_rg([100])
    second_rg = _make_rg([90])

    calls = 0

    def routing_fn():
        nonlocal calls
        calls += 1

    result = PressureUnroller(
        [first_rg, second_rg], routing_fn, PressureUnrollConfig(2, 2)
    ).run()

    assert calls == 2
    assert result.n_try == 2
    assert result.n_commit == 2
    assert result.stop_reason is PressureUnrollStopReason.MAX_TRY
    assert len(first_rg.core_placements) == 2
    assert len(second_rg.core_placements) == 2


def test_unroll_pressure_falls_back_to_next_group_after_routing_failure(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(6)

    high_rg = _make_rg([100])
    high_original_cores = list(high_rg.core_placements)
    next_rg = _make_rg([90])

    calls = 0

    def routing_fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("highest pressure group cannot route")

    result = PressureUnroller(
        [high_rg, next_rg], routing_fn, PressureUnrollConfig(2)
    ).run()

    assert calls == 2
    assert result.n_try == 2
    assert result.n_commit == 1
    assert result.peak_drops == [(next_rg.name, _pressure(90), _pressure(45))]
    assert high_rg.core_placements == high_original_cores
    assert len(next_rg.core_placements) == 2


def test_unroll_pressure_max_try_limits_route_probes(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(20)

    first_rg = _make_rg([100])
    second_rg = _make_rg([90])

    result = PressureUnroller(
        [first_rg, second_rg],
        lambda: (_ for _ in ()).throw(RuntimeError("no route")),
        PressureUnrollConfig(1),
    ).run()

    assert result.n_try == 1
    assert result.n_commit == 0
    assert result.stop_reason is PressureUnrollStopReason.ROUTING_FAILED
    assert len(first_rg.core_placements) == 1
    assert len(second_rg.core_placements) == 1


def test_unroll_pressure_default_max_try_limits_route_probes(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(20)

    rgs = [_make_rg([pressure]) for pressure in (100, 90, 80, 70, 60)]

    calls = 0

    def routing_fn():
        nonlocal calls
        calls += 1
        raise RuntimeError("no route")

    result = PressureUnroller(rgs, routing_fn).run()

    assert calls == 4
    assert result.n_try == 4
    assert result.stop_reason is PressureUnrollStopReason.ROUTING_FAILED
    assert all(len(rg.core_placements) == 1 for rg in rgs)


def test_unroll_pressure_computes_split_factor_from_budget_and_config(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(6)

    rg = RoutingGroup([], [])
    rg.core_placements = [_make_pressure_core(8)]

    result = PressureUnroller(
        [rg],
        lambda: None,
        PressureUnrollConfig(max_split_factor=4),
    ).run()

    assert result.n_try == 1
    assert result.n_commit == 1
    assert len(rg.core_placements) == 4


def test_unroll_pressure_can_cap_split_factor_to_two(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(6)

    rg = RoutingGroup([], [])
    rg.core_placements = [_make_pressure_core(8)]

    result = PressureUnroller([rg], lambda: None, PressureUnrollConfig(1, 2)).run()

    assert result.n_try == 1
    assert result.n_commit == 1
    assert len(rg.core_placements) == 2


def test_unroll_pressure_skips_peak_group_when_free_cores_are_insufficient(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(5)

    peak_rg = RoutingGroup([], [])
    peak_rg.core_placements = [
        _make_pressure_core(100),
        _make_pressure_core(100),
        _make_pressure_core(100),
    ]
    next_rg = RoutingGroup([], [])
    next_rg.core_placements = [_make_pressure_core(80)]

    calls = 0

    def routing_fn():
        nonlocal calls
        calls += 1

    result = PressureUnroller([peak_rg, next_rg], routing_fn).run()

    assert calls == 1
    assert result.n_try == 1
    assert result.n_commit == 1
    assert result.peak_drops == [(next_rg.name, _pressure(80), _pressure(40))]
    assert len(peak_rg.core_placements) == 3
    assert len(next_rg.core_placements) == 2


def test_unroll_pressure_rolls_back_all_failed_group_attempts(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(4)

    first_rg = RoutingGroup([], [])
    first_original_cores = [_make_pressure_core(100)]
    first_rg.core_placements = list(first_original_cores)
    second_rg = RoutingGroup([], [])
    second_original_cores = [_make_pressure_core(90)]
    second_rg.core_placements = list(second_original_cores)

    calls = 0

    def routing_fn():
        nonlocal calls
        calls += 1
        raise RuntimeError("no route")

    result = PressureUnroller([first_rg, second_rg], routing_fn).run()

    assert calls == 2
    assert result.n_try == 2
    assert result.n_commit == 0
    assert result.stop_reason is PressureUnrollStopReason.ROUTING_FAILED
    assert first_rg.core_placements == first_original_cores
    assert second_rg.core_placements == second_original_cores


def test_unroll_pressure_rolls_back_touched_layer_groups(
    offline_core_count: Callable[[int], None],
):
    offline_core_count(8)

    target = _core_node("layer", 1)
    first_rg = _make_rg([100], target=target)
    second_rg = _make_rg([100], target=target)
    first_original_cores = list(first_rg.core_placements)
    second_original_cores = list(second_rg.core_placements)

    result = PressureUnroller(
        [first_rg, second_rg],
        lambda: (_ for _ in ()).throw(RuntimeError("no route")),
        PressureUnrollConfig(1, 2),
    ).run()

    assert result.n_try == 1
    assert result.n_commit == 0
    assert result.stop_reason is PressureUnrollStopReason.ROUTING_FAILED
    assert first_rg.core_placements == first_original_cores
    assert second_rg.core_placements == second_original_cores
