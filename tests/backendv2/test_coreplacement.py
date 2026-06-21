import pytest
from paicorelib import (
    CoordXY,
    CoordZXYOffset,
    CSCAccelerateMode,
    DataWidth,
    FoldType,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    NeuronType,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
    OutputType,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressType,
)
from paicorelib.neuron_defs import ResetMode

from paibox.backendv2.coreplacement import (
    EmptyOfflineCorePlacementV2,
    EmptyOnlineCorePlacementV2,
    OfflineCorePlacementV2,
)
from paibox.backendv2.neuron import OfflineNeuronPlacement
from paibox.backendv2.weight import Weight


def _attrs_part1(neuron_type: NeuronType = NeuronType.FULL):
    return OfflineNeuFullAttrsV2Part1(
        weight_skew=0,
        weight_address_start=0,
        weight_address_end=0,
        fold_type=FoldType.UNFOLDED,
        neuron_type=neuron_type,
        output_type=OutputType.VALUE,
    )


def _attrs_part2(
    weight_compress: WeightCompressType,
    vjt_initial: int = 0,
):
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


def _placement(
    *,
    weight_compress: WeightCompressType,
    vjt_initial: int = 0,
    neuron_type: NeuronType = NeuronType.FULL,
) -> OfflineNeuronPlacement:
    return OfflineNeuronPlacement(
        neu=[],
        attrs_part1=_attrs_part1(neuron_type),
        attrs_part2=_attrs_part2(weight_compress, vjt_initial),
    )


def _weight(compress_type: WeightCompressType) -> Weight:
    return Weight(
        data=[1, 0, 2, 0, 0],
        compress_type=compress_type,
        weight_width=DataWidth.WIDTH_8BIT,
        input_width=DataWidth.WIDTH_8BIT,
    )


def _core_with_single_neuron(
    neuron: OfflineNeuronPlacement, weight: Weight
) -> OfflineCorePlacementV2:
    core = OfflineCorePlacementV2()
    core.neus = [neuron]
    core.weights = [weight]
    core.neu_weight_map = {0: 0}
    return core


def test_set_weight_address_backfills_vjt_initial_for_csc_sparse_full_neuron():
    neuron = _placement(weight_compress=WeightCompressType.SPARSE, vjt_initial=0)
    core = _core_with_single_neuron(neuron, _weight(WeightCompressType.SPARSE))

    core.set_weight_address()

    assert core.default_core_config.csc_accelerate == CSCAccelerateMode.ENABLE
    assert neuron.neu_attrs_part1.weight_address_start == neuron.n_sram_required
    assert (
        neuron.neu_attrs_part2.vjt_initial
        == neuron.neu_attrs_part1.weight_address_start
    )


def test_set_weight_address_disables_csc_accelerate_for_sparse_nonzero_init_v():
    neuron = _placement(weight_compress=WeightCompressType.SPARSE, vjt_initial=7)
    core = _core_with_single_neuron(neuron, _weight(WeightCompressType.SPARSE))

    core.set_weight_address()

    assert core.default_core_config.csc_accelerate == CSCAccelerateMode.DISABLE
    assert neuron.neu_attrs_part2.vjt_initial == 7


def test_set_weight_address_leaves_dense_vjt_initial_unchanged():
    neuron = _placement(weight_compress=WeightCompressType.DENSE, vjt_initial=0)
    core = _core_with_single_neuron(neuron, _weight(WeightCompressType.DENSE))

    core.set_weight_address()

    assert core.default_core_config.csc_accelerate == CSCAccelerateMode.ENABLE
    assert neuron.neu_attrs_part1.weight_address_start == neuron.n_sram_required
    assert neuron.neu_attrs_part2.vjt_initial == 0


def test_set_weight_address_does_not_touch_half_neuron_part2_semantics():
    neuron = _placement(
        weight_compress=WeightCompressType.SPARSE,
        vjt_initial=0,
        neuron_type=NeuronType.HALF,
    )
    core = _core_with_single_neuron(neuron, _weight(WeightCompressType.SPARSE))

    core.set_weight_address()

    assert core.default_core_config.csc_accelerate == CSCAccelerateMode.ENABLE
    assert neuron.neu_attrs_part2 is None


def test_empty_offline_core_exports_only_frame1():
    core = EmptyOfflineCorePlacementV2()
    core._coord = CoordXY(1, 2)
    core.set_auto_core_config(CoordZXYOffset(-1, 0, -1))

    frame1, frame2, frame3 = core.to_frame()

    assert frame1 is not None
    assert frame2 is None
    assert frame3 is None


def test_empty_online_core_exports_minimal_online_frame1():
    core = EmptyOnlineCorePlacementV2()
    core._coord = CoordXY(1, 2)
    core.set_auto_core_config(CoordZXYOffset(-1, 0, -1))

    assert (
        core.auto_core_config.test_core_xy,
        core.auto_core_config.test_core_x,
        core.auto_core_config.test_core_y,
    ) == (-1, 0, -1)

    frame1, frame2, frame3 = core.to_frame()
    assert frame1 is not None
    assert frame2 is None
    assert frame3 is None


@pytest.mark.parametrize(
    "accessor",
    [
        lambda core: core.n_sram_required,
        lambda core: core.weight_sram_required,
        lambda core: core.neuron_sram_required,
        lambda core: core.output_width,
    ],
)
def test_empty_online_core_unsupported_properties_raise(accessor):
    core = EmptyOnlineCorePlacementV2()
    core._coord = CoordXY(1, 2)

    with pytest.raises(NotImplementedError):
        accessor(core)
