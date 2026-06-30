from paicorelib import (
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

from paibox.backendv2.neuron import OfflineNeuronPlacement


def _attrs_part1(
    neuron_type: NeuronType = NeuronType.FULL,
    weight_start: int = 8,
    weight_end: int = 12,
    vjt: int = 5,
) -> OfflineNeuFullAttrsV2Part1:
    return OfflineNeuFullAttrsV2Part1(
        weight_skew=0,
        weight_address_start=weight_start,
        weight_address_end=weight_end,
        fold_type=FoldType.UNFOLDED,
        neuron_type=neuron_type,
        output_type=OutputType.VALUE,
        vjt=vjt,
    )


def _attrs_part2(vjt_initial: int = 8) -> OfflineNeuFullAttrsV2Part2:
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
        weight_compress=WeightCompressType.SPARSE,
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
        fold_vjt_0=1, fold_vjt_1=2, fold_vjt_2=3, fold_vjt_3=4
    )


def test_offline_neuron_copy_for_readdress_resets_derived_attrs():
    raw_neus = []
    placement = OfflineNeuronPlacement(
        raw_neus,
        _attrs_part1(weight_start=8, weight_end=12, vjt=6),
        _attrs_part2(vjt_initial=8),
        _fold_attrs_part1(),
        [_fold_attrs_part2()],
    )

    copied = placement.copy_for_readdress(clear_vjt_initial=True)

    assert copied is not placement
    assert copied.raw_neus is raw_neus
    assert copied.dest_info is None
    assert copied.neu_attrs_part1 is not placement.neu_attrs_part1
    assert copied.neu_attrs_part1.weight_address_start == 0
    assert copied.neu_attrs_part1.weight_address_end == 0
    assert copied.neu_attrs_part1.vjt == 0
    assert placement.neu_attrs_part1.weight_address_start == 8
    assert copied.neu_attrs_part2 is not placement.neu_attrs_part2
    assert copied.neu_attrs_part2.vjt_initial == 0
    assert copied.folded_neu_attrs_part1 is not placement.folded_neu_attrs_part1
    assert copied.folded_neu_attrs_part1 == placement.folded_neu_attrs_part1
    assert copied.folded_neu_attrs_part2s[0] is not placement.folded_neu_attrs_part2s[0]
    assert (
        copied.folded_neu_attrs_part2s[0].fold_vjt_0,
        copied.folded_neu_attrs_part2s[0].fold_vjt_1,
        copied.folded_neu_attrs_part2s[0].fold_vjt_2,
        copied.folded_neu_attrs_part2s[0].fold_vjt_3,
    ) == (0, 0, 0, 0)
    assert placement.folded_neu_attrs_part2s[0].fold_vjt_0 == 1


def test_offline_neuron_copy_for_readdress_keeps_real_vjt_initial():
    placement = OfflineNeuronPlacement([], _attrs_part1(), _attrs_part2(7))

    assert placement.copy_for_readdress().neu_attrs_part2.vjt_initial == 7
    assert (
        placement.copy_for_readdress(clear_vjt_initial=True).neu_attrs_part2.vjt_initial
        == 7
    )


def test_offline_neuron_copy_for_readdress_preserves_equal_real_vjt_by_default():
    placement = OfflineNeuronPlacement(
        [], _attrs_part1(weight_start=8), _attrs_part2(vjt_initial=8)
    )

    assert placement.copy_for_readdress().neu_attrs_part2.vjt_initial == 8


def test_offline_neuron_copy_for_readdress_handles_half_neuron():
    placement = OfflineNeuronPlacement(
        [], _attrs_part1(neuron_type=NeuronType.HALF), _attrs_part2()
    )

    copied = placement.copy_for_readdress()

    assert copied.neuron_type == NeuronType.HALF
    assert copied.neu_attrs_part2 is None
    assert copied.neu_attrs_part1.weight_address_start == 0


def test_offline_neuron_fold_attrs_list_is_not_aliased():
    first = OfflineNeuronPlacement([], _attrs_part1(), _attrs_part2())
    second = OfflineNeuronPlacement([], _attrs_part1(), _attrs_part2())
    first.folded_neu_attrs_part2s.append(_fold_attrs_part2())

    external_list = [_fold_attrs_part2()]
    third = OfflineNeuronPlacement(
        [], _attrs_part1(), _attrs_part2(), fold_attrs_part2s=external_list
    )
    external_list.append(_fold_attrs_part2())

    assert second.folded_neu_attrs_part2s == []
    assert len(third.folded_neu_attrs_part2s) == 1
