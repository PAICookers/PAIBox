from time import perf_counter

import numpy as np
import pytest
from paicorelib import (
    AERPacketZXYCopy,
    CoordXY,
    CoordZXYOffset,
    DataWidth,
    FoldType,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    NeuronType,
    OfflineFrameGenV2,
    OutputType,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressType,
)
from paicorelib.framelib.parser_v2 import decode_core_config, parse_frame_stream
from paicorelib.neuron_defs import ResetMode

from paibox.backendv2.compute_pressure import compute_weight_pressure
from paibox.visualizer.backends.v2.errors import FrameDecodeError
from paibox.visualizer.backends.v2.offline_v2 import (
    decode_neuron_destinations,
    decode_neurons,
    decode_offline_core,
)

from ..helpers import make_core_frame


def test_decode_core_config_semantic_fields() -> None:
    frames = make_core_frame(
        CoordXY(3, 2),
        global_send=(1 << 6) | (1 << 3),
        global_receive=1 << 2,
    )
    parsed = parse_frame_stream(frames)
    core = parsed.cores[(3, 2)]
    config = decode_core_config(core.frame_type1_payloads)
    decoded = decode_offline_core(config, core.packages)

    mode_fields = {
        field.name: field for field in decoded.core_config.groups["Mode/Data"]
    }
    assert mode_fields["snn_ann"].label == "SNN"
    assert mode_fields["input_sign"].label == "SIGNED"
    assert mode_fields["input_width"].label == "WIDTH_8BIT"

    signal_fields = {
        field.name: field for field in decoded.core_config.groups["Signal bits"]
    }
    assert "local" in signal_fields["global_send"].label
    assert "+x" in signal_fields["global_send"].label


def test_decode_lut_int_entries() -> None:
    offset = CoordZXYOffset(0, 2, 2)
    lut = OfflineFrameGenV2.gen_config_frame2(
        offset,
        np.arange(-128, 128, dtype=np.int32),
        np.arange(-128, 128, dtype=np.int8),
    )
    parsed = parse_frame_stream(lut)
    core = parsed.cores[(2, 2)]

    decoded = decode_offline_core({"output_sign": 1}, core.packages)

    assert decoded.lut.present
    assert decoded.lut.summary["entry_count"] == 256
    assert decoded.lut.entries[0].potential == -128
    assert decoded.lut.entries[0].activation == -128
    assert decoded.lut.entries[-1].potential == 127
    assert decoded.lut.entries[-1].activation == 127


def test_decode_full_neuron_and_weight_summary() -> None:
    offset = CoordZXYOffset(0, 2, 2)
    dest = {
        "tick_relative": 1,
        "addr_axon": 12,
        "addr_core_xy": 0,
        "addr_core_x": 1,
        "addr_core_y": -1,
        "addr_copy_xy": 0,
        "addr_copy_x": 0,
        "addr_copy_y": 0,
    }
    attrs1 = {
        "weight_skew": 0,
        "weight_address_start": 2,
        "weight_address_end": 3,
        "output_type": OutputType.VALUE,
        "fold_type": FoldType.UNFOLDED,
        "neuron_type": NeuronType.FULL,
        "vjt": 0,
    }
    attrs2 = {
        "reset_mode": ResetMode.MODE_NORMAL,
        "reset_v": -2,
        "threshold_neg_mode": ThresholdNegMode.FIRE,
        "threshold_pos_mode": ThresholdPosMode.FIRE,
        "threshold_neg": -8,
        "threshold_pos": 9,
        "lateral_inhibition": LateralInhibitionMode.DISABLE,
        "leak_multi_sequence": LeakMultiComparisonOrder.BEFORE_COMPARE,
        "leak_multi_input": LeakMultiInputMode.DISABLE,
        "leak_multi_mode": LeakMultiMode.DISABLE,
        "leak_add_mode": LeakAddMode.FORWARD,
        "leak_tau": -1,
        "leak_v": -3,
        "weight_compress": WeightCompressType.DENSE,
        "vjt_initial": -4,
    }
    neuron = OfflineFrameGenV2.gen_config_frame3_pkg_full(dest, attrs1, attrs2)
    weight = OfflineFrameGenV2.gen_config_frame3_weight_pkg(
        np.arange(32, dtype=np.uint8),
        weight_width=DataWidth.WIDTH_8BIT,
        input_width=DataWidth.WIDTH_8BIT,
        csc_compress=False,
    )
    frame3 = OfflineFrameGenV2.gen_config_frame3_pkg_header(
        offset,
        start_addr=0,
        n_package=len(neuron) + len(weight),
        pkt_ncopy=AERPacketZXYCopy(),
    )
    parsed = parse_frame_stream(np.concatenate([frame3, neuron, weight]))
    core = parsed.cores[(2, 2)]

    decoded = decode_offline_core(
        {
            "neuron_number": len(neuron) // 2,
            "weight_sign": 0,
            "weight_width": DataWidth.WIDTH_8BIT,
            "input_width": DataWidth.WIDTH_8BIT,
        },
        core.packages,
    )

    assert decoded.neurons.summary.full_count == 1
    expected_sops = compute_weight_pressure(
        fold_count=1,
        input_width=DataWidth.WIDTH_8BIT,
        weight_width=DataWidth.WIDTH_8BIT,
        weight_sram_records=2,
        is_csc=False,
    )
    assert expected_sops == 2048
    assert decoded.neurons.summary.synops_pressure == expected_sops
    assert decoded.neurons.summary.sops_with_padding == expected_sops
    assert decoded.neurons.summary.sops_without_padding == 2048
    assert decoded.neurons.summary.weight_sram_pressure == 2
    full_attrs = decoded.neurons.records[0].fields["full attrs"]
    assert next(field for field in full_attrs if field.name == "reset_v").decoded == -2
    assert (
        next(field for field in full_attrs if field.name == "threshold_neg").decoded
        == -8
    )
    assert decoded.weights.summary.total == 1
    assert decoded.weights.summary.data_type == "uint8"
    assert decoded.weights.summary.sram_records == 2
    assert decoded.weights.summary.bytes == 32
    weight_record = decoded.weights.records[0]
    assert weight_record.storage_value_count == 32
    assert weight_record.storage_values == list(range(32))
    assert weight_record.storage_values_raw == list(range(32))


def test_decode_folded_neuron_attaches_fold_attrs() -> None:
    offset = CoordZXYOffset(0, 2, 2)
    dest = {
        "tick_relative": 1,
        "addr_axon": 12,
        "addr_core_xy": 0,
        "addr_core_x": 1,
        "addr_core_y": -1,
        "addr_copy_xy": 0,
        "addr_copy_x": 0,
        "addr_copy_y": 0,
    }
    attrs1 = {
        "weight_skew": 0,
        "weight_address_start": 4,
        "weight_address_end": 5,
        "output_type": OutputType.VALUE,
        "fold_type": FoldType.FOLDED,
        "neuron_type": NeuronType.HALF,
        "vjt": 0,
    }
    folded_attrs1 = {
        "fold_range_xy": 1,
        "fold_range_x": 2,
        "fold_range_y": 2,
        "fold_skew_xy": 0,
        "fold_skew_x": 1,
        "fold_skew_y": 2,
        "fold_axon_xy": 0,
        "fold_axon_x": 1,
        "fold_axon_y": 2,
        "fold_number": 4,
    }
    folded_attrs2 = [
        {
            "fold_vjt_0": 0,
            "fold_vjt_1": 1,
            "fold_vjt_2": 2,
            "fold_vjt_3": 3,
        }
    ]
    half, _, folded = OfflineFrameGenV2.gen_config_frame3_pkg_neu(
        dest,
        attrs1,
        None,
        folded_attrs1,
        folded_attrs2,
    )
    neuron = np.concatenate([half, folded])
    weight = OfflineFrameGenV2.gen_config_frame3_weight_pkg(
        np.arange(32, dtype=np.uint8),
        weight_width=DataWidth.WIDTH_8BIT,
        input_width=DataWidth.WIDTH_8BIT,
        csc_compress=False,
    )
    frame3 = OfflineFrameGenV2.gen_config_frame3_pkg_header(
        offset,
        start_addr=0,
        n_package=len(neuron) + len(weight),
        pkt_ncopy=AERPacketZXYCopy(),
    )
    parsed = parse_frame_stream(np.concatenate([frame3, neuron, weight]))
    core = parsed.cores[(2, 2)]

    decoded = decode_offline_core(
        {
            "neuron_number": len(neuron) // 2,
            "weight_sign": 0,
            "weight_width": DataWidth.WIDTH_8BIT,
            "input_width": DataWidth.WIDTH_8BIT,
        },
        core.packages,
    )

    assert decoded.neurons.summary.total == 1
    assert decoded.neurons.summary.half_count == 1
    assert decoded.neurons.summary.folded_count == 1
    assert decoded.neurons.summary.synops_pressure == 8192
    assert decoded.neurons.summary.sops_with_padding == 8192
    assert decoded.neurons.summary.sops_without_padding == 8192
    assert decoded.neurons.summary.weight_sram_pressure == 8
    record = decoded.neurons.records[0]
    assert record.kind == "half+folded"
    fold_attrs = record.fields["fold attrs"]
    assert (
        next(field for field in fold_attrs if field.name == "fold_number").decoded == 4
    )
    assert decoded.weights.summary.total == 1
    assert decoded.weights.summary.sram_records == 2


def test_folded_neuron_extra_vjt_records_match_backend_layout() -> None:
    offset = CoordZXYOffset(0, 2, 2)
    folded_dest = {
        "tick_relative": 1,
        "addr_axon": 12,
        "addr_core_xy": 0,
        "addr_core_x": 1,
        "addr_core_y": 0,
        "addr_copy_xy": 0,
        "addr_copy_x": 0,
        "addr_copy_y": 0,
    }
    folded_attrs1 = {
        "weight_skew": 0,
        "weight_address_start": 3,
        "weight_address_end": 3,
        "output_type": OutputType.VALUE,
        "fold_type": FoldType.FOLDED,
        "neuron_type": NeuronType.HALF,
        "vjt": 0,
    }
    fold_attrs1 = {
        "fold_range_xy": 1,
        "fold_range_x": 3,
        "fold_range_y": 3,
        "fold_skew_xy": 0,
        "fold_skew_x": 1,
        "fold_skew_y": 2,
        "fold_axon_xy": 0,
        "fold_axon_x": 1,
        "fold_axon_y": 2,
        "fold_number": 9,
    }
    fold_attrs2 = [
        {
            "fold_vjt_0": 0,
            "fold_vjt_1": 1,
            "fold_vjt_2": 2,
            "fold_vjt_3": 3,
        },
        {
            "fold_vjt_0": 4,
            "fold_vjt_1": 5,
            "fold_vjt_2": 6,
            "fold_vjt_3": 7,
        },
    ]
    folded_half, _, folded_extra = OfflineFrameGenV2.gen_config_frame3_pkg_neu(
        folded_dest,
        folded_attrs1,
        None,
        fold_attrs1,
        fold_attrs2,
    )
    next_dest = {
        "tick_relative": 0,
        "addr_axon": 1,
        "addr_core_xy": 0,
        "addr_core_x": -1,
        "addr_core_y": 0,
        "addr_copy_xy": 0,
        "addr_copy_x": 0,
        "addr_copy_y": 0,
    }
    next_attrs1 = {
        "weight_skew": 0,
        "weight_address_start": 4,
        "weight_address_end": 4,
        "output_type": OutputType.VALUE,
        "fold_type": FoldType.UNFOLDED,
        "neuron_type": NeuronType.HALF,
        "vjt": 0,
    }
    next_neuron = OfflineFrameGenV2.gen_config_frame3_pkg_half(next_dest, next_attrs1)
    neurons = np.concatenate([folded_half, folded_extra, next_neuron])
    weight = OfflineFrameGenV2.gen_config_frame3_weight_pkg(
        np.arange(32, dtype=np.uint8),
        weight_width=DataWidth.WIDTH_8BIT,
        input_width=DataWidth.WIDTH_8BIT,
        csc_compress=False,
    )
    frame3 = OfflineFrameGenV2.gen_config_frame3_pkg_header(
        offset,
        start_addr=0,
        n_package=len(neurons) + len(weight),
        pkt_ncopy=AERPacketZXYCopy(),
    )
    parsed = parse_frame_stream(np.concatenate([frame3, neurons, weight]))
    core = parsed.cores[(2, 2)]

    decoded = decode_offline_core(
        {
            "neuron_number": len(neurons) // 2,
            "weight_sign": 0,
            "weight_width": DataWidth.WIDTH_8BIT,
            "input_width": DataWidth.WIDTH_8BIT,
        },
        core.packages,
        core_coord=(2, 2),
        grid_width=9,
        grid_height=9,
    )

    assert decoded.neurons.summary.total == 2
    assert decoded.neurons.records[0].kind == "half+folded"
    assert decoded.neurons.records[0].frame_indices == [
        core.packages[0].frame_start + 1 + index for index in range(8)
    ]
    assert decoded.neurons.records[1].kind == "half"
    assert decoded.neurons.records[1].frame_indices == [
        core.packages[0].frame_start + 9,
        core.packages[0].frame_start + 10,
    ]
    assert [
        (destination.target_x, destination.target_y)
        for destination in decoded.neurons.records[1].destinations
    ] == [(1, 2)]


def test_csc_accelerate_relabels_vjt_initial_as_weight_address_start() -> None:
    offset = CoordZXYOffset(0, 2, 2)
    dest = {
        "tick_relative": 1,
        "addr_axon": 12,
        "addr_core_xy": 0,
        "addr_core_x": 1,
        "addr_core_y": -1,
        "addr_copy_xy": 0,
        "addr_copy_x": 0,
        "addr_copy_y": 0,
    }
    attrs1 = {
        "weight_skew": 0,
        "weight_address_start": 2,
        "weight_address_end": 2,
        "output_type": OutputType.VALUE,
        "fold_type": FoldType.UNFOLDED,
        "neuron_type": NeuronType.FULL,
        "vjt": 0,
    }
    attrs2 = {
        "reset_mode": ResetMode.MODE_NORMAL,
        "reset_v": 0,
        "threshold_neg_mode": ThresholdNegMode.FIRE,
        "threshold_pos_mode": ThresholdPosMode.FIRE,
        "threshold_neg": -8,
        "threshold_pos": 9,
        "lateral_inhibition": LateralInhibitionMode.DISABLE,
        "leak_multi_sequence": LeakMultiComparisonOrder.BEFORE_COMPARE,
        "leak_multi_input": LeakMultiInputMode.DISABLE,
        "leak_multi_mode": LeakMultiMode.DISABLE,
        "leak_add_mode": LeakAddMode.FORWARD,
        "leak_tau": 0,
        "leak_v": 0,
        "weight_compress": WeightCompressType.SPARSE,
        "vjt_initial": 2,
    }
    neuron = OfflineFrameGenV2.gen_config_frame3_pkg_full(dest, attrs1, attrs2)
    frame3 = OfflineFrameGenV2.gen_config_frame3_pkg_header(
        offset,
        start_addr=0,
        n_package=len(neuron),
        pkt_ncopy=AERPacketZXYCopy(),
    )
    parsed = parse_frame_stream(np.concatenate([frame3, neuron]))
    core = parsed.cores[(2, 2)]

    decoded = decode_offline_core(
        {
            "neuron_number": len(neuron) // 2,
            "csc_accelerate": 1,
            "weight_sign": 0,
            "weight_width": DataWidth.WIDTH_8BIT,
            "input_width": DataWidth.WIDTH_8BIT,
        },
        core.packages,
    )

    full_attrs = decoded.neurons.records[0].fields["full attrs"]
    field = next(field for field in full_attrs if field.name == "vjt_initial")
    assert field.decoded == 2
    assert field.numeric_kind == "uint"
    assert "weight_address_start" in field.description


def test_decode_csc_weight_storage_entries_and_padding() -> None:
    offset = CoordZXYOffset(0, 2, 2)
    dest = {
        "tick_relative": 1,
        "addr_axon": 12,
        "addr_core_xy": 0,
        "addr_core_x": 1,
        "addr_core_y": -1,
        "addr_copy_xy": 0,
        "addr_copy_x": 0,
        "addr_copy_y": 0,
    }
    attrs1 = {
        "weight_skew": 0,
        "weight_address_start": 2,
        "weight_address_end": 2,
        "output_type": OutputType.VALUE,
        "fold_type": FoldType.UNFOLDED,
        "neuron_type": NeuronType.FULL,
        "vjt": 0,
    }
    attrs2 = {
        "reset_mode": ResetMode.MODE_NORMAL,
        "reset_v": 0,
        "threshold_neg_mode": ThresholdNegMode.FIRE,
        "threshold_pos_mode": ThresholdPosMode.FIRE,
        "threshold_neg": -8,
        "threshold_pos": 9,
        "lateral_inhibition": LateralInhibitionMode.DISABLE,
        "leak_multi_sequence": LeakMultiComparisonOrder.BEFORE_COMPARE,
        "leak_multi_input": LeakMultiInputMode.DISABLE,
        "leak_multi_mode": LeakMultiMode.DISABLE,
        "leak_add_mode": LeakAddMode.FORWARD,
        "leak_tau": 0,
        "leak_v": 0,
        "weight_compress": WeightCompressType.SPARSE,
        "vjt_initial": 0,
    }
    neuron = OfflineFrameGenV2.gen_config_frame3_pkg_full(dest, attrs1, attrs2)
    weight = OfflineFrameGenV2.gen_config_frame3_weight_pkg(
        np.array([1, 0, 2, 0, 3, 0, 4], dtype=np.uint8),
        weight_width=DataWidth.WIDTH_8BIT,
        input_width=DataWidth.WIDTH_8BIT,
        csc_compress=True,
        weight_skews=(0,),
    )
    frame3 = OfflineFrameGenV2.gen_config_frame3_pkg_header(
        offset,
        start_addr=0,
        n_package=len(neuron) + len(weight),
        pkt_ncopy=AERPacketZXYCopy(),
    )
    parsed = parse_frame_stream(np.concatenate([frame3, neuron, weight]))
    core = parsed.cores[(2, 2)]

    decoded = decode_offline_core(
        {
            "neuron_number": len(neuron) // 2,
            "weight_sign": 0,
            "weight_width": DataWidth.WIDTH_8BIT,
            "input_width": DataWidth.WIDTH_8BIT,
        },
        core.packages,
    )

    assert decoded.weights.summary.csc_count == 1
    assert decoded.weights.summary.nonzero_count == 4
    assert decoded.weights.summary.padding_count == 1
    expected_padded_sops = compute_weight_pressure(
        fold_count=1,
        input_width=DataWidth.WIDTH_8BIT,
        weight_width=DataWidth.WIDTH_8BIT,
        weight_sram_records=1,
        is_csc=True,
    )
    expected_unpadded_sops = compute_weight_pressure(
        fold_count=1,
        input_width=DataWidth.WIDTH_8BIT,
        weight_width=DataWidth.WIDTH_8BIT,
        weight_sram_records=1,
        is_csc=True,
        weight_slots=4,
    )
    assert expected_padded_sops == 320
    assert expected_unpadded_sops == 256
    assert decoded.neurons.summary.sops_with_padding == expected_padded_sops
    assert decoded.neurons.summary.sops_without_padding == expected_unpadded_sops
    assert decoded.neurons.summary.weight_sram_pressure == 1
    weight_record = decoded.weights.records[0]
    assert weight_record.kind == "sparse"
    assert weight_record.storage_value_count == 5
    assert [entry.value for entry in weight_record.storage_entries] == [1, 2, 3, 4, 0]
    assert [entry.logical_index for entry in weight_record.storage_entries] == [
        0,
        2,
        4,
        6,
        6,
    ]
    assert [entry.is_padding for entry in weight_record.storage_entries] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_out_of_grid_neuron_destination_route_fails_with_provenance() -> None:
    offset = CoordZXYOffset(0, 2, 2)
    dest = {
        "tick_relative": 1,
        "addr_axon": 12,
        "addr_core_xy": 0,
        "addr_core_x": 7,
        "addr_core_y": 0,
        "addr_copy_xy": 0,
        "addr_copy_x": 0,
        "addr_copy_y": 0,
    }
    attrs1 = {
        "weight_skew": 0,
        "weight_address_start": 2,
        "weight_address_end": 2,
        "output_type": OutputType.VALUE,
        "fold_type": FoldType.UNFOLDED,
        "neuron_type": NeuronType.HALF,
        "vjt": 0,
    }
    neuron = OfflineFrameGenV2.gen_config_frame3_pkg_half(dest, attrs1)
    frame3 = OfflineFrameGenV2.gen_config_frame3_pkg_header(
        offset,
        start_addr=0,
        n_package=len(neuron),
        pkt_ncopy=AERPacketZXYCopy(),
    )
    parsed = parse_frame_stream(np.concatenate([frame3, neuron]))
    core = parsed.cores[(2, 2)]

    with pytest.raises(
        FrameDecodeError, match="neuron destination route leaves chip grid"
    ) as exc_info:
        decode_offline_core(
            {"neuron_number": len(neuron) // 2},
            core.packages,
            core_coord=(2, 2),
            grid_width=9,
            grid_height=9,
        )

    assert exc_info.value.frame_index == core.packages[0].frame_start + 2
    assert exc_info.value.raw_frame == core.packages[0].payloads[1]
    assert exc_info.value.context["source_core"] == "(2,2)"
    assert exc_info.value.context["target_core"] == "(9,2)"
    assert exc_info.value.context["illegal_core"] == "(9,2)"
    assert exc_info.value.context["route_kind"] == "offset"
    assert exc_info.value.context["route_axis"] == "x"
    assert exc_info.value.context["addr_core_x"] == 7


def test_intermediate_neuron_route_foothold_fails_even_when_final_target_is_valid() -> (
    None
):
    offset = CoordZXYOffset(0, 0, 8)
    dest = {
        "tick_relative": 1,
        "addr_axon": 12,
        "addr_core_xy": -1,
        "addr_core_x": 1,
        "addr_core_y": 0,
        "addr_copy_xy": 0,
        "addr_copy_x": 0,
        "addr_copy_y": 0,
    }
    attrs1 = {
        "weight_skew": 0,
        "weight_address_start": 2,
        "weight_address_end": 2,
        "output_type": OutputType.VALUE,
        "fold_type": FoldType.UNFOLDED,
        "neuron_type": NeuronType.HALF,
        "vjt": 0,
    }
    neuron = OfflineFrameGenV2.gen_config_frame3_pkg_half(dest, attrs1)
    frame3 = OfflineFrameGenV2.gen_config_frame3_pkg_header(
        offset,
        start_addr=0,
        n_package=len(neuron),
        pkt_ncopy=AERPacketZXYCopy(),
    )
    parsed = parse_frame_stream(np.concatenate([frame3, neuron]))
    core = parsed.cores[(0, 8)]

    with pytest.raises(
        FrameDecodeError, match="neuron destination route leaves chip grid"
    ) as exc_info:
        decode_offline_core(
            {"neuron_number": len(neuron) // 2},
            core.packages,
            core_coord=(0, 8),
            grid_width=9,
            grid_height=9,
        )

    assert exc_info.value.context["source_core"] == "(0,8)"
    assert exc_info.value.context["target_core"] == "(0,7)"
    assert exc_info.value.context["illegal_core"] == "(-1,7)"
    assert exc_info.value.context["route_kind"] == "offset"
    assert exc_info.value.context["route_axis"] == "z"
    assert exc_info.value.context["route_step"] == 1


def test_neuron_route_copy_foothold_fails_when_multicast_branch_leaves_grid() -> None:
    offset = CoordZXYOffset(0, 8, 8)
    dest = {
        "tick_relative": 1,
        "addr_axon": 12,
        "addr_core_xy": 0,
        "addr_core_x": 0,
        "addr_core_y": 0,
        "addr_copy_xy": 1,
        "addr_copy_x": 0,
        "addr_copy_y": 0,
    }
    attrs1 = {
        "weight_skew": 0,
        "weight_address_start": 2,
        "weight_address_end": 2,
        "output_type": OutputType.VALUE,
        "fold_type": FoldType.UNFOLDED,
        "neuron_type": NeuronType.HALF,
        "vjt": 0,
    }
    neuron = OfflineFrameGenV2.gen_config_frame3_pkg_half(dest, attrs1)
    frame3 = OfflineFrameGenV2.gen_config_frame3_pkg_header(
        offset,
        start_addr=0,
        n_package=len(neuron),
        pkt_ncopy=AERPacketZXYCopy(),
    )
    parsed = parse_frame_stream(np.concatenate([frame3, neuron]))
    core = parsed.cores[(8, 8)]

    with pytest.raises(
        FrameDecodeError, match="neuron destination route leaves chip grid"
    ) as exc_info:
        decode_offline_core(
            {"neuron_number": len(neuron) // 2},
            core.packages,
            core_coord=(8, 8),
            grid_width=9,
            grid_height=9,
        )

    assert exc_info.value.context["source_core"] == "(8,8)"
    assert exc_info.value.context["target_core"] == "(9,9)"
    assert exc_info.value.context["base_target_core"] == "(8,8)"
    assert exc_info.value.context["illegal_core"] == "(9,9)"
    assert exc_info.value.context["route_kind"] == "copy"
    assert exc_info.value.context["route_axis"] == "z"
    assert exc_info.value.context["addr_copy_xy"] == 1


def test_neuron_destinations_expand_multicast_copy_targets() -> None:
    offset = CoordZXYOffset(0, 3, 2)
    dest = {
        "tick_relative": 0,
        "addr_axon": 0,
        "addr_core_xy": 0,
        "addr_core_x": 1,
        "addr_core_y": 0,
        "addr_copy_xy": 1,
        "addr_copy_x": 0,
        "addr_copy_y": 0,
    }
    attrs1 = {
        "weight_skew": 0,
        "weight_address_start": 2,
        "weight_address_end": 2,
        "output_type": OutputType.VALUE,
        "fold_type": FoldType.UNFOLDED,
        "neuron_type": NeuronType.HALF,
        "vjt": 0,
    }
    neuron = OfflineFrameGenV2.gen_config_frame3_pkg_half(dest, attrs1)
    frame3 = OfflineFrameGenV2.gen_config_frame3_pkg_header(
        offset,
        start_addr=0,
        n_package=len(neuron),
        pkt_ncopy=AERPacketZXYCopy(),
    )
    parsed = parse_frame_stream(np.concatenate([frame3, neuron]))
    core = parsed.cores[(3, 2)]

    neurons = decode_neurons({"neuron_number": len(neuron) // 2}, core.packages)
    destinations = decode_neuron_destinations(neurons.records[0], (3, 2))

    assert [
        (destination.target_x, destination.target_y) for destination in destinations
    ] == [(4, 2), (5, 3)]


def test_neuron_destination_decode_stays_lightweight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_walk(*args: object, **kwargs: object) -> None:
        raise AssertionError("target-only decode must not walk the complete packet")

    monkeypatch.setattr("paibox.backendv2.route_scope.aer_packet_walk", fail_walk)
    offset = CoordZXYOffset(0, 3, 2)
    dest = {
        "tick_relative": 0,
        "addr_axon": 0,
        "addr_core_xy": 0,
        "addr_core_x": 1,
        "addr_core_y": 0,
        "addr_copy_xy": 1,
        "addr_copy_x": 1,
        "addr_copy_y": 1,
    }
    attrs1 = {
        "weight_skew": 0,
        "weight_address_start": 2,
        "weight_address_end": 2,
        "output_type": OutputType.VALUE,
        "fold_type": FoldType.UNFOLDED,
        "neuron_type": NeuronType.HALF,
        "vjt": 0,
    }
    neuron = OfflineFrameGenV2.gen_config_frame3_pkg_half(dest, attrs1)
    frame3 = OfflineFrameGenV2.gen_config_frame3_pkg_header(
        offset,
        start_addr=0,
        n_package=len(neuron),
        pkt_ncopy=AERPacketZXYCopy(),
    )
    parsed = parse_frame_stream(np.concatenate([frame3, neuron]))
    core = parsed.cores[(3, 2)]

    neurons = decode_neurons({"neuron_number": len(neuron) // 2}, core.packages)
    started = perf_counter()
    for _ in range(500):
        destinations = decode_neuron_destinations(neurons.records[0], (3, 2))
    elapsed = perf_counter() - started

    assert [
        (destination.target_x, destination.target_y) for destination in destinations
    ] == [
        (4, 2),
        (5, 3),
        (5, 2),
        (4, 3),
        (6, 3),
        (5, 4),
        (6, 4),
    ]
    assert elapsed < 1.0
