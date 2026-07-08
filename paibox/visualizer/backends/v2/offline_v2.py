from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import Literal

from paicorelib import (
    LCN_EX,
    AddPotentialMode,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
    global_signal_direction_names,
)
from paicorelib.framelib.frame_defs import OfflineConfigFrame3FormatV2
from paicorelib.framelib.parser_v2 import (
    FramePackageInfo,
)
from paicorelib.framelib.parser_v2 import (
    decode_lut_entries as parse_lut_entries,
)
from paicorelib.neuron_defs import ResetMode
from paicorelib.neuron_defs_v2 import (
    FoldType,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    NeuronType,
    OutputType,
    ThresholdNegMode,
    ThresholdPosMode,
    WeightCompressType,
)

from paibox.backendv2.compute_pressure import (
    SRAM_RECORD_BITS,
    compute_weight_pressure,
    csc_weight_slots_per_sram,
)

from ...model import (
    CoreConfigView,
    DecodedField,
    LutEntryView,
    LutView,
    NeuronDestinationView,
    NeuronRecordView,
    NeuronSummaryView,
    NeuronView,
    RawFrameRecord,
    WeightRecordView,
    WeightStorageEntryView,
    WeightSummaryView,
    WeightView,
)
from .errors import FrameDecodeError

SRAM_WORDS = 2
SRAM_RECORD_BYTES = 16
WEIGHT_STORAGE_PREVIEW_LIMIT = 256

_RouteKind = Literal["offset", "copy"]
_RouteAxis = Literal["z", "x", "y"]


@dataclass(frozen=True)
class _RouteFields:
    offset_z: int
    offset_x: int
    offset_y: int
    copy_z: int
    copy_x: int
    copy_y: int


@dataclass(frozen=True)
class _RouteMove:
    kind: _RouteKind
    axis: _RouteAxis
    step: int
    x: int
    y: int
    target_x: int
    target_y: int


@dataclass(frozen=True)
class _RouteState:
    x: int
    y: int
    copy_z: int
    copy_x: int
    copy_y: int


@dataclass(frozen=True)
class OfflineDecodeResult:
    core_config: CoreConfigView
    lut: LutView
    neurons: NeuronView
    weights: WeightView
    raw_frames: list[RawFrameRecord]


@dataclass(frozen=True)
class _DenseStorageDecode:
    values: list[int]
    values_raw: list[int]
    nonzero_count: int
    storage_value_count: int
    storage_preview_limit: int


@dataclass(frozen=True)
class _CscStorageDecode:
    entries: list[WeightStorageEntryView]
    nonzero_count: int
    padding_count: int
    storage_value_count: int
    storage_preview_limit: int


def decode_offline_core(
    core_config: dict[str, int],
    packages: list[FramePackageInfo],
    *,
    core_coord: tuple[int, int] | None = None,
    grid_width: int = 9,
    grid_height: int = 9,
) -> OfflineDecodeResult:
    """Decode offline-core integer config structures from parsed frame packages.

    This module is kept independent of FastAPI and React so the frame decoder can
    later move toward `paicorelib` with minimal coupling. It currently covers the
    offline integer path only; online-core and float interpretation stay outside
    this first visualizer decoder.
    """
    decoded_core = decode_core_config_view(core_config, packages)
    lut = decode_lut_view(core_config, packages)
    neurons = decode_neurons(core_config, packages)
    if core_coord is not None:
        neurons = attach_neuron_destinations(neurons, core_coord)
        validate_neuron_destinations(neurons, core_coord, grid_width, grid_height)
    weights = decode_weights(core_config, packages, neurons)
    neurons = apply_sops_summary(core_config, neurons, weights)
    raw_frames = build_raw_frame_records(packages)
    return OfflineDecodeResult(
        core_config=decoded_core,
        lut=lut,
        neurons=neurons,
        weights=weights,
        raw_frames=raw_frames,
    )


def decode_core_config_view(
    core_config: dict[str, int],
    packages: list[FramePackageInfo],
) -> CoreConfigView:
    source_index = _first_package_start(packages, 1)
    groups: dict[str, list[DecodedField]] = {
        "Mode/Data": [
            _enum_field("snn_ann", core_config, SNNMode, "SNN/ANN mode", source_index),
            _enum_field(
                "max_pooling", core_config, PoolingMode, "Pooling mode", source_index
            ),
            _enum_field(
                "add_potential",
                core_config,
                AddPotentialMode,
                "Accumulation mode",
                source_index,
            ),
            _enum_field(
                "zero_output",
                core_config,
                ZeroOutputMode,
                "Zero output mode",
                source_index,
            ),
            _enum_field(
                "input_sign", core_config, DataSign, "Input sign", source_index
            ),
            _enum_field(
                "input_width", core_config, DataWidth, "Input width", source_index
            ),
            _enum_field(
                "output_sign", core_config, DataSign, "Output sign", source_index
            ),
            _enum_field(
                "output_width", core_config, DataWidth, "Output width", source_index
            ),
            _enum_field(
                "weight_sign", core_config, DataSign, "Weight sign", source_index
            ),
            _enum_field(
                "weight_width", core_config, DataWidth, "Weight width", source_index
            ),
        ],
        "LCN/Axon": [
            _enum_field("lcn", core_config, LCN_EX, "Fan-in extension", source_index),
            _enum_field(
                "target_lcn",
                core_config,
                LCN_EX,
                "Target fan-in extension",
                source_index,
            ),
            _int_field(
                "axon_skew",
                core_config,
                "Axon address offset",
                source_index,
                bits=16,
            ),
            _uint_field(
                "neuron_number",
                core_config,
                "Number of valid neuron SRAM words",
                source_index,
            ),
        ],
        "Timing": [
            _uint_field("thread_number", core_config, "Thread number", source_index),
            _uint_field(
                "busy_cycle", core_config, "Busy-cycle threshold", source_index
            ),
            _uint_field(
                "delay_cycle", core_config, "Control delay cycles", source_index
            ),
            _uint_field(
                "width_cycle", core_config, "Signal width cycles", source_index
            ),
            _uint_field("tick_start", core_config, "Start sync tick", source_index),
            _uint_field(
                "tick_duration", core_config, "Active sync duration", source_index
            ),
            _uint_field(
                "tick_initial", core_config, "State reinitialization tick", source_index
            ),
        ],
        "Test target": [
            _sign_magnitude_field(
                "test_core_xy",
                core_config,
                "Relative test/control XY offset",
                source_index,
            ),
            _sign_magnitude_field(
                "test_core_x",
                core_config,
                "Relative test/control X offset",
                source_index,
            ),
            _sign_magnitude_field(
                "test_core_y",
                core_config,
                "Relative test/control Y offset",
                source_index,
            ),
        ],
        "Signal bits": [
            _bitset_field(
                "global_send",
                core_config,
                "Global signal send directions",
                source_index,
                include_local=True,
            ),
            _enum_field(
                "csc_accelerate",
                core_config,
                CSCAccelerateMode,
                "CSC acceleration mode",
                source_index,
            ),
            _bitset_field(
                "global_receive",
                core_config,
                "Global signal receive directions",
                source_index,
                include_local=False,
            ),
        ],
    }
    return CoreConfigView(groups=groups)


def decode_lut_view(
    core_config: dict[str, int], packages: list[FramePackageInfo]
) -> LutView:
    lut_packages = [package for package in packages if package.frame_type == 2]
    if not lut_packages:
        return LutView()

    output_signed = core_config.get("output_sign", 0) == DataSign.SIGNED
    entries: list[LutEntryView] = []
    for package in lut_packages:
        parsed_entries = parse_lut_entries(package.payloads)
        for parsed in parsed_entries:
            activation_raw = parsed.activation
            entries.append(
                LutEntryView(
                    index=package.start_addr + parsed.index,
                    potential_raw=parsed.potential & 0xFFFFFFFF,
                    potential=parsed.potential,
                    activation_raw=activation_raw,
                    activation=(
                        twos_complement_to_int(activation_raw, 8)
                        if output_signed
                        else activation_raw
                    ),
                    raw_hex=f"0x{parsed.raw:016x}",
                    source_frame_index=package.frame_start + 1 + parsed.index,
                )
            )

    if entries:
        potentials = [entry.potential for entry in entries]
        activations = [entry.activation for entry in entries]
        summary: dict[str, int | str] = {
            "entry_count": len(entries),
            "potential_min": min(potentials),
            "potential_max": max(potentials),
            "activation_min": min(activations),
            "activation_max": max(activations),
            "activation_sign": "signed" if output_signed else "unsigned",
        }
    else:
        summary = {"entry_count": 0}
    return LutView(present=True, entries=entries, summary=summary)


def decode_neurons(
    core_config: dict[str, int], packages: list[FramePackageInfo]
) -> NeuronView:
    neuron_sram_records = core_config.get("neuron_number", 0)
    if neuron_sram_records <= 0:
        return NeuronView()

    frame3_payload = _frame3_payload(packages)
    neuron_words = neuron_sram_records * SRAM_WORDS
    neuron_payload = frame3_payload[:neuron_words]
    records: list[NeuronRecordView] = []
    cursor = 0
    csc_accelerate_enabled = (
        core_config.get("csc_accelerate", CSCAccelerateMode.DISABLE)
        == CSCAccelerateMode.ENABLE
    )
    while cursor + 1 < len(neuron_payload):
        w1 = int(neuron_payload[cursor][1], 16)
        kind_raw = bit_field(
            w1, _F3.Full.Word1.NEURON_TYPE_OFFSET, _F3.Full.Word1.NEURON_TYPE_MASK
        )
        fold_raw = bit_field(
            w1, _F3.Full.Word1.FOLD_TYPE_OFFSET, _F3.Full.Word1.FOLD_TYPE_MASK
        )
        if kind_raw == NeuronType.FULL and cursor + 3 < len(neuron_payload):
            record = _decode_half_or_full_record(
                neuron_payload,
                cursor,
                full=True,
                csc_accelerate_enabled=csc_accelerate_enabled,
            )
            cursor += 4
        else:
            record = _decode_half_or_full_record(
                neuron_payload,
                cursor,
                full=False,
                kind_override=kind_raw,
                fold_override=fold_raw,
                csc_accelerate_enabled=csc_accelerate_enabled,
            )
            cursor += 2
        if fold_raw == FoldType.FOLDED:
            base_word_count = len(record.raw_hex)
            fold_record = _decode_fold_record(neuron_payload, cursor, record)
            record = fold_record
            cursor += len(fold_record.raw_hex) - base_word_count
        records.append(record)

    compress_counts = Counter()
    for record in records:
        compress_counts[_record_weight_compress_label(record)] += 1

    summary = NeuronSummaryView(
        total=len(records),
        half_count=sum(1 for record in records if record.kind.startswith("half")),
        full_count=sum(1 for record in records if record.kind.startswith("full")),
        folded_count=sum(1 for record in records if "folded" in record.kind),
        weight_compress_counts=dict(compress_counts),
    )
    return NeuronView(summary=summary, records=records)


def apply_sops_summary(
    core_config: dict[str, int], neurons: NeuronView, weights: WeightView
) -> NeuronView:
    """Estimate per-core SOPS from decoded neuron weight address ranges.

    Sparse weights can include storage padding slots. The summary keeps both
    padded and non-padded counts so the UI can expose that distinction instead
    of hiding storage pressure inside one number.
    """
    input_width = _data_width(core_config, "input_width")
    weight_width = _data_width(core_config, "weight_width")
    weight_records = {
        (record.start_address, record.end_address, record.kind): record
        for record in weights.records
    }
    sops_with_padding = 0
    sops_without_padding = 0
    weight_sram_pressure = 0
    for record in neurons.records:
        common = record.fields.get("common attrs", [])
        start = _field_decoded_int(common, "weight_address_start")
        end = _field_decoded_int(common, "weight_address_end")
        if start is None or end is None or end < start:
            continue

        fold_number = _fold_multiplier(record)
        sram_record_count = end - start + 1
        kind = _record_weight_kind(record)
        is_sparse = kind == "sparse"
        weight_sram_pressure += fold_number * sram_record_count
        sops_with_padding += compute_weight_pressure(
            fold_number, input_width, weight_width, sram_record_count, is_sparse
        )

        if is_sparse:
            weight_record = weight_records.get((start, end, kind))
            slots_without_padding = (
                weight_record.nonzero_count
                if weight_record is not None and weight_record.nonzero_count is not None
                else sram_record_count * csc_weight_slots_per_sram(weight_width)
            )
        else:
            slots_without_padding = None

        sops_without_padding += compute_weight_pressure(
            fold_number,
            input_width,
            weight_width,
            sram_record_count,
            is_sparse,
            slots_without_padding,
        )

    summary = replace(
        neurons.summary,
        synops_pressure=sops_with_padding,
        sops_with_padding=sops_with_padding,
        sops_without_padding=sops_without_padding,
        weight_sram_pressure=weight_sram_pressure,
    )
    return NeuronView(summary=summary, records=neurons.records)


def validate_neuron_destinations(
    neurons: NeuronView, core_coord: tuple[int, int], grid_width: int, grid_height: int
) -> None:
    """Fail fast when any decoded neuron route leaves the chip grid.

    The hardware route is checked as a path, not only as a final coordinate:
    offset routing moves through Z/XY, then X, then Y segments, and multicast copy
    destinations are validated after expansion.
    """
    source_x, source_y = core_coord
    for record in neurons.records:
        route = _route_fields(record)
        target_x = source_x + route.offset_z + route.offset_x
        target_y = source_y + route.offset_z + route.offset_y
        move = _first_offset_route_violation(
            source_x, source_y, route, grid_width, grid_height
        )
        if move is None:
            move = _first_copy_destination_violation(record, grid_width, grid_height)
        if move is not None:
            _raise_neuron_route_error(
                record,
                source_x,
                source_y,
                target_x,
                target_y,
                grid_width,
                grid_height,
                route,
                move,
            )


def attach_neuron_destinations(
    neurons: NeuronView, core_coord: tuple[int, int]
) -> NeuronView:
    records = [
        replace(record, destinations=decode_neuron_destinations(record, core_coord))
        for record in neurons.records
    ]
    return replace(neurons, records=records)


def decode_neuron_destinations(
    record: NeuronRecordView, core_coord: tuple[int, int]
) -> list[NeuronDestinationView]:
    """Expand one neuron's base route and multicast copy shape to destinations."""
    route = _route_fields(record)
    source_x, source_y = core_coord
    base_x = source_x + route.offset_z + route.offset_x
    base_y = source_y + route.offset_z + route.offset_y
    return [
        NeuronDestinationView(
            target_x=base_x + copy_offset_x,
            target_y=base_y + copy_offset_y,
            copy_offset_x=copy_offset_x,
            copy_offset_y=copy_offset_y,
        )
        for copy_offset_x, copy_offset_y in _copy_offsets(
            route.copy_z, route.copy_x, route.copy_y
        )
    ]


def _route_fields(record: NeuronRecordView) -> _RouteFields:
    fields = record.fields.get("dest info", [])
    return _RouteFields(
        offset_z=_field_decoded_int(fields, "addr_core_xy") or 0,
        offset_x=_field_decoded_int(fields, "addr_core_x") or 0,
        offset_y=_field_decoded_int(fields, "addr_core_y") or 0,
        copy_z=_field_decoded_int(fields, "addr_copy_xy") or 0,
        copy_x=_field_decoded_int(fields, "addr_copy_x") or 0,
        copy_y=_field_decoded_int(fields, "addr_copy_y") or 0,
    )


def _raise_neuron_route_error(
    record: NeuronRecordView,
    source_x: int,
    source_y: int,
    target_x: int,
    target_y: int,
    grid_width: int,
    grid_height: int,
    route: _RouteFields,
    move: _RouteMove,
) -> None:
    base_target = f"({target_x},{target_y})"
    route_target = f"({move.target_x},{move.target_y})"
    context = {
        "source_core": f"({source_x},{source_y})",
        "target_core": route_target,
        "illegal_core": f"({move.x},{move.y})",
        "route_kind": move.kind,
        "route_axis": move.axis,
        "route_step": move.step,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "neuron_index": record.index,
        "sram_address": record.sram_address,
        "addr_core_xy": route.offset_z,
        "addr_core_x": route.offset_x,
        "addr_core_y": route.offset_y,
        "addr_copy_xy": route.copy_z,
        "addr_copy_x": route.copy_x,
        "addr_copy_y": route.copy_y,
    }
    if route_target != base_target:
        context["base_target_core"] = base_target
    raise FrameDecodeError(
        "neuron destination route leaves chip grid",
        frame_index=record.frame_indices[1] if len(record.frame_indices) > 1 else None,
        raw_frame=int(record.raw_hex[1], 16) if len(record.raw_hex) > 1 else None,
        context=context,
    )


def _first_offset_route_violation(
    source_x: int, source_y: int, route: _RouteFields, grid_width: int, grid_height: int
) -> _RouteMove | None:
    x, y = source_x, source_y
    target_x = source_x + route.offset_z + route.offset_x
    target_y = source_y + route.offset_z + route.offset_y
    for axis, steps in (
        ("z", route.offset_z),
        ("x", route.offset_x),
        ("y", route.offset_y),
    ):
        violation = _first_axis_violation(x, y, axis, steps, grid_width, grid_height)
        if violation is not None:
            step, move_x, move_y = violation
            return _RouteMove("offset", axis, step, move_x, move_y, target_x, target_y)
        dx, dy = _axis_delta(axis, steps)
        x += dx
        y += dy
    return None


def _first_copy_destination_violation(
    record: NeuronRecordView, grid_width: int, grid_height: int
) -> _RouteMove | None:
    for destination in record.destinations:
        target_x = destination.target_x
        target_y = destination.target_y
        if 0 <= target_x < grid_width and 0 <= target_y < grid_height:
            continue
        return _RouteMove(
            "copy",
            _copy_violation_axis(destination.copy_offset_x, destination.copy_offset_y),
            1,
            target_x,
            target_y,
            target_x,
            target_y,
        )
    return None


def _first_axis_violation(
    x: int, y: int, axis: _RouteAxis, steps: int, grid_width: int, grid_height: int
) -> tuple[int, int, int] | None:
    if steps == 0:
        return None
    dx, dy = _axis_unit(axis, steps)
    step_count = abs(steps)
    start_step = _first_axis_boundary_crossing_step(
        x, y, dx, dy, grid_width, grid_height
    )
    if start_step is None or start_step > step_count:
        return None
    return start_step, x + dx * start_step, y + dy * start_step


def _first_axis_boundary_crossing_step(
    x: int, y: int, dx: int, dy: int, grid_width: int, grid_height: int
) -> int | None:
    first_step: int | None = None
    if dx > 0:
        first_step = _min_defined(first_step, grid_width - x)
    elif dx < 0:
        first_step = _min_defined(first_step, x + 1)
    if dy > 0:
        first_step = _min_defined(first_step, grid_height - y)
    elif dy < 0:
        first_step = _min_defined(first_step, y + 1)
    return first_step


def _min_defined(current: int | None, candidate: int) -> int:
    if current is None:
        return candidate
    return min(current, candidate)


@lru_cache(maxsize=128)
def _copy_offsets(copy_z: int, copy_x: int, copy_y: int) -> tuple[tuple[int, int], ...]:
    """Return ordered multicast offsets for one V2 copy shape.

    Many neurons share identical copy fields. Caching avoids rebuilding the same
    breadth-first expansion while preserving the ordered, deduplicated offsets
    expected by the UI.
    """
    queue = [_RouteState(0, 0, copy_z, copy_x, copy_y)]
    offsets: list[tuple[int, int]] = []
    visited_offsets: set[tuple[int, int]] = set()
    visited_states: set[_RouteState] = set()
    cursor = 0
    while cursor < len(queue):
        state = queue[cursor]
        cursor += 1
        if state in visited_states:
            continue
        visited_states.add(state)
        for foothold, copied in _walk_copy_state(state):
            if foothold not in visited_offsets:
                visited_offsets.add(foothold)
                offsets.append(foothold)
            queue.append(copied)
    return tuple(offsets)


def _walk_copy_state(
    state: _RouteState,
) -> list[tuple[tuple[int, int], _RouteState]]:
    copied_states: list[tuple[tuple[int, int], _RouteState]] = []
    copy_z = state.copy_z
    copy_x = state.copy_x
    copy_y = state.copy_y
    while copy_z or copy_x or copy_y:
        if copy_z:
            dx, dy = _axis_unit("z", copy_z)
            copy_z -= _sign(copy_z)
        elif copy_x:
            dx, dy = _axis_unit("x", copy_x)
            copy_x -= _sign(copy_x)
        else:
            dx, dy = _axis_unit("y", copy_y)
            copy_y -= _sign(copy_y)
        copied_states.append(
            (
                (state.x, state.y),
                _RouteState(state.x + dx, state.y + dy, copy_z, copy_x, copy_y),
            )
        )
    copied_states.append(((state.x, state.y), state))
    return copied_states


def _copy_violation_axis(copy_offset_x: int, copy_offset_y: int) -> _RouteAxis:
    if copy_offset_x == copy_offset_y and copy_offset_x != 0:
        return "z"
    if copy_offset_x != 0:
        return "x"
    return "y"


def _axis_delta(axis: _RouteAxis, steps: int) -> tuple[int, int]:
    dx, dy = _axis_unit(axis, steps)
    return dx * abs(steps), dy * abs(steps)


def _axis_unit(axis: _RouteAxis, steps: int) -> tuple[int, int]:
    sign = _sign(steps)
    if axis == "z":
        return sign, sign
    if axis == "x":
        return sign, 0
    return 0, sign


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def decode_weights(
    core_config: dict[str, int],
    packages: list[FramePackageInfo],
    neurons: NeuronView,
) -> WeightView:
    """Decode storage-level dense/CSC weight records referenced by neurons.

    The visualizer can recover values as laid out in SRAM. It intentionally does
    not claim to reconstruct the original logical tensor/kernel layout without
    extra compile-time mapping metadata.
    """
    neuron_words = core_config.get("neuron_number", 0) * SRAM_WORDS
    frame3_payload = _frame3_payload(packages)
    if neuron_words >= len(frame3_payload):
        return WeightView()

    weight_payload = frame3_payload[neuron_words:]
    weight_ranges = _collect_weight_ranges(neurons.records)
    records: list[WeightRecordView] = []
    for index, (start, end, kind) in enumerate(weight_ranges):
        word_start = max(0, (start * SRAM_WORDS) - neuron_words)
        word_end = max(word_start, ((end + 1) * SRAM_WORDS) - neuron_words)
        words = weight_payload[word_start:word_end]
        sram_records = max(0, end - start + 1)
        raw_hex = [raw for _, raw in words]
        nonzero_count: int | None = None
        padding_count: int | None = None
        storage_values: list[int] = []
        storage_values_raw: list[int] = []
        storage_entries: list[WeightStorageEntryView] = []
        storage_value_count = 0
        storage_preview_limit = 0
        weight_width = _width_bits(core_config, "weight_width")
        input_width = _width_bits(core_config, "input_width")
        signed = core_config.get("weight_sign", DataSign.UNSIGNED) == DataSign.SIGNED
        if kind == "sparse":
            decoded_csc = _decode_csc_storage(words, weight_width, input_width, signed)
            nonzero_count = decoded_csc.nonzero_count
            padding_count = decoded_csc.padding_count
            storage_value_count = decoded_csc.storage_value_count
            storage_preview_limit = decoded_csc.storage_preview_limit
            storage_entries = decoded_csc.entries
        else:
            decoded_dense = _decode_dense_storage(words, weight_width, signed)
            nonzero_count = decoded_dense.nonzero_count
            storage_value_count = decoded_dense.storage_value_count
            storage_preview_limit = decoded_dense.storage_preview_limit
            storage_values = decoded_dense.values
            storage_values_raw = decoded_dense.values_raw
        records.append(
            WeightRecordView(
                index=index,
                kind=kind,
                start_address=start,
                end_address=end,
                sram_records=sram_records,
                bits=sram_records * SRAM_RECORD_BITS,
                bytes=sram_records * SRAM_RECORD_BYTES,
                nonzero_count=nonzero_count,
                padding_count=padding_count,
                storage_value_count=storage_value_count,
                storage_preview_limit=storage_preview_limit,
                storage_values=storage_values,
                storage_values_raw=storage_values_raw,
                storage_entries=storage_entries,
                frame_indices=[frame_idx for frame_idx, _ in words],
                raw_hex=raw_hex,
            )
        )

    summary = WeightSummaryView(
        total=len(records),
        data_type=_weight_data_type(core_config),
        dense_count=sum(1 for item in records if item.kind == "dense"),
        csc_count=sum(1 for item in records if item.kind == "sparse"),
        sram_records=sum(item.sram_records for item in records),
        bits=sum(item.bits for item in records),
        bytes=sum(item.bytes for item in records),
        nonzero_count=sum(item.nonzero_count or 0 for item in records),
        padding_count=sum(item.padding_count or 0 for item in records),
    )
    return WeightView(summary=summary, records=records)


def build_raw_frame_records(
    packages: list[FramePackageInfo],
) -> list[RawFrameRecord]:
    records: list[RawFrameRecord] = []
    for package in packages:
        for offset, raw in enumerate(package.payloads):
            sram_address = (
                package.start_addr + offset // SRAM_WORDS
                if package.frame_type == 3
                else package.start_addr + offset
            )
            records.append(
                RawFrameRecord(
                    frame_index=package.frame_start + 1 + offset,
                    frame_type=package.frame_type,
                    start_addr=package.start_addr,
                    package_type=package.package_type,
                    sram_address=sram_address,
                    word_offset=offset,
                    raw_hex=f"0x{raw:016x}",
                )
            )
    return records


_F3 = OfflineConfigFrame3FormatV2


def bit_field(value: int, offset: int, mask: int) -> int:
    return (value >> offset) & mask


def sign_magnitude_to_int(value: int, bits: int = 6) -> int:
    sign = value >> (bits - 1)
    magnitude = value & ((1 << (bits - 1)) - 1)
    return -magnitude if sign else magnitude


def _decode_half_or_full_record(
    payload: list[tuple[int, str]],
    cursor: int,
    *,
    full: bool,
    csc_accelerate_enabled: bool,
    kind_override: int | None = None,
    fold_override: int | None = None,
) -> NeuronRecordView:
    w1 = int(payload[cursor][1], 16)
    w2 = int(payload[cursor + 1][1], 16)
    fields = {
        "common attrs": _common_neuron_fields(
            w1, w2, payload[cursor][0], kind_override, fold_override
        ),
        "dest info": _dest_fields(w2, payload[cursor + 1][0]),
    }
    raw_hex = [payload[cursor][1], payload[cursor + 1][1]]
    frame_indices = [payload[cursor][0], payload[cursor + 1][0]]
    kind = "half"
    if full:
        w3 = int(payload[cursor + 2][1], 16)
        w4 = int(payload[cursor + 3][1], 16)
        fields["full attrs"] = _full_neuron_fields(
            w3, w4, payload[cursor + 2][0], csc_accelerate_enabled
        )
        raw_hex.extend([payload[cursor + 2][1], payload[cursor + 3][1]])
        frame_indices.extend([payload[cursor + 2][0], payload[cursor + 3][0]])
        kind = "full"
    return NeuronRecordView(
        index=cursor // SRAM_WORDS,
        kind=kind,
        sram_address=cursor // SRAM_WORDS,
        frame_indices=frame_indices,
        fields=fields,
        raw_hex=raw_hex,
    )


def _decode_fold_record(
    payload: list[tuple[int, str]], cursor: int, base_record: NeuronRecordView
) -> NeuronRecordView:
    if cursor + 1 >= len(payload):
        raise FrameDecodeError(
            "folded neuron attrs payload is truncated",
            frame_index=(
                base_record.frame_indices[0] if base_record.frame_indices else None
            ),
            raw_frame=int(base_record.raw_hex[0], 16) if base_record.raw_hex else None,
            context={"sram_address": base_record.sram_address},
        )
    w1 = int(payload[cursor][1], 16)
    w2 = int(payload[cursor + 1][1], 16)
    fields = dict(base_record.fields)
    fields["fold attrs"] = _fold_neuron_fields(w1, w2, payload[cursor][0])
    raw_hex = [*base_record.raw_hex, payload[cursor][1], payload[cursor + 1][1]]
    frame_indices = [
        *base_record.frame_indices,
        payload[cursor][0],
        payload[cursor + 1][0],
    ]
    fold_number = _field_decoded_int(fields["fold attrs"], "fold_number") or 0
    n_extra_records = ((fold_number - 1) + 3) // 4 if fold_number else 0
    for idx in range(n_extra_records):
        base = cursor + 2 + idx * 2
        if base + 1 >= len(payload):
            raise FrameDecodeError(
                "folded neuron VJT payload is truncated",
                frame_index=payload[cursor][0],
                raw_frame=int(payload[cursor][1], 16),
                context={"fold_number": fold_number},
            )
        raw_hex.extend([payload[base][1], payload[base + 1][1]])
        frame_indices.extend([payload[base][0], payload[base + 1][0]])
    return NeuronRecordView(
        index=base_record.index,
        kind=f"{base_record.kind}+folded",
        sram_address=base_record.sram_address,
        frame_indices=frame_indices,
        fields=fields,
        raw_hex=raw_hex,
    )


def _common_neuron_fields(
    w1: int,
    w2: int,
    source_index: int,
    kind_override: int | None,
    fold_override: int | None,
) -> list[DecodedField]:
    weight_skew = (
        bit_field(
            w2,
            _F3.Full.Word2.WEIGHT_SKEW_HIGH11_OFFSET,
            _F3.Full.Word2.WEIGHT_SKEW_HIGH11_MASK,
        )
        << 5
    ) | bit_field(
        w1, _F3.Full.Word1.WEIGHT_SKEW_LOW5_OFFSET, _F3.Full.Word1.WEIGHT_SKEW_LOW5_MASK
    )
    kind_raw = (
        kind_override
        if kind_override is not None
        else bit_field(
            w1, _F3.Full.Word1.NEURON_TYPE_OFFSET, _F3.Full.Word1.NEURON_TYPE_MASK
        )
    )
    fold_raw = (
        fold_override
        if fold_override is not None
        else bit_field(
            w1, _F3.Full.Word1.FOLD_TYPE_OFFSET, _F3.Full.Word1.FOLD_TYPE_MASK
        )
    )
    return [
        _decoded_field(
            "weight_skew", weight_skew, weight_skew, "Weight skew", "uint", source_index
        ),
        _decoded_field(
            "weight_address_start",
            bit_field(
                w1,
                _F3.Full.Word1.WEIGHT_ADDRESS_START_OFFSET,
                _F3.Full.Word1.WEIGHT_ADDRESS_START_MASK,
            ),
            bit_field(
                w1,
                _F3.Full.Word1.WEIGHT_ADDRESS_START_OFFSET,
                _F3.Full.Word1.WEIGHT_ADDRESS_START_MASK,
            ),
            "Weight start",
            "uint",
            source_index,
        ),
        _decoded_field(
            "weight_address_end",
            bit_field(
                w1,
                _F3.Full.Word1.WEIGHT_ADDRESS_END_OFFSET,
                _F3.Full.Word1.WEIGHT_ADDRESS_END_MASK,
            ),
            bit_field(
                w1,
                _F3.Full.Word1.WEIGHT_ADDRESS_END_OFFSET,
                _F3.Full.Word1.WEIGHT_ADDRESS_END_MASK,
            ),
            "Weight end",
            "uint",
            source_index,
        ),
        _enum_value_field(
            "output_type",
            bit_field(
                w1, _F3.Full.Word1.OUTPUT_TYPE_OFFSET, _F3.Full.Word1.OUTPUT_TYPE_MASK
            ),
            OutputType,
            "Output type",
            source_index,
        ),
        _enum_value_field("fold_type", fold_raw, FoldType, "Fold type", source_index),
        _enum_value_field(
            "neuron_type", kind_raw, NeuronType, "Neuron type", source_index
        ),
        _decoded_field(
            "vjt",
            bit_field(w1, _F3.Full.Word1.VJT_OFFSET, _F3.Full.Word1.VJT_MASK),
            twos_complement_to_int(
                bit_field(w1, _F3.Full.Word1.VJT_OFFSET, _F3.Full.Word1.VJT_MASK), 32
            ),
            "Membrane potential",
            "int",
            source_index,
        ),
    ]


def _dest_fields(w2: int, source_index: int) -> list[DecodedField]:
    return [
        _decoded_field(
            "tick_relative",
            bit_field(
                w2,
                _F3.Full.Word2.TICK_RELATIVE_OFFSET,
                _F3.Full.Word2.TICK_RELATIVE_MASK,
            ),
            bit_field(
                w2,
                _F3.Full.Word2.TICK_RELATIVE_OFFSET,
                _F3.Full.Word2.TICK_RELATIVE_MASK,
            ),
            "Relative tick",
            "uint",
            source_index,
        ),
        _decoded_field(
            "addr_axon",
            bit_field(
                w2, _F3.Full.Word2.ADDR_AXON_OFFSET, _F3.Full.Word2.ADDR_AXON_MASK
            ),
            bit_field(
                w2, _F3.Full.Word2.ADDR_AXON_OFFSET, _F3.Full.Word2.ADDR_AXON_MASK
            ),
            "Target axon",
            "uint",
            source_index,
        ),
        _sign_field_from_raw(
            "addr_core_xy",
            bit_field(
                w2, _F3.Full.Word2.ADDR_CORE_XY_OFFSET, _F3.Full.Word2.ADDR_CORE_XY_MASK
            ),
            "Target core XY",
            source_index,
        ),
        _sign_field_from_raw(
            "addr_core_x",
            bit_field(
                w2, _F3.Full.Word2.ADDR_CORE_X_OFFSET, _F3.Full.Word2.ADDR_CORE_X_MASK
            ),
            "Target core X",
            source_index,
        ),
        _sign_field_from_raw(
            "addr_core_y",
            bit_field(
                w2, _F3.Full.Word2.ADDR_CORE_Y_OFFSET, _F3.Full.Word2.ADDR_CORE_Y_MASK
            ),
            "Target core Y",
            source_index,
        ),
        _sign_field_from_raw(
            "addr_copy_xy",
            bit_field(
                w2, _F3.Full.Word2.ADDR_COPY_XY_OFFSET, _F3.Full.Word2.ADDR_COPY_XY_MASK
            ),
            "Copy XY",
            source_index,
        ),
        _sign_field_from_raw(
            "addr_copy_x",
            bit_field(
                w2, _F3.Full.Word2.ADDR_COPY_X_OFFSET, _F3.Full.Word2.ADDR_COPY_X_MASK
            ),
            "Copy X",
            source_index,
        ),
        _sign_field_from_raw(
            "addr_copy_y",
            bit_field(
                w2, _F3.Full.Word2.ADDR_COPY_Y_OFFSET, _F3.Full.Word2.ADDR_COPY_Y_MASK
            ),
            "Copy Y",
            source_index,
        ),
    ]


def _full_neuron_fields(
    w3: int, w4: int, source_index: int, csc_accelerate_enabled: bool
) -> list[DecodedField]:
    threshold_pos = (
        bit_field(
            w4,
            _F3.Full.Word4.THRESHOLD_POS_HIGH12_OFFSET,
            _F3.Full.Word4.THRESHOLD_POS_HIGH12_MASK,
        )
        << 20
    ) | bit_field(
        w3,
        _F3.Full.Word3.THRESHOLD_POS_LOW20_OFFSET,
        _F3.Full.Word3.THRESHOLD_POS_LOW20_MASK,
    )
    vjt_initial_raw = bit_field(
        w3, _F3.Full.Word3.VJT_INITIAL_OFFSET, _F3.Full.Word3.VJT_INITIAL_MASK
    )
    if csc_accelerate_enabled:
        vjt_initial_field = _decoded_field(
            "vjt_initial",
            vjt_initial_raw,
            vjt_initial_raw,
            "weight_address_start stored in vjt_initial bits when CSC accelerate is enabled",
            "uint",
            source_index,
        )
    else:
        vjt_initial_field = _decoded_field(
            "vjt_initial",
            vjt_initial_raw,
            twos_complement_to_int(vjt_initial_raw, 12),
            "Initial membrane potential",
            "int",
            source_index,
        )
    return [
        _enum_value_field(
            "lateral_inhibition",
            bit_field(
                w3,
                _F3.Full.Word3.LATERAL_INHIBITION_OFFSET,
                _F3.Full.Word3.LATERAL_INHIBITION_MASK,
            ),
            LateralInhibitionMode,
            "Lateral inhibition",
            source_index,
        ),
        _enum_value_field(
            "leak_multi_sequence",
            bit_field(
                w3,
                _F3.Full.Word3.LEAK_MULTI_SEQUENCE_OFFSET,
                _F3.Full.Word3.LEAK_MULTI_SEQUENCE_MASK,
            ),
            LeakMultiComparisonOrder,
            "Leak sequence",
            source_index,
        ),
        _enum_value_field(
            "leak_multi_input",
            bit_field(
                w3,
                _F3.Full.Word3.LEAK_MULTI_INPUT_OFFSET,
                _F3.Full.Word3.LEAK_MULTI_INPUT_MASK,
            ),
            LeakMultiInputMode,
            "Leak input",
            source_index,
        ),
        _enum_value_field(
            "leak_multi_mode",
            bit_field(
                w3,
                _F3.Full.Word3.LEAK_MULTI_MODE_OFFSET,
                _F3.Full.Word3.LEAK_MULTI_MODE_MASK,
            ),
            LeakMultiMode,
            "Leak multiply mode",
            source_index,
        ),
        _enum_value_field(
            "leak_add_mode",
            bit_field(
                w3,
                _F3.Full.Word3.LEAK_ADD_MODE_OFFSET,
                _F3.Full.Word3.LEAK_ADD_MODE_MASK,
            ),
            LeakAddMode,
            "Leak add mode",
            source_index,
        ),
        _decoded_field(
            "leak_tau",
            bit_field(w3, _F3.Full.Word3.LEAK_TAU_OFFSET, _F3.Full.Word3.LEAK_TAU_MASK),
            twos_complement_to_int(
                bit_field(
                    w3, _F3.Full.Word3.LEAK_TAU_OFFSET, _F3.Full.Word3.LEAK_TAU_MASK
                ),
                6,
            ),
            "Leak tau",
            "int",
            source_index,
        ),
        _decoded_field(
            "leak_v",
            bit_field(w3, _F3.Full.Word3.LEAK_V_OFFSET, _F3.Full.Word3.LEAK_V_MASK),
            twos_complement_to_int(
                bit_field(w3, _F3.Full.Word3.LEAK_V_OFFSET, _F3.Full.Word3.LEAK_V_MASK),
                20,
            ),
            "Leak voltage",
            "int",
            source_index,
        ),
        _enum_value_field(
            "weight_compress",
            bit_field(
                w3,
                _F3.Full.Word3.WEIGHT_COMPRESS_OFFSET,
                _F3.Full.Word3.WEIGHT_COMPRESS_MASK,
            ),
            WeightCompressType,
            "Weight compression",
            source_index,
        ),
        vjt_initial_field,
        _enum_value_field(
            "reset_mode",
            bit_field(
                w4, _F3.Full.Word4.RESET_MODE_OFFSET, _F3.Full.Word4.RESET_MODE_MASK
            ),
            ResetMode,
            "Reset mode",
            source_index,
        ),
        _decoded_field(
            "reset_v",
            bit_field(w4, _F3.Full.Word4.RESET_V_OFFSET, _F3.Full.Word4.RESET_V_MASK),
            twos_complement_to_int(
                bit_field(
                    w4, _F3.Full.Word4.RESET_V_OFFSET, _F3.Full.Word4.RESET_V_MASK
                ),
                16,
            ),
            "Reset voltage",
            "int",
            source_index,
        ),
        _enum_value_field(
            "threshold_neg_mode",
            bit_field(
                w4,
                _F3.Full.Word4.THRESHOLD_NEG_MODE_OFFSET,
                _F3.Full.Word4.THRESHOLD_NEG_MODE_MASK,
            ),
            ThresholdNegMode,
            "Negative threshold mode",
            source_index,
        ),
        _enum_value_field(
            "threshold_pos_mode",
            bit_field(
                w4,
                _F3.Full.Word4.THRESHOLD_POS_MODE_OFFSET,
                _F3.Full.Word4.THRESHOLD_POS_MODE_MASK,
            ),
            ThresholdPosMode,
            "Positive threshold mode",
            source_index,
        ),
        _decoded_field(
            "threshold_neg",
            bit_field(
                w4,
                _F3.Full.Word4.THRESHOLD_NEG_OFFSET,
                _F3.Full.Word4.THRESHOLD_NEG_MASK,
            ),
            twos_complement_to_int(
                bit_field(
                    w4,
                    _F3.Full.Word4.THRESHOLD_NEG_OFFSET,
                    _F3.Full.Word4.THRESHOLD_NEG_MASK,
                ),
                32,
            ),
            "Negative threshold",
            "int",
            source_index,
        ),
        _decoded_field(
            "threshold_pos",
            threshold_pos,
            twos_complement_to_int(threshold_pos, 32),
            "Positive threshold",
            "int",
            source_index,
        ),
    ]


def _fold_neuron_fields(w1: int, w2: int, source_index: int) -> list[DecodedField]:
    fold_skew_y = (
        bit_field(
            w2,
            _F3.Fold.Word2.FOLD_SKEW_Y_HIGH9_OFFSET,
            _F3.Fold.Word2.FOLD_SKEW_Y_HIGH9_MASK,
        )
        << 2
    ) | bit_field(
        w1, _F3.Fold.Word1.FOLD_SKEW_Y_LOW2_OFFSET, _F3.Fold.Word1.FOLD_SKEW_Y_LOW2_MASK
    )
    fields = [
        (
            "fold_range_xy",
            bit_field(
                w2,
                _F3.Fold.Word2.FOLD_RANGE_XY_OFFSET,
                _F3.Fold.Word2.FOLD_RANGE_XY_MASK,
            ),
            "Fold XY range",
        ),
        (
            "fold_range_x",
            bit_field(
                w2, _F3.Fold.Word2.FOLD_RANGE_X_OFFSET, _F3.Fold.Word2.FOLD_RANGE_X_MASK
            ),
            "Fold X range",
        ),
        (
            "fold_range_y",
            bit_field(
                w2, _F3.Fold.Word2.FOLD_RANGE_Y_OFFSET, _F3.Fold.Word2.FOLD_RANGE_Y_MASK
            ),
            "Fold Y range",
        ),
        (
            "fold_skew_xy",
            bit_field(
                w2, _F3.Fold.Word2.FOLD_SKEW_XY_OFFSET, _F3.Fold.Word2.FOLD_SKEW_XY_MASK
            ),
            "Fold XY weight skew",
        ),
        (
            "fold_skew_x",
            bit_field(
                w2, _F3.Fold.Word2.FOLD_SKEW_X_OFFSET, _F3.Fold.Word2.FOLD_SKEW_X_MASK
            ),
            "Fold X weight skew",
        ),
        ("fold_skew_y", fold_skew_y, "Fold Y weight skew"),
        (
            "fold_axon_xy",
            bit_field(
                w1, _F3.Fold.Word1.FOLD_AXON_XY_OFFSET, _F3.Fold.Word1.FOLD_AXON_XY_MASK
            ),
            "Fold XY axon skew",
        ),
        (
            "fold_axon_x",
            bit_field(
                w1, _F3.Fold.Word1.FOLD_AXON_X_OFFSET, _F3.Fold.Word1.FOLD_AXON_X_MASK
            ),
            "Fold X axon skew",
        ),
        (
            "fold_axon_y",
            bit_field(
                w1, _F3.Fold.Word1.FOLD_AXON_Y_OFFSET, _F3.Fold.Word1.FOLD_AXON_Y_MASK
            ),
            "Fold Y axon skew",
        ),
        (
            "fold_number",
            bit_field(
                w1, _F3.Fold.Word1.FOLD_NUMBER_OFFSET, _F3.Fold.Word1.FOLD_NUMBER_MASK
            ),
            "Folded neuron count",
        ),
    ]
    return [
        _decoded_field(name, raw, raw, label, "uint", source_index)
        for name, raw, label in fields
    ]


def _frame3_payload(packages: list[FramePackageInfo]) -> list[tuple[int, str]]:
    payload: list[tuple[int, str]] = []
    for package in packages:
        if package.frame_type != 3:
            continue
        payload.extend(
            (package.frame_start + 1 + offset, f"0x{raw:016x}")
            for offset, raw in enumerate(package.payloads)
        )
    return payload


def _collect_weight_ranges(
    records: list[NeuronRecordView],
) -> list[tuple[int, int, str]]:
    ranges: set[tuple[int, int, str]] = set()
    for record in records:
        common = record.fields.get("common attrs", [])
        start = _field_decoded_int(common, "weight_address_start")
        end = _field_decoded_int(common, "weight_address_end")
        if start is None or end is None or end < start:
            continue
        ranges.add((start, end, _record_weight_kind(record)))
    return sorted(ranges)


def _record_weight_compress_label(record: NeuronRecordView) -> str:
    return next(
        (
            field.label
            for field in record.fields.get("full attrs", [])
            if field.name == "weight_compress"
        ),
        "DENSE",
    )


def _record_weight_kind(record: NeuronRecordView) -> str:
    return "sparse" if "SPARSE" in _record_weight_compress_label(record) else "dense"


def _fold_multiplier(record: NeuronRecordView) -> int:
    return _field_decoded_int(record.fields.get("fold attrs", []), "fold_number") or 1


def _decode_dense_storage(
    words: list[tuple[int, str]], weight_width: int, signed: bool
) -> _DenseStorageDecode:
    values: list[int] = []
    values_raw: list[int] = []
    nonzero_count = 0
    values_per_word = 64 // weight_width
    value_mask = (1 << weight_width) - 1
    for _, raw in words:
        word = int(raw, 16)
        for value_idx in range(values_per_word):
            raw_value = (word >> (value_idx * weight_width)) & value_mask
            if raw_value != 0:
                nonzero_count += 1
            if len(values) < WEIGHT_STORAGE_PREVIEW_LIMIT:
                values_raw.append(raw_value)
                values.append(_decode_weight_value(raw_value, weight_width, signed))

    return _DenseStorageDecode(
        values=values,
        values_raw=values_raw,
        nonzero_count=nonzero_count,
        storage_value_count=len(words) * values_per_word,
        storage_preview_limit=WEIGHT_STORAGE_PREVIEW_LIMIT,
    )


def _decode_csc_storage(
    words: list[tuple[int, str]], weight_width: int, input_width: int, signed: bool
) -> _CscStorageDecode:
    n_per_addr = {1: 7, 2: 7, 4: 6, 8: 5}.get(weight_width, 5)
    entries: list[WeightStorageEntryView] = []
    nonzero_count = 0
    padding_count = 0
    slot = 0

    for idx in range(0, len(words), SRAM_WORDS):
        if idx + 1 >= len(words):
            break
        low_frame_index, low_raw = words[idx]
        low = int(low_raw, 16)
        high = int(words[idx + 1][1], 16)
        indices = _decode_csc_indices(low, high, weight_width, n_per_addr)
        value_mask = (1 << weight_width) - 1

        for value_idx in range(n_per_addr):
            raw_value = (low >> (value_idx * weight_width)) & value_mask
            is_padding = raw_value == 0
            if is_padding:
                padding_count += 1
            else:
                nonzero_count += 1
            bit_index = indices[value_idx]
            logical_index = (
                bit_index // input_width
                if input_width > 0 and bit_index % input_width == 0
                else None
            )
            if len(entries) < WEIGHT_STORAGE_PREVIEW_LIMIT:
                entries.append(
                    WeightStorageEntryView(
                        slot=slot,
                        value_raw=raw_value,
                        value=_decode_weight_value(raw_value, weight_width, signed),
                        bit_index=bit_index,
                        logical_index=logical_index,
                        is_padding=is_padding,
                        source_frame_index=low_frame_index,
                    )
                )
            slot += 1

    return _CscStorageDecode(
        entries=entries,
        nonzero_count=nonzero_count,
        padding_count=padding_count,
        storage_value_count=slot,
        storage_preview_limit=WEIGHT_STORAGE_PREVIEW_LIMIT,
    )


def _decode_csc_indices(
    low: int, high: int, weight_width: int, n_per_addr: int
) -> list[int]:
    offset = {1: 16, 2: 16, 4: 32, 8: 48}.get(weight_width, 48)
    n_low = n_per_addr - 4
    indices = []
    low_indices = low >> offset
    for idx in range(n_low):
        indices.append((low_indices >> (idx * 16)) & 0xFFFF)
    for idx in range(4):
        indices.append((high >> (idx * 16)) & 0xFFFF)
    return indices


def _decode_weight_value(raw_value: int, weight_width: int, signed: bool) -> int:
    if not signed:
        return raw_value
    return twos_complement_to_int(raw_value, weight_width)


def _width_bits(core_config: dict[str, int], key: str) -> int:
    raw = core_config.get(key, DataWidth.WIDTH_8BIT)
    return 1 << raw


def _data_width(core_config: dict[str, int], key: str) -> DataWidth:
    raw = core_config.get(key, DataWidth.WIDTH_8BIT)
    return DataWidth(int(raw))


def _weight_data_type(core_config: dict[str, int]) -> str:
    sign_raw = core_config.get("weight_sign", DataSign.UNSIGNED)
    width_raw = core_config.get("weight_width", DataWidth.WIDTH_8BIT)
    prefix = "int" if sign_raw == DataSign.SIGNED else "uint"
    return f"{prefix}{1 << width_raw}"


def _first_package_start(
    packages: list[FramePackageInfo], frame_type: int
) -> int | None:
    for package in packages:
        if package.frame_type == frame_type:
            return package.frame_start
    return None


def _enum_field(
    name: str,
    values: dict[str, int],
    enum_cls: type[Enum],
    description: str,
    source_index: int | None,
) -> DecodedField:
    raw = values.get(name, 0)
    return _enum_value_field(name, raw, enum_cls, description, source_index)


def _enum_value_field(
    name: str,
    raw: int,
    enum_cls: type[Enum],
    description: str,
    source_index: int | None,
) -> DecodedField:
    try:
        enum_value = enum_cls(raw)
    except ValueError as exc:
        raise FrameDecodeError(
            "decoded enum value is outside paicorelib definition",
            frame_index=source_index,
            context={"field": name, "value": raw, "enum": enum_cls.__name__},
        ) from exc
    return DecodedField(
        name=name,
        raw=raw,
        decoded=enum_value.value,
        label=enum_value.name,
        description=description,
        numeric_kind="enum",
        source_frame_index=source_index,
    )


def _uint_field(
    name: str, values: dict[str, int], description: str, source_index: int | None
) -> DecodedField:
    raw = values.get(name, 0)
    return _decoded_field(name, raw, raw, description, "uint", source_index)


def _int_field(
    name: str,
    values: dict[str, int],
    description: str,
    source_index: int | None,
    *,
    bits: int,
) -> DecodedField:
    raw = values.get(name, 0)
    return _decoded_field(
        name, raw, twos_complement_to_int(raw, bits), description, "int", source_index
    )


def _sign_magnitude_field(
    name: str, values: dict[str, int], description: str, source_index: int | None
) -> DecodedField:
    raw = values.get(name, 0)
    return _sign_field_from_raw(name, raw, description, source_index)


def _sign_field_from_raw(
    name: str, raw: int, description: str, source_index: int | None
) -> DecodedField:
    decoded = sign_magnitude_to_int(raw)
    return DecodedField(
        name=name,
        raw=raw,
        decoded=decoded,
        label=str(decoded),
        description=description,
        numeric_kind="sign_magnitude",
        source_frame_index=source_index,
    )


def _bitset_field(
    name: str,
    values: dict[str, int],
    description: str,
    source_index: int | None,
    *,
    include_local: bool,
) -> DecodedField:
    raw = values.get(name, 0)
    dirs = global_signal_direction_names(raw, include_local=include_local)
    label = ", ".join(dirs) if dirs else "none"
    return DecodedField(
        name=name,
        raw=raw,
        decoded=",".join(dirs),
        label=label,
        description=description,
        numeric_kind="bitset",
        source_frame_index=source_index,
    )


def _decoded_field(
    name: str,
    raw: int | str,
    decoded: int | str | None,
    description: str,
    kind: str,
    source_index: int | None,
) -> DecodedField:
    return DecodedField(
        name=name,
        raw=raw,
        decoded=decoded,
        label=str(decoded),
        description=description,
        numeric_kind=kind,  # type: ignore[arg-type]
        source_frame_index=source_index,
    )


def _field_decoded_int(fields: list[DecodedField], name: str) -> int | None:
    for field in fields:
        if field.name == name and isinstance(field.decoded, int):
            return field.decoded
    return None


def twos_complement_to_int(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    mask = (1 << bits) - 1
    value &= mask
    return value - (1 << bits) if value & sign_bit else value
