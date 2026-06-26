from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import replace
from math import prod
from typing import Any

from paicorelib.framelib.parser_v2 import (
    decode_aer_route_fields,
    expand_aer_destinations,
)
from paicorelib.framelib.utils import LCN_TO_TS_AXON_WIDTHS

from paibox.backendv2.proto.compile_artifacts_pb2 import (
    CompileArtifacts,
    DataType,
    OutputTensorMapping,
)

from ...model import (
    BufferSpanView,
    CoreView,
    DecodedField,
    IoCoreSummary,
    IoEntryView,
    IoTensorView,
    IoView,
    NeuronRecordView,
    TensorPlane,
    TensorRegionRun,
    TensorRegionView,
    ValidationEntry,
)
from .frame_parser import CHIP_ID, GRID_HEIGHT, GRID_WIDTH

FANIN_BASE = 512
INPUT_BUFFER_DEPTH = 256
INPUT_BUFFER_WIDTH = 512


def build_io_view(
    artifacts: CompileArtifacts | None, validation: list[ValidationEntry]
) -> IoView:
    """Build static IO views from protobuf mapping metadata.

    The view connects three spaces for debugging: original tensor coordinates,
    physical target cores, and per-core input-buffer slots. It does not load or
    display runtime sample payload values.
    """
    if artifacts is None:
        return IoView()

    tensors: list[IoTensorView] = []
    input_entries: list[IoEntryView] = []
    output_entries: list[IoEntryView] = []

    for thread in artifacts.io_mapping.threads:
        thread_id = thread.thread_id
        for mapping in thread.input_mappings.items:
            shape = list(mapping.shape.size)
            plane = _tensor_plane(shape)
            dtype = _input_dtype(mapping)
            tensor_entries: list[IoEntryView] = []
            for entry in mapping.entries:
                elem_idx = entry.elem_idx
                tick_relative = entry.tick_relative
                addr_axon = entry.addr_axon
                target_lcn = entry.target_lcn
                if _validate_elem_idx(
                    validation,
                    "input",
                    thread_id,
                    mapping.name,
                    elem_idx,
                    shape,
                ):
                    continue
                coord = _tensor_coord(elem_idx, shape)
                plane_y, plane_x = _plane_point(coord, plane)
                slice_key = _slice_key(coord, plane)
                full_addr = tick_relative * FANIN_BASE + addr_axon
                work_timestep, work_axon = _split_input_address(target_lcn, full_addr)
                for copy_offset_x, copy_offset_y in _copy_offsets(
                    entry.copy_count.xy,
                    entry.copy_count.x,
                    entry.copy_count.y,
                ):
                    target_x = (
                        entry.core_offset.xy + entry.core_offset.x + copy_offset_x
                    )
                    target_y = (
                        entry.core_offset.xy + entry.core_offset.y + copy_offset_y
                    )
                    _validate_input_target(
                        validation,
                        thread_id,
                        mapping.name,
                        elem_idx,
                        target_x,
                        target_y,
                        tick_relative,
                        addr_axon,
                    )
                    tensor_entries.append(
                        IoEntryView(
                            index=len(input_entries) + len(tensor_entries),
                            direction="input",
                            thread_id=thread_id,
                            tensor_name=mapping.name,
                            elem_idx=elem_idx,
                            tensor_coord=coord,
                            slice_key=slice_key,
                            plane_y=plane_y,
                            plane_x=plane_x,
                            chip_id=CHIP_ID,
                            target_x=target_x,
                            target_y=target_y,
                            copy_offset_x=copy_offset_x,
                            copy_offset_y=copy_offset_y,
                            buffer_row=tick_relative,
                            buffer_bit=addr_axon,
                            full_addr=full_addr,
                            work_timestep=work_timestep,
                            work_axon=work_axon,
                            copy_id=entry.copy_id,
                            target_lcn=target_lcn,
                            dtype=dtype,
                        )
                    )
            input_entries.extend(tensor_entries)
            tensors.append(
                IoTensorView(
                    direction="input",
                    thread_id=thread_id,
                    name=mapping.name,
                    shape=shape,
                    dim_names=_dim_names(shape),
                    bit_width=mapping.bit_width,
                    dtype=dtype,
                    entry_count=len(mapping.entries),
                    expanded_entry_count=len(tensor_entries),
                    plane=plane,
                    slice_keys=_slice_keys(tensor_entries),
                )
            )

        output_target_lcn = (
            thread.output_mappings.target_lcn
            if thread.output_mappings.HasField("target_lcn")
            else None
        )
        for mapping in thread.output_mappings.items:
            shape = list(mapping.shape.size)
            plane = _tensor_plane(shape)
            output_kind = OutputTensorMapping.OutputKind.Name(mapping.kind)
            dtype = _output_dtype(mapping)
            tensor_entries = []
            for entry in mapping.entries:
                elem_idx = entry.elem_idx
                if _validate_elem_idx(
                    validation,
                    "output",
                    thread_id,
                    mapping.name,
                    elem_idx,
                    shape,
                ):
                    continue
                coord = _tensor_coord(elem_idx, shape)
                plane_y, plane_x = _plane_point(coord, plane)
                tensor_entries.append(
                    IoEntryView(
                        index=len(output_entries) + len(tensor_entries),
                        direction="output",
                        thread_id=thread_id,
                        tensor_name=mapping.name,
                        elem_idx=elem_idx,
                        tensor_coord=coord,
                        slice_key=_slice_key(coord, plane),
                        plane_y=plane_y,
                        plane_x=plane_x,
                        axon_bit_idx=entry.axon_bit_idx,
                        copy_id=entry.copy_id,
                        target_lcn=output_target_lcn,
                        dtype=dtype,
                        output_kind=output_kind,
                    )
                )
            output_entries.extend(tensor_entries)
            tensors.append(
                IoTensorView(
                    direction="output",
                    thread_id=thread_id,
                    name=mapping.name,
                    shape=shape,
                    dim_names=_dim_names(shape),
                    bit_width=mapping.bit_width,
                    dtype=dtype,
                    output_kind=output_kind,
                    target_lcn=output_target_lcn,
                    entry_count=len(mapping.entries),
                    expanded_entry_count=len(tensor_entries),
                    plane=plane,
                    slice_keys=_slice_keys(tensor_entries),
                )
            )

    input_regions = _build_regions(input_entries)
    output_regions = _build_regions(output_entries)
    input_buffer_spans = _build_buffer_spans(input_entries)
    core_summaries = _build_core_summaries(input_entries, output_entries)
    _validate_duplicate_input_slots(validation, input_entries)

    return IoView(
        available=bool(tensors),
        tensors=tensors,
        core_summaries=core_summaries,
        input_regions=input_regions,
        output_regions=output_regions,
        input_entries=input_entries,
        output_entries=output_entries,
        input_buffer_spans=input_buffer_spans,
    )


def io_core_summary_map(io_view: IoView) -> dict[tuple[int, int, int], IoCoreSummary]:
    return {
        (summary.chip_id, summary.x, summary.y): summary
        for summary in io_view.core_summaries
    }


def attribute_output_entries_to_cores(
    io_view: IoView, cores: Sequence[CoreView]
) -> IoView:
    """Attach output tensor entries to physical cores when PB omits coordinates.

    Backendv2 output mappings describe host-visible output addresses, but they do
    not always carry a core coordinate. The visualizer infers that coordinate by
    matching output tensor names to decoded core node names and `axon_bit_idx` to
    decoded neuron destination axons.
    """
    output_core_by_addr = _output_core_by_tensor_addr(cores)
    if not output_core_by_addr:
        return io_view

    changed = False
    output_entries: list[IoEntryView] = []
    for entry in io_view.output_entries:
        if entry.axon_bit_idx is None:
            output_entries.append(entry)
            continue
        target = output_core_by_addr.get(
            (entry.thread_id, entry.tensor_name, entry.axon_bit_idx)
        )
        if target is None:
            output_entries.append(entry)
            continue
        chip_id, x, y = target
        output_entries.append(replace(entry, chip_id=chip_id, target_x=x, target_y=y))
        changed = True

    if not changed:
        return io_view

    return replace(
        io_view,
        output_entries=output_entries,
        output_regions=_build_regions(output_entries),
        core_summaries=_build_core_summaries(io_view.input_entries, output_entries),
    )


def _output_core_by_tensor_addr(
    cores: Sequence[CoreView],
) -> dict[tuple[int, str, int], tuple[int, int, int]]:
    result: dict[tuple[int, str, int], tuple[int, int, int]] = {}
    for core in cores:
        if core.thread_id is None or not core.nodes:
            continue
        addrs = _core_neuron_output_addrs(core.neurons.records)
        if not addrs:
            continue
        for node_name in core.nodes:
            for addr in addrs:
                result[(core.thread_id, node_name, addr)] = (
                    core.chip_id,
                    core.x,
                    core.y,
                )
    return result


def _core_neuron_output_addrs(records: Sequence[NeuronRecordView]) -> set[int]:
    addrs: set[int] = set()
    for record in records:
        addr = _decoded_field_int(record.fields.get("dest info", []), "addr_axon")
        if addr is not None:
            addrs.add(addr)
    return addrs


def _decoded_field_int(fields: Sequence[DecodedField], name: str) -> int | None:
    for field in fields:
        if field.name == name and isinstance(field.decoded, int):
            return field.decoded
    return None


def _tensor_plane(shape: list[int]) -> TensorPlane:
    """Project arbitrary-rank tensor coordinates onto a generic 2D plane.

    No layout name such as NCHW/NHWC is assumed. The last two dimensions become
    the display plane and all preceding dimensions become slice selectors.
    """
    if len(shape) >= 2:
        y_dim = len(shape) - 2
        x_dim = len(shape) - 1
        return TensorPlane(
            outer_dims=list(range(0, len(shape) - 2)),
            y_dim=y_dim,
            x_dim=x_dim,
            height=max(shape[y_dim], 1),
            width=max(shape[x_dim], 1),
        )
    return TensorPlane(
        outer_dims=[],
        y_dim=None,
        x_dim=0 if shape else None,
        height=1,
        width=max(shape[0], 1) if shape else 1,
    )


def _dim_names(shape: list[int]) -> list[str]:
    return [f"dim{idx}" for idx, _ in enumerate(shape)]


def _tensor_coord(elem_idx: int, shape: list[int]) -> list[int]:
    if not shape:
        return []
    coord = [0] * len(shape)
    remaining = elem_idx
    for idx in range(len(shape) - 1, -1, -1):
        extent = max(shape[idx], 1)
        coord[idx] = remaining % extent
        remaining //= extent
    return coord


def _plane_point(coord: list[int], plane: TensorPlane) -> tuple[int, int]:
    if plane.y_dim is None:
        return 0, coord[plane.x_dim] if plane.x_dim is not None and coord else 0
    x = coord[plane.x_dim] if plane.x_dim is not None else 0
    y = coord[plane.y_dim]
    return y, x


def _slice_key(coord: list[int], plane: TensorPlane) -> str:
    if not plane.outer_dims:
        return "all"
    return ",".join(f"dim{dim}={coord[dim]}" for dim in plane.outer_dims)


def _slice_keys(entries: list[IoEntryView]) -> list[str]:
    return sorted({entry.slice_key for entry in entries})


def _validate_elem_idx(
    validation: list[ValidationEntry],
    direction: str,
    thread_id: int,
    tensor_name: str,
    elem_idx: int,
    shape: list[int],
) -> bool:
    total = prod(shape) if shape else 1
    if 0 <= elem_idx < total:
        return False
    validation.append(
        ValidationEntry(
            severity="error",
            code=f"{direction}_elem_idx_out_of_range",
            message=(
                f"{direction} tensor {tensor_name} thread {thread_id} elem_idx "
                f"{elem_idx} is outside shape {shape}"
            ),
        )
    )
    return True


def _validate_input_target(
    validation: list[ValidationEntry],
    thread_id: int,
    tensor_name: str,
    elem_idx: int,
    x: int,
    y: int,
    buffer_row: int,
    buffer_bit: int,
) -> None:
    if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
        validation.append(
            ValidationEntry(
                severity="error",
                code="input_core_out_of_grid",
                message=(
                    f"input tensor {tensor_name} thread {thread_id} elem_idx "
                    f"{elem_idx} targets core ({x},{y}) outside 9x9 grid"
                ),
                chip_id=CHIP_ID,
                x=x,
                y=y,
            )
        )
    if not (0 <= buffer_row < INPUT_BUFFER_DEPTH):
        validation.append(
            ValidationEntry(
                severity="error",
                code="input_buffer_row_out_of_range",
                message=(
                    f"input tensor {tensor_name} thread {thread_id} elem_idx "
                    f"{elem_idx} targets input buffer row {buffer_row}, expected 0..255"
                ),
                chip_id=CHIP_ID,
                x=x,
                y=y,
            )
        )
    if not (0 <= buffer_bit < INPUT_BUFFER_WIDTH):
        validation.append(
            ValidationEntry(
                severity="error",
                code="input_buffer_bit_out_of_range",
                message=(
                    f"input tensor {tensor_name} thread {thread_id} elem_idx "
                    f"{elem_idx} targets input buffer bit {buffer_bit}, expected 0..511"
                ),
                chip_id=CHIP_ID,
                x=x,
                y=y,
            )
        )


def _split_input_address(target_lcn: int, full_addr: int) -> tuple[int, int]:
    widths = LCN_TO_TS_AXON_WIDTHS[target_lcn]
    ax_width = widths[1]
    timestep = full_addr >> ax_width
    axon = full_addr & ((1 << ax_width) - 1)
    return timestep, axon


def _input_dtype(mapping) -> str:
    for entry in mapping.entries:
        if entry.HasField("dtype"):
            return DataType.Code.Name(entry.dtype)
    return ""


def _output_dtype(mapping) -> str:
    if mapping.kind == OutputTensorMapping.VOLTAGE:
        return "INT32"
    for entry in mapping.entries:
        if entry.HasField("dtype"):
            return DataType.Code.Name(entry.dtype)
    return ""


def _build_regions(entries: list[IoEntryView]) -> list[TensorRegionView]:
    """Aggregate IO entries into drawable tensor masks per core/tensor/slice."""
    grouped: dict[tuple[Any, ...], list[IoEntryView]] = defaultdict(list)
    for entry in entries:
        grouped[
            (
                entry.direction,
                entry.thread_id,
                entry.tensor_name,
                entry.slice_key,
                entry.chip_id,
                entry.target_x,
                entry.target_y,
                entry.dtype,
                entry.output_kind,
            )
        ].append(entry)

    regions: list[TensorRegionView] = []
    for key, group_entries in grouped.items():
        ys = [entry.plane_y for entry in group_entries]
        xs = [entry.plane_x for entry in group_entries]
        points = {(entry.plane_y, entry.plane_x) for entry in group_entries}
        min_y, max_y = min(ys), max(ys)
        min_x, max_x = min(xs), max(xs)
        bbox_width = max_x - min_x + 1
        bbox_height = max_y - min_y + 1
        bbox_area = bbox_width * bbox_height
        active_count = len(points)
        runs = _runs_from_points(points)
        (
            direction,
            thread_id,
            tensor_name,
            slice_key,
            chip_id,
            target_x,
            target_y,
            dtype,
            output_kind,
        ) = key
        regions.append(
            TensorRegionView(
                direction=direction,
                thread_id=thread_id,
                tensor_name=tensor_name,
                slice_key=slice_key,
                chip_id=chip_id,
                target_x=target_x,
                target_y=target_y,
                dtype=dtype,
                output_kind=output_kind,
                elem_start=min(entry.elem_idx for entry in group_entries),
                elem_end=max(entry.elem_idx for entry in group_entries),
                bbox_min=[min_y, min_x],
                bbox_max=[max_y, max_x],
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                active_element_count=active_count,
                bbox_area=bbox_area,
                coverage_ratio=active_count / bbox_area if bbox_area else 0.0,
                is_rectangular=active_count == bbox_area,
                component_count=_component_count(points),
                buffer_row_min=_min_optional(
                    entry.buffer_row for entry in group_entries
                ),
                buffer_row_max=_max_optional(
                    entry.buffer_row for entry in group_entries
                ),
                buffer_bit_min=_min_optional(
                    entry.buffer_bit for entry in group_entries
                ),
                buffer_bit_max=_max_optional(
                    entry.buffer_bit for entry in group_entries
                ),
                axon_bit_min=_min_optional(
                    entry.axon_bit_idx for entry in group_entries
                ),
                axon_bit_max=_max_optional(
                    entry.axon_bit_idx for entry in group_entries
                ),
                runs=runs,
            )
        )
    return sorted(
        regions,
        key=lambda item: (
            item.direction,
            item.thread_id,
            item.tensor_name,
            item.slice_key,
            item.target_y if item.target_y is not None else -1,
            item.target_x if item.target_x is not None else -1,
        ),
    )


def _runs_from_points(points: set[tuple[int, int]]) -> list[TensorRegionRun]:
    runs: list[TensorRegionRun] = []
    by_y: dict[int, list[int]] = defaultdict(list)
    for y, x in points:
        by_y[y].append(x)
    for y in sorted(by_y):
        values = sorted(by_y[y])
        start = end = values[0]
        for value in values[1:]:
            if value == end + 1:
                end = value
                continue
            runs.append(TensorRegionRun(y=y, x_start=start, x_end=end))
            start = end = value
        runs.append(TensorRegionRun(y=y, x_start=start, x_end=end))
    return runs


def _component_count(points: set[tuple[int, int]]) -> int:
    """Count connected mask components with 4-neighbor connectivity."""
    if not points:
        return 0
    remaining = set(points)
    count = 0
    while remaining:
        count += 1
        start = remaining.pop()
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            y, x = queue.popleft()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return count


def _build_buffer_spans(entries: list[IoEntryView]) -> list[BufferSpanView]:
    """Compress occupied input-buffer bits into row-local contiguous spans."""
    grouped: dict[tuple[int, int, int, int, str, str], list[int]] = defaultdict(list)
    for entry in entries:
        if (
            entry.chip_id is None
            or entry.target_x is None
            or entry.target_y is None
            or entry.buffer_row is None
            or entry.buffer_bit is None
        ):
            continue

        grouped[
            (
                entry.chip_id,
                entry.target_x,
                entry.target_y,
                entry.buffer_row,
                entry.tensor_name,
                entry.dtype,
            )
        ].append(entry.buffer_bit)

    spans: list[BufferSpanView] = []
    for (chip_id, x, y, row, tensor_name, dtype), bits in grouped.items():
        values = sorted(set(bits))
        start = end = values[0]
        count = 1
        for bit in values[1:]:
            if bit == end + 1:
                end = bit
                count += 1
                continue
            spans.append(
                BufferSpanView(
                    chip_id=chip_id,
                    x=x,
                    y=y,
                    row=row,
                    bit_start=start,
                    bit_end=end,
                    count=count,
                    tensor_name=tensor_name,
                    dtype=dtype,
                )
            )
            start = end = bit
            count = 1
        spans.append(
            BufferSpanView(
                chip_id=chip_id,
                x=x,
                y=y,
                row=row,
                bit_start=start,
                bit_end=end,
                count=count,
                tensor_name=tensor_name,
                dtype=dtype,
            )
        )
    return sorted(
        spans,
        key=lambda item: (
            item.chip_id,
            item.y,
            item.x,
            item.tensor_name,
            item.row,
            item.bit_start,
        ),
    )


def _build_core_summaries(
    input_entries: list[IoEntryView], output_entries: list[IoEntryView]
) -> list[IoCoreSummary]:
    core_inputs: dict[tuple[int, int, int], list[IoEntryView]] = defaultdict(list)
    for entry in input_entries:
        if (
            entry.chip_id is not None
            and entry.target_x is not None
            and entry.target_y is not None
        ):
            core_inputs[(entry.chip_id, entry.target_x, entry.target_y)].append(entry)
    core_outputs: dict[tuple[int, int, int], list[IoEntryView]] = defaultdict(list)
    for entry in output_entries:
        if (
            entry.chip_id is not None
            and entry.target_x is not None
            and entry.target_y is not None
        ):
            core_outputs[(entry.chip_id, entry.target_x, entry.target_y)].append(entry)

    summaries: list[IoCoreSummary] = []
    for key in sorted(set(core_inputs) | set(core_outputs)):
        chip_id, x, y = key
        inputs = core_inputs.get(key, [])
        outputs = core_outputs.get(key, [])
        summaries.append(
            IoCoreSummary(
                chip_id=chip_id,
                x=x,
                y=y,
                input_count=len(inputs),
                output_count=len(outputs),
                input_tensors=sorted({entry.tensor_name for entry in inputs}),
                output_tensors=sorted({entry.tensor_name for entry in outputs}),
            )
        )
    return summaries


def _validate_duplicate_input_slots(
    validation: list[ValidationEntry], entries: list[IoEntryView]
) -> None:
    seen: dict[tuple[int, int, int, int, int], IoEntryView] = {}
    for entry in entries:
        if (
            entry.chip_id is None
            or entry.target_x is None
            or entry.target_y is None
            or entry.buffer_row is None
            or entry.buffer_bit is None
        ):
            continue
        key = (
            entry.chip_id,
            entry.target_x,
            entry.target_y,
            entry.buffer_row,
            entry.buffer_bit,
        )
        existing = seen.get(key)
        if existing is None:
            seen[key] = entry
            continue
        validation.append(
            ValidationEntry(
                severity="error",
                code="input_buffer_slot_duplicate",
                message=(
                    f"input buffer slot ({entry.target_x},{entry.target_y}) "
                    f"row {entry.buffer_row} bit {entry.buffer_bit} receives both "
                    f"{existing.tensor_name}[{existing.elem_idx}] and "
                    f"{entry.tensor_name}[{entry.elem_idx}]"
                ),
                chip_id=entry.chip_id,
                x=entry.target_x,
                y=entry.target_y,
            )
        )


def _min_optional(values) -> int | None:
    filtered = [value for value in values if value is not None]
    return min(filtered) if filtered else None


def _max_optional(values) -> int | None:
    filtered = [value for value in values if value is not None]
    return max(filtered) if filtered else None


def _copy_offsets(copy_z: int, copy_x: int, copy_y: int) -> tuple[tuple[int, int], ...]:
    route = decode_aer_route_fields(0, 0, 0, copy_z, copy_x, copy_y)
    return tuple((coord.x, coord.y) for coord in expand_aer_destinations((0, 0), route))
