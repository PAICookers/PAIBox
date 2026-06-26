import pytest

from paibox.backendv2.proto.compile_artifacts_pb2 import (
    CompileArtifacts,
    DataType,
    OutputTensorMapping,
)
from paibox.visualizer.backends.v2.io_mapping import (
    attribute_output_entries_to_cores,
    build_io_view,
)
from paibox.visualizer.model import (
    CoreView,
    DecodedField,
    NeuronRecordView,
    NeuronView,
    ValidationEntry,
)


def test_input_tensor_coords_regions_and_buffer_spans() -> None:
    artifacts = CompileArtifacts()
    thread = artifacts.io_mapping.threads.add()
    thread.thread_id = 0
    mapping = thread.input_mappings.items.add()
    mapping.name = "input0"
    mapping.shape.size.extend([1, 2, 4, 5])
    mapping.bit_width = 8

    for elem_idx, addr_axon in ((5, 10), (6, 11), (10, 12), (11, 13)):
        entry = mapping.entries.add()
        entry.elem_idx = elem_idx
        entry.core_offset.xy = 0
        entry.core_offset.x = 2
        entry.core_offset.y = 3
        entry.tick_relative = 7
        entry.addr_axon = addr_axon
        entry.target_lcn = 0
        entry.dtype = DataType.UINT8

    validation: list[ValidationEntry] = []
    io_view = build_io_view(artifacts, validation)

    assert not validation
    assert io_view.available
    tensor = io_view.tensors[0]
    assert tensor.shape == [1, 2, 4, 5]
    assert tensor.dim_names == ["dim0", "dim1", "dim2", "dim3"]
    assert tensor.plane.outer_dims == [0, 1]
    assert tensor.plane.height == 4
    assert tensor.plane.width == 5
    assert tensor.slice_keys == ["dim0=0,dim1=0"]

    region = io_view.input_regions[0]
    assert region.bbox_min == [1, 0]
    assert region.bbox_max == [2, 1]
    assert region.bbox_width == 2
    assert region.bbox_height == 2
    assert region.active_element_count == 4
    assert region.bbox_area == 4
    assert region.coverage_ratio == pytest.approx(1.0)
    assert region.is_rectangular
    assert region.component_count == 1
    assert region.buffer_row_min == 7
    assert region.buffer_bit_min == 10
    assert region.buffer_bit_max == 13
    assert [(run.y, run.x_start, run.x_end) for run in region.runs] == [
        (1, 0, 1),
        (2, 0, 1),
    ]

    spans = io_view.input_buffer_spans
    assert len(spans) == 1
    assert spans[0].row == 7
    assert spans[0].bit_start == 10
    assert spans[0].bit_end == 13


def test_output_tensor_region_marks_data_and_voltage() -> None:
    artifacts = CompileArtifacts()
    thread = artifacts.io_mapping.threads.add()
    thread.thread_id = 0
    thread.output_mappings.target_lcn = 0
    data_mapping = thread.output_mappings.items.add()
    data_mapping.name = "out_data"
    data_mapping.shape.size.extend([1, 4])
    data_mapping.kind = OutputTensorMapping.DATA
    data_mapping.bit_width = 4
    for elem_idx in range(4):
        entry = data_mapping.entries.add()
        entry.elem_idx = elem_idx
        entry.axon_bit_idx = elem_idx
        entry.dtype = DataType.UINT4

    voltage_mapping = thread.output_mappings.items.add()
    voltage_mapping.name = "out_v"
    voltage_mapping.shape.size.extend([2])
    voltage_mapping.kind = OutputTensorMapping.VOLTAGE
    voltage_mapping.bit_width = 32
    entry = voltage_mapping.entries.add()
    entry.elem_idx = 1
    entry.axon_bit_idx = 16

    validation: list[ValidationEntry] = []
    io_view = build_io_view(artifacts, validation)

    assert not validation
    data_region = next(
        region for region in io_view.output_regions if region.tensor_name == "out_data"
    )
    assert data_region.output_kind == "DATA"
    assert data_region.axon_bit_min == 0
    assert data_region.axon_bit_max == 3
    voltage_region = next(
        region for region in io_view.output_regions if region.tensor_name == "out_v"
    )
    assert voltage_region.output_kind == "VOLTAGE"
    assert voltage_region.dtype == "INT32"
    assert voltage_region.axon_bit_min == 16
    assert voltage_region.axon_bit_max == 16


def test_output_tensor_entries_are_attributed_to_output_core() -> None:
    artifacts = CompileArtifacts()
    thread = artifacts.io_mapping.threads.add()
    thread.thread_id = 0
    mapping = thread.output_mappings.items.add()
    mapping.name = "out_data"
    mapping.shape.size.extend([1, 4])
    mapping.kind = OutputTensorMapping.DATA
    mapping.bit_width = 4
    for elem_idx in range(4):
        entry = mapping.entries.add()
        entry.elem_idx = elem_idx
        entry.axon_bit_idx = elem_idx
        entry.dtype = DataType.UINT4

    io_view = build_io_view(artifacts, [])
    output_core = CoreView(
        chip_id=0,
        x=6,
        y=2,
        role="offline",
        used=True,
        nodes=["out_data"],
        thread_id=0,
        neurons=NeuronView(
            records=[
                NeuronRecordView(
                    index=idx,
                    kind="full",
                    sram_address=idx,
                    frame_indices=[],
                    fields={
                        "dest info": [
                            DecodedField(
                                name="addr_axon",
                                raw=idx,
                                decoded=idx,
                                label=str(idx),
                            )
                        ]
                    },
                    destinations=[],
                    raw_hex=[],
                )
                for idx in range(4)
            ]
        ),
    )

    attributed = attribute_output_entries_to_cores(io_view, [output_core])

    assert {
        (entry.target_x, entry.target_y) for entry in attributed.output_entries
    } == {(6, 2)}
    summary = attributed.core_summaries[0]
    assert (summary.x, summary.y) == (6, 2)
    assert summary.output_count == 4
    assert summary.output_tensors == ["out_data"]
    region = attributed.output_regions[0]
    assert (region.target_x, region.target_y) == (6, 2)
    assert region.active_element_count == 4


def test_input_mapping_validation_for_bounds_and_duplicates() -> None:
    artifacts = CompileArtifacts()
    thread = artifacts.io_mapping.threads.add()
    thread.thread_id = 0
    mapping = thread.input_mappings.items.add()
    mapping.name = "bad_input"
    mapping.shape.size.extend([3])
    mapping.bit_width = 8

    first = mapping.entries.add()
    first.elem_idx = 0
    first.core_offset.x = 1
    first.core_offset.y = 1
    first.tick_relative = 2
    first.addr_axon = 8
    first.target_lcn = 0
    first.dtype = DataType.UINT8

    duplicate = mapping.entries.add()
    duplicate.elem_idx = 1
    duplicate.core_offset.x = 1
    duplicate.core_offset.y = 1
    duplicate.tick_relative = 2
    duplicate.addr_axon = 8
    duplicate.target_lcn = 0
    duplicate.dtype = DataType.UINT8

    out_of_range = mapping.entries.add()
    out_of_range.elem_idx = 5
    out_of_range.core_offset.x = 10
    out_of_range.tick_relative = 300
    out_of_range.addr_axon = 512
    out_of_range.target_lcn = 0
    out_of_range.dtype = DataType.UINT8

    validation: list[ValidationEntry] = []
    build_io_view(artifacts, validation)

    codes = {item.code for item in validation}
    assert "input_elem_idx_out_of_range" in codes
    assert "input_buffer_slot_duplicate" in codes
