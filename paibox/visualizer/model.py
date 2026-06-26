from dataclasses import asdict, dataclass, field
from typing import Any, Literal

CoreRole = Literal["cpu", "online", "offline"]
IoDirection = Literal["input", "output"]
ValidationSeverity = Literal["info", "warning", "error"]
NumericKind = Literal[
    "uint", "int", "enum", "bitset", "sign_magnitude", "hex", "float32", "bf16"
]


@dataclass(frozen=True)
class ArtifactInfo:
    path: str
    kind: str
    has_pb: bool = False
    has_json: bool = False
    has_merged_npy: bool = False
    has_typed_npy: bool = False


@dataclass(frozen=True)
class ValidationEntry:
    severity: ValidationSeverity
    code: str
    message: str
    chip_id: int | None = None
    x: int | None = None
    y: int | None = None


@dataclass(frozen=True)
class RuntimeInfo:
    thread_id: int
    root_core_offset: dict[str, int]
    runtime: dict[str, int | str]
    input_count: int
    output_count: int
    core_tick_count: int


@dataclass(frozen=True)
class TensorPlane:
    outer_dims: list[int] = field(default_factory=list)
    y_dim: int | None = None
    x_dim: int | None = None
    height: int = 1
    width: int = 1


@dataclass(frozen=True)
class IoEntryView:
    index: int
    direction: IoDirection
    thread_id: int
    tensor_name: str
    elem_idx: int
    tensor_coord: list[int]
    slice_key: str
    plane_y: int
    plane_x: int
    chip_id: int | None = None
    target_x: int | None = None
    target_y: int | None = None
    copy_offset_x: int = 0
    copy_offset_y: int = 0
    buffer_row: int | None = None
    buffer_bit: int | None = None
    full_addr: int | None = None
    work_timestep: int | None = None
    work_axon: int | None = None
    axon_bit_idx: int | None = None
    copy_id: int = 0
    target_lcn: int | None = None
    dtype: str = ""
    output_kind: str = ""


@dataclass(frozen=True)
class TensorRegionRun:
    y: int
    x_start: int
    x_end: int


@dataclass(frozen=True)
class TensorRegionView:
    direction: IoDirection
    thread_id: int
    tensor_name: str
    slice_key: str
    chip_id: int | None = None
    target_x: int | None = None
    target_y: int | None = None
    dtype: str = ""
    output_kind: str = ""
    elem_start: int = 0
    elem_end: int = 0
    bbox_min: list[int] = field(default_factory=list)
    bbox_max: list[int] = field(default_factory=list)
    bbox_width: int = 0
    bbox_height: int = 0
    active_element_count: int = 0
    bbox_area: int = 0
    coverage_ratio: float = 0.0
    is_rectangular: bool = False
    component_count: int = 0
    buffer_row_min: int | None = None
    buffer_row_max: int | None = None
    buffer_bit_min: int | None = None
    buffer_bit_max: int | None = None
    axon_bit_min: int | None = None
    axon_bit_max: int | None = None
    runs: list[TensorRegionRun] = field(default_factory=list)


@dataclass(frozen=True)
class BufferSpanView:
    chip_id: int
    x: int
    y: int
    row: int
    bit_start: int
    bit_end: int
    count: int
    tensor_name: str
    dtype: str


@dataclass(frozen=True)
class IoTensorView:
    direction: IoDirection
    thread_id: int
    name: str
    shape: list[int]
    dim_names: list[str]
    bit_width: int
    dtype: str
    output_kind: str = ""
    target_lcn: int | None = None
    entry_count: int = 0
    expanded_entry_count: int = 0
    plane: TensorPlane = field(default_factory=TensorPlane)
    slice_keys: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IoCoreSummary:
    chip_id: int
    x: int
    y: int
    input_count: int = 0
    output_count: int = 0
    input_tensors: list[str] = field(default_factory=list)
    output_tensors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IoCoreView:
    chip_id: int
    x: int
    y: int
    summary: IoCoreSummary
    input_regions: list[TensorRegionView] = field(default_factory=list)
    output_regions: list[TensorRegionView] = field(default_factory=list)
    input_buffer_spans: list[BufferSpanView] = field(default_factory=list)
    input_entries: list[IoEntryView] = field(default_factory=list)
    output_entries: list[IoEntryView] = field(default_factory=list)


@dataclass(frozen=True)
class IoView:
    available: bool = False
    tensors: list[IoTensorView] = field(default_factory=list)
    core_summaries: list[IoCoreSummary] = field(default_factory=list)
    input_regions: list[TensorRegionView] = field(default_factory=list)
    output_regions: list[TensorRegionView] = field(default_factory=list)
    input_entries: list[IoEntryView] = field(default_factory=list)
    output_entries: list[IoEntryView] = field(default_factory=list)
    input_buffer_spans: list[BufferSpanView] = field(default_factory=list)


@dataclass(frozen=True)
class FramePackageSummary:
    frame_type: int
    start_addr: int
    package_type: int
    package_count: int
    frame_start: int
    frame_end: int
    header: int | None = None
    header_hex: str | None = None
    payload_hex: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DecodedField:
    name: str
    raw: int | str
    decoded: int | str | None
    label: str
    description: str = ""
    numeric_kind: NumericKind = "uint"
    source_frame_index: int | None = None


@dataclass(frozen=True)
class CoreConfigView:
    groups: dict[str, list[DecodedField]] = field(default_factory=dict)


@dataclass(frozen=True)
class LutEntryView:
    index: int
    potential_raw: int
    potential: int
    activation_raw: int
    activation: int
    raw_hex: str
    source_frame_index: int | None = None


@dataclass(frozen=True)
class LutView:
    present: bool = False
    entries: list[LutEntryView] = field(default_factory=list)
    summary: dict[str, int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class NeuronDestinationView:
    target_x: int
    target_y: int
    copy_offset_x: int = 0
    copy_offset_y: int = 0


@dataclass(frozen=True)
class NeuronRecordView:
    index: int
    kind: str
    sram_address: int
    frame_indices: list[int] = field(default_factory=list)
    fields: dict[str, list[DecodedField]] = field(default_factory=dict)
    destinations: list[NeuronDestinationView] = field(default_factory=list)
    raw_hex: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NeuronSummaryView:
    total: int = 0
    half_count: int = 0
    full_count: int = 0
    folded_count: int = 0
    weight_compress_counts: dict[str, int] = field(default_factory=dict)
    synops_pressure: int = 0
    sops_with_padding: int = 0
    sops_without_padding: int = 0
    weight_sram_pressure: int = 0


@dataclass(frozen=True)
class NeuronView:
    summary: NeuronSummaryView = field(default_factory=NeuronSummaryView)
    records: list[NeuronRecordView] = field(default_factory=list)


@dataclass(frozen=True)
class WeightStorageEntryView:
    slot: int
    value_raw: int
    value: int
    bit_index: int | None = None
    logical_index: int | None = None
    is_padding: bool = False
    source_frame_index: int | None = None


@dataclass(frozen=True)
class WeightRecordView:
    index: int
    kind: str
    start_address: int
    end_address: int
    sram_records: int
    bits: int
    bytes: int
    nonzero_count: int | None = None
    padding_count: int | None = None
    storage_value_count: int = 0
    storage_preview_limit: int = 0
    storage_values: list[int] = field(default_factory=list)
    storage_values_raw: list[int] = field(default_factory=list)
    storage_entries: list[WeightStorageEntryView] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    raw_hex: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WeightSummaryView:
    total: int = 0
    data_type: str = ""
    dense_count: int = 0
    csc_count: int = 0
    sram_records: int = 0
    bits: int = 0
    bytes: int = 0
    nonzero_count: int = 0
    padding_count: int = 0


@dataclass(frozen=True)
class WeightView:
    summary: WeightSummaryView = field(default_factory=WeightSummaryView)
    records: list[WeightRecordView] = field(default_factory=list)


@dataclass(frozen=True)
class RawFrameRecord:
    frame_index: int
    frame_type: int
    start_addr: int
    package_type: int
    sram_address: int | None
    word_offset: int
    raw_hex: str


@dataclass(frozen=True)
class CoreFrameSummary:
    frame_type1_count: int = 0
    frame_type2_count: int = 0
    frame_type3_count: int = 0
    package_count: int = 0


@dataclass(frozen=True)
class RoutePointView:
    x: int
    y: int


@dataclass(frozen=True)
class ControlPathView:
    thread_id: int
    target_x: int
    target_y: int
    offset_xy: int
    offset_x: int
    offset_y: int
    points: list[RoutePointView] = field(default_factory=list)
    source_frame_index: int | None = None


@dataclass(frozen=True)
class GlobalSignal:
    send_bits: int = 0
    receive_bits: int = 0
    send_dirs: list[str] = field(default_factory=list)
    receive_dirs: list[str] = field(default_factory=list)
    sends_local: bool = False
    is_source: bool = False
    source_thread_ids: list[int] = field(default_factory=list)
    control_paths: list[ControlPathView] = field(default_factory=list)


@dataclass(frozen=True)
class CoreView:
    chip_id: int
    x: int
    y: int
    role: CoreRole
    used: bool = False
    source: str = "layout"
    nodes: list[str] = field(default_factory=list)
    thread_id: int | None = None
    core_config: dict[str, int] = field(default_factory=dict)
    io_summary: IoCoreSummary | None = None
    global_signal: GlobalSignal = field(default_factory=GlobalSignal)
    frames: CoreFrameSummary = field(default_factory=CoreFrameSummary)
    packages: list[FramePackageSummary] = field(default_factory=list)
    decoded_core_config: CoreConfigView = field(default_factory=CoreConfigView)
    lut: LutView = field(default_factory=LutView)
    neurons: NeuronView = field(default_factory=NeuronView)
    weights: WeightView = field(default_factory=WeightView)
    raw_frames: list[RawFrameRecord] = field(default_factory=list)
    validation: list[ValidationEntry] = field(default_factory=list)


@dataclass(frozen=True)
class LinkView:
    chip_id: int
    source: dict[str, int]
    target: dict[str, int]
    kind: str
    direction: str


@dataclass(frozen=True)
class ChipView:
    chip_id: int
    grid_width: int
    grid_height: int
    cores: list[CoreView]


@dataclass(frozen=True)
class ViewerSummary:
    chip_count: int
    core_count: int
    used_core_count: int
    frame_count: int
    package_count: int
    validation_error_count: int
    validation_warning_count: int


@dataclass(frozen=True)
class ViewerModel:
    """Serializable app-level schema consumed by the React visualizer.

    Backend adapters should translate their native artifacts into this model so
    the UI can remain backend-agnostic.
    """

    schema_version: int
    artifact: ArtifactInfo
    chips: list[ChipView]
    links: list[LinkView]
    io: IoView
    validation: list[ValidationEntry]
    runtime: list[RuntimeInfo]
    summary: ViewerSummary

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
