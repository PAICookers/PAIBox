export type CoreRole = 'cpu' | 'online' | 'offline'
export type IoDirection = 'input' | 'output'
export type ValidationSeverity = 'info' | 'warning' | 'error'
export type NumericKind = 'uint' | 'int' | 'enum' | 'bitset' | 'sign_magnitude' | 'hex' | 'float32' | 'bf16'

// These interfaces intentionally mirror `paibox.visualizer.model` dataclasses.
// Keep snake_case field names so the UI can consume backend JSON directly and
// preserve source-frame provenance without a conversion layer.

export interface ValidationEntry {
  severity: ValidationSeverity
  code: string
  message: string
  chip_id: number | null
  x: number | null
  y: number | null
}

export interface FramePackageSummary {
  frame_type: number
  start_addr: number
  package_type: number
  package_count: number
  frame_start: number
  frame_end: number
  header: number | null
  header_hex: string | null
  payload_hex: string[]
}

export interface DecodedField {
  name: string
  raw: number | string
  decoded: number | string | null
  label: string
  description: string
  numeric_kind: NumericKind
  source_frame_index: number | null
}

export interface CoreConfigView {
  groups: Record<string, DecodedField[]>
}

export interface LutEntryView {
  index: number
  potential_raw: number
  potential: number
  activation_raw: number
  activation: number
  raw_hex: string
  source_frame_index: number | null
}

export interface LutView {
  present: boolean
  entries: LutEntryView[]
  summary: Record<string, number | string>
}

export interface NeuronDestinationView {
  target_x: number
  target_y: number
  copy_offset_x: number
  copy_offset_y: number
}

export interface NeuronRecordView {
  index: number
  kind: string
  sram_address: number
  frame_indices: number[]
  fields: Record<string, DecodedField[]>
  destinations: NeuronDestinationView[]
  raw_hex: string[]
}

export interface NeuronSummaryView {
  total: number
  half_count: number
  full_count: number
  folded_count: number
  weight_compress_counts: Record<string, number>
  synops_pressure: number
  sops_with_padding: number
  sops_without_padding: number
  weight_sram_pressure: number
}

export interface NeuronView {
  summary: NeuronSummaryView
  records: NeuronRecordView[]
}

export interface WeightStorageEntryView {
  slot: number
  value_raw: number
  value: number
  bit_index: number | null
  logical_index: number | null
  is_padding: boolean
  source_frame_index: number | null
}

export interface WeightRecordView {
  index: number
  kind: string
  start_address: number
  end_address: number
  sram_records: number
  bits: number
  bytes: number
  nonzero_count: number | null
  padding_count: number | null
  storage_value_count: number
  storage_preview_limit: number
  storage_values: number[]
  storage_values_raw: number[]
  storage_entries: WeightStorageEntryView[]
  frame_indices: number[]
  raw_hex: string[]
}

export interface WeightSummaryView {
  total: number
  data_type: string
  dense_count: number
  csc_count: number
  sram_records: number
  bits: number
  bytes: number
  nonzero_count: number
  padding_count: number
}

export interface WeightView {
  summary: WeightSummaryView
  records: WeightRecordView[]
}

export interface RawFrameRecord {
  frame_index: number
  frame_type: number
  start_addr: number
  package_type: number
  sram_address: number | null
  word_offset: number
  raw_hex: string
}

export interface CoreFrameSummary {
  frame_type1_count: number
  frame_type2_count: number
  frame_type3_count: number
  package_count: number
}

export interface RoutePointView {
  x: number
  y: number
}

export interface ControlPathView {
  thread_id: number
  target_x: number
  target_y: number
  offset_xy: number
  offset_x: number
  offset_y: number
  points: RoutePointView[]
  source_frame_index: number | null
}

export interface GlobalSignal {
  send_bits: number
  receive_bits: number
  send_dirs: string[]
  receive_dirs: string[]
  sends_local: boolean
  is_source: boolean
  source_thread_ids: number[]
  control_paths: ControlPathView[]
}

export interface TensorPlane {
  outer_dims: number[]
  y_dim: number | null
  x_dim: number | null
  height: number
  width: number
}

export interface TensorRegionRun {
  y: number
  x_start: number
  x_end: number
}

export interface TensorRegionView {
  direction: IoDirection
  thread_id: number
  tensor_name: string
  slice_key: string
  chip_id: number | null
  target_x: number | null
  target_y: number | null
  dtype: string
  output_kind: string
  elem_start: number
  elem_end: number
  bbox_min: number[]
  bbox_max: number[]
  bbox_width: number
  bbox_height: number
  active_element_count: number
  bbox_area: number
  coverage_ratio: number
  is_rectangular: boolean
  component_count: number
  buffer_row_min: number | null
  buffer_row_max: number | null
  buffer_bit_min: number | null
  buffer_bit_max: number | null
  axon_bit_min: number | null
  axon_bit_max: number | null
  runs: TensorRegionRun[]
}

export interface BufferSpanView {
  chip_id: number
  x: number
  y: number
  row: number
  bit_start: number
  bit_end: number
  count: number
  tensor_name: string
  dtype: string
}

export interface IoEntryView {
  index: number
  direction: IoDirection
  thread_id: number
  tensor_name: string
  elem_idx: number
  tensor_coord: number[]
  slice_key: string
  plane_y: number
  plane_x: number
  chip_id: number | null
  target_x: number | null
  target_y: number | null
  copy_offset_x: number
  copy_offset_y: number
  buffer_row: number | null
  buffer_bit: number | null
  full_addr: number | null
  work_timestep: number | null
  work_axon: number | null
  axon_bit_idx: number | null
  copy_id: number
  target_lcn: number | null
  dtype: string
  output_kind: string
}

export interface IoTensorView {
  direction: IoDirection
  thread_id: number
  name: string
  shape: number[]
  dim_names: string[]
  bit_width: number
  dtype: string
  output_kind: string
  target_lcn: number | null
  entry_count: number
  expanded_entry_count: number
  plane: TensorPlane
  slice_keys: string[]
}

export interface IoCoreSummary {
  chip_id: number
  x: number
  y: number
  input_count: number
  output_count: number
  input_tensors: string[]
  output_tensors: string[]
}

export interface IoCoreView {
  chip_id: number
  x: number
  y: number
  summary: IoCoreSummary
  input_regions: TensorRegionView[]
  output_regions: TensorRegionView[]
  input_buffer_spans: BufferSpanView[]
  input_entries: IoEntryView[]
  output_entries: IoEntryView[]
}

export interface IoSummary {
  available: boolean
  tensors: IoTensorView[]
  core_summaries: IoCoreSummary[]
}

export interface CoreOverview {
  chip_id: number
  x: number
  y: number
  role: CoreRole
  used: boolean
  source: string
  nodes: string[]
  thread_id: number | null
  core_config: Record<string, number | string>
  io_summary: IoCoreSummary | null
  global_signal: GlobalSignal
  frames: CoreFrameSummary
  neurons: Pick<NeuronView, 'summary'>
}

export interface CoreView extends CoreOverview {
  packages: FramePackageSummary[]
  decoded_core_config: CoreConfigView
  lut: LutView
  neurons: NeuronView
  weights: WeightView
  raw_frames: RawFrameRecord[]
  validation: ValidationEntry[]
}

export interface LinkView {
  chip_id: number
  source: { x: number; y: number }
  target: { x: number; y: number }
  kind: string
  direction: string
}

export interface ChipView {
  chip_id: number
  grid_width: number
  grid_height: number
  cores: CoreView[]
}

export interface ChipOverview {
  chip_id: number
  grid_width: number
  grid_height: number
  cores: CoreOverview[]
}

export interface ViewerSummary {
  chip_count: number
  core_count: number
  used_core_count: number
  frame_count: number
  package_count: number
  validation_error_count: number
  validation_warning_count: number
}

export interface PageResponse<T> {
  offset: number
  limit: number
  total: number
  items: T[]
}
