import { useEffect, useState } from 'react'

import type {
  CoreView,
  DecodedField,
  IoCoreView,
  IoSummary,
  LutEntryView,
  NeuronRecordView,
  RawFrameRecord,
  WeightRecordView,
} from '../../types'
import { getSops } from '../../shared/core'
import { formatMaybe } from '../../shared/format'
import { Pager, RawHexStrip, SimpleKv, SourceFrameButton, TinyMetric } from '../../shared/components'
import type { InspectorTab, IoSelection, MapMode, NonRawInspectorTab, ValueMode } from '../../shared/view-types'
import { IoTab } from '../io-map/IoMap'

const NEURON_PAGE_SIZE = 4
const WEIGHT_PAGE_SIZE = 10
const RAW_PAGE_SIZE = 24

export function Inspector({
  core,
  ioCore,
  ioSummary,
  ioSelection,
  mapMode,
  onIoSelectionChange,
  includePaddingInSops,
  onIncludePaddingInSopsChange,
}: {
  core: CoreView | null
  ioCore: IoCoreView | null
  ioSummary: IoSummary | null
  ioSelection: IoSelection | null
  mapMode: MapMode
  onIoSelectionChange: (selection: IoSelection) => void
  includePaddingInSops: boolean
  onIncludePaddingInSopsChange: (includePadding: boolean) => void
}) {
  const [activeTab, setActiveTab] = useState<InspectorTab>('core')
  const [valueMode, setValueMode] = useState<ValueMode>('semantic')
  const [rawReturnTab, setRawReturnTab] = useState<NonRawInspectorTab>('core')
  const [rawFilters, setRawFilters] = useState<RawFilters>({
    frameIndex: '',
    frameType: 'all',
    packageType: '',
    sramAddress: '',
  })

  useEffect(() => {
    setRawReturnTab('core')
    setRawFilters({ frameIndex: '', frameType: 'all', packageType: '', sramAddress: '' })
    setActiveTab((current) => (current === 'raw' ? 'core' : current))
  }, [core?.chip_id, core?.x, core?.y])

  if (!core)
    return (
      <aside className="inspector">
        <h2>Inspector</h2>
        <p>Select a core.</p>
      </aside>
    )
  const selectedCore = core

  if (mapMode === 'io') {
    // IO Map is a main view mode, not a Core inspector tab. In this mode the
    // right pane becomes an IO-focused inspector instead of showing
    // Core/LUT/Neurons/Weights tabs.
    return (
      <aside className="inspector io-inspector">
        <div className="inspector-title-row">
          <h2>IO Map</h2>
        </div>
        <div className="badge-row">
          <span className="badge">
            core ({selectedCore.x}, {selectedCore.y})
          </span>
          <span className={`badge role-${selectedCore.role}`}>{selectedCore.role}</span>
          {selectedCore.used && <span className="badge">used</span>}
          {selectedCore.thread_id !== null && <span className="badge">thread {selectedCore.thread_id}</span>}
        </div>
        <IoTab
          core={selectedCore}
          ioCore={ioCore}
          ioSummary={ioSummary}
          ioSelection={ioSelection}
          onIoSelectionChange={onIoSelectionChange}
        />
      </aside>
    )
  }

  function goToTab(tab: InspectorTab) {
    if (tab === 'lut' && !selectedCore.lut.present) return
    if (tab === 'raw' && activeTab !== 'raw') {
      setRawReturnTab(activeTab)
    }
    setActiveTab(tab)
  }

  function openRawFrom(frameIndex: number | null, returnTab: NonRawInspectorTab) {
    if (frameIndex !== null) {
      setRawFilters({
        frameIndex: String(frameIndex),
        frameType: 'all',
        packageType: '',
        sramAddress: '',
      })
    }
    setRawReturnTab(returnTab)
    setActiveTab('raw')
  }

  return (
    <aside className="inspector">
      <div className="inspector-title-row">
        <h2>
          Core ({selectedCore.x}, {selectedCore.y})
        </h2>
        <div className="inspector-actions">
          <div className="value-toggle" aria-label="field value mode">
            <button
              type="button"
              className={valueMode === 'semantic' ? 'active' : ''}
              onClick={() => setValueMode('semantic')}
            >
              Semantic
            </button>
            <button type="button" className={valueMode === 'raw' ? 'active' : ''} onClick={() => setValueMode('raw')}>
              Raw
            </button>
          </div>
          <button type="button" className="secondary-action" onClick={() => goToTab('raw')}>
            Raw
          </button>
        </div>
      </div>
      <div className="badge-row">
        <span className={`badge role-${selectedCore.role}`}>{selectedCore.role}</span>
        <span className="badge">{selectedCore.source}</span>
        {selectedCore.used && <span className="badge">used</span>}
        {selectedCore.thread_id !== null && <span className="badge">thread {selectedCore.thread_id}</span>}
      </div>
      <label className="inspector-checkbox">
        <input
          type="checkbox"
          checked={includePaddingInSops}
          onChange={(event) => onIncludePaddingInSopsChange(event.target.checked)}
        />
        Include padding in SOPS
      </label>
      <div className="inspector-tabs" role="tablist">
        <button type="button" className={activeTab === 'core' ? 'active' : ''} onClick={() => goToTab('core')}>
          Core
        </button>
        <button
          type="button"
          className={activeTab === 'lut' ? 'active' : ''}
          disabled={!selectedCore.lut.present}
          onClick={() => goToTab('lut')}
        >
          LUT
        </button>
        <button type="button" className={activeTab === 'neurons' ? 'active' : ''} onClick={() => goToTab('neurons')}>
          Neurons
        </button>
        <button type="button" className={activeTab === 'weights' ? 'active' : ''} onClick={() => goToTab('weights')}>
          Weights
        </button>
      </div>
      {activeTab === 'core' && (
        <CoreTab
          core={selectedCore}
          includePaddingInSops={includePaddingInSops}
          valueMode={valueMode}
          onOpenRaw={(frameIndex) => openRawFrom(frameIndex, 'core')}
        />
      )}
      {activeTab === 'lut' && (
        <LutTab core={selectedCore} valueMode={valueMode} onOpenRaw={(frameIndex) => openRawFrom(frameIndex, 'lut')} />
      )}
      {activeTab === 'neurons' && (
        <NeuronsTab
          core={selectedCore}
          includePaddingInSops={includePaddingInSops}
          valueMode={valueMode}
          onOpenRaw={(frameIndex) => openRawFrom(frameIndex, 'neurons')}
        />
      )}
      {activeTab === 'weights' && <WeightsTab core={selectedCore} valueMode={valueMode} />}
      {activeTab === 'raw' && (
        <RawTab
          core={selectedCore}
          filters={rawFilters}
          onFiltersChange={setRawFilters}
          onExit={() => setActiveTab(rawReturnTab)}
        />
      )}
    </aside>
  )
}

interface RawFilters {
  frameIndex: string
  frameType: string
  packageType: string
  sramAddress: string
}

function CoreTab({
  core,
  includePaddingInSops,
  valueMode,
  onOpenRaw,
}: {
  core: CoreView
  includePaddingInSops: boolean
  valueMode: ValueMode
  onOpenRaw: (frameIndex: number | null) => void
}) {
  const groups = Object.entries(core.decoded_core_config.groups)
  const sops = getSops(core, includePaddingInSops)
  return (
    <>
      <section>
        <div className="inspector-metrics">
          <TinyMetric label="Tick start" value={formatMaybe(core.core_config.tick_start)} />
          <TinyMetric label="Neuron SRAM" value={formatMaybe(core.core_config.neuron_number)} />
          <TinyMetric label="SOPS" value={sops.toLocaleString()} />
        </div>
      </section>
      <section>
        <h3>Nodes</h3>
        <p>{core.nodes.length ? core.nodes.join(', ') : 'None'}</p>
      </section>
      {groups.length ? (
        groups.map(([groupName, fields]) => (
          <section key={groupName}>
            <h3>{groupName}</h3>
            <FieldTable fields={fields} valueMode={valueMode} onOpenRaw={onOpenRaw} />
          </section>
        ))
      ) : (
        <section>
          <h3>Core Config</h3>
          <p className="muted">Offline integer decoder is not applied to this core.</p>
        </section>
      )}
    </>
  )
}

function LutTab({
  core,
  valueMode,
  onOpenRaw,
}: {
  core: CoreView
  valueMode: ValueMode
  onOpenRaw: (frameIndex: number | null) => void
}) {
  if (!core.lut.present) {
    return (
      <section>
        <h3>LUT</h3>
        <p className="muted">No config frame type2 LUT for this core.</p>
      </section>
    )
  }
  return (
    <>
      <section>
        <h3>LUT Summary</h3>
        <SimpleKv entries={core.lut.summary} />
      </section>
      <section>
        <h3>Function</h3>
        <LutChart entries={core.lut.entries} />
      </section>
      <section>
        <h3>Entries</h3>
        <div className="lut-table">
          <div className="lut-row lut-head">
            <span>idx</span>
            <span>potential</span>
            <span>activation</span>
            <span>source</span>
          </div>
          {core.lut.entries.map((entry) => (
            <div className="lut-row" key={entry.index}>
              <span>{entry.index}</span>
              <span>{valueMode === 'raw' ? entry.potential_raw : entry.potential}</span>
              <span>{valueMode === 'raw' ? entry.activation_raw : entry.activation}</span>
              <SourceFrameButton frameIndex={entry.source_frame_index} onOpenRaw={onOpenRaw} />
            </div>
          ))}
        </div>
      </section>
    </>
  )
}

function NeuronsTab({
  core,
  includePaddingInSops,
  valueMode,
  onOpenRaw,
}: {
  core: CoreView
  includePaddingInSops: boolean
  valueMode: ValueMode
  onOpenRaw: (frameIndex: number | null) => void
}) {
  const [page, setPage] = useState(0)
  useEffect(() => setPage(0), [core.chip_id, core.x, core.y])
  const records = core.neurons.records
  const pageCount = Math.max(1, Math.ceil(records.length / NEURON_PAGE_SIZE))
  const pageItems = records.slice(page * NEURON_PAGE_SIZE, (page + 1) * NEURON_PAGE_SIZE)
  const sops = getSops(core, includePaddingInSops)
  return (
    <>
      <section>
        <h3>Neuron Summary</h3>
        <div className="inspector-metrics">
          <TinyMetric label="Total" value={core.neurons.summary.total} />
          <TinyMetric label="Half" value={core.neurons.summary.half_count} />
          <TinyMetric label="Full" value={core.neurons.summary.full_count} />
          <TinyMetric label="Folded" value={core.neurons.summary.folded_count} />
          <TinyMetric label="SOPS" value={sops.toLocaleString()} />
          <TinyMetric label="Weight records" value={core.neurons.summary.weight_sram_pressure.toLocaleString()} />
        </div>
        {Object.keys(core.neurons.summary.weight_compress_counts).length > 0 && (
          <SimpleKv entries={core.neurons.summary.weight_compress_counts} />
        )}
      </section>
      <Pager page={page} pageCount={pageCount} total={records.length} onPage={setPage} />
      <div className="record-list compact-scroll-list">
        {pageItems.map((record) => (
          <NeuronRecord
            key={`${record.index}-${record.sram_address}`}
            record={record}
            valueMode={valueMode}
            onOpenRaw={onOpenRaw}
          />
        ))}
        {!records.length && <p className="muted">No decoded neuron records.</p>}
      </div>
    </>
  )
}

function NeuronRecord({
  record,
  valueMode,
  onOpenRaw,
}: {
  record: NeuronRecordView
  valueMode: ValueMode
  onOpenRaw: (frameIndex: number | null) => void
}) {
  return (
    <section className="record-block">
      <div className="record-title">
        <strong>
          #{record.index} {record.kind}
        </strong>
        <span>SRAM {record.sram_address}</span>
        {record.frame_indices[0] !== undefined && (
          <SourceFrameButton frameIndex={record.frame_indices[0]} onOpenRaw={onOpenRaw} />
        )}
      </div>
      {Object.entries(record.fields).map(([group, fields]) => (
        <div className="record-group" key={group}>
          <h4>{group}</h4>
          <FieldTable fields={fields} valueMode={valueMode} onOpenRaw={onOpenRaw} />
        </div>
      ))}
      {valueMode === 'raw' && <RawHexStrip values={record.raw_hex} />}
    </section>
  )
}

function WeightsTab({ core, valueMode }: { core: CoreView; valueMode: ValueMode }) {
  const [page, setPage] = useState(0)
  useEffect(() => setPage(0), [core.chip_id, core.x, core.y])
  const records = core.weights.records
  const pageCount = Math.max(1, Math.ceil(records.length / WEIGHT_PAGE_SIZE))
  const pageItems = records.slice(page * WEIGHT_PAGE_SIZE, (page + 1) * WEIGHT_PAGE_SIZE)
  return (
    <>
      <section>
        <h3>Weight Summary</h3>
        <div className="inspector-metrics">
          <TinyMetric label="Total" value={core.weights.summary.total} />
          <TinyMetric label="Data type" value={core.weights.summary.data_type || '-'} />
          <TinyMetric label="Dense" value={core.weights.summary.dense_count} />
          <TinyMetric label="CSC" value={core.weights.summary.csc_count} />
          <TinyMetric label="SRAM records" value={core.weights.summary.sram_records} />
          <TinyMetric label="Bits" value={core.weights.summary.bits.toLocaleString()} />
          <TinyMetric label="Bytes" value={core.weights.summary.bytes.toLocaleString()} />
          <TinyMetric label="Nonzero" value={core.weights.summary.nonzero_count.toLocaleString()} />
          <TinyMetric label="Padding" value={core.weights.summary.padding_count.toLocaleString()} />
        </div>
        <p className="muted detail-note">
          Weight SRAM ranges are grouped by neuron address metadata. Dense and CSC storage slots are decoded from SRAM
          records; full logical matrix layout still needs shape and mapping metadata. Padding means CSC zero-value
          storage slots inserted only to align compressed records, not convolution padding.
        </p>
      </section>
      <Pager page={page} pageCount={pageCount} total={records.length} onPage={setPage} />
      <div className="record-list">
        {pageItems.map((record) => (
          <WeightRecord
            key={`${record.index}-${record.start_address}-${record.end_address}`}
            record={record}
            valueMode={valueMode}
          />
        ))}
        {!records.length && <p className="muted">No decoded weight records.</p>}
      </div>
    </>
  )
}

function WeightRecord({ record, valueMode }: { record: WeightRecordView; valueMode: ValueMode }) {
  return (
    <section className="record-block">
      <div className="record-title">
        <strong>
          #{record.index} {record.kind}
        </strong>
        <span>
          SRAM {record.start_address}-{record.end_address}
        </span>
      </div>
      <SimpleKv
        entries={{
          sram_records: record.sram_records,
          bits: record.bits,
          bytes: record.bytes,
          storage_slots: record.storage_value_count,
          nonzero: record.nonzero_count ?? 'n/a',
          csc_padding_slots: record.padding_count ?? 'n/a',
        }}
      />
      <WeightStoragePreview record={record} valueMode={valueMode} />
      {valueMode === 'raw' && <RawHexStrip values={record.raw_hex} />}
    </section>
  )
}

function WeightStoragePreview({ record, valueMode }: { record: WeightRecordView; valueMode: ValueMode }) {
  const storageValueCount = record.storage_value_count ?? 0
  const storagePreviewLimit = record.storage_preview_limit ?? 0
  const truncated = storagePreviewLimit > 0 && storageValueCount > storagePreviewLimit
  if (record.kind === 'sparse') {
    const entries = record.storage_entries ?? []
    return (
      <div className="storage-preview">
        <div className="storage-row storage-head">
          <span>slot</span>
          <span>value</span>
          <span>bit index</span>
          <span>logical index</span>
          <span>padding</span>
        </div>
        {entries.map((entry) => (
          <div className="storage-row" key={`${record.index}-${entry.slot}`}>
            <span>{entry.slot}</span>
            <span>{valueMode === 'raw' ? entry.value_raw : entry.value}</span>
            <span>{entry.bit_index ?? '-'}</span>
            <span>{entry.logical_index ?? '-'}</span>
            <span>{entry.is_padding ? 'yes' : 'no'}</span>
          </div>
        ))}
        {!entries.length && <p className="muted storage-note">No decoded CSC storage slots.</p>}
        {truncated && (
          <p className="muted storage-note">
            Showing first {storagePreviewLimit} of {storageValueCount} slots.
          </p>
        )}
      </div>
    )
  }
  const semanticValues = record.storage_values ?? []
  const rawValues = record.storage_values_raw ?? []
  const values = valueMode === 'raw' ? rawValues : semanticValues
  return (
    <div className="dense-values" aria-label="dense weight storage values">
      {values.map((value, index) => (
        <span key={`${record.index}-${index}`}>
          {index}:{value}
        </span>
      ))}
      {!values.length && <span>No decoded dense storage values.</span>}
      {truncated && <span>... {storageValueCount - storagePreviewLimit} more</span>}
    </div>
  )
}

function RawTab({
  core,
  filters,
  onFiltersChange,
  onExit,
}: {
  core: CoreView
  filters: RawFilters
  onFiltersChange: (filters: RawFilters) => void
  onExit: () => void
}) {
  const [page, setPage] = useState(0)
  useEffect(
    () => setPage(0),
    [core.chip_id, core.x, core.y, filters.frameIndex, filters.frameType, filters.packageType, filters.sramAddress],
  )
  const records = filterRawFrames(core.raw_frames, filters)
  const pageCount = Math.max(1, Math.ceil(records.length / RAW_PAGE_SIZE))
  const pageItems = records.slice(page * RAW_PAGE_SIZE, (page + 1) * RAW_PAGE_SIZE)
  return (
    <>
      <section>
        <div className="raw-title-row">
          <h3>Raw Frames</h3>
          <button type="button" className="secondary-action" onClick={onExit}>
            Back
          </button>
        </div>
        <div className="raw-help">
          <p>
            Raw Frames lists decoded payload words for the selected core. <strong>frame</strong> is the absolute frame
            index, <strong>type</strong> is config frame type 1/2/3, <strong>pkg</strong> is the package type from the
            package header, <strong>sram</strong> is the derived SRAM record address when applicable, and{' '}
            <strong>raw</strong> is the original 64-bit payload word.
          </p>
          <p>
            Filters are exact matches: frame type, frame index, SRAM address, and package type. Empty fields mean no
            filter on that column.
          </p>
        </div>
        <div className="raw-filters">
          <label>
            Frame type
            <select
              value={filters.frameType}
              onChange={(event) => onFiltersChange({ ...filters, frameType: event.target.value })}
            >
              <option value="all">all</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
            </select>
          </label>
          <label>
            Frame index
            <input
              value={filters.frameIndex}
              onChange={(event) => onFiltersChange({ ...filters, frameIndex: event.target.value })}
              inputMode="numeric"
            />
          </label>
          <label>
            SRAM address
            <input
              value={filters.sramAddress}
              onChange={(event) => onFiltersChange({ ...filters, sramAddress: event.target.value })}
              inputMode="numeric"
            />
          </label>
          <label>
            Package type
            <input
              value={filters.packageType}
              onChange={(event) => onFiltersChange({ ...filters, packageType: event.target.value })}
              inputMode="numeric"
            />
          </label>
        </div>
      </section>
      <Pager page={page} pageCount={pageCount} total={records.length} onPage={setPage} />
      <div className="raw-table">
        <div className="raw-row raw-head">
          <span>frame</span>
          <span>type</span>
          <span>pkg</span>
          <span>sram</span>
          <span>raw</span>
        </div>
        {pageItems.map((record) => (
          <div className="raw-row" key={`${record.frame_index}-${record.word_offset}`}>
            <span>{record.frame_index}</span>
            <span>{record.frame_type}</span>
            <span>{record.package_type}</span>
            <span>{record.sram_address ?? '-'}</span>
            <span>{record.raw_hex}</span>
          </div>
        ))}
        {!records.length && <p className="muted">No raw frames match the filters.</p>}
      </div>
    </>
  )
}

function FieldTable({
  fields,
  valueMode,
  onOpenRaw,
}: {
  fields: DecodedField[]
  valueMode: ValueMode
  onOpenRaw: (frameIndex: number | null) => void
}) {
  return (
    <div className="field-table">
      {fields.map((field) => (
        <div className="field-row" key={field.name}>
          <div>
            <strong>{field.name}</strong>
            <span>{field.description}</span>
          </div>
          <code>{formatFieldValue(field, valueMode)}</code>
          <SourceFrameButton frameIndex={field.source_frame_index} onOpenRaw={onOpenRaw} />
        </div>
      ))}
    </div>
  )
}

function LutChart({ entries }: { entries: LutEntryView[] }) {
  if (!entries.length) return <p className="muted">No LUT entries.</p>
  const width = 560
  const height = 260
  const pad = 30
  const potentials = entries.map((entry) => entry.potential)
  const activations = entries.map((entry) => entry.activation)
  const minX = Math.min(...potentials)
  const maxX = Math.max(...potentials)
  const minY = Math.min(...activations)
  const maxY = Math.max(...activations)
  const xSpan = Math.max(maxX - minX, 1)
  const ySpan = Math.max(maxY - minY, 1)
  const chartEntries = [...entries]
    .sort((a, b) => a.potential - b.potential)
    .map((entry) => {
      const x = pad + ((entry.potential - minX) / xSpan) * (width - pad * 2)
      const y = height - pad - ((entry.activation - minY) / ySpan) * (height - pad * 2)
      return { ...entry, x, y }
    })
  const points = chartEntries.map((entry) => `${entry.x.toFixed(1)},${entry.y.toFixed(1)}`).join(' ')
  return (
    <svg
      className="lut-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="LUT potential to activation function"
    >
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} />
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} />
      <polyline points={points} />
      <g className="lut-points">
        {chartEntries.map((entry) => (
          <circle key={entry.index} cx={entry.x} cy={entry.y} r={1.45}>
            <title>{`idx ${entry.index}: ${entry.potential} -> ${entry.activation}`}</title>
          </circle>
        ))}
      </g>
      <text x={pad} y={height - 4}>
        {minX}
      </text>
      <text x={width - pad - 60} y={height - 4}>
        {maxX}
      </text>
      <text x={5} y={pad + 4}>
        {maxY}
      </text>
      <text x={5} y={height - pad}>
        {minY}
      </text>
    </svg>
  )
}

function filterRawFrames(records: RawFrameRecord[], filters: RawFilters): RawFrameRecord[] {
  const frameType = parseFilterInt(filters.frameType)
  const frameIndex = parseFilterInt(filters.frameIndex)
  const sramAddress = parseFilterInt(filters.sramAddress)
  const packageType = parseFilterInt(filters.packageType)
  return records.filter((record) => {
    if (frameType !== null && record.frame_type !== frameType) return false
    if (frameIndex !== null && record.frame_index !== frameIndex) return false
    if (sramAddress !== null && record.sram_address !== sramAddress) return false
    if (packageType !== null && record.package_type !== packageType) return false
    return true
  })
}

function parseFilterInt(value: string): number | null {
  if (value === 'all' || value.trim() === '') return null
  const parsed = Number(value)
  return Number.isInteger(parsed) ? parsed : Number.NaN
}

function formatFieldValue(field: DecodedField, valueMode: ValueMode): string {
  if (valueMode === 'raw') {
    return String(field.raw)
  }
  if (field.label) {
    return field.label.replace(/\s+\([^)]*\)$/u, '')
  }
  return String(field.decoded ?? '')
}
