import { useEffect, useState } from 'react'

import { fetchIoRegions } from '../../api'
import type {
  BufferSpanView,
  CoreView,
  IoCoreView,
  IoDirection,
  IoEntryView,
  IoSummary,
  IoTensorView,
  TensorRegionView,
} from '../../types'
import { Pager, TinyMetric } from '../../shared/components'
import type { IoSelection } from '../../shared/view-types'

const IO_ENTRY_PAGE_SIZE = 12

function findIoTensor(ioSummary: IoSummary | null, selection: IoSelection | null): IoTensorView | null {
  if (!ioSummary || !selection) return null
  return (
    ioSummary.tensors.find(
      (tensor) => tensor.direction === selection.direction && tensor.name === selection.tensorName,
    ) ?? null
  )
}

function regionMatchesSelection(region: TensorRegionView, selection: IoSelection | null): boolean {
  if (!selection) return false
  return (
    region.direction === selection.direction &&
    region.tensor_name === selection.tensorName &&
    region.slice_key === selection.sliceKey
  )
}

function entryMatchesSelection(entry: IoEntryView, selection: IoSelection | null): boolean {
  if (!selection) return false
  return (
    entry.direction === selection.direction &&
    entry.tensor_name === selection.tensorName &&
    entry.slice_key === selection.sliceKey
  )
}

function regionKey(region: TensorRegionView, index: number): string {
  return [
    region.direction,
    region.thread_id,
    region.tensor_name,
    region.slice_key,
    region.target_x,
    region.target_y,
    index,
  ].join(':')
}

function dimensionRange(region: TensorRegionView, axis: 0 | 1): string {
  const dim = axis === 0 ? 'dimY' : 'dimX'
  return `${dim}: ${region.bbox_min[axis]}..${region.bbox_max[axis]}`
}

function formatRatio(value: number): string {
  if (!Number.isFinite(value)) return '-'
  return `${(value * 100).toFixed(1)}%`
}

function regionTooltip(region: TensorRegionView): string {
  const lines = [
    `${region.tensor_name} ${region.slice_key}`,
    `bbox ${dimensionRange(region, 0)}, ${dimensionRange(region, 1)}`,
    `bbox size ${region.bbox_width} x ${region.bbox_height}`,
    `active ${region.active_element_count} / ${region.bbox_area} (${formatRatio(region.coverage_ratio)})`,
    `components ${region.component_count}`,
  ]
  if (region.target_x !== null) lines.push(`core (${region.target_x},${region.target_y})`)
  if (region.buffer_row_min !== null) {
    lines.push(
      `buffer row ${region.buffer_row_min}..${region.buffer_row_max}, bit ${region.buffer_bit_min}..${region.buffer_bit_max}`,
    )
  }
  if (region.axon_bit_min !== null) lines.push(`output addr ${region.axon_bit_min}..${region.axon_bit_max}`)
  return lines.join('\n')
}

export function IoTab({
  core,
  ioCore,
  ioSummary,
  ioSelection,
  onIoSelectionChange,
}: {
  core: CoreView
  ioCore: IoCoreView | null
  ioSummary: IoSummary | null
  ioSelection: IoSelection | null
  onIoSelectionChange: (selection: IoSelection) => void
}) {
  const [selectedRegion, setSelectedRegion] = useState<TensorRegionView | null>(null)
  const [outputRegionsGlobal, setOutputRegionsGlobal] = useState<TensorRegionView[]>([])
  const [page, setPage] = useState(0)
  useEffect(() => {
    setSelectedRegion(null)
    setPage(0)
  }, [core.chip_id, core.x, core.y, ioSelection?.direction, ioSelection?.tensorName, ioSelection?.sliceKey])
  useEffect(() => {
    if (!ioSelection || ioSelection.direction !== 'output') {
      setOutputRegionsGlobal([])
      return
    }
    // Output mappings can be global host scatter data when no core attribution
    // exists. Fetch a bounded page of global regions for that fallback display.
    fetchIoRegions({
      direction: 'output',
      tensorName: ioSelection.tensorName,
      sliceKey: ioSelection.sliceKey,
      limit: 500,
    })
      .then((pageData) => setOutputRegionsGlobal(pageData.items))
      .catch(() => setOutputRegionsGlobal([]))
  }, [ioSelection?.direction, ioSelection?.tensorName, ioSelection?.sliceKey])

  if (!ioSummary?.available) {
    return (
      <section>
        <h3>IO</h3>
        <p className="muted">No protobuf IO mapping is available for this artifact.</p>
      </section>
    )
  }

  const activeTensor = findIoTensor(ioSummary, ioSelection)
  const inputRegions = (ioCore?.input_regions ?? []).filter((region) => regionMatchesSelection(region, ioSelection))
  const coreOutputRegions = (ioCore?.output_regions ?? []).filter((region) =>
    regionMatchesSelection(region, ioSelection),
  )
  const globalOutputRegions = outputRegionsGlobal.filter((region) => regionMatchesSelection(region, ioSelection))
  const hasOutputCoreAttribution = Boolean(
    ioSelection?.direction === 'output' &&
    ioSummary.core_summaries.some((summary) => summary.output_tensors.includes(ioSelection.tensorName)),
  )
  // If output entries can be attributed to cores, selecting a core should show
  // only that core's output region. Otherwise show the explicit global fallback.
  const outputRegions = hasOutputCoreAttribution ? coreOutputRegions : globalOutputRegions
  const regions = ioSelection?.direction === 'output' ? outputRegions : inputRegions
  const entries =
    (ioSelection?.direction === 'output' ? ioCore?.output_entries : ioCore?.input_entries)?.filter((entry) =>
      entryMatchesSelection(entry, ioSelection),
    ) ?? []
  const pageCount = Math.max(1, Math.ceil(entries.length / IO_ENTRY_PAGE_SIZE))
  const pageItems = entries.slice(page * IO_ENTRY_PAGE_SIZE, (page + 1) * IO_ENTRY_PAGE_SIZE)

  return (
    <>
      <section>
        <h3>IO Summary</h3>
        <div className="inspector-metrics">
          <TinyMetric label="Input entries" value={ioCore?.summary.input_count ?? 0} />
          <TinyMetric label="Output entries" value={ioCore?.summary.output_count ?? 0} />
          <TinyMetric label="Input tensors" value={(ioCore?.summary.input_tensors ?? []).length} />
          <TinyMetric label="Output tensors" value={(ioCore?.summary.output_tensors ?? []).length} />
        </div>
        <IoSelectionControls ioSummary={ioSummary} selection={ioSelection} onChange={onIoSelectionChange} />
      </section>
      <section>
        <h3>Source Plane</h3>
        {activeTensor ? (
          <SourcePlane
            tensor={activeTensor}
            regions={regions}
            selectedRegion={selectedRegion}
            onSelect={setSelectedRegion}
          />
        ) : (
          <p className="muted">Select an IO tensor.</p>
        )}
        {ioSelection?.direction === 'output' && hasOutputCoreAttribution && !regions.length && (
          <p className="muted detail-note">Select a highlighted output core to inspect this tensor's output region.</p>
        )}
        {ioSelection?.direction === 'output' && !hasOutputCoreAttribution && (
          <p className="muted detail-note">
            No output core attribution was found; showing global host scatter output regions.
          </p>
        )}
        <RegionDetail region={selectedRegion ?? regions[0] ?? null} />
      </section>
      {ioSelection?.direction !== 'output' && (
        <section>
          <h3>Core Input Buffer</h3>
          <InputBufferMap spans={ioCore?.input_buffer_spans ?? []} tensorName={ioSelection?.tensorName ?? ''} />
        </section>
      )}
      <section>
        <h3>Tensor Strip</h3>
        <TensorStrip regions={regions} />
      </section>
      <section>
        <h3>Entry Detail</h3>
        <Pager page={page} pageCount={pageCount} total={entries.length} onPage={setPage} />
        <IoEntryTable entries={pageItems} direction={ioSelection?.direction ?? 'input'} />
      </section>
    </>
  )
}

function IoSelectionControls({
  ioSummary,
  selection,
  onChange,
}: {
  ioSummary: IoSummary
  selection: IoSelection | null
  onChange: (selection: IoSelection) => void
}) {
  const direction = selection?.direction ?? 'input'
  const tensors = ioSummary.tensors.filter((tensor) => tensor.direction === direction)
  const selectedTensor = tensors.find((tensor) => tensor.name === selection?.tensorName) ?? tensors[0]
  function changeDirection(nextDirection: 'input' | 'output') {
    const tensor = ioSummary.tensors.find((item) => item.direction === nextDirection)
    if (tensor) onChange({ direction: nextDirection, tensorName: tensor.name, sliceKey: tensor.slice_keys[0] ?? 'all' })
  }
  function changeTensor(tensorName: string) {
    const tensor = tensors.find((item) => item.name === tensorName)
    if (tensor) onChange({ direction, tensorName: tensor.name, sliceKey: tensor.slice_keys[0] ?? 'all' })
  }
  return (
    <div className="io-selectors">
      <label>
        Direction
        <select
          value={direction}
          onChange={(event) => changeDirection(event.target.value === 'output' ? 'output' : 'input')}
        >
          <option value="input">Input</option>
          <option value="output">Output</option>
        </select>
      </label>
      <label>
        Tensor
        <select value={selectedTensor?.name ?? ''} onChange={(event) => changeTensor(event.target.value)}>
          {tensors.map((tensor) => (
            <option key={`${tensor.direction}-${tensor.thread_id}-${tensor.name}`} value={tensor.name}>
              {tensor.name}
            </option>
          ))}
        </select>
      </label>
      <label>
        Slice
        <select
          value={selection?.sliceKey ?? selectedTensor?.slice_keys[0] ?? 'all'}
          onChange={(event) =>
            selectedTensor && onChange({ direction, tensorName: selectedTensor.name, sliceKey: event.target.value })
          }
        >
          {(selectedTensor?.slice_keys.length ? selectedTensor.slice_keys : ['all']).map((sliceKey) => (
            <option key={sliceKey} value={sliceKey}>
              {sliceKey}
            </option>
          ))}
        </select>
      </label>
    </div>
  )
}

function SourcePlane({
  tensor,
  regions,
  selectedRegion,
  onSelect,
}: {
  tensor: IoTensorView
  regions: TensorRegionView[]
  selectedRegion: TensorRegionView | null
  onSelect: (region: TensorRegionView) => void
}) {
  if (tensor.shape.length < 2) {
    return <OutputStrip tensor={tensor} regions={regions} selectedRegion={selectedRegion} onSelect={onSelect} />
  }
  const width = 280
  const height = Math.max(120, Math.round((width * tensor.plane.height) / Math.max(tensor.plane.width, 1)))
  const xScale = width / Math.max(tensor.plane.width, 1)
  const yScale = height / Math.max(tensor.plane.height, 1)
  return (
    <svg className="source-plane" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="source tensor plane">
      <rect className="source-plane-bg" x={0} y={0} width={width} height={height} />
      {regions.map((region, index) => (
        <g key={regionKey(region, index)} className={selectedRegion === region ? 'region selected' : 'region'}>
          {/* Runs draw the exact mask; the bbox rectangle is only a readable size cue. */}
          {region.runs.map((run) => (
            <rect
              key={`${run.y}-${run.x_start}-${run.x_end}`}
              x={run.x_start * xScale}
              y={run.y * yScale}
              width={(run.x_end - run.x_start + 1) * xScale}
              height={Math.max(yScale, 1)}
              onMouseEnter={() => onSelect(region)}
              onClick={() => onSelect(region)}
            >
              <title>{regionTooltip(region)}</title>
            </rect>
          ))}
          <rect
            className="bbox"
            x={region.bbox_min[1] * xScale}
            y={region.bbox_min[0] * yScale}
            width={region.bbox_width * xScale}
            height={region.bbox_height * yScale}
          />
        </g>
      ))}
      {!regions.length && (
        <text x={10} y={24}>
          No region for this core/slice
        </text>
      )}
    </svg>
  )
}

function OutputStrip({
  tensor,
  regions,
  selectedRegion,
  onSelect,
}: {
  tensor: IoTensorView
  regions: TensorRegionView[]
  selectedRegion: TensorRegionView | null
  onSelect: (region: TensorRegionView) => void
}) {
  // Rank-1 outputs are easier to inspect as a flat strip than as a forced 2D
  // plane. The region detail panel still reports exact element ranges.
  const width = 280
  const height = 56
  const total = Math.max(
    tensor.shape.reduce((value, item) => value * item, 1),
    1,
  )
  return (
    <svg
      className="source-plane output-strip"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="output tensor strip"
    >
      <rect className="source-plane-bg" x={0} y={0} width={width} height={height} />
      {regions.map((region, index) => {
        const x = (region.elem_start / total) * width
        const w = Math.max(((region.elem_end - region.elem_start + 1) / total) * width, 4)
        return (
          <rect
            key={regionKey(region, index)}
            className={selectedRegion === region ? 'strip-region selected' : 'strip-region'}
            x={x}
            y={12}
            width={w}
            height={28}
            onMouseEnter={() => onSelect(region)}
            onClick={() => onSelect(region)}
          >
            <title>{regionTooltip(region)}</title>
          </rect>
        )
      })}
      {!regions.length && (
        <text x={10} y={30}>
          No output region for this core
        </text>
      )}
    </svg>
  )
}

function RegionDetail({ region }: { region: TensorRegionView | null }) {
  if (!region) return <p className="muted detail-note">Hover or click a mask to inspect region size.</p>
  const bbox = `${dimensionRange(region, 0)}, ${dimensionRange(region, 1)}`
  const bufferRange =
    region.buffer_row_min === null
      ? '-'
      : `row ${region.buffer_row_min}..${region.buffer_row_max}, bit ${region.buffer_bit_min}..${region.buffer_bit_max}`
  const outputRange = region.axon_bit_min === null ? '-' : `${region.axon_bit_min}..${region.axon_bit_max}`
  return (
    <dl className="kv compact region-detail">
      <dt>tensor</dt>
      <dd>{region.tensor_name}</dd>
      <dt>slice</dt>
      <dd>{region.slice_key}</dd>
      <dt>bbox</dt>
      <dd>{bbox}</dd>
      <dt>bbox size</dt>
      <dd>
        {region.bbox_width} x {region.bbox_height}
      </dd>
      <dt>active</dt>
      <dd>
        {region.active_element_count} / {region.bbox_area} ({formatRatio(region.coverage_ratio)})
      </dd>
      <dt>components</dt>
      <dd>{region.component_count}</dd>
      <dt>core</dt>
      <dd>{region.target_x === null ? '-' : `(${region.target_x},${region.target_y})`}</dd>
      <dt>buffer</dt>
      <dd>{bufferRange}</dd>
      <dt>output addr</dt>
      <dd>{outputRange}</dd>
      <dt>dtype</dt>
      <dd>{region.output_kind ? `${region.output_kind} ${region.dtype}` : region.dtype}</dd>
    </dl>
  )
}

function InputBufferMap({ spans, tensorName }: { spans: BufferSpanView[]; tensorName: string }) {
  // Each core input buffer is 256 rows by 512 bits; spans are pre-aggregated by
  // the backend so large inputs can render without one SVG node per element.
  const filtered = spans.filter((span) => span.tensor_name === tensorName)
  const width = 280
  const height = 140
  const xScale = width / 512
  const yScale = height / 256
  return (
    <svg className="buffer-map" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="core input buffer occupancy">
      <rect className="source-plane-bg" x={0} y={0} width={width} height={height} />
      {filtered.map((span, index) => (
        <rect
          key={`${span.row}-${span.bit_start}-${span.bit_end}-${index}`}
          x={span.bit_start * xScale}
          y={span.row * yScale}
          width={Math.max((span.bit_end - span.bit_start + 1) * xScale, 1)}
          height={Math.max(yScale, 1)}
        >
          <title>{`row ${span.row}, bit ${span.bit_start}..${span.bit_end}, ${span.count} entries`}</title>
        </rect>
      ))}
      {!filtered.length && (
        <text x={10} y={24}>
          No input buffer spans for this tensor
        </text>
      )}
    </svg>
  )
}

function TensorStrip({ regions }: { regions: TensorRegionView[] }) {
  const maxElem = Math.max(...regions.map((region) => region.elem_end), 0)
  const width = 280
  return (
    <div className="tensor-strip" aria-label="tensor flat index coverage">
      {regions.map((region, index) => {
        const left = maxElem > 0 ? (region.elem_start / (maxElem + 1)) * 100 : 0
        const stripWidth = maxElem > 0 ? ((region.elem_end - region.elem_start + 1) / (maxElem + 1)) * 100 : 100
        return (
          <span
            key={regionKey(region, index)}
            style={{ left: `${left}%`, width: `${Math.max(stripWidth, 1)}%` }}
            title={regionTooltip(region)}
          />
        )
      })}
      <div className="strip-axis" style={{ width }} />
      {!regions.length && <p className="muted">No tensor regions.</p>}
    </div>
  )
}

function IoEntryTable({ entries, direction }: { entries: IoEntryView[]; direction: IoDirection }) {
  return (
    <div className="io-entry-table">
      <div className="io-entry-row io-entry-head">
        <span>elem</span>
        <span>coord</span>
        <span>{direction === 'output' ? 'axon' : 'buffer'}</span>
        <span>dtype</span>
      </div>
      {entries.map((entry) => (
        <div className="io-entry-row" key={`${entry.direction}-${entry.index}`}>
          <span>{entry.elem_idx}</span>
          <span>[{entry.tensor_coord.join(',')}]</span>
          <span>
            {direction === 'output'
              ? (entry.axon_bit_idx ?? '-')
              : `r${entry.buffer_row ?? '-'} b${entry.buffer_bit ?? '-'}`}
          </span>
          <span>{entry.output_kind ? `${entry.output_kind} ${entry.dtype}` : entry.dtype}</span>
        </div>
      ))}
      {!entries.length && <p className="muted">No IO entries for this selection.</p>}
    </div>
  )
}
