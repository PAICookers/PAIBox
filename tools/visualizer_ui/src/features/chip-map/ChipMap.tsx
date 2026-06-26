import type {
  ChipOverview,
  ControlPathView,
  CoreOverview,
  CoreView,
  DecodedField,
  IoDirection,
  LinkView,
  NeuronRecordView,
} from '../../types'
import { getTickStart } from '../../shared/core'
import { formatCompactInteger } from '../../shared/format'
import type { IoSelection, MapMode, OverlayOptions } from '../../shared/view-types'

const CELL = 58
const GAP = 8
const PAD = 54
const DIRECTIONS: Record<string, [number, number]> = {
  '+xy': [1, 1],
  '-xy': [-1, -1],
  '+x': [1, 0],
  '-x': [-1, 0],
  '+y': [0, 1],
  '-y': [0, -1],
}

interface DestinationArrow {
  targetX: number
  targetY: number
  neuronCount: number
}

interface CpuInputFlowArrow {
  sourceX: number
  sourceY: number
  targetX: number
  targetY: number
  count: number
  tensors: string[]
}

export function buildLinks(chip: ChipOverview | undefined): LinkView[] {
  if (!chip) return []
  const links: LinkView[] = []
  for (const core of chip.cores) {
    for (const dir of core.global_signal.send_dirs) {
      if (dir === 'local') continue
      const delta = DIRECTIONS[dir]
      if (!delta) continue
      const targetX = core.x + delta[0]
      const targetY = core.y + delta[1]
      if (!isCoordInChip(chip, targetX, targetY)) continue
      links.push({
        chip_id: core.chip_id,
        source: { x: core.x, y: core.y },
        target: { x: targetX, y: targetY },
        kind: 'global_send',
        direction: dir,
      })
    }
  }
  return links
}

export function buildControlPaths(chip: ChipOverview | undefined): ControlPathView[] {
  if (!chip) return []
  return chip.cores.flatMap((core) => (core.global_signal.is_source ? core.global_signal.control_paths : []))
}

function isCoordInChip(chip: ChipOverview, x: number, y: number): boolean {
  return x >= 0 && x < chip.grid_width && y >= 0 && y < chip.grid_height
}

function coordToPoint(chip: ChipOverview, x: number, y: number) {
  const sx = PAD + x * (CELL + GAP) + CELL / 2
  const sy = PAD + (chip.grid_height - 1 - y) * (CELL + GAP) + CELL / 2
  return [sx, sy] as const
}

export function ChipGrid({
  chip,
  links,
  controlPaths,
  selected,
  overlays,
  mapMode,
  ioSelection,
  tickColorMap,
  threadFilter,
  highlightTickStart,
  onSelect,
}: {
  chip: ChipOverview
  links: LinkView[]
  controlPaths: ControlPathView[]
  selected: CoreView | null
  overlays: OverlayOptions
  mapMode: MapMode
  ioSelection: IoSelection | null
  tickColorMap: Map<number, string>
  threadFilter: number | 'all'
  highlightTickStart: number | null
  onSelect: (core: CoreOverview) => void
}) {
  const width = PAD * 2 + chip.grid_width * CELL + (chip.grid_width - 1) * GAP
  const height = PAD * 2 + chip.grid_height * CELL + (chip.grid_height - 1) * GAP
  const maxIoCount = Math.max(
    ...chip.cores.map((core) => ioCountForCore(core, ioSelection)).filter((value) => value > 0),
    0,
  )
  return (
    <div className="chip-scroll">
      <svg className="chip-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="PAICORE chip layout">
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="#5b7cfa" />
          </marker>
          <marker id="control-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="#b45309" />
          </marker>
          <marker id="dest-arrow" markerWidth="6" markerHeight="6" refX="5.4" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#c2410c" />
          </marker>
          <marker id="host-input-arrow" markerWidth="6" markerHeight="6" refX="5.4" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#0f766e" />
          </marker>
        </defs>
        {chip.cores.map((core) => {
          const x = PAD + core.x * (CELL + GAP)
          const y = PAD + (chip.grid_height - 1 - core.y) * (CELL + GAP)
          const isSelected = selected?.chip_id === core.chip_id && selected.x === core.x && selected.y === core.y
          const tickStart = getTickStart(core)
          const ioCount = ioCountForCore(core, ioSelection)
          const ioFill =
            mapMode === 'io' ? ioHeatColor(ioCount, maxIoCount, ioSelection?.direction ?? 'input') : undefined
          const fill =
            ioFill ?? (overlays.colorByTickStart && tickStart !== null ? tickColorMap.get(tickStart) : undefined)
          const isMuted = threadFilter !== 'all' && core.thread_id !== threadFilter
          const sameTick = tickStart !== null && tickStart === highlightTickStart
          const isSourceCore = core.global_signal.is_source
          const isHighlighted = sameTick || isSelected
          return (
            <g
              key={`${core.chip_id}-${core.x}-${core.y}`}
              className={[
                'core',
                `core-${core.role}`,
                core.used ? 'used' : 'idle',
                isSelected ? 'selected' : '',
                sameTick ? 'same-tick' : '',
                isMuted ? 'muted' : '',
              ].join(' ')}
              onClick={() => onSelect(core)}
              tabIndex={0}
              role="button"
            >
              {isSourceCore && (
                <rect className="core-source-halo" x={x - 5} y={y - 5} width={CELL + 10} height={CELL + 10} rx={11} />
              )}
              <rect x={x} y={y} width={CELL} height={CELL} rx={6} style={fill ? { fill } : undefined} />
              {isHighlighted && (
                <rect className="core-highlight" x={x - 3} y={y - 3} width={CELL + 6} height={CELL + 6} rx={8} />
              )}
              <text className="coord-label" x={x + 7} y={y + 15}>
                {core.x},{core.y}
              </text>
              {mapMode === 'io' && ioCount > 0 ? (
                <text className="tick-label" x={x + 7} y={y + 36}>
                  {formatCompactInteger(ioCount)}
                </text>
              ) : overlays.showTickStartLabels && tickStart !== null ? (
                <text className="tick-label" x={x + 7} y={y + 36}>{`tick ${tickStart}`}</text>
              ) : (
                <text className="state-label" x={x + 7} y={y + 36}>
                  {core.used ? 'used' : 'idle'}
                </text>
              )}
            </g>
          )
        })}
        {mapMode !== 'io' && overlays.showSignalArrows && (
          <g className="signal-layer">
            {links.map((link, index) => {
              const [sx, sy] = coordToPoint(chip, link.source.x, link.source.y)
              const [tx, ty] = coordToPoint(chip, link.target.x, link.target.y)
              return (
                <line
                  key={`${link.source.x}-${link.source.y}-${link.direction}-${index}`}
                  x1={sx}
                  y1={sy}
                  x2={tx}
                  y2={ty}
                  className="signal-link"
                  markerEnd="url(#arrow)"
                />
              )
            })}
          </g>
        )}
        {mapMode !== 'io' && overlays.showSignalArrows && <ControlPathLayer chip={chip} paths={controlPaths} />}
        {mapMode !== 'io' &&
          overlays.showNeuronDestinations &&
          selected &&
          (selected.role === 'cpu' ? (
            <CpuInputFlowLayer chip={chip} flows={buildCpuInputFlows(chip, selected)} />
          ) : (
            <NeuronDestinationLayer
              chip={chip}
              source={selected}
              destinations={buildNeuronDestinations(chip, selected)}
            />
          ))}
        <DirectionLabels width={width} height={height} />
      </svg>
    </div>
  )
}

function CpuInputFlowLayer({ chip, flows }: { chip: ChipOverview; flows: CpuInputFlowArrow[] }) {
  if (!flows.length) return null
  return (
    <g className="cpu-input-layer" aria-label="CPU input flow">
      {flows.map((flow) => {
        const [sx, sy] = coordToPoint(chip, flow.sourceX, flow.sourceY)
        const [tx, ty] = coordToPoint(chip, flow.targetX, flow.targetY)
        const curve = destinationCurve(sx, sy, tx, ty, 0, 1)
        const midX = (sx + tx) / 2 + curve.labelDx
        const midY = (sy + ty) / 2 + curve.labelDy
        return (
          <g key={`${flow.sourceX}-${flow.sourceY}-${flow.targetX}-${flow.targetY}`}>
            <path d={`M ${sx} ${sy} Q ${curve.cx} ${curve.cy} ${tx} ${ty}`} markerEnd="url(#host-input-arrow)">
              <title>{cpuInputFlowTitle(flow)}</title>
            </path>
            {flow.count > 1 && (
              <text x={midX} y={midY}>
                {formatCompactInteger(flow.count)}
              </text>
            )}
          </g>
        )
      })}
    </g>
  )
}

function ControlPathLayer({ chip, paths }: { chip: ChipOverview; paths: ControlPathView[] }) {
  if (!paths.length) return null
  return (
    <g className="control-path-layer" aria-label="global signal source control paths">
      {paths.map((path, pathIndex) => {
        if (path.points.length < 2) return null
        return (
          <g key={`${path.thread_id}-${path.source_frame_index ?? 'na'}-${pathIndex}`}>
            {path.points.slice(1).map((point, pointIndex) => {
              const previous = path.points[pointIndex]
              const [sx, sy] = coordToPoint(chip, previous.x, previous.y)
              const [tx, ty] = coordToPoint(chip, point.x, point.y)
              const isLast = pointIndex === path.points.length - 2
              return (
                <line
                  key={`${previous.x}-${previous.y}-${point.x}-${point.y}-${pointIndex}`}
                  x1={sx}
                  y1={sy}
                  x2={tx}
                  y2={ty}
                  markerEnd={isLast ? 'url(#control-arrow)' : undefined}
                >
                  <title>{controlPathTitle(path)}</title>
                </line>
              )
            })}
          </g>
        )
      })}
    </g>
  )
}

function controlPathTitle(path: ControlPathView): string {
  const source = path.points[0]
  return [
    `global source thread ${path.thread_id}`,
    `control path (${source?.x ?? '-'},${source?.y ?? '-'}) -> (${path.target_x},${path.target_y})`,
    `test offset zxy=(${path.offset_xy},${path.offset_x},${path.offset_y})`,
  ].join('\n')
}

function NeuronDestinationLayer({
  chip,
  source,
  destinations,
}: {
  chip: ChipOverview
  source: CoreView
  destinations: DestinationArrow[]
}) {
  if (!destinations.length) return null
  // Destination arrows are a selected-core overlay. They are intentionally
  // omitted in IO Map mode so tensor/buffer heat colors stay visually dominant.
  const [sx, sy] = coordToPoint(chip, source.x, source.y)
  return (
    <g className="destination-layer" aria-label="selected core neuron destinations">
      {destinations.map((destination, index) => {
        const [tx, ty] = coordToPoint(chip, destination.targetX, destination.targetY)
        const curve = destinationCurve(sx, sy, tx, ty, index, destinations.length)
        const midX = (sx + tx) / 2 + curve.labelDx
        const midY = (sy + ty) / 2 + curve.labelDy
        return (
          <g key={`${destination.targetX}-${destination.targetY}-${index}`}>
            <path d={`M ${sx} ${sy} Q ${curve.cx} ${curve.cy} ${tx} ${ty}`} markerEnd="url(#dest-arrow)" />
            {destination.neuronCount > 1 && (
              <text x={midX} y={midY}>
                {formatCompactInteger(destination.neuronCount)}
              </text>
            )}
          </g>
        )
      })}
    </g>
  )
}

function buildCpuInputFlows(chip: ChipOverview, cpu: CoreOverview): CpuInputFlowArrow[] {
  const flows: CpuInputFlowArrow[] = []
  for (const core of chip.cores) {
    if (core.chip_id !== cpu.chip_id || (core.x === cpu.x && core.y === cpu.y)) continue
    const summary = core.io_summary
    if (!summary) continue
    if (summary.input_count > 0) {
      flows.push({
        sourceX: cpu.x,
        sourceY: cpu.y,
        targetX: core.x,
        targetY: core.y,
        count: summary.input_count,
        tensors: summary.input_tensors,
      })
    }
  }
  return flows.sort((a, b) => a.targetY - b.targetY || a.targetX - b.targetX)
}

function cpuInputFlowTitle(flow: CpuInputFlowArrow): string {
  return [
    `CPU -> input core (${flow.targetX},${flow.targetY})`,
    `entries: ${flow.count}`,
    `tensors: ${flow.tensors.join(', ') || '-'}`,
  ].join('\n')
}

function destinationCurve(sx: number, sy: number, tx: number, ty: number, index: number, total: number) {
  // Parallel destination arrows need a small deterministic bend; otherwise
  // routes with the same source/target overlap and hide their labels.
  const dx = tx - sx
  const dy = ty - sy
  const length = Math.hypot(dx, dy) || 1
  const normalX = -dy / length
  const normalY = dx / length
  const spread = total > 1 ? index - (total - 1) / 2 : 0
  const bend = Math.min(42, Math.max(14, length * 0.14)) * (spread || 0.85)
  return {
    cx: (sx + tx) / 2 + normalX * bend,
    cy: (sy + ty) / 2 + normalY * bend,
    labelDx: normalX * bend * 0.6,
    labelDy: normalY * bend * 0.6,
  }
}

function DirectionLabels({ width, height }: { width: number; height: number }) {
  return (
    <g className="direction-labels" aria-hidden="true">
      <text className="direction-y" x={12} y={22}>
        +Y
      </text>
      <text className="direction-xy" x={width - 42} y={22}>
        +XY
      </text>
      <text className="direction-x" x={width - 34} y={height - 14}>
        +X
      </text>
    </g>
  )
}

function buildNeuronDestinations(chip: ChipOverview, core: CoreView): DestinationArrow[] {
  const counts = new Map<string, DestinationArrow>()
  for (const record of core.neurons.records) {
    const neuronCount = getFoldedNeuronCount(record)
    for (const target of record.destinations) {
      const targetX = target.target_x
      const targetY = target.target_y
      // Backend validation rejects out-of-grid routes. This guard only protects
      // the SVG from stale or externally edited viewer JSON.
      if (targetX < 0 || targetX >= chip.grid_width || targetY < 0 || targetY >= chip.grid_height) {
        continue
      }
      const key = `${targetX},${targetY}`
      const existing = counts.get(key)
      if (existing) {
        existing.neuronCount += neuronCount
      } else {
        counts.set(key, { targetX, targetY, neuronCount })
      }
    }
  }
  return [...counts.values()].sort((a, b) => a.targetY - b.targetY || a.targetX - b.targetX)
}

function getFoldedNeuronCount(record: NeuronRecordView): number {
  const foldNumber = decodedFieldNumber(record.fields['fold attrs'] ?? [], 'fold_number')
  return foldNumber && foldNumber > 0 ? foldNumber : 1
}

function decodedFieldNumber(fields: DecodedField[], name: string): number | null {
  const field = fields.find((item) => item.name === name)
  return typeof field?.decoded === 'number' ? field.decoded : null
}

function ioCountForCore(core: CoreOverview, selection: IoSelection | null): number {
  const summary = core.io_summary
  if (!summary || !selection) return 0
  if (selection.direction === 'output') {
    if (!summary.output_tensors.includes(selection.tensorName)) return 0
    return summary.output_count
  }
  if (!summary.input_tensors.includes(selection.tensorName)) return 0
  return summary.input_count
}

function ioHeatColor(count: number, maxCount: number, direction: IoDirection): string | undefined {
  if (count <= 0 || maxCount <= 0) return undefined
  const ratio = Math.max(0.16, Math.min(count / maxCount, 1))
  if (direction === 'output') {
    return `rgba(217, 119, 6, ${0.22 + ratio * 0.58})`
  }
  return `rgba(13, 148, 136, ${0.22 + ratio * 0.58})`
}

export function MiniMap({ chip, selected }: { chip: ChipOverview; selected: CoreView | null }) {
  return (
    <div className="minimap">
      {chip.cores.map((core) => {
        const selectedClass = selected?.x === core.x && selected.y === core.y ? 'mini-selected' : ''
        return (
          <span
            key={`${core.x}-${core.y}`}
            className={`mini-core mini-${core.role} ${core.used ? 'mini-used' : ''} ${selectedClass}`}
            style={{ gridColumn: core.x + 1, gridRow: chip.grid_height - core.y }}
            title={`${core.x},${core.y}`}
          />
        )
      })}
    </div>
  )
}

export function RoleLegend() {
  return (
    <div className="legend-list">
      <div className="legend-row">
        <span className="legend-swatch role-cpu-swatch" />
        CPU
      </div>
      <div className="legend-row">
        <span className="legend-swatch role-online-swatch" />
        Online core
      </div>
      <div className="legend-row">
        <span className="legend-swatch role-offline-swatch" />
        Offline core
      </div>
    </div>
  )
}

export function ThreadPanel({
  threads,
  selected,
  onChange,
}: {
  threads: number[]
  selected: number | 'all'
  onChange: (thread: number | 'all') => void
}) {
  return (
    <select
      className="thread-select"
      value={selected}
      onChange={(event) => onChange(event.target.value === 'all' ? 'all' : Number(event.target.value))}
    >
      <option value="all">All threads</option>
      {threads.map((thread) => (
        <option value={thread} key={thread}>
          Thread {thread}
        </option>
      ))}
    </select>
  )
}
