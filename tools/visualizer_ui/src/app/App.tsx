import { useEffect, useMemo, useState } from 'react'
import type { CSSProperties, PointerEvent as ReactPointerEvent } from 'react'

import { fetchChips, fetchCore, fetchIoCore, fetchIoSummary, fetchSummary, fetchValidation } from '../api'
import type {
  ChipOverview,
  CoreOverview,
  CoreView,
  IoCoreView,
  IoDirection,
  IoSummary,
  ValidationEntry,
  ViewerSummary,
} from '../types'
import { buildControlPaths, buildLinks, ChipGrid, MiniMap, RoleLegend, ThreadPanel } from '../features/chip-map/ChipMap'
import { Inspector } from '../features/inspector/Inspector'
import {
  buildTickColorMap,
  buildTickComputeRows,
  TickComputePanel,
  TickPressureBarChart,
} from '../features/tick-compute/TickComputePanel'
import { collectThreads, getTickStart } from '../shared/core'
import { Metric, ValidationList } from '../shared/components'
import { clamp } from '../shared/format'
import type { IoSelection, MapMode, OverlayOptions } from '../shared/view-types'

const DEFAULT_INSPECTOR_WIDTH = 380
const MIN_INSPECTOR_WIDTH = 320
const MAX_INSPECTOR_WIDTH = 720
const DEFAULT_SIDEBAR_WIDTH = 280
const MIN_SIDEBAR_WIDTH = 220
const MAX_SIDEBAR_WIDTH = 380

export function App() {
  const [summary, setSummary] = useState<ViewerSummary | null>(null)
  const [chips, setChips] = useState<ChipOverview[]>([])
  const [ioSummary, setIoSummary] = useState<IoSummary | null>(null)
  const [selectedIoCore, setSelectedIoCore] = useState<IoCoreView | null>(null)
  const [validation, setValidation] = useState<ValidationEntry[]>([])
  const [selected, setSelected] = useState<CoreView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [threadFilter, setThreadFilter] = useState<number | 'all'>('all')
  const [highlightTickStart, setHighlightTickStart] = useState<number | null>(null)
  const [sidebarWidth, setSidebarWidth] = useState(DEFAULT_SIDEBAR_WIDTH)
  const [inspectorWidth, setInspectorWidth] = useState(DEFAULT_INSPECTOR_WIDTH)
  const [includePaddingInSops, setIncludePaddingInSops] = useState(true)
  const [mapMode, setMapMode] = useState<MapMode>('chip')
  const [ioSelection, setIoSelection] = useState<IoSelection | null>(null)
  const [overlays, setOverlays] = useState<OverlayOptions>({
    showSignalArrows: false,
    showNeuronDestinations: true,
    colorByTickStart: true,
    showTickStartLabels: true,
    highlightSameTickStart: true,
  })

  useEffect(() => {
    // Bootstrap summary-level data first, then lazily load one core detail.
    // Core detail is larger because it includes decoded records and raw frames.
    Promise.all([fetchSummary(), fetchChips(), fetchValidation(), fetchIoSummary()])
      .then(([summaryData, chipData, validationData, ioData]) => {
        setSummary(summaryData)
        setChips(chipData)
        setValidation(validationData)
        setIoSummary(ioData)
        const firstInput = ioData.tensors.find((tensor) => tensor.direction === 'input')
        if (firstInput) {
          setIoSelection({
            direction: 'input',
            tensorName: firstInput.name,
            sliceKey: firstInput.slice_keys[0] ?? 'all',
          })
        }
        const firstUsed = chipData[0]?.cores.find((core) => core.used)
        if (firstUsed) void selectCore(firstUsed)
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  useEffect(() => {
    if (!overlays.highlightSameTickStart) {
      setHighlightTickStart(null)
    }
  }, [overlays.highlightSameTickStart])

  const chip = chips[0]
  const links = useMemo(() => buildLinks(chip), [chip])
  const controlPaths = useMemo(() => buildControlPaths(chip), [chip])
  const tickColorMap = useMemo(() => (chip ? buildTickColorMap(chip) : new Map<number, string>()), [chip])
  const tickComputeRows = chip ? buildTickComputeRows(chip, includePaddingInSops, tickColorMap) : []
  const threads = useMemo(() => (chip ? collectThreads(chip) : []), [chip])

  async function selectCore(core: CoreOverview) {
    const tickStart = getTickStart(core)
    setHighlightTickStart(overlays.highlightSameTickStart ? tickStart : null)
    try {
      const detail = await fetchCore(core.chip_id, core.x, core.y)
      setSelected(detail)
      if (mapMode === 'io') {
        setSelectedIoCore(await fetchIoCore(core.chip_id, core.x, core.y))
      }
    } catch {
      setSelected(null)
      setSelectedIoCore(null)
    }
  }

  useEffect(() => {
    if (!selected || mapMode !== 'io') return
    // Core config and IO detail are separate endpoints. Switching to IO Map
    // should refresh the selected core's IO subset without changing selection.
    fetchIoCore(selected.chip_id, selected.x, selected.y)
      .then(setSelectedIoCore)
      .catch(() => setSelectedIoCore(null))
  }, [selected?.chip_id, selected?.x, selected?.y, mapMode])

  if (error) {
    return <main className="app error-state">Failed to load visualizer data: {error}</main>
  }

  function startInspectorResize(event: ReactPointerEvent<HTMLDivElement>) {
    event.preventDefault()
    const startX = event.clientX
    const startWidth = inspectorWidth
    const handleMove = (moveEvent: globalThis.PointerEvent) => {
      const nextWidth = startWidth - (moveEvent.clientX - startX)
      setInspectorWidth(clamp(nextWidth, MIN_INSPECTOR_WIDTH, MAX_INSPECTOR_WIDTH))
    }
    const handleUp = () => {
      window.removeEventListener('pointermove', handleMove)
      window.removeEventListener('pointerup', handleUp)
    }
    window.addEventListener('pointermove', handleMove)
    window.addEventListener('pointerup', handleUp)
  }

  function startSidebarResize(event: ReactPointerEvent<HTMLDivElement>) {
    event.preventDefault()
    const startX = event.clientX
    const startWidth = sidebarWidth
    const handleMove = (moveEvent: globalThis.PointerEvent) => {
      const nextWidth = startWidth + (moveEvent.clientX - startX)
      setSidebarWidth(clamp(nextWidth, MIN_SIDEBAR_WIDTH, MAX_SIDEBAR_WIDTH))
    }
    const handleUp = () => {
      window.removeEventListener('pointermove', handleMove)
      window.removeEventListener('pointerup', handleUp)
    }
    window.addEventListener('pointermove', handleMove)
    window.addEventListener('pointerup', handleUp)
  }

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <h1>PAIBox Visualizer</h1>
          <p>Frame-first chip layout inspection</p>
        </div>
        {summary && (
          <div className="summary-strip">
            <Metric label="Cores" value={`${summary.used_core_count}/${summary.core_count}`} />
            <Metric label="Frames" value={summary.frame_count.toLocaleString()} />
            <Metric label="Packages" value={summary.package_count.toLocaleString()} />
            <Metric
              label="Validation"
              value={`${summary.validation_error_count}E ${summary.validation_warning_count}W`}
            />
          </div>
        )}
      </header>
      <section
        className="workspace"
        style={
          {
            '--sidebar-width': `${sidebarWidth}px`,
            '--inspector-width': `${inspectorWidth}px`,
          } as CSSProperties
        }
      >
        <aside className="sidebar">
          <h2>Minimap</h2>
          {chip && <MiniMap chip={chip} selected={selected} />}
          <h2>Core Roles</h2>
          <RoleLegend />
          <h2>Threads</h2>
          <ThreadPanel threads={threads} selected={threadFilter} onChange={setThreadFilter} />
          <h2>Tick Compute</h2>
          {chip && (
            <TickComputePanel
              rows={tickComputeRows}
              highlighted={highlightTickStart}
              onSelect={setHighlightTickStart}
            />
          )}
          <h2>Validation</h2>
          <ValidationList entries={validation} />
        </aside>
        <div
          className="sidebar-resizer"
          role="separator"
          aria-label="Resize sidebar"
          aria-orientation="vertical"
          onPointerDown={startSidebarResize}
        />
        <section className="canvas-panel">
          {chip ? (
            <>
              <ViewControls
                mapMode={mapMode}
                onMapModeChange={setMapMode}
                options={overlays}
                onChange={setOverlays}
                ioSummary={ioSummary}
                ioSelection={ioSelection}
                onIoSelectionChange={setIoSelection}
              />
              <ChipGrid
                chip={chip}
                links={links}
                controlPaths={controlPaths}
                selected={selected}
                overlays={overlays}
                mapMode={mapMode}
                ioSelection={ioSelection}
                tickColorMap={tickColorMap}
                threadFilter={threadFilter}
                highlightTickStart={highlightTickStart}
                onSelect={selectCore}
              />
              <TickPressureBarChart rows={tickComputeRows} />
            </>
          ) : (
            <div className="loading">Loading artifact...</div>
          )}
        </section>
        <div
          className="inspector-resizer"
          role="separator"
          aria-label="Resize inspector"
          aria-orientation="vertical"
          onPointerDown={startInspectorResize}
        />
        <Inspector
          core={selected}
          ioCore={selectedIoCore}
          ioSummary={ioSummary}
          ioSelection={ioSelection}
          mapMode={mapMode}
          onIoSelectionChange={setIoSelection}
          includePaddingInSops={includePaddingInSops}
          onIncludePaddingInSopsChange={setIncludePaddingInSops}
        />
      </section>
    </main>
  )
}

function ViewControls({
  mapMode,
  onMapModeChange,
  options,
  onChange,
  ioSummary,
  ioSelection,
  onIoSelectionChange,
}: {
  mapMode: MapMode
  onMapModeChange: (mode: MapMode) => void
  options: OverlayOptions
  onChange: (options: OverlayOptions) => void
  ioSummary: IoSummary | null
  ioSelection: IoSelection | null
  onIoSelectionChange: (selection: IoSelection) => void
}) {
  function toggle(key: keyof OverlayOptions) {
    onChange({ ...options, [key]: !options[key] })
  }
  const inputTensors = ioSummary?.tensors.filter((tensor) => tensor.direction === 'input') ?? []
  const outputTensors = ioSummary?.tensors.filter((tensor) => tensor.direction === 'output') ?? []
  const activeTensors = ioSelection?.direction === 'output' ? outputTensors : inputTensors
  const selectedTensor = activeTensors.find((tensor) => tensor.name === ioSelection?.tensorName) ?? activeTensors[0]

  // `Chip` and `IO Map` are mutually exclusive main-view modes. Keep their
  // controls separate so chip overlays and IO tensor selectors do not compete.
  function changeIoDirection(direction: IoDirection) {
    const tensor = (direction === 'input' ? inputTensors : outputTensors)[0]
    if (!tensor) return
    onIoSelectionChange({ direction, tensorName: tensor.name, sliceKey: tensor.slice_keys[0] ?? 'all' })
  }

  function changeIoTensor(tensorName: string) {
    const tensor = activeTensors.find((item) => item.name === tensorName)
    if (!tensor || !ioSelection) return
    onIoSelectionChange({ ...ioSelection, tensorName: tensor.name, sliceKey: tensor.slice_keys[0] ?? 'all' })
  }

  function changeIoSlice(sliceKey: string) {
    if (!ioSelection) return
    onIoSelectionChange({ ...ioSelection, sliceKey })
  }

  return (
    <div className="view-controls" aria-label="chip map overlays">
      <div className="segmented-control" aria-label="map mode">
        <button type="button" className={mapMode === 'chip' ? 'active' : ''} onClick={() => onMapModeChange('chip')}>
          Chip
        </button>
        <button type="button" className={mapMode === 'io' ? 'active' : ''} onClick={() => onMapModeChange('io')}>
          IO Map
        </button>
      </div>
      {mapMode === 'io' ? (
        <>
          <label>
            Direction
            <select
              value={ioSelection?.direction ?? 'input'}
              disabled={!ioSummary?.available}
              onChange={(event) => changeIoDirection(event.target.value === 'output' ? 'output' : 'input')}
            >
              <option value="input">Input</option>
              <option value="output">Output</option>
            </select>
          </label>
          <label>
            Tensor
            <select
              value={selectedTensor?.name ?? ''}
              disabled={!selectedTensor}
              onChange={(event) => changeIoTensor(event.target.value)}
            >
              {activeTensors.map((tensor) => (
                <option key={`${tensor.direction}-${tensor.thread_id}-${tensor.name}`} value={tensor.name}>
                  {tensor.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Slice
            <select
              value={ioSelection?.sliceKey ?? selectedTensor?.slice_keys[0] ?? 'all'}
              disabled={!selectedTensor}
              onChange={(event) => changeIoSlice(event.target.value)}
            >
              {(selectedTensor?.slice_keys.length ? selectedTensor.slice_keys : ['all']).map((sliceKey) => (
                <option key={sliceKey} value={sliceKey}>
                  {sliceKey}
                </option>
              ))}
            </select>
          </label>
        </>
      ) : (
        <>
          <label>
            <input type="checkbox" checked={options.showSignalArrows} onChange={() => toggle('showSignalArrows')} />
            Signal arrows
          </label>
          <label>
            <input
              type="checkbox"
              checked={options.showNeuronDestinations}
              onChange={() => toggle('showNeuronDestinations')}
            />
            Flow arrows
          </label>
          <label>
            <input type="checkbox" checked={options.colorByTickStart} onChange={() => toggle('colorByTickStart')} />
            Tick colors
          </label>
          <label>
            <input
              type="checkbox"
              checked={options.showTickStartLabels}
              onChange={() => toggle('showTickStartLabels')}
            />
            Tick labels
          </label>
          <label>
            <input
              type="checkbox"
              checked={options.highlightSameTickStart}
              onChange={() => toggle('highlightSameTickStart')}
            />
            Link tick select
          </label>
        </>
      )}
    </div>
  )
}
