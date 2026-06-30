import { useState } from 'react'

import type { ChipOverview } from '../../types'
import { getSops, getTickStart } from '../../shared/core'
import { TinyMetric } from '../../shared/components'
import { formatCompactNumber } from '../../shared/format'

const TICK_COLORS = [
  '#c7ddff',
  '#cdeec4',
  '#ffe0a6',
  '#f3c3c8',
  '#bae8e4',
  '#e3d1ff',
  '#d7e2a2',
  '#ffcdb2',
  '#b8e0ff',
  '#f4d0ff',
  '#bde6c8',
  '#f0d6a7',
  '#cbd5ff',
  '#ffd0dd',
  '#b7ead8',
  '#e9e2b3',
  '#d2e9ff',
  '#e0d7c6',
  '#bfe0c1',
  '#ffd8ba',
  '#cbd9d4',
  '#efcbd2',
  '#c3e7f1',
  '#ded8ff',
  '#d9e6b4',
  '#ffc7a8',
  '#c9eeef',
  '#eed1b3',
  '#bfcfff',
  '#f6d3ea',
  '#c9ead4',
  '#ffe7a8',
  '#bddfee',
  '#d7dcc1',
  '#ffd1c4',
  '#c6e4b4',
]
const GOLDEN_ANGLE = 137.508
const PRESSURE_CHART_WIDTH = 1080
const PRESSURE_CHART_HEIGHT = 260
const PRESSURE_CHART_PAD_LEFT = 76
const PRESSURE_CHART_PAD_RIGHT = 30
const PRESSURE_CHART_PAD_TOP = 44
const PRESSURE_CHART_PAD_BOTTOM = 56
const PRESSURE_CHART_GROUP_WIDTH = 54
const PRESSURE_CHART_GROUP_GAP = 18
const PRESSURE_CHART_BAR_WIDTH = 18

interface TickComputeCore {
  x: number
  y: number
  sops: number
}

export interface TickComputeRow {
  tickStart: number
  color: string
  totalSops: number
  averageSops: number | null
  medianSops: number | null
  maxCoreSops: number
  activeCoreCount: number
  topCores: TickComputeCore[]
}

interface TickComputeSummary {
  overallTotalSops: number
  peakTickByTotal: TickComputeRow | null
  peakTickByMaxCore: TickComputeRow | null
  maxTotalSops: number
}

type TickComputeSortKey = 'tick' | 'total' | 'average' | 'median' | 'core'
type SortDirection = 'asc' | 'desc'

export function buildTickComputeRows(
  chip: ChipOverview,
  includePaddingInSops: boolean,
  tickColorMap: Map<number, string>,
): TickComputeRow[] {
  const maxTickStart = Math.max(...chip.cores.map(getTickStart).filter((value): value is number => value !== null), 0)
  if (maxTickStart <= 0) return []

  return Array.from({ length: maxTickStart }, (_, index) => {
    const tickStart = index + 1
    const cores = chip.cores
      .filter((core) => getTickStart(core) === tickStart)
      .map((core) => ({ x: core.x, y: core.y, sops: getSops(core, includePaddingInSops) }))
      .sort((a, b) => b.sops - a.sops || a.y - b.y || a.x - b.x)
    const totalSops = cores.reduce((total, core) => total + core.sops, 0)
    const coreSops = cores.map((core) => core.sops)
    return {
      tickStart,
      color: tickColorMap.get(tickStart) ?? '#e6ebf2',
      totalSops,
      averageSops: cores.length ? totalSops / cores.length : null,
      medianSops: median(coreSops),
      maxCoreSops: cores[0]?.sops ?? 0,
      activeCoreCount: cores.length,
      topCores: cores.slice(0, 3),
    }
  })
}

function buildComputeSummary(rows: TickComputeRow[]): TickComputeSummary {
  const peakTickByTotal = rows.reduce<TickComputeRow | null>(
    (peak, row) => (peak === null || row.totalSops > peak.totalSops ? row : peak),
    null,
  )
  const peakTickByMaxCore = rows.reduce<TickComputeRow | null>(
    (peak, row) => (peak === null || row.maxCoreSops > peak.maxCoreSops ? row : peak),
    null,
  )
  return {
    overallTotalSops: rows.reduce((total, row) => total + row.totalSops, 0),
    peakTickByTotal,
    peakTickByMaxCore,
    maxTotalSops: peakTickByTotal?.totalSops ?? 0,
  }
}

function sortTickComputeRows(
  rows: TickComputeRow[],
  sortKey: TickComputeSortKey,
  sortDirection: SortDirection,
): TickComputeRow[] {
  const multiplier = sortDirection === 'asc' ? 1 : -1
  return [...rows].sort((a, b) => {
    const valueA = tickComputeSortValue(a, sortKey)
    const valueB = tickComputeSortValue(b, sortKey)
    if (valueA !== valueB) return (valueA - valueB) * multiplier
    return a.tickStart - b.tickStart
  })
}

function tickComputeSortValue(row: TickComputeRow, sortKey: TickComputeSortKey): number {
  if (sortKey === 'total') return row.totalSops
  if (sortKey === 'average') return row.averageSops ?? Number.NEGATIVE_INFINITY
  if (sortKey === 'median') return row.medianSops ?? Number.NEGATIVE_INFINITY
  if (sortKey === 'core') return row.activeCoreCount
  return row.tickStart
}

function median(values: number[]): number | null {
  if (!values.length) return null
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  if (sorted.length % 2 === 1) return sorted[middle]
  return (sorted[middle - 1] + sorted[middle]) / 2
}

function tickBarWidth(row: TickComputeRow, maxTotalSops: number): string {
  if (row.totalSops <= 0 || maxTotalSops <= 0) return '0%'
  return `${Math.max((row.totalSops / maxTotalSops) * 100, 6).toFixed(1)}%`
}

function tickComputeTitle(row: TickComputeRow): string {
  const topCores = row.topCores.length
    ? row.topCores.map((core) => `(${core.x},${core.y}) ${formatCompactNumber(core.sops)}`).join(', ')
    : 'none'
  return [
    `tick ${row.tickStart}`,
    `total SOPS: ${row.totalSops.toLocaleString()}`,
    `average SOPS: ${formatPressureValue(row.averageSops)}`,
    `median SOPS: ${formatPressureValue(row.medianSops)}`,
    `max core SOPS: ${row.maxCoreSops.toLocaleString()}`,
    `active cores: ${row.activeCoreCount}`,
    `top cores: ${topCores}`,
  ].join('\n')
}

function formatPeakTick(row: TickComputeRow | null): string {
  if (row === null || row.totalSops <= 0) return '-'
  return `${row.tickStart} ${formatCompactNumber(row.totalSops)}`
}

function formatPeakCore(row: TickComputeRow | null): string {
  const top = row?.topCores[0]
  if (!top || top.sops <= 0) return '-'
  return `(${top.x},${top.y}) ${formatCompactNumber(top.sops)}`
}

function formatPressureValue(value: number | null): string {
  if (value === null) return '-'
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(1)
}

function formatCompactPressure(value: number | null): string {
  return value === null ? '-' : formatCompactNumber(value)
}

function tickColorAt(index: number): string {
  if (index < TICK_COLORS.length) return TICK_COLORS[index]
  const hue = (index * GOLDEN_ANGLE) % 360
  return `hsl(${hue.toFixed(1)} 68% 86%)`
}

export function buildTickColorMap(chip: ChipOverview): Map<number, string> {
  const values = [...new Set(chip.cores.map(getTickStart).filter((value): value is number => value !== null))].sort(
    (a, b) => a - b,
  )
  return new Map(values.map((value, index) => [value, tickColorAt(index)]))
}

export function TickComputePanel({
  rows,
  highlighted,
  onSelect,
}: {
  rows: TickComputeRow[]
  highlighted: number | null
  onSelect: (tickStart: number | null) => void
}) {
  const [sortKey, setSortKey] = useState<TickComputeSortKey>('tick')
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc')
  const computeSummary = buildComputeSummary(rows)
  const sortedRows = sortTickComputeRows(rows, sortKey, sortDirection)
  function changeSort(nextKey: TickComputeSortKey) {
    if (sortKey === nextKey) {
      setSortDirection((current) => (current === 'asc' ? 'desc' : 'asc'))
      return
    }
    setSortKey(nextKey)
    setSortDirection(nextKey === 'tick' ? 'asc' : 'desc')
  }
  if (!rows.length) return <p className="muted">No tick_start values.</p>
  return (
    <div className="tick-panel">
      <div className="tick-compute-summary">
        <TinyMetric label="Total SOPS" value={formatCompactNumber(computeSummary.overallTotalSops)} />
        <TinyMetric label="Peak tick" value={formatPeakTick(computeSummary.peakTickByTotal)} />
        <TinyMetric label="Peak core" value={formatPeakCore(computeSummary.peakTickByMaxCore)} />
      </div>
      <div className="tick-row tick-head" aria-label="Tick compute sort controls">
        <TickSortButton
          label="tick"
          sortKey="tick"
          activeKey={sortKey}
          direction={sortDirection}
          onClick={changeSort}
        />
        <TickSortButton
          label="total"
          sortKey="total"
          activeKey={sortKey}
          direction={sortDirection}
          onClick={changeSort}
        />
        <TickSortButton
          label="avg"
          sortKey="average"
          activeKey={sortKey}
          direction={sortDirection}
          onClick={changeSort}
        />
        <TickSortButton
          label="med"
          sortKey="median"
          activeKey={sortKey}
          direction={sortDirection}
          onClick={changeSort}
        />
        <TickSortButton
          label="core"
          sortKey="core"
          activeKey={sortKey}
          direction={sortDirection}
          onClick={changeSort}
        />
      </div>
      {sortedRows.map((row) => (
        <button
          type="button"
          className={[
            'tick-row',
            row.activeCoreCount ? '' : 'empty',
            highlighted === row.tickStart ? 'active' : '',
          ].join(' ')}
          key={row.tickStart}
          onClick={() => onSelect(highlighted === row.tickStart ? null : row.tickStart)}
          title={tickComputeTitle(row)}
        >
          <span className="tick-name">
            <span className="tick-swatch" style={{ background: row.color }} />
            {row.tickStart}
          </span>
          <span className="tick-sops">
            <span style={{ width: tickBarWidth(row, computeSummary.maxTotalSops), background: row.color }} />
            <strong>{formatCompactNumber(row.totalSops)}</strong>
            <small>
              avg {formatCompactPressure(row.averageSops)} / med {formatCompactPressure(row.medianSops)}
            </small>
          </span>
          <span className="tick-count">{row.activeCoreCount}</span>
        </button>
      ))}
    </div>
  )
}

export function TickPressureBarChart({ rows }: { rows: TickComputeRow[] }) {
  if (!rows.length) return null
  const maxPressure = Math.max(
    ...rows.flatMap((row) => [row.averageSops ?? 0, row.medianSops ?? 0]).filter((value) => value > 0),
    0,
  )
  const chartWidth = Math.max(
    PRESSURE_CHART_WIDTH,
    PRESSURE_CHART_PAD_LEFT +
      PRESSURE_CHART_PAD_RIGHT +
      rows.length * PRESSURE_CHART_GROUP_WIDTH +
      Math.max(rows.length - 1, 0) * PRESSURE_CHART_GROUP_GAP,
  )
  const plotHeight = PRESSURE_CHART_HEIGHT - PRESSURE_CHART_PAD_TOP - PRESSURE_CHART_PAD_BOTTOM
  const baselineY = PRESSURE_CHART_HEIGHT - PRESSURE_CHART_PAD_BOTTOM
  const chartId = 'tick-pressure-bar-chart-title'
  const subtitleId = 'tick-pressure-bar-chart-desc'

  function barHeight(value: number | null): number {
    if (value === null || value <= 0 || maxPressure <= 0) return 0
    return Math.max((value / maxPressure) * plotHeight, 1)
  }

  return (
    <section className="tick-pressure-chart" aria-labelledby={chartId} aria-describedby={subtitleId}>
      <div className="tick-pressure-chart-title">
        <div>
          <h2 id={chartId}>Layer Pressure</h2>
          <p id={subtitleId}>Average and median SOPS by tick_start</p>
        </div>
        <div className="tick-pressure-legend" aria-label="pressure series">
          <span>
            <span className="tick-pressure-key avg" />
            avg
          </span>
          <span>
            <span className="tick-pressure-key median" />
            median
          </span>
        </div>
      </div>
      <div className="tick-pressure-scroll">
        <svg
          className="tick-pressure-svg"
          viewBox={`0 0 ${chartWidth} ${PRESSURE_CHART_HEIGHT}`}
          role="img"
          aria-labelledby={chartId}
          aria-describedby={subtitleId}
        >
          <line
            className="tick-pressure-axis"
            x1={PRESSURE_CHART_PAD_LEFT}
            y1={baselineY}
            x2={chartWidth - PRESSURE_CHART_PAD_RIGHT}
            y2={baselineY}
          />
          <line
            className="tick-pressure-axis"
            x1={PRESSURE_CHART_PAD_LEFT}
            y1={PRESSURE_CHART_PAD_TOP}
            x2={PRESSURE_CHART_PAD_LEFT}
            y2={baselineY}
          />
          <line
            className="tick-pressure-axis muted"
            x1={PRESSURE_CHART_PAD_LEFT}
            y1={PRESSURE_CHART_PAD_TOP}
            x2={chartWidth - PRESSURE_CHART_PAD_RIGHT}
            y2={PRESSURE_CHART_PAD_TOP}
          />
          <text className="tick-pressure-yvalue" x={PRESSURE_CHART_PAD_LEFT - 10} y={PRESSURE_CHART_PAD_TOP + 4}>
            {formatCompactNumber(maxPressure)}
          </text>
          <text
            className="tick-pressure-axis-label y"
            x={18}
            y={(PRESSURE_CHART_PAD_TOP + baselineY) / 2}
            transform={`rotate(-90 18 ${(PRESSURE_CHART_PAD_TOP + baselineY) / 2})`}
          >
            SOPS per active core
          </text>
          <text className="tick-pressure-axis-label x" x={chartWidth / 2} y={PRESSURE_CHART_HEIGHT - 14}>
            tick_start layer
          </text>
          {rows.map((row, index) => {
            const groupX = PRESSURE_CHART_PAD_LEFT + index * (PRESSURE_CHART_GROUP_WIDTH + PRESSURE_CHART_GROUP_GAP)
            const averageHeight = barHeight(row.averageSops)
            const medianHeight = barHeight(row.medianSops)
            const hasPressure = row.averageSops !== null || row.medianSops !== null
            return (
              <g className={hasPressure ? 'tick-pressure-group' : 'tick-pressure-group empty'} key={row.tickStart}>
                <rect
                  className="tick-pressure-slot"
                  x={groupX - 3}
                  y={PRESSURE_CHART_PAD_TOP}
                  width={PRESSURE_CHART_GROUP_WIDTH + 6}
                  height={plotHeight}
                  rx={4}
                />
                {row.averageSops !== null && (
                  <rect
                    className="tick-pressure-bar avg"
                    x={groupX + 7}
                    y={baselineY - averageHeight}
                    width={PRESSURE_CHART_BAR_WIDTH}
                    height={averageHeight}
                    rx={3}
                  >
                    <title>{tickComputeTitle(row)}</title>
                  </rect>
                )}
                {row.averageSops !== null && (
                  <text
                    className="tick-pressure-value avg"
                    x={groupX + 7 + PRESSURE_CHART_BAR_WIDTH / 2}
                    y={baselineY - averageHeight - 6}
                  >
                    {formatCompactPressure(row.averageSops)}
                  </text>
                )}
                {row.medianSops !== null && (
                  <rect
                    className="tick-pressure-bar median"
                    x={groupX + 29}
                    y={baselineY - medianHeight}
                    width={PRESSURE_CHART_BAR_WIDTH}
                    height={medianHeight}
                    rx={3}
                  >
                    <title>{tickComputeTitle(row)}</title>
                  </rect>
                )}
                {row.medianSops !== null && (
                  <text
                    className="tick-pressure-value median"
                    x={groupX + 29 + PRESSURE_CHART_BAR_WIDTH / 2}
                    y={baselineY - medianHeight - 6}
                  >
                    {formatCompactPressure(row.medianSops)}
                  </text>
                )}
                <text className="tick-pressure-xlabel" x={groupX + PRESSURE_CHART_GROUP_WIDTH / 2} y={baselineY + 17}>
                  {row.tickStart}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
    </section>
  )
}

function TickSortButton({
  label,
  sortKey,
  activeKey,
  direction,
  onClick,
}: {
  label: string
  sortKey: TickComputeSortKey
  activeKey: TickComputeSortKey
  direction: SortDirection
  onClick: (sortKey: TickComputeSortKey) => void
}) {
  const active = sortKey === activeKey
  return (
    <button
      type="button"
      className={active ? 'active' : ''}
      aria-label={`Sort by ${label} ${active ? direction : ''}`.trim()}
      onClick={() => onClick(sortKey)}
    >
      <span>{label}</span>
      <span className="sort-arrows" aria-hidden="true">
        <span className={active && direction === 'asc' ? 'sort-up active' : 'sort-up'} />
        <span className={active && direction === 'desc' ? 'sort-down active' : 'sort-down'} />
      </span>
    </button>
  )
}
