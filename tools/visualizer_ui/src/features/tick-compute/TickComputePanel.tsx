import { useState } from 'react'

import type { ChipOverview } from '../../types'
import { getSops, getTickStart } from '../../shared/core'
import { TinyMetric } from '../../shared/components'
import { formatCompactNumber } from '../../shared/format'

const TICK_COLORS = [
  '#cfe1ff',
  '#d8f0d2',
  '#ffe2b8',
  '#e7d8ff',
  '#bfe8e4',
  '#ffd3d8',
  '#d8ecff',
  '#f3e6bd',
  '#cdebd8',
  '#f4d5ff',
  '#d8dde8',
  '#bfe1ff',
]

interface TickComputeCore {
  x: number
  y: number
  sops: number
}

interface TickComputeRow {
  tickStart: number
  color: string
  totalSops: number
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

type TickComputeSortKey = 'tick' | 'sops' | 'core'
type SortDirection = 'asc' | 'desc'

function buildTickComputeRows(
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
    return {
      tickStart,
      color: tickColorMap.get(tickStart) ?? '#e6ebf2',
      totalSops,
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
  if (sortKey === 'sops') return row.totalSops
  if (sortKey === 'core') return row.activeCoreCount
  return row.tickStart
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

export function buildTickColorMap(chip: ChipOverview): Map<number, string> {
  const values = [...new Set(chip.cores.map(getTickStart).filter((value): value is number => value !== null))].sort(
    (a, b) => a - b,
  )
  return new Map(values.map((value, index) => [value, TICK_COLORS[index % TICK_COLORS.length]]))
}

export function TickComputePanel({
  chip,
  includePaddingInSops,
  tickColorMap,
  highlighted,
  onSelect,
}: {
  chip: ChipOverview
  includePaddingInSops: boolean
  tickColorMap: Map<number, string>
  highlighted: number | null
  onSelect: (tickStart: number | null) => void
}) {
  const [sortKey, setSortKey] = useState<TickComputeSortKey>('tick')
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc')
  const rows = buildTickComputeRows(chip, includePaddingInSops, tickColorMap)
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
          label="SOPS"
          sortKey="sops"
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
          </span>
          <span className="tick-count">{row.activeCoreCount}</span>
        </button>
      ))}
    </div>
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
