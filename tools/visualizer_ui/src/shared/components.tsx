import { Fragment, useEffect, useState } from 'react'
import type { SubmitEventHandler } from 'react'

import type { ValidationEntry } from '../types'
import { clamp } from './format'

export function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export function TinyMetric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="tiny-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export function SimpleKv({ entries }: { entries: Record<string, number | string> }) {
  return (
    <dl className="kv compact">
      {Object.entries(entries).map(([key, value]) => (
        <Fragment key={key}>
          <dt>{key}</dt>
          <dd>{String(value)}</dd>
        </Fragment>
      ))}
    </dl>
  )
}

export function Pager({
  page,
  pageCount,
  total,
  onPage,
}: {
  page: number
  pageCount: number
  total: number
  onPage: (page: number) => void
}) {
  const [pageText, setPageText] = useState(String(page + 1))

  useEffect(() => {
    setPageText(String(page + 1))
  }, [page])

  function commitPageValue() {
    const parsed = Number(pageText)
    if (!Number.isInteger(parsed)) {
      setPageText(String(page + 1))
      return
    }
    onPage(clamp(parsed - 1, 0, pageCount - 1))
  }

  const commitPage: SubmitEventHandler<HTMLFormElement> = (event) => {
    event.preventDefault()
    commitPageValue()
  }

  return (
    <div className="pager">
      <span>{total} items</span>
      <button type="button" disabled={page <= 0} onClick={() => onPage(page - 1)}>
        Prev
      </button>
      <form className="page-jump" onSubmit={commitPage}>
        <input
          aria-label="Page number"
          inputMode="numeric"
          min={1}
          max={pageCount}
          type="number"
          value={pageText}
          onBlur={commitPageValue}
          onChange={(event) => setPageText(event.target.value)}
        />
        <span>/ {pageCount}</span>
      </form>
      <button type="button" disabled={page + 1 >= pageCount} onClick={() => onPage(page + 1)}>
        Next
      </button>
    </div>
  )
}

export function SourceFrameButton({
  frameIndex,
  onOpenRaw,
}: {
  frameIndex: number | null
  onOpenRaw: (frameIndex: number | null) => void
}) {
  const disabled = frameIndex === null
  return (
    <button
      type="button"
      className="source-link"
      disabled={disabled}
      title={disabled ? 'No source frame' : `Open raw frame ${frameIndex}`}
      onClick={() => onOpenRaw(frameIndex)}
    >
      {disabled ? 'F#-' : `F#${frameIndex}`}
    </button>
  )
}

export function RawHexStrip({ values }: { values: string[] }) {
  return (
    <div className="raw-hex-strip">
      {values.map((value, index) => (
        <code key={`${value}-${index}`}>{value}</code>
      ))}
    </div>
  )
}

export function ValidationList({ entries }: { entries: ValidationEntry[] }) {
  if (!entries.length) return <p className="muted">No validation messages.</p>
  return (
    <div className="validation-list">
      {entries.slice(0, 20).map((entry, index) => (
        <div className={`validation validation-${entry.severity}`} key={`${entry.code}-${index}`}>
          <strong>{entry.severity}</strong>
          <span>{entry.code}</span>
        </div>
      ))}
    </div>
  )
}
