import type {
  ChipOverview,
  CoreView,
  IoDirection,
  IoCoreView,
  IoSummary,
  PageResponse,
  TensorRegionView,
  ValidationEntry,
  ViewerSummary,
} from './types'

async function getJson<T>(url: string): Promise<T> {
  // The packaged UI is served by the same FastAPI app as `/api`, so relative
  // URLs work both in wheel installs and in Vite dev-server proxy mode.
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export async function fetchSummary(): Promise<ViewerSummary> {
  return getJson<ViewerSummary>('/api/summary')
}

export async function fetchChips(): Promise<ChipOverview[]> {
  return getJson<ChipOverview[]>('/api/chips')
}

export async function fetchCore(chipId: number, x: number, y: number): Promise<CoreView> {
  return getJson<CoreView>(`/api/cores/${chipId}/${x}/${y}`)
}

export async function fetchIoSummary(): Promise<IoSummary> {
  return getJson<IoSummary>('/api/io/summary')
}

export async function fetchIoCore(chipId: number, x: number, y: number): Promise<IoCoreView> {
  return getJson<IoCoreView>(`/api/io/cores/${chipId}/${x}/${y}`)
}

export async function fetchIoRegions(params: {
  direction: IoDirection
  threadId?: number
  tensorName?: string
  sliceKey?: string
  chipId?: number
  x?: number
  y?: number
  offset?: number
  limit?: number
}): Promise<PageResponse<TensorRegionView>> {
  // Per-core IO uses `/api/io/cores/...`. This paged endpoint is for global
  // region fallback, mainly when output mappings cannot be attributed to a core.
  const query = new URLSearchParams()
  query.set('direction', params.direction)
  query.set('offset', String(params.offset ?? 0))
  query.set('limit', String(params.limit ?? 100))
  if (params.threadId !== undefined) query.set('thread_id', String(params.threadId))
  if (params.tensorName !== undefined) query.set('tensor_name', params.tensorName)
  if (params.sliceKey !== undefined) query.set('slice_key', params.sliceKey)
  if (params.chipId !== undefined) query.set('chip_id', String(params.chipId))
  if (params.x !== undefined) query.set('x', String(params.x))
  if (params.y !== undefined) query.set('y', String(params.y))
  return getJson<PageResponse<TensorRegionView>>(`/api/io/regions?${query}`)
}

export async function fetchValidation(): Promise<ValidationEntry[]> {
  return getJson<ValidationEntry[]>('/api/validation')
}
