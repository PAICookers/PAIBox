import type { ChipOverview, CoreOverview } from '../types'

export function getTickStart(core: CoreOverview): number | null {
  const raw = core.core_config.tick_start
  return typeof raw === 'number' ? raw : null
}

export function getSops(core: CoreOverview, includePadding: boolean): number {
  const summary = core.neurons.summary
  if (includePadding) {
    return summary.sops_with_padding ?? summary.synops_pressure
  }
  return summary.sops_without_padding ?? summary.synops_pressure
}

export function collectThreads(chip: ChipOverview): number[] {
  return [...new Set(chip.cores.map((core) => core.thread_id).filter((value): value is number => value !== null))].sort(
    (a, b) => a - b,
  )
}
