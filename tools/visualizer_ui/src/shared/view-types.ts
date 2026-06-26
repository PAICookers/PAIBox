import type { IoDirection } from '../types'

export type CoreInspectorTab = 'core' | 'lut' | 'neurons' | 'weights'
export type InspectorTab = CoreInspectorTab | 'raw'
export type NonRawInspectorTab = CoreInspectorTab
export type ValueMode = 'semantic' | 'raw'
export type MapMode = 'chip' | 'io'

export interface OverlayOptions {
  showSignalArrows: boolean
  showNeuronDestinations: boolean
  colorByTickStart: boolean
  showTickStartLabels: boolean
  highlightSameTickStart: boolean
}

export interface IoSelection {
  direction: IoDirection
  tensorName: string
  sliceKey: string
}
