import { cp, rm } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const uiRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repoRoot = resolve(uiRoot, '..', '..')
const source = resolve(uiRoot, 'dist')
const target = resolve(repoRoot, 'paibox', 'visualizer', 'static')

await rm(target, { recursive: true, force: true })
await cp(source, target, { recursive: true })
console.log(`copied visualizer UI assets to ${target}`)
