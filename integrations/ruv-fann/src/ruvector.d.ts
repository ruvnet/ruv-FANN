import type { Metric, VectorStore } from './index.js';
export interface RuvectorOptions {
  /** Explicit bytes of the pinned published RuVector WASM. No implicit fetch. */
  wasmBytes: Uint8Array;
  dimensions: number;
  metric?: Metric;
  useHnsw?: boolean;
  maxEntries?: number;
}
export function createRuvectorStore(options: RuvectorOptions): Promise<VectorStore>;
