import type * as Bindings from '../dist/ruv_swarm_wasm.js';
export interface RuntimeOptions { simd?: boolean | 'auto' }
export interface WasmRuntime {
  readonly backend: 'wasm' | 'wasm-simd';
  /** Actual generated wasm-bindgen exports. Raw objects require free() by the owner. */
  readonly bindings: typeof Bindings;
  readonly memory: WebAssembly.Memory;
}
export function supportsSIMD(): boolean;
export function validateRuntimeOptions(options?: RuntimeOptions): boolean | 'auto';
export function loadWasm(options?: RuntimeOptions): Promise<WasmRuntime>;
