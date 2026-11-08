/** A trusted external cuda2rust executable is required. Nothing is downloaded on install. */
export interface CompilerOptions {
  /** Executable path or name; defaults to CUDA2RUST_BINARY, then cuda2rust on PATH. */
  binary?: string;
  /** Per-process timeout: 1–120000 ms, default 30000. */
  timeoutMs?: number;
  /** Per-stream output cap: 1–16777216 bytes, default 4194304. */
  maxOutputBytes?: number;
}
export interface TranspileOptions extends CompilerOptions {
  /** Generate source text only. Defaults to rust. */
  target?: 'rust' | 'wgsl';
}
export interface TranspileResult {
  code: string;
  target: 'rust' | 'wgsl';
  compilerVersion: string;
}
export class CudaWasmError extends Error {
  constructor(code: string, message: string, details?: { exitCode?: number; compilerStderr?: string });
  code: string;
  exitCode?: number;
  compilerStderr?: string;
}
export const MAX_INPUT_BYTES: number;
/** Transpile at most 1 MiB of CUDA source; no kernel compilation or execution occurs. */
export function transpileCuda(source: string, options?: TranspileOptions): Promise<TranspileResult>;
/** Read a regular UTF-8 file with the same input bound, then generate source. */
export function transpileFile(inputPath: string, options?: TranspileOptions): Promise<TranspileResult>;
/** Verify the external compiler identifies itself as cuda2rust 0.2.0 or newer. */
export function getCompilerVersion(options?: CompilerOptions): Promise<string>;
/** Return this npm package's version without starting a compiler. */
export function getVersion(): string;
declare const api: {
  transpileCuda: typeof transpileCuda;
  transpileFile: typeof transpileFile;
  getCompilerVersion: typeof getCompilerVersion;
  getVersion: typeof getVersion;
  CudaWasmError: typeof CudaWasmError;
  MAX_INPUT_BYTES: typeof MAX_INPUT_BYTES;
};
export default api;
