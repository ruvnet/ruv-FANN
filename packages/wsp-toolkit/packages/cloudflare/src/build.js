import { build } from 'esbuild';
import { mkdir, copyFile, stat } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** Bundle the exact installed bindings and statically import both precompiled Wasm modules. */
export async function buildWorker(outDirectory) {
  if (typeof outDirectory !== 'string' || !outDirectory.length) throw new TypeError('outDirectory is required');
  const output = resolve(outDirectory);
  const wasmRoot = resolve(dirname(fileURLToPath(import.meta.resolve('ruv-fann-wasm'))), '..');
  const assets = [
    [join(wasmRoot, 'dist', 'ruv_swarm_wasm_bg.wasm'), join(output, 'core.wasm')],
    [join(wasmRoot, 'dist', 'ruv_swarm_simd.wasm'), join(output, 'simd.wasm')]
  ];
  // Fail before touching output if an installed runtime is incomplete.
  await Promise.all(assets.map(([source]) => stat(source)));
  await mkdir(output, { recursive: true });
  await Promise.all(assets.map(([source, destination]) => copyFile(source, destination)));
  const loader = `
import * as core from ${JSON.stringify(join(wasmRoot, 'dist', 'core.js'))};
import * as simd from ${JSON.stringify(join(wasmRoot, 'dist', 'simd.js'))};
import coreModule from './core.wasm';
import simdModule from './simd.wasm';
const flights = new Map();
export function supportsSIMD() { return true; }
export function validateRuntimeOptions(options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options) || Object.keys(options).some(k => k !== 'simd'))
    throw new TypeError('unknown runtime options');
  const value = options.simd === undefined ? 'auto' : options.simd;
  if (value !== true && value !== false && value !== 'auto') throw new TypeError('simd must be boolean or auto');
  return value;
}
export async function loadWasm(options = {}) {
  const requested = validateRuntimeOptions(options);
  const accelerated = requested !== false;
  const key = accelerated ? 'simd' : 'core';
  if (!flights.has(key)) {
    const bindings = accelerated ? simd : core;
    const module = accelerated ? simdModule : coreModule;
    const flight = Promise.resolve().then(() => {
      const wasm = bindings.initSync({ module });
      if (!(wasm.memory instanceof WebAssembly.Memory)) throw new Error('WASM memory export missing');
      return Object.freeze({ backend: accelerated ? 'wasm-simd' : 'wasm', bindings, memory: wasm.memory });
    });
    flights.set(key, flight);
    flight.catch(() => { if (flights.get(key) === flight) flights.delete(key); });
  }
  return flights.get(key);
}
`;
  await build({
    entryPoints: [fileURLToPath(new URL('./worker.js', import.meta.url))],
    outfile: join(output, 'worker.js'), bundle: true, format: 'esm', platform: 'browser',
    target: 'es2022', minify: true, legalComments: 'none', define: { process: 'undefined' },
    plugins: [{ name: 'wsp-static-wasm', setup(builder) {
      builder.onResolve({ filter: /^ruv-fann-wasm$/ }, () => ({ path: 'runtime', namespace: 'wsp-static-wasm' }));
      builder.onLoad({ filter: /.*/, namespace: 'wsp-static-wasm' }, () => ({ contents: loader, resolveDir: wasmRoot }));
      builder.onResolve({ filter: /^\.\/(core|simd)\.wasm$/ }, args => ({ path: args.path, external: true }));
    } }]
  });
  return Object.freeze({ script: join(output, 'worker.js'), coreWasm: join(output, 'core.wasm'), simdWasm: join(output, 'simd.wasm') });
}
