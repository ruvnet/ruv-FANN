const flights = new Map();

/** Runtime feature probe, independent of host flags and user-agent strings. */
export function supportsSIMD() {
  return typeof WebAssembly === 'object' && WebAssembly.validate(Uint8Array.of(
    0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11));
}

export function validateRuntimeOptions(options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options) ||
      Object.keys(options).some(key => key !== 'simd')) throw new TypeError('unknown runtime options');
  const simd = options.simd === undefined ? 'auto' : options.simd;
  if (simd !== true && simd !== false && simd !== 'auto') throw new TypeError('simd must be boolean or auto');
  return simd;
}

/** Single-flight per backend. Rejections are removed so a later call can retry. */
export async function loadWasm(options = {}) {
  const requested = validateRuntimeOptions(options);
  const simd = requested === 'auto' ? supportsSIMD() : requested;
  const key = simd ? 'simd' : 'core';
  if (!flights.has(key)) {
    const flight = initialize(simd);
    flights.set(key, flight);
    flight.catch(() => { if (flights.get(key) === flight) flights.delete(key); });
  }
  return flights.get(key);
}

async function initialize(simd) {
  const bindings = simd ? await import('../dist/simd.js') : await import('../dist/core.js');
  const url = simd ? new URL('../dist/ruv_swarm_simd.wasm', import.meta.url) :
    new URL('../dist/ruv_swarm_wasm_bg.wasm', import.meta.url);
  let source;
  if (typeof process === 'object' && process.versions?.node) {
    // Always read the installed asset directly. Node initialization never uses fetch.
    const nodeFS = 'node:fs/promises';
    source = await (await import(nodeFS)).readFile(url);
  } else {
    source = await fetch(url);
  }
  const wasm = await bindings.default({ module_or_path: source });
  if (!(wasm.memory instanceof WebAssembly.Memory)) throw new Error('WASM memory export missing');
  return Object.freeze({ backend: simd ? 'wasm-simd' : 'wasm', bindings, memory: wasm.memory });
}
