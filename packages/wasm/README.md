# ruv-fann-wasm

Real ruv-FANN WebAssembly and generated wasm-bindgen bindings, packaged without
npm dependencies. ESM for Node >=20.9 and browsers with WebAssembly support.

```js
import { loadWasm } from 'ruv-fann-wasm';
const { bindings, backend } = await loadWasm({ simd: 'auto' });
const prepared = bindings.WasmPreparedNetwork.from_dense_layers_json(modelJSON);
try { console.log(backend, prepared.run(Float64Array.of(1, 2))); }
finally { prepared.free(); }
```

`loadWasm({simd?: boolean | 'auto'})` returns `{bindings, backend, memory}`.
The default is `auto`, using a WebAssembly SIMD feature probe. `false` selects
`wasm`; `true` selects `wasm-simd` and rejects if unavailable. Core and SIMD use
separate generated module state, class constructors, memories and single-flight
initialization promises. Failed initialization can be retried. Raw objects belong
to their backend and must be freed by their owner. Do not pass raw objects between
backends, or use objects after `free()`.

Node reads installed binaries through `node:fs/promises`; it never downloads a
binary. Browser loaders fetch assets relative to the package. Serve `dist` beside
`src` (or configure your bundler to preserve their relative WASM URLs and dynamic
binding imports). Serve `.wasm` as `application/wasm`; CSP must permit WebAssembly.
No CommonJS entry point or native addon is provided.

The binary contains existing compiled swarm/forecasting exports, but this package
loads no swarm runtime, SQLite or vector dependencies. For an ergonomic dense
inference/training API use `ruv-fann`.

From the repository, `npm run build` verifies fixed SHA-256 hashes before copying
the previously built bindings and core/SIMD binaries from `ruv-swarm/npm/wasm`.
It does not compile Rust or replace unavailable operations. `dist/SHA256SUMS.json`
records the original source hashes; generated `core.js`/`simd.js` are byte-for-byte
binding copies. Build tools/tests are repository tooling and are excluded from the
published package.

MIT licensed. Source: https://github.com/ruvnet/ruv-FANN/tree/main/packages/wasm
