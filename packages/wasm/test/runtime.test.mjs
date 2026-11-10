import assert from 'node:assert/strict';
import { test } from 'node:test';
import { cp, mkdtemp, rm, rename, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';
import { loadWasm, supportsSIMD } from '../src/index.js';

test('real binaries initialize once per backend without Node fetch, and have independent state', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = () => { throw new Error('Node loader used network'); };
  try {
    const core = await Promise.all(Array.from({ length: 12 }, () => loadWasm({ simd: false })));
    assert.ok(core.every(value => value === core[0]));
    assert.equal(core[0].backend, 'wasm');
    assert.ok(core[0].memory instanceof WebAssembly.Memory);
    assert.equal(typeof core[0].bindings.WasmTrainableNetwork, 'function');
    if (supportsSIMD()) {
      const simd = await loadWasm({ simd: true });
      assert.equal(simd.backend, 'wasm-simd');
      assert.notEqual(core[0].memory, simd.memory);
      assert.notEqual(core[0].bindings.WasmPreparedNetwork, simd.bindings.WasmPreparedNetwork);
      assert.equal(await loadWasm(), simd);
    }
  } finally { globalThis.fetch = originalFetch; }
});

test('unknown and coercible loader options reject', async () => {
  for (const value of [null, [], { simd: null }, { simd: 'yes' }, { simd: 1 }, { other: true }])
    await assert.rejects(loadWasm(value), TypeError);
});

test('missing and corrupt actual binary failures can be repaired and retried', async () => {
  const temporary = await mkdtemp(join(tmpdir(), 'fann-loader-retry-'));
  try {
    await cp(new URL('../', import.meta.url), temporary, { recursive: true });
    const url = pathToFileURL(join(temporary, 'src/index.js'));
    const runtime = await import(url.href);
    const binary = join(temporary, 'dist/ruv_swarm_wasm_bg.wasm');
    await rename(binary, `${binary}.saved`);
    const failures = await Promise.allSettled([runtime.loadWasm({ simd: false }), runtime.loadWasm({ simd: false })]);
    assert.ok(failures.every(result => result.status === 'rejected'));
    await writeFile(binary, 'invalid actual WASM');
    await assert.rejects(runtime.loadWasm({ simd: false }), WebAssembly.CompileError);
    await rename(`${binary}.saved`, binary);
    const repaired = await runtime.loadWasm({ simd: false });
    assert.equal(repaired.backend, 'wasm');
    assert.ok(repaired.memory instanceof WebAssembly.Memory);
  } finally { await rm(temporary, { recursive: true, force: true }); }
});
