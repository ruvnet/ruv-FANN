/** Copy already-verified upstream assets; no Rust build, downloads or substitutions. */
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const root = new URL('../', import.meta.url);
const source = new URL('ruv-swarm/npm/wasm/', root);
const destination = new URL('packages/wasm/dist/', root);
const expected = {
  'ruv_swarm_wasm.js': '1d1d834f2d114e4ccfdb30286b06f9c10d7bbb13af6e34159c5618e2e31f4c1a',
  'ruv_swarm_wasm_bg.wasm': 'a46f30ba9f7abafa7e77fbd8b686ab8b4785dd325f3e792b5a0fb39628a894f9',
  'ruv_swarm_simd.wasm': '4aec422bae909a8e3b765af396834522520c1432c61b6c37aac72e093d27ee01',
  'ruv_swarm_wasm.d.ts': 'd4075989bd5c677878436c22f1aa36472540af1c1c151e628998f579424bed0f',
  'ruv_swarm_wasm_bg.wasm.d.ts': 'c72c5517098b6717aac53873cddb926d0cf83baedf8f48c552dd9eb71adbc5d1'
};
// Validate every source before touching the generated output.
for (const [file, hash] of Object.entries(expected)) {
  const actual = createHash('sha256').update(await readFile(new URL(file, source))).digest('hex');
  if (actual !== hash) throw new Error(`Verified asset hash mismatch: ${file}: ${actual}`);
}
await mkdir(destination, { recursive: true });
for (const file of Object.keys(expected)) await copyFile(new URL(file, source), new URL(file, destination));
// wasm-bindgen stores module state at module scope. Distinct files are essential:
// using a single import for both binaries silently makes the second load reuse the first.
await copyFile(new URL('ruv_swarm_wasm.js', source), new URL('core.js', destination));
await copyFile(new URL('ruv_swarm_wasm.js', source), new URL('simd.js', destination));
await writeFile(new URL('SHA256SUMS.json', destination), `${JSON.stringify(expected, null, 2)}\n`);
console.log('Verified and copied 5 upstream WASM/binding assets; generated isolated core/SIMD bindings.');
