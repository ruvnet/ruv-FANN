/** Offline installed-tarball proof. Optional RUV_FANN_TSC points at a host tsc binary. */
import { execFileSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import './npm-modules-build.mjs';
const root = fileURLToPath(new URL('../', import.meta.url));
const destination = process.argv[2] ? resolve(process.argv[2]) : await mkdtemp(join(tmpdir(), 'fann-npm-packs-'));
await mkdir(destination, { recursive: true });
const tarballs = [];
for (const name of ['wasm', 'core']) {
  const info = JSON.parse(execFileSync('npm', ['pack', join(root, 'packages', name), '--ignore-scripts',
    '--json', '--pack-destination', destination], { encoding: 'utf8' }))[0];
  if (info.files.some(file => /node_modules|test\/|scripts\//.test(file.path))) throw new Error('Unexpected packed developer files');
  if (name === 'wasm' && !['dist/core.js', 'dist/simd.js', 'dist/ruv_swarm_wasm_bg.wasm', 'dist/ruv_swarm_simd.wasm',
    'dist/ruv_swarm_wasm.d.ts', 'dist/SHA256SUMS.json'].every(file => info.files.some(item => item.path === file)))
    throw new Error('WASM package missing required actual assets');
  tarballs.push(join(destination, info.filename));
  console.log(JSON.stringify({ name: info.name, version: info.version, filename: tarballs.at(-1),
    integrity: info.integrity, packedBytes: info.size, unpackedBytes: info.unpackedSize }));
}
const consumer = await mkdtemp(join(tmpdir(), 'fann-npm-consumer-'));
try {
  await writeFile(join(consumer, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  execFileSync('npm', ['install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund', '--package-lock=false', ...tarballs],
    { cwd: consumer, stdio: 'pipe' });
  const tree = JSON.parse(execFileSync('npm', ['ls', '--all', '--json'], { cwd: consumer, encoding: 'utf8' }));
  if (Object.keys(tree.dependencies).sort().join(',') !== 'ruv-fann,ruv-fann-wasm') throw new Error('Unexpected dependency tree');
  const wasmManifest = JSON.parse(await readFile(join(consumer, 'node_modules/ruv-fann-wasm/package.json')));
  if (Object.keys(wasmManifest.dependencies ?? {}).length) throw new Error('WASM package has dependencies');
  const coreManifest = JSON.parse(await readFile(join(consumer, 'node_modules/ruv-fann/package.json')));
  if (JSON.stringify(coreManifest.dependencies) !== JSON.stringify({ 'ruv-fann-wasm': '0.1.1' })) throw new Error('SDK dependency drift');
  await writeFile(join(consumer, 'test.mjs'), `
import assert from 'node:assert/strict';
import { loadRuntime } from 'ruv-fann';
import { loadWasm, supportsSIMD } from 'ruv-fann-wasm';
globalThis.fetch = () => { throw new Error('Packed Node consumer attempted network'); };
const model = { schema: 'ruv-fann.dense-layers/1', layers: [{inputs:1,outputs:1,weights:[2],bias:[1],activation:'Linear',steepness:1}] };
for (const simd of [false, ...(supportsSIMD() ? [true] : [])]) {
  const runtime = await loadRuntime({simd});
  const prepared = runtime.prepare(model, {maxRows:2});
  assert.deepEqual(Array.from(prepared.predict(Float64Array.of(1,2))),[3,5]);
  const trained = runtime.train(model, {inputs:Float64Array.of(1,2),targets:Float64Array.of(2,4),rows:2,epochs:10,learningRate:0.05,optimizer:'adamw'});
  assert.ok(trained.finalLoss < trained.initialLoss);
  prepared.dispose(); assert.throws(() => prepared.predict(Float64Array.of(1)),/disposed/);
  assert.ok((await loadWasm({simd})).memory instanceof WebAssembly.Memory);
}
console.log('PASS: isolated packed consumer; exact dependency tree; real core/SIMD inference and training; no Node network.');
`);
  process.stdout.write(execFileSync(process.execPath, ['test.mjs'], { cwd: consumer, encoding: 'utf8' }));
  await writeFile(join(consumer, 'types.mts'), `
import { loadRuntime, type DenseModel, type PreparedModel, type TrainResult } from 'ruv-fann';
import { loadWasm, type WasmRuntime } from 'ruv-fann-wasm';
const model: DenseModel = {schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[1],activation:'Linear',steepness:1}]};
const runtime = await loadRuntime({simd:'auto'});
const prepared: PreparedModel = runtime.prepare(model);
prepared.predictInto(new Float64Array(1), new Float64Array(1)); prepared.dispose();
const result: TrainResult = runtime.train(model,{inputs:new Float64Array(1),targets:new Float64Array(1),rows:1,epochs:1,learningRate:.1});
const raw: WasmRuntime = await loadWasm();
const trainable = raw.bindings.WasmTrainableNetwork.from_dense_layers_json(JSON.stringify(result.model),1); trainable.free();
`);
  if (process.env.RUV_FANN_TSC) {
    execFileSync(process.execPath, [process.env.RUV_FANN_TSC, '--noEmit', '--strict', '--target', 'ES2022',
      '--module', 'NodeNext', '--moduleResolution', 'NodeNext', '--lib', 'ES2023,DOM,ESNext.Disposable', 'types.mts'],
      { cwd: consumer, stdio: 'pipe' });
    console.log('PASS: installed public TypeScript declarations (strict NodeNext).');
  } else console.log('TypeScript declaration check not run: set RUV_FANN_TSC to a host typescript/bin/tsc.');
} finally { await rm(consumer, { recursive: true, force: true }); }
console.log(`Tarballs retained in ${destination}`);
