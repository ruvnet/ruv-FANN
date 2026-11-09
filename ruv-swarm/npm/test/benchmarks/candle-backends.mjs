#!/usr/bin/env node
/** Actual checked backend timings. Baseline must be an archived matching
 * wasm-bindgen JS/binary pair, not an old binary paired with new glue.
 * Run serially on a quiet host; timings include JS boundary validation/copies.
 */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import os from 'node:os';
import { externalEvidencePath, sourceSnapshot, repo } from '../../../../integrations/ruv-fann/test/e2e/fixture.mjs';

const args = process.argv.slice(2), options = {};
for (let i = 0; i < args.length; i += 2) {
  if (!['--output', '--baseline'].includes(args[i]) || !args[i + 1]) {
    throw new Error('Use --output NEW_EXTERNAL_JSON [--baseline ARCHIVED_WASM_DIRECTORY]');
  }
  if (options[args[i]]) {
    throw new Error('Duplicate option');
  }
  options[args[i]] = args[i + 1];
}
if (!options['--output'] || !process.env.RUV_FANN_CANDLE_FIXTURE) {
  throw new Error('An external --output and RUV_FANN_CANDLE_FIXTURE are required');
}
const output = await externalEvidencePath(options['--output']);
const before = await sourceSnapshot();
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const nativePath = join(repo, 'bindings/napi');
async function nativeSnapshot() {
  const receipt = JSON.parse(await readFile(join(nativePath, 'build-info.json'), 'utf8'));
  assert.equal(receipt.binarySha256, sha(await readFile(join(nativePath, 'ruv_fann.node'))));
  for (const [file, hash] of Object.entries(receipt.inputsAfter)) {
    assert.equal(sha(await readFile(join(repo, file))), hash, `Native build input ${file}`);
  }
  return { receiptSha256: sha(await readFile(join(nativePath, 'build-info.json'))), binarySha256: receipt.binarySha256 };
}
const nativeBefore = await nativeSnapshot();
const native = await import(pathToFileURL(join(nativePath, 'index.mjs')));
const fixture = JSON.parse(await readFile(process.env.RUV_FANN_CANDLE_FIXTURE, 'utf8'));
assert.match(fixture.oracle, /candle-core\/candle-nn 0\.11\.1 CPU f64/);
const backends = [];
for (const variant of ['core', 'simd']) {
  for (const generation of options['--baseline'] ? ['baseline', 'candidate'] : ['candidate']) {
    const directory = generation === 'baseline' ? resolve(options['--baseline']) : join(repo, 'ruv-swarm/npm/wasm');
    const glue = join(directory, 'ruv_swarm_wasm.js');
    const url = pathToFileURL(glue); url.searchParams.set('benchmark', `${variant}-${generation}`);
    const mod = await import(url);
    const filename = variant === 'simd' ? 'ruv_swarm_simd.wasm' : 'ruv_swarm_wasm_bg.wasm';
    const bytes = await readFile(join(directory, filename));
    await mod.default({ module_or_path: bytes });
    assert.equal(JSON.parse(mod.get_features()).prepared_batch_simd, variant === 'simd');
    backends.push({ name: `${generation}-${variant}`, generation, variant, mod,
      binarySha256: sha(bytes), glueSha256: sha(await readFile(glue)) });
  }
}
backends.push({ name: 'native-napi', generation: 'candidate', variant: 'napi',
  mod: native, binarySha256: nativeBefore.binarySha256 });

function measure(operation) {
  for (let i = 0; i < 20; i++) {
    operation();
  }
  let iterations = 1;
  for (;;) {
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      operation();
    }
    if (performance.now() - start >= 2 || iterations === 4096) {
      break;
    }
    iterations = Math.min(4096, iterations * 4);
  }
  const samplesNs = [];
  for (let block = 0; block < 20; block++) {
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      operation();
    }
    samplesNs.push((performance.now() - start) * 1e6 / iterations);
  }
  return { iterations, samplesNs };
}

function specification(shape, activation) {
  const weights = Float64Array.from({ length: shape.slice(1).reduce((n, width, i) => n + width * shape[i], 0) },
    (_, i) => ((i * 17 % 31) - 15) / 64);
  let offset = 0;
  const layers = shape.slice(1).map((outputs, i) => {
    const inputs = shape[i], start = offset; offset += inputs * outputs;
    return { inputs, outputs, weights: Array.from({ length: inputs * outputs }, (_, k) =>
      weights[start + (k % inputs) * outputs + Math.floor(k / inputs)]),
    bias: null, activation, steepness: 1 };
  });
  return { weights, json: JSON.stringify({ schema: 'ruv-fann.dense-layers/1', layers }) };
}
function prepare(backend, shape, activation, specification) {
  if (backend.variant === 'napi') {
    return native.NativePreparedNetwork.from_dense_layers_json(specification.json);
  }
  const model = new backend.mod.WasmNeuralNetwork(Uint32Array.from(shape), backend.mod.ActivationFunction[activation]);
  try {
    model.set_weights(specification.weights); return model.prepare();
  } finally {
    model.free();
  }
}

const inference = [], references = new Map(), wasmReferences = new Map();
let caseId = 0;
for (const activation of ['Linear', 'Sigmoid']) {
  for (const shape of [[5, 7, 3], [100, 50, 25, 10], [784, 128, 64, 10]]) {
    const spec = specification(shape, activation);
    for (const rows of [1, 3, 8, 32, 128]) {
      const inputs = Float64Array.from({ length: rows * shape[0] }, (_, i) =>
        ((i % shape[0] * 7 % 13) - 6) / 8 + (Math.floor(i / shape[0]) % 5) / 64);
      const order = caseId % 2 ? [...backends].reverse() : backends;
      for (const backend of order) {
        const start = performance.now();
        const prepared = prepare(backend, shape, activation, spec);
        const batch = prepared.create_batch(rows);
        const setupNs = (performance.now() - start) * 1e6;
        const output = new Float64Array(rows * shape.at(-1));
        try {
          batch.run_into(inputs, output, rows);
          if (!references.has(caseId)) {
            references.set(caseId, Array.from(output));
          }
          // Native and WASM exp() implementations can differ by an ulp.
          // Each engine's single/batch and WASM before/after stay exact.
          const expected = references.get(caseId);
          let maximumAbsoluteError = 0;
          for (let i = 0; i < output.length; i++) {
            const error = Math.abs(output[i] - expected[i]);
            assert.ok(Number.isFinite(output[i]) && error <= 1e-12 * (1 + Math.abs(expected[i])),
              `${backend.name},case${caseId},element${i}`);
            maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
          }
          if (backend.variant !== 'napi') {
            if (!wasmReferences.has(caseId)) {
              wasmReferences.set(caseId, Array.from(output));
            }
            assert.deepEqual(Array.from(output), wasmReferences.get(caseId));
          }
          const ownReference = Array.from({ length: rows }, (_, row) => Array.from(prepared.run(
            inputs.subarray(row * shape[0], (row + 1) * shape[0])))).flat();
          assert.deepEqual(Array.from(output), ownReference);
          const timing = measure(() => batch.run_into(inputs, output, rows));
          assert.deepEqual(Array.from(output), ownReference);
          inference.push({ case: caseId, backend: backend.name, shape, activation, precision: 'f64', rows,
            ...timing, setupNs, estimatedBytes: batch.estimated_bytes,
            maximumAbsoluteError, checksum: output.reduce((a, b) => a + b, 0) });
        } finally {
          batch.free(); prepared.free();
        }
      }
      caseId++;
    }
  }
}

const training = [];
for (const backend of backends.filter(item => item.generation === 'candidate')) {
  for (const rows of [1, 8, 32, 128]) {
    const model = backend.variant === 'napi'
      ? native.NativeTrainableNetwork.from_dense_layers_json(fixture.dense_json, rows)
      : backend.mod.WasmTrainableNetwork.from_dense_layers_json(fixture.dense_json, rows);
    const inputs = Float64Array.from({ length: rows * fixture.input.length }, (_, i) => fixture.input[i % fixture.input.length]);
    const targets = Float64Array.from({ length: rows * fixture.training_target.length }, (_, i) => fixture.training_target[i % fixture.training_target.length]);
    try {
      const initial = model.mse_backward(inputs, targets, rows);
      assert.ok(Math.abs(initial - fixture.training_loss) < 1e-12);
      for (const operation of ['backward_mse', 'backward_mse_and_sgd']) {
        let loss = initial;
        const timing = measure(() => {
          loss = model.mse_backward(inputs, targets, rows);
          if (operation === 'backward_mse_and_sgd') {
            model.sgd_step(1e-12);
          }
        });
        assert.ok(Number.isFinite(loss));
        training.push({ backend: backend.name, precision: 'f64', shape: [3, 4, 2], rows, operation,
          ...timing, initialMse: initial, finalMse: loss, estimatedBytes: model.estimated_bytes });
      }
    } finally {
      model.free();
    }
  }
}
assert.deepEqual(await sourceSnapshot(), before);
assert.deepEqual(await nativeSnapshot(), nativeBefore);
const receipt = { schema: 'ruv-fann.candle-backends-bench/1', status: 'passed', node: process.version,
  platform: os.platform(), arch: os.arch(), cpu: os.cpus()[0].model, logicalCpus: os.cpus().length,
  method: '20 warmups, adaptive blocks >=2ms or4096calls,20 raw blocks; case-order reversal. Sequential/unpaired shared-host observations.',
  scope: 'Warm API calls include intrinsic JS validation and staging/copy costs. Setup is one API-specific observation, not a comparable packing-only metric. Single/batch and WASM before/after outputs agree exactly. Cross-native/WASM comparison uses abs+relative1e-12 because exp implementations differ. Training has no prior-backend baseline.',
  numericScope: 'f64; ordinary multiply/add; no fastmath. Maximum cross-backend absolute error retained per case. Training first loss compared with actual Candle; small SGD rate keeps subsequent benchmark iterations finite.',
  inputsBefore: before, inputsAfter: before, native: nativeBefore,
  backends: backends.map(({ name, binarySha256, glueSha256 }) => ({ name, binarySha256, glueSha256 })),
  fixtureSha256: sha(await readFile(process.env.RUV_FANN_CANDLE_FIXTURE)), inference, training };
await writeFile(output, `${JSON.stringify(receipt, null, 2) }\n`, { flag: 'wx' });
console.log(JSON.stringify({ status: 'passed', inferenceCases: inference.length, trainingCases: training.length, output }));
