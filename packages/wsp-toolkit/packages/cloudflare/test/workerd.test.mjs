import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Miniflare } from 'miniflare';
import { buildWorker } from '../src/build.js';
import { fingerprintModel } from 'ruv-fann-wsp';

const token = 'workerd-test-token-do-not-use-in-production';
const model = { schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 2, outputs: 2, weights: [1, -1, -1, 1], bias: [0, 0], activation: 'ReLU', steepness: 1 },
  { inputs: 2, outputs: 1, weights: [0, 0], bias: [0], activation: 'Linear', steepness: 1 }
] };
const dataset = { inputs: [0, 0, 0, 1, 1, 0, 1, 1], targets: [0, 1, 1, 0], rows: 4 };

test('built Worker executes both static Wasm backends in real workerd', { timeout: 60000 }, async t => {
  const output = await mkdtemp(join(tmpdir(), 'wsp-workerd-'));
  const artifact = await buildWorker(output);
  const code = await readFile(artifact.script, 'utf8');
  assert.match(code, /core\.wasm/);
  assert.match(code, /simd\.wasm/);
  assert.doesNotMatch(code, /node:fs/);
  const worker = new Miniflare({
    modules: true, modulesRoot: output, scriptPath: artifact.script, compatibilityDate: '2026-07-30',
    modulesRules: [{ type: 'CompiledWasm', include: ['**/*.wasm'], fallthrough: true }],
    bindings: { WSP_BEARER_TOKEN: token }
  });
  t.after(async () => { await worker.dispose(); await rm(output, { recursive: true, force: true }); });
  const invoke = (path, body, authorization = `Bearer ${token}`) => worker.dispatchFetch(`https://worker.test${path}`, {
    method: 'POST', headers: { 'content-type': 'application/json', authorization }, body: JSON.stringify(body)
  });
  const baseHash = await fingerprintModel(model);
  const patch = { schema: 'ruv-fann.weight-patch/1', baseHash, edits: [
    { layer: 1, parameter: 'weights', index: 0, value: 1 },
    { layer: 1, parameter: 'weights', index: 1, value: 1 }
  ] };
  for (const simd of [false, true]) {
    const response = await invoke('/evaluate', { model, patch, dataset,
      policy: { maxMSE: 0, maxRegressionMSE: 0, maxAbsoluteError: 0 }, options: { simd } });
    const result = await response.json();
    assert.equal(response.status, 200, JSON.stringify(result));
    assert.equal(result.backend, simd ? 'wasm-simd' : 'wasm');
    assert.equal(result.accepted, true);
    assert.equal(result.baseline.mse, 0.5);
    assert.equal(result.candidate.mse, 0);
    assert.equal(result.baseHash, baseHash);
    const trace = await (await invoke('/trace', { model: result.model,
      batch: { inputs: dataset.inputs, rows: 4 }, options: { simd } })).json();
    assert.deepEqual(trace.layers[1].activations, dataset.targets);
    const ablation = await (await invoke('/ablation', { model: result.model, layer: 0, neuron: 0 })).json();
    assert.equal(ablation.edits[0].value, 0);
    const rejection = await (await invoke('/evaluate', { model: result.model, patch: ablation, dataset,
      policy: { maxMSE: 0, maxRegressionMSE: 0 }, options: { simd } })).json();
    assert.equal(rejection.accepted, false);
    assert.equal(rejection.candidate.mse, 0.25);
  }
  const linear = { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 1, weights: [1], bias: [0], activation: 'Linear', steepness: 1 }
  ] };
  const derivative = await (await invoke('/sensitivity', { model: linear,
    dataset: { inputs: [1, 2], targets: [2, 4], rows: 2 },
    locations: [{ layer: 0, parameter: 'weights', index: 0 }], options: { simd: false } })).json();
  assert.ok(Math.abs(derivative.entries[0].derivative + 5) < 1e-8);
  assert.equal((await invoke('/trace', {}, 'Bearer incorrect')).status, 401);
  assert.equal((await invoke('/promote', {})).status, 404);
});
