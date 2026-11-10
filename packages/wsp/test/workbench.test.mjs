import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { loadRuntime } from 'ruv-fann';
import { inspectWeightModel, traceWeightModel, analyzeWeightSensitivity, createNeuronAblationPatch,
  fingerprintModel, applyWeightPatch, evaluateWeightPatch, WSP_LIMITS } from '../src/index.js';
import { linearModel, linearData } from './fixtures.mjs';
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const close = (actual, expected, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) <= tolerance,
  `${actual} differs from ${expected}`);
const network = (activation = 'Tanh', steepness = .7) => ({ schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 2, outputs: 4, weights: [1, -2, .5, .25, -1, 1, 0, 2], bias: [.5, -.25, 1, -1], activation, steepness },
  { inputs: 4, outputs: 2, weights: [1, -1, .5, 0, .25, .75, -.25, 1], bias: [.1, -.2], activation: 'Linear', steepness: 1.3 }
] });
const activations = ['Linear', 'Threshold', 'ThresholdSymmetric', 'Sigmoid', 'SigmoidSymmetric', 'Tanh',
  'Gaussian', 'GaussianSymmetric', 'Elliot', 'ElliotSymmetric', 'LinearPiece', 'LinearPieceSymmetric',
  'ReLU', 'ReLULeaky', 'Sin', 'Cos', 'SinSymmetric', 'CosSymmetric', 'Gelu', 'Swish', 'LeakyRelu'];

test('inspection canonicalizes missing bias and reports independently counted dimensions/parameters', async () => {
  const model = linearModel(2), result = await inspectWeightModel(model);
  assert.deepEqual(result, { schema: 'ruv-fann.weight-model-inspection/1', baseHash: await fingerprintModel(model),
    parameterCount: 2, inputSize: 1, outputSize: 1, layers: [{ layer: 0, inputs: 1, outputs: 1,
      activation: 'Linear', steepness: 1, weightCount: 1, biasCount: 1, parameterCount: 2 }] });
  result.layers[0].inputs = 99;
  assert.equal((await inspectWeightModel(model)).layers[0].inputs, 1);
  assert.deepEqual(await inspectWeightModel(JSON.stringify(model)), await inspectWeightModel(model));
});

for (const simd of [false, true]) {
  test(`traces have independent affine/nonunit-steepness oracle and exact hashes (SIMD=${simd})`, async () => {
    const model = { schema: 'ruv-fann.dense-layers/1', layers: [
      { inputs: 2, outputs: 2, weights: [2, -1, 1, 3], bias: [1, -2], activation: 'Linear', steepness: 2 },
      { inputs: 2, outputs: 1, weights: [1, -.5], bias: [3], activation: 'Linear', steepness: .5 }
    ] };
    const batch = { inputs: [1, 2, -1, 3], rows: 2 };
    const trace = await traceWeightModel(model, batch, { simd });
    assert.equal(trace.schema, 'ruv-fann.weight-trace/1');
    assert.equal(trace.backend, simd ? 'wasm-simd' : 'wasm');
    assert.equal(trace.baseHash, await fingerprintModel(model));
    assert.equal(trace.inputHash, hash({ schema: 'ruv-fann.weight-input/1', inputSize: 2, rows: batch.rows, inputs: batch.inputs }));
    assert.deepEqual(trace.layers[0].preactivations, [1, 5, -4, 6]);
    assert.deepEqual(trace.layers[0].activations, [2, 10, -8, 12]);
    assert.deepEqual(trace.layers[1].preactivations, [0, -11]);
    assert.deepEqual(trace.layers[1].activations, [0, -5.5]);
    const changed = await traceWeightModel(model, { inputs: [1, 2], rows: 1 }, { simd });
    assert.notEqual(changed.inputHash, trace.inputHash);
  });
  test(`every activation and batch/tile boundary traces match complete WASM inference (SIMD=${simd})`, async () => {
    const runtime = await loadRuntime({ simd });
    for (const activation of activations) for (const rows of [1, 3, 7, 8, 9, 32, 33, 65]) {
      const model = network(activation, -.75), inputs = Array.from({ length: rows * 2 }, (_, i) => Math.sin(i * .7));
      const trace = await traceWeightModel(model, { inputs, rows }, { simd });
      const prepared = runtime.prepare(model, { maxRows: rows });
      try {
        const expected = [...prepared.predict(Float64Array.from(inputs), rows)];
        assert.deepEqual(trace.layers.at(-1).activations, expected, `${activation},rows=${rows}`);
      } finally { prepared.dispose(); }
      assert.equal(trace.layers[0].preactivations.length, rows * 4);
      assert.equal(trace.layers[1].activations.length, rows * 2);
    }
  });
  test(`sensitivity matches analytic MSE derivatives for weights and implicit bias (SIMD=${simd})`, async () => {
    const model = linearModel(1);
    const result = await analyzeWeightSensitivity(model, linearData(), [
      { layer: 0, parameter: 'weights', index: 0 }, { layer: 0, parameter: 'bias', index: 0 }
    ], { simd });
    assert.equal(result.schema, 'ruv-fann.weight-sensitivity/1');
    assert.equal(result.method, 'central-finite-difference'); assert.equal(result.epsilon, 1e-4);
    assert.equal(result.backend, simd ? 'wasm-simd' : 'wasm');
    assert.deepEqual(result.baseline, { mse: 2.5, maxAbsoluteError: 2 });
    close(result.entries[0].derivative, -5); close(result.entries[1].derivative, -3);
    close(result.entries[0].minus.mse, 2.500500025);
    close(result.entries[0].plus.mse, 2.499500025);
    assert.equal(result.baseHash, await fingerprintModel(model));
    assert.equal(result.datasetHash, hash({ schema: 'ruv-fann.weight-dataset/1', inputSize: 1, outputSize: 1,
      rows: 2, inputs: [1, 2], targets: [2, 4] }));
  });
}

test('central differences divide by the actual represented probe span', async () => {
  const value = 999999, epsilon = 1e-8;
  const result = await analyzeWeightSensitivity(linearModel(value), { inputs: [1e-3], targets: [0], rows: 1 },
    [{ layer: 0, parameter: 'weights', index: 0 }], { epsilon });
  const entry = result.entries[0];
  assert.equal(entry.derivative, (entry.plus.mse - entry.minus.mse) / ((value + epsilon) - (value - epsilon)));
  assert.notEqual(entry.derivative, (entry.plus.mse - entry.minus.mse) / (2 * epsilon));
});

test('ablation disconnects outgoing edges, preserves nonlinear neuron and supports normal evaluation/repair', async () => {
  const model = network('Sigmoid'), before = JSON.stringify(model);
  const patch = await createNeuronAblationPatch(model, 0, 1);
  assert.equal(patch.baseHash, await fingerprintModel(model));
  assert.deepEqual(patch.edits, [{ layer: 1, parameter: 'weights', index: 1, value: 0 },
    { layer: 1, parameter: 'weights', index: 5, value: 0 }]);
  const candidate = await applyWeightPatch(model, patch);
  assert.deepEqual(candidate.layers[0], model.layers[0]);
  const runtime = await loadRuntime(), prepared = runtime.prepare(candidate);
  try {
    const inputs = [1, 2, 3, 4], targets = [...prepared.predict(Float64Array.from(inputs), 2)];
    const evaluation = await evaluateWeightPatch(model, patch, { inputs, targets, rows: 2 }, { maxMSE: 0, maxRegressionMSE: 0 });
    assert.equal(evaluation.accepted, true); assert.equal(evaluation.candidate.mse, 0);
  } finally { prepared.dispose(); }
  assert.equal(JSON.stringify(model), before);
  await assert.rejects(createNeuronAblationPatch(model, 1, 0), /hidden layer/);
  await assert.rejects(createNeuronAblationPatch(model, 0, 4), /neuron/);
  await assert.rejects(createNeuronAblationPatch(model, 0, 1, { maxEdits: 1 }), /edits/);
  await assert.rejects(createNeuronAblationPatch(model, 0, 1, { maxAbsDelta: .1 }), /maxAbsDelta/);
  const zero = network(); zero.layers[1].weights.fill(0);
  await assert.rejects(createNeuronAblationPatch(zero, 0, 1), /no nonzero/);
  await assert.rejects(createNeuronAblationPatch(linearModel(), 0, 0), /hidden layer/);
  const manyOutgoing = { schema: model.schema, layers: [linearModel(1).layers[0], {
    inputs: 1, outputs: 129, weights: Array(129).fill(1), bias: Array(129).fill(0), activation: 'Linear', steepness: 1
  }] };
  await assert.rejects(createNeuronAblationPatch(manyOutgoing, 0, 0), /maxEdits/);
});

test('all new APIs snapshot model, batch, locations, options and dataset before yielding', async () => {
  const model = network(), original = structuredClone(model), inputs = [1, 2], settings = { simd: false };
  const expectedHash = await fingerprintModel(model);
  const inspection = inspectWeightModel(model), trace = traceWeightModel(model, { inputs, rows: 1 }, settings);
  const patch = createNeuronAblationPatch(model, 0, 1);
  model.layers[0].weights[0] = 999; inputs[0] = 999; settings.simd = true;
  assert.equal((await inspection).baseHash, expectedHash);
  const traced = await trace; assert.equal(traced.baseHash, expectedHash); assert.equal(traced.backend, 'wasm');
  assert.equal(traced.layers[0].preactivations[0], -2.5);
  assert.equal((await patch).baseHash, expectedHash);
  const data = linearData(), locations = [{ layer: 0, parameter: 'weights', index: 0 }], options = { epsilon: .01 };
  const sensitivity = analyzeWeightSensitivity(linearModel(1), data, locations, options);
  data.targets[0] = 999; locations[0].index = 99; options.epsilon = 1;
  const result = await sensitivity; close(result.entries[0].derivative, -5); assert.equal(result.epsilon, .01);
  traced.layers[0].activations[0] = 999;
  assert.notEqual((await traceWeightModel(original, { inputs: [1, 2], rows: 1 })).layers[0].activations[0], 999);
});

test('hostile own arrays/objects reject getters, symbols, iterators and unknown fields without invocation', async () => {
  let calls = 0; const getter = () => { calls++; throw new Error('caller code executed'); };
  const accessor = (object, key) => Object.defineProperty(object, key, { get: getter, enumerable: true });
  for (const corrupt of [array => accessor(array, '0'), array => accessor(array, Symbol.iterator),
    array => { array.extra = 1; }, array => { array[Symbol('extra')] = 1; }, array => { delete array[0]; }]) {
    const inputs = [1, 2]; corrupt(inputs);
    await assert.rejects(traceWeightModel(linearModel(1), { inputs, rows: 2 }));
    const model = linearModel(); corrupt(model.layers[0].weights);
    await assert.rejects(inspectWeightModel(model));
    const locations = [{ layer: 0, parameter: 'weights', index: 0 }]; corrupt(locations);
    await assert.rejects(analyzeWeightSensitivity(linearModel(1), linearData(), locations));
  }
  await assert.rejects(traceWeightModel(linearModel(), accessor({ inputs: [1], rows: 1 }, 'rows')), /accessors/);
  await assert.rejects(traceWeightModel(linearModel(), { inputs: [1], rows: 1 }, accessor({}, 'simd')), /accessors/);
  await assert.rejects(analyzeWeightSensitivity(linearModel(), linearData(),
    [accessor({ layer: 0, parameter: 'weights', index: 0 }, 'index')]), /accessors/);
  await assert.rejects(createNeuronAblationPatch(network(), 0, 0, accessor({}, 'maxEdits')), /accessors/);
  await assert.rejects(traceWeightModel(linearModel(), { inputs: [1], rows: 1, targets: [1] }), /unknown/);
  await assert.rejects(traceWeightModel(linearModel(), { inputs: [1], rows: 1 }, { epsilon: .1 }), /unknown/);
  await assert.rejects(analyzeWeightSensitivity(linearModel(), linearData(),
    [{ layer: 0, parameter: 'weights', index: 0, value: 1 }]), /unknown/);
  assert.equal(calls, 0);
});

test('new boundaries enforce finite numeric values, strict options and location limits', async () => {
  for (const rows of [0, 1.5, 1025, NaN, '1'])
    await assert.rejects(traceWeightModel(linearModel(), { inputs: [1], rows }));
  for (const value of [NaN, Infinity, -Infinity, '1', null])
    await assert.rejects(traceWeightModel(linearModel(), { inputs: [value], rows: 1 }));
  await assert.rejects(traceWeightModel(linearModel(), { inputs: [1, 2], rows: 1 }));
  for (const simd of ['yes', 1, null]) await assert.rejects(traceWeightModel(linearModel(), { inputs: [1], rows: 1 }, { simd }));
  const location = { layer: 0, parameter: 'weights', index: 0 };
  for (const epsilon of [0, 1e-9, 1.01, NaN, Infinity, '0.1', null])
    await assert.rejects(analyzeWeightSensitivity(linearModel(), linearData(), [location], { epsilon }));
  for (const locations of [[], Array(17).fill(location), [location, location], [{ ...location, layer: 1 }],
    [{ ...location, index: 1 }], [{ ...location, parameter: 'activation' }]])
    await assert.rejects(analyzeWeightSensitivity(linearModel(), linearData(), locations));
  await assert.rejects(analyzeWeightSensitivity(linearModel(1e6), linearData(), [location]), /probe/);
  await assert.rejects(analyzeWeightSensitivity(linearModel(1e100), linearData(), [location]), /probe/);
});

test('aggregate trace memory and two-pass work reject before reading hostile elements', async () => {
  let calls = 0; const badInputs = length => { const array = new Array(length); Object.defineProperty(array, '0', {
    get() { calls++; throw new Error('oversized elements read'); } }); return array; };
  const wide = { schema: 'ruv-fann.dense-layers/1', layers: [{ inputs: 1, outputs: 200,
    weights: Array(200).fill(1), bias: Array(200).fill(0), activation: 'Linear', steepness: 1 }] };
  await assert.rejects(traceWeightModel(wide, { inputs: badInputs(164), rows: 164 }), /aggregate trace elements/);
  const dense = { schema: 'ruv-fann.dense-layers/1', layers: [{ inputs: 250, outputs: 250,
    weights: Array(62500).fill(1), bias: Array(250).fill(0), activation: 'Linear', steepness: 1 }] };
  await assert.rejects(traceWeightModel(dense, { inputs: badInputs(250 * 80), rows: 80 }), /aggregate work/);
  // A single pass fits 5M units, two passes exceed the 10M bound.
  assert.ok(80 * 62750 <= WSP_LIMITS.maxWork);
  assert.equal(calls, 0);
});

test('aggregate sensitivity amplification is bounded across all probes', async () => {
  const model = { schema: 'ruv-fann.dense-layers/1', layers: [{ inputs: 100, outputs: 100,
    weights: Array(10000).fill(.01), bias: Array(100).fill(0), activation: 'Linear', steepness: 1 }] };
  const rows = 40, data = { rows, inputs: Array(rows * 100).fill(.1), targets: Array(rows * 100).fill(0) };
  await assert.rejects(analyzeWeightSensitivity(model, data,
    Array.from({ length: 16 }, (_, index) => ({ layer: 0, parameter: 'weights', index }))), /aggregate work/);
});

test('actual WASM overflow rejects and subsequent operations remain healthy', async () => {
  const bad = linearModel(1e308);
  await assert.rejects(traceWeightModel(bad, { inputs: [2], rows: 1 }));
  await assert.rejects(analyzeWeightSensitivity(linearModel(1), { inputs: [1e155], targets: [0], rows: 1 },
    [{ layer: 0, parameter: 'weights', index: 0 }]), /finite/);
  // Both probe MSEs remain finite while their divided difference overflows.
  await assert.rejects(analyzeWeightSensitivity(linearModel(.9), { inputs: [1e154], targets: [0], rows: 1 },
    [{ layer: 0, parameter: 'weights', index: 0 }]), /sensitivity derivative/);
  assert.deepEqual((await traceWeightModel(linearModel(2), { inputs: [3], rows: 1 })).layers[0].activations, [6]);
});

// Load the production module with only the runtime acquisition replaced. This isolates
// native-resource ownership branches; numerical correctness above uses actual WASM.
async function instrumented(runtime, mutant = false) {
  let source = await readFile(new URL('../src/workbench.js', import.meta.url), 'utf8');
  const slot = `wsp-runtime-${Math.random()}`;
  globalThis[slot] = runtime;
  source = source.replace("import { loadRuntime } from 'ruv-fann';", `const runtime = globalThis[${JSON.stringify(slot)}]; const loadRuntime = async () => runtime;`);
  for (const name of ['validation', 'crypto']) source = source.replace(`'./${name}.js'`, JSON.stringify(new URL(`../src/${name}.js`, import.meta.url).href));
  if (mutant) source = source.replace('finally { prepared.dispose(); }', 'finally { /* missing dispose mutant */ }');
  try { return await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`); }
  finally { delete globalThis[slot]; }
}
function fakeRuntime(failAt, counts) {
  return { backend: 'wasm', prepare() {
    counts.created++; let disposed = false;
    return { predict(input, rows) {
      counts.predictions++;
      if (counts.predictions === failAt) throw new Error('injected native prediction failure');
      return Float64Array.from({ length: rows }, () => 0);
    }, dispose() { assert.equal(disposed, false); disposed = true; counts.disposed++; } };
  } };
}
for (const scenario of ['trace first', 'trace activation', 'sensitivity baseline', 'sensitivity minus', 'sensitivity plus']) {
  test(`every prepared resource is disposed after ${scenario} failure; disposal mutant is detected`, async () => {
    const failAt = scenario === 'trace activation' || scenario === 'sensitivity minus' ? 2 : scenario === 'sensitivity plus' ? 3 : 1;
    const execute = module => scenario.startsWith('trace') ? module.traceWeightModel(linearModel(1), { inputs: [1], rows: 1 }) :
      module.analyzeWeightSensitivity(linearModel(1), { inputs: [1], targets: [0], rows: 1 }, [{ layer: 0, parameter: 'weights', index: 0 }]);
    for (const mutant of [false, true]) {
      const counts = { created: 0, disposed: 0, predictions: 0 }, runtime = fakeRuntime(failAt, counts);
      // Capture runtime in the loader before global registration is removed.
      const original = await instrumented(runtime, mutant);
      await assert.rejects(execute(original), /injected native prediction failure/);
      if (mutant) assert.notEqual(counts.created, counts.disposed, 'missing-dispose mutation must fail resource invariant');
      else assert.equal(counts.created, counts.disposed);
    }
  });
}
