import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { loadRuntime } from 'ruv-fann';
import { applyWeightPatch, createWeightWorkspace, fingerprintModel } from 'ruv-fann-wsp';
import { searchWeightPatch, compileBooleanTable, createHebbianPatch, SEARCH_LIMITS } from '../src/index.js';
const linear = (weight = 0) => ({ schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 1, outputs: 1, weights: [weight], bias: [0], activation: 'Linear', steepness: 1 }
] });
const data = () => ({ inputs: [1, 2], targets: [2, 4], rows: 2 });
const policy = () => ({ maxMSE: 0, maxRegressionMSE: 0 });
const location = (values = [0, 1, 2]) => ({ layer: 0, parameter: 'weights', index: 0, values });
const options = extra => ({ locations: [location()], simd: false, ...extra });
function predict(runtime, model, inputs, rows) {
  const prepared = runtime.prepare(model, { maxRows: rows });
  try { return Array.from(prepared.predict(Float64Array.from(inputs), rows)); }
  finally { prepared.dispose(); }
}

test('deterministic search caches baseline, matches analytic MSE and requires explicit promotion', async () => {
  const model = linear(), original = structuredClone(model);
  const result = await searchWeightPatch(model, data(), policy(), options());
  assert.equal(result.baselineScore, 10); // mean([(-2)^2,(-4)^2])
  assert.equal(result.bestScore, 0);
  assert.equal(result.evaluation.candidate.mse, 0);
  assert.equal(result.evaluation.accepted, true);
  assert.equal(result.backend, 'wasm');
  assert.equal(result.evaluations, 2);
  assert.equal(result.inferencePasses, 5);
  assert.deepEqual(result.patch.edits, [{ layer: 0, parameter: 'weights', index: 0, value: 2 }]);
  assert.deepEqual(model, original);
  const workspace = await createWeightWorkspace(model, { simd: false });
  assert.equal(workspace.snapshot().revision, 1);
  const proposal = await workspace.propose(result.patch, data(), policy());
  assert.equal(workspace.promote(proposal.id, 1).revision, 2);
});

test('two-coordinate XOR repair, bounded no-gradient search', async () => {
  const model = { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 2, outputs: 2, weights: [1,-1,-1,1], bias: [0,0], activation: 'ReLU', steepness: 1 },
    { inputs: 2, outputs: 1, weights: [0,0], bias: [0], activation: 'Linear', steepness: 1 }
  ] };
  const result = await searchWeightPatch(model, { inputs: [0,0,0,1,1,0,1,1], targets: [0,1,1,0], rows: 4 }, policy(), {
    locations: [0,1].map(index => ({ layer: 1, parameter: 'weights', index, values: [0,1] })), simd: true
  });
  assert.equal(result.evaluation.accepted, true);
  assert.equal(result.evaluation.candidate.mse, 0);
  assert.equal(result.patch.edits.length, 2);
  assert.equal(result.backend, 'wasm-simd');
});

test('discontinuous trusted objective can prefer a candidate rejected by authoritative MSE gate', async () => {
  let calls = 0;
  const result = await searchWeightPatch(linear(), data(), policy(), options({
    locations: [location([0,1])], objective(frame) {
      calls++;
      assert(Object.isFrozen(frame) && Object.isFrozen(frame.predictions) && Object.isFrozen(frame.model.layers[0].weights));
      return frame.predictions.filter(x => x < 0.5).length;
    }
  }));
  assert.equal(result.objective, 'custom');
  assert.equal(result.bestScore, 0);
  assert.equal(result.evaluation.accepted, false);
  assert.equal(result.evaluation.candidate.mse, 2.5);
  assert.equal(calls, 1 + result.evaluations);
});

test('no improvement returns no patch; strict ties are stable', async () => {
  const result = await searchWeightPatch(linear(2), data(), policy(), options({ locations: [location([1,2,3])] }));
  assert.equal(result.patch, null); assert.equal(result.evaluation, null);
  assert.equal(result.bestScore, 0); assert.equal(result.evaluations, 2);
  const constant = await searchWeightPatch(linear(), data(), policy(), options({ objective: () => 1 }));
  assert.equal(constant.patch, null);
});

test('snapshots all mutable caller data before first await', async () => {
  const model = linear(), dataset = data(), criteria = policy(), config = options();
  const pending = searchWeightPatch(model, dataset, criteria, config);
  model.layers[0].weights[0] = 100;
  dataset.inputs[0] = 100; dataset.targets[0] = 100;
  criteria.maxMSE = 1000; config.locations[0].values[2] = 100;
  const result = await pending;
  assert.equal(result.baselineScore, 10); assert.equal(result.bestScore, 0);
  assert.equal(result.patch.edits[0].value, 2);
});

test('budget is global and cannot be raised; candidate evaluation cap is exact', async () => {
  const result = await searchWeightPatch(linear(), data(), policy(), options({ maxEvaluations: 1 }));
  assert.equal(result.evaluations, 1); assert.equal(result.budgetExhausted, true);
  assert.equal(result.evaluation.accepted, false);
  await assert.rejects(searchWeightPatch(linear(), data(), policy(), options({ maxWork: 1 })), /aggregate work/);
  await assert.rejects(searchWeightPatch(linear(), data(), policy(), options({ maxWork: SEARCH_LIMITS.maxWork + 1 })), /maxWork/);
  await assert.rejects(searchWeightPatch(linear(), data(), policy(), options({ maxEvaluations: 257 })), /maxEvaluations/);
  await assert.rejects(searchWeightPatch(linear(), data(), policy(), options({ maxPasses: 17 })), /maxPasses/);
});

test('malformed choices, nonfinite scores, getters, sparse arrays and relaxed patch bounds reject', async () => {
  const bad = [
    { locations: [] }, { locations: [location([])] }, { locations: [location([1,1])] },
    { locations: [location(), location()] }, { locations: [location([Infinity])] },
    { locations: [{ ...location(), index: 1 }] }, { locations: [{ ...location(), parameter: 'activation' }] },
    { locations: [location([1001])] }, { maxEvaluations: 0 }, { maxEvaluations: null }, { maxPasses: null }, { maxWork: null }, { limits: { maxAbsDelta: null } }, { simd: 'yes' }, { unknown: true },
    { limits: { maxEdits: 129 } }, { limits: { maxAbsDelta: 1001 } }, { objective: 'return 0' }
  ];
  for (const config of bad) await assert.rejects(searchWeightPatch(linear(), data(), policy(), options(config)));
  await assert.rejects(searchWeightPatch(linear(), data(), policy(), options({ objective: () => Infinity })), /objective score/);
  await assert.rejects(searchWeightPatch(linear(), data(), policy(), options({ objective: async () => 1 })), /objective score/);
  let called = 0;
  const accessor = { get locations() { called++; return [location()]; } };
  await assert.rejects(searchWeightPatch(linear(), data(), policy(), accessor), /accessors/);
  const sparse = options({ locations: new Array(1) });
  await assert.rejects(searchWeightPatch(linear(), data(), policy(), sparse), /own data/);
  const values = [0,1]; Object.defineProperty(values, '1', { get() { called++; return 2; } });
  await assert.rejects(searchWeightPatch(linear(), data(), policy(), options({ locations: [location(values)] })), /own data/);
  assert.equal(called, 0);
});

test('complete Boolean compilation agrees with exhaustive independent parity and conjunction oracles', async () => {
  const runtime = await loadRuntime({ simd: false });
  for (const n of [1,2,3,6]) {
    const inputs = [], targets = [];
    for (let row = 0; row < 2 ** n; row++) {
      const bits = Array.from({ length: n }, (_, i) => (row >> (n - i - 1)) & 1);
      inputs.push(...bits);
      targets.push(bits.reduce((sum, x) => sum + x, 0) % 2, bits.every(Boolean) ? 1 : 0);
    }
    const model = compileBooleanTable({ inputs: n, outputs: 2, targets });
    assert.deepEqual(predict(runtime, model, inputs, 2 ** n), targets);
  }
});

test('Boolean compiler rejects partial, nonbinary and oversized tables without reading getters', () => {
  for (const table of [
    { inputs: 0, outputs: 1, targets: [] }, { inputs: 7, outputs: 1, targets: [] },
    { inputs: 1, outputs: 17, targets: [] }, { inputs: 1, outputs: 1, targets: [0] },
    { inputs: 1, outputs: 1, targets: [0,2] }, { inputs: 1, outputs: 1, targets: [false,true] }
  ]) assert.throws(() => compileBooleanTable(table));
  let calls = 0;
  assert.throws(() => compileBooleanTable({ inputs: 1, outputs: 1, get targets() { calls++; return [0,1]; } }), /accessors/);
  assert.equal(calls, 0);
});

test('Hebbian correlation matches analytic oracle, clips limits and leaves model untouched', async () => {
  const model = linear(1), original = structuredClone(model);
  const patch = await createHebbianPatch(model, { inputs: [1,2], rows: 2 }, { layer: 0, indices: [0], learningRate: 0.1, simd: false });
  // pre=post=[1,2]; mean(pre*post)=2.5; delta=0.25.
  assert.equal(patch.edits[0].value, 1.25);
  assert.equal(patch.baseHash, await fingerprintModel(model));
  assert.equal((await applyWeightPatch(model, patch)).layers[0].weights[0], 1.25);
  assert.deepEqual(model, original);
  const clipped = await createHebbianPatch(model, { inputs: [1,2], rows: 2 }, {
    layer: 0, indices: [0], learningRate: 1, limits: { maxAbsDelta: 0.1 }, simd: false
  });
  assert(Math.abs(clipped.edits[0].value - 1.1) < 1e-14);
  assert(Math.abs(clipped.edits[0].value - 1) <= 0.1);
});

test('Hebbian hidden-layer inputs use prior actual activations and snapshot caller values', async () => {
  const model = { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 1, weights: [-1], bias: [0], activation: 'ReLU', steepness: 1 },
    { inputs: 1, outputs: 1, weights: [1], bias: [0], activation: 'Linear', steepness: 1 }
  ] };
  const batch = { inputs: [-2,1], rows: 2 }, config = { layer: 1, indices: [0], learningRate: 0.1, simd: false };
  const pending = createHebbianPatch(model, batch, config);
  model.layers[1].weights[0] = 100; batch.inputs[0] = 100; config.indices[0] = 100;
  const patch = await pending;
  assert(Math.abs(patch.edits[0].value - 1.2) < 1e-14); // pre=post=[2,0], mean product=2.
});

test('Hebbian invalid, duplicate, overflow and zero updates reject', async () => {
  for (const config of [
    { layer: 1, indices: [0], learningRate: 1 }, { layer: 0, indices: [1], learningRate: 1 },
    { layer: 0, indices: [0,0], learningRate: 1 }, { layer: 0, indices: [], learningRate: 1 },
    { layer: 0, indices: [0], learningRate: 0 }, { layer: 0, indices: [0], learningRate: 2 }
  ]) await assert.rejects(createHebbianPatch(linear(1), { inputs: [1], rows: 1 }, config));
  await assert.rejects(createHebbianPatch(linear(0), { inputs: [1], rows: 1 }, { layer: 0, indices: [0], learningRate: 1 }), /no representable/);
  await assert.rejects(createHebbianPatch(linear(1), { inputs: [1e200], rows: 1 }, { layer: 0, indices: [0], learningRate: 1 }), /finite|overflow/);
});

test('custom objective cannot bypass the regression policy', async () => {
  const result = await searchWeightPatch(linear(2), data(), { maxMSE: 100, maxRegressionMSE: 0 }, options({
    locations: [location([2,3])], objective: ({ predictions }) => -predictions[0]
  }));
  assert.equal(result.bestScore, -3);
  assert.equal(result.evaluation.baseline.mse, 0);
  assert.equal(result.evaluation.candidate.mse, 2.5);
  assert.equal(result.evaluation.accepted, false);
  assert.deepEqual(result.evaluation.reasons, ['candidate MSE regression exceeds maxRegressionMSE']);
});

test('source schema and aggregate multi-candidate work reject before callbacks execute', async () => {
  let calls = 0;
  const big = { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 64, weights: Array(64).fill(0), bias: Array(64).fill(0), activation: 'Linear', steepness: 1 },
    { inputs: 64, outputs: 1, weights: Array(64).fill(0), bias: [0], activation: 'Linear', steepness: 1 }
  ] };
  await assert.rejects(searchWeightPatch(big, { inputs: Array(1024).fill(0), targets: Array(1024).fill(0), rows: 1024 }, policy(), {
    locations: [location(Array.from({ length: 32 }, (_, i) => i))], maxPasses: 16, objective: () => { calls++; return 0; }
  }), /aggregate work/);
  const malformed = linear(); malformed.layers[0].activation = 'made-up';
  await assert.rejects(searchWeightPatch(malformed, data(), policy(), options({ objective: () => { calls++; return 0; } })), /activation/);
  assert.equal(calls, 0);
});

test('invalid objective thenables reject without reading their executable then getter', async () => {
  let calls=0;
  const thenable=Object.defineProperty({},'then',{get(){calls++;throw new Error('then getter executed');}});
  await assert.rejects(searchWeightPatch(linear(),data(),policy(),options({objective:()=>thenable})),/objective score/);
  assert.equal(calls,0);
});

test('invalid asynchronous objective rejection is contained after the caller catches validation', () => {
  const source = `import assert from 'node:assert/strict';
    import {searchWeightPatch} from ${JSON.stringify(new URL('../src/index.js', import.meta.url).href)};
    const model=${JSON.stringify(linear())}, dataset=${JSON.stringify(data())}, policy=${JSON.stringify(policy())};
    for (const objective of [async()=>{throw new Error('async failure');},()=>Promise.reject(new Error('promise failure'))]) {
      await assert.rejects(searchWeightPatch(model,dataset,policy,{...${JSON.stringify(options())},objective}),/objective score/);
    }
    await new Promise(resolve=>setTimeout(resolve,10));
    const valid=await searchWeightPatch(model,dataset,policy,${JSON.stringify(options())});
    assert.equal(valid.evaluation.accepted,true);console.log('contained');`;
  assert.equal(execFileSync(process.execPath,['--input-type=module','-e',source],{encoding:'utf8',timeout:10000}).trim(),'contained');
});
