#!/usr/bin/env node
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { cpus, arch } from 'node:os';
import { isAbsolute } from 'node:path';
import { performance } from 'node:perf_hooks';
import { ROOT, SEEDS, asDataset, digestJSON, loadDatasets, metrics, rng, sha256, standardizer, stratifiedSplit } from './wsp-benchmark-lib.mjs';
import { coordinateSearch, groupSearch, medianClip, refitReadout } from './wsp-blind-repair-controls.mjs';
import { applyWeightPatch, createWeightWorkspace, fingerprintModel, traceWeightModel } from '../packages/wsp/src/index.js';

const LOAD_RUNTIME = new URL('../packages/wsp/node_modules/ruv-fann/src/index.js', import.meta.url);
const MAX_EVALUATIONS = 20;
const FAMILIES = ['outliers', 'sign-flip', 'zero-row'];
const METHODS = ['no-edit', 'clipping', 'coordinate', 'refit', 'group-search', 'agent'];
const INPUT_SCHEMA = 'ruv-fann.blind-repair-input/1';
const PLAN_SCHEMA = 'ruv-fann.blind-repair-plan/1';

function options(args) {
  const [mode, output, plan, ...extra] = args;
  const usage = 'usage: node scripts/wsp-blind-repair.mjs prepare ABSOLUTE_INPUT | score ABSOLUTE_REPORT ABSOLUTE_PLAN';
  assert.ok(extra.length === 0 && output && isAbsolute(output), usage);
  assert.ok((mode === 'prepare' && !plan) || (mode === 'score' && plan && isAbsolute(plan) && plan !== output), usage);
  return { mode, output, plan };
}

function initialModel(width, seed) {
  const random = rng(seed ^ 0xa5a5a5a5);
  function layer(inputs, outputs, activation) {
    return { inputs, outputs, activation, steepness: 1,
      weights: Array.from({ length: inputs * outputs }, () => (random() * 2 - 1) * Math.sqrt(6 / (inputs + outputs))),
      bias: Array(outputs).fill(0) };
  }
  return { schema: 'ruv-fann.dense-layers/1', layers: [layer(width, 8, 'Tanh'), layer(8, 3, 'Linear')] };
}

function partitionValidation(validation) {
  const development = [], gate = [];
  for (const label of [0, 1, 2]) {
    const rows = validation.filter(row => row.label === label);
    const midpoint = Math.floor(rows.length / 2);
    development.push(...rows.slice(0, midpoint));
    gate.push(...rows.slice(midpoint));
  }
  return { development, gate };
}

function makeEvaluator(runtime, rows) {
  const features = Float64Array.from(rows.flatMap(row => row.features));
  const labels = rows.map(row => row.label);
  return model => {
    const prepared = runtime.prepare(model, { maxRows: rows.length });
    try {
      return metrics(labels, prepared.predict(features, rows.length));
    } finally {
      prepared.dispose();
    }
  };
}

function corrupt(model, seed, dataset, family) {
  const damaged = structuredClone(model);
  // Precommitted varied locations with both layers represented.
  const seedIndex = SEEDS.indexOf(seed);
  const familyIndex = FAMILIES.indexOf(family);
  const datasetIndex = Number(dataset === 'wine');
  const random = rng((seed * 71_297) ^ (familyIndex * 9_418) ^ (datasetIndex * 3_481));
  const layerIndex = (seedIndex + familyIndex + datasetIndex) % 2;
  const layer = damaged.layers[layerIndex];
  const row = Math.floor(random() * layer.outputs);
  const offset = row * layer.inputs;
  for (let column = 0; column < layer.inputs; column++) {
    const index = offset + column;
    if (family === 'outliers') layer.weights[index] = (random() < 0.5 ? -1 : 1) * (8 + 8 * random());
    if (family === 'sign-flip') layer.weights[index] *= -1;
    if (family === 'zero-row') layer.weights[index] = 0;
  }
  return damaged;
}

function rowsOf(flat, width) {
  return Array.from({ length: flat.length / width }, (_, row) => Array.from(flat.slice(row * width, (row + 1) * width)));
}

async function describeCase(model, development, evaluate, id) {
  const baseline = evaluate(model);
  const { inputs, rows } = asDataset(development);
  const trace = await traceWeightModel(model, { inputs, rows }, { simd: false });
  return {
    id,
    development: { examples: rows, accuracy: baseline.accuracy, mse: baseline.mse, confusion: baseline.confusion },
    layers: model.layers.map((layer, index) => {
      const activations = rowsOf(trace.layers[index].activations, layer.outputs);
      const columns = Array.from({ length: layer.outputs }, (_, column) => activations.map(row => row[column]));
      return {
        layer: index, weightsByOutput: rowsOf(layer.weights, layer.inputs), bias: layer.bias,
        activation: layer.activation,
        activationMean: columns.map(values => values.reduce((sum, value) => sum + value, 0) / values.length),
        activationAbsMean: columns.map(values => values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length)
      };
    })
  };
}

async function buildCases(runtime) {
  const { datasets, lockSha256 } = await loadDatasets();
  const cases = [];
  for (const { provenance, rows } of datasets) {
    const dataset = provenance.id;
    for (const seed of SEEDS) {
      const split = stratifiedSplit(rows, seed);
      const scaler = standardizer(split.train);
      const data = asDataset(scaler.transform(split.train));
      const { development, gate } = partitionValidation(scaler.transform(split.validation));
      const test = scaler.transform(split.test);
      for (const [familyIndex, family] of [...FAMILIES, 'clean'].entries()) {
        // Each case starts from a separately initialized checkpoint. Matching
        // weights across cases must never expose the intact values of a fault.
        const modelSeed = seed + 1024 * (familyIndex + 4 * Number(dataset === 'wine'));
        const { model } = runtime.train(initialModel(rows[0].features.length, modelSeed), {
          inputs: Float64Array.from(data.inputs), targets: Float64Array.from(data.targets), rows: data.rows,
          epochs: 200, learningRate: 0.01, optimizer: 'adamw'
        });
        const id = digestJSON(['blind-repair-v1', dataset, seed, family]).slice(0, 16);
        const damaged = family === 'clean' ? model : corrupt(model, seed, dataset, family);
        const evaluate = makeEvaluator(runtime, development);
        const description = await describeCase(damaged, development, evaluate, id);
        cases.push({ id, dataset, seed, family, splitHash: split.hash, damaged, intact: model, development, gate, test, evaluate, description });
      }
    }
  }
  return { cases: cases.toSorted((left, right) => left.id.localeCompare(right.id)), lockSha256 };
}

function proposerInput(cases) {
  return {
    schema: INPUT_SCHEMA,
    contract: {
      output: { schema: PLAN_SCHEMA, inputSha256: '<SHA256 of input bytes>', cases: { '<every provided id>': ['<0 to 20 operator objects>'] } },
      note: 'Propose only from these diagnostics. Empty arrays abstain. No evaluator access. Each selected candidate is later scored on development, and only the best qualifying candidate proceeds to an independent gate.',
      operators: {
        flip: { op: 'flip', layer: 'integer layer', row: 'integer output row' },
        scale: { op: 'scale', layer: 'integer layer', row: 'integer output row', factor: 'one of 0, 0.05, 0.1, 0.25, 0.5, 2' },
        clip: { op: 'clip', layer: 'integer layer', row: 'integer output row', bound: 'positive finite magnitude <= 2' },
        refit: { op: 'refit', ridge: 'one of 0.01, 0.1, 1' }
      }
    },
    cases: cases.map(item => item.description)
  };
}

function exactKeys(value, keys) {
  assert.ok(value && typeof value === 'object' && !Array.isArray(value), 'expected object');
  assert.deepEqual(Object.keys(value).sort(), keys.toSorted(), 'unexpected object fields');
}

async function loadPlan(path, inputBytes, cases) {
  assert.ok((await stat(path)).size <= 150_000, 'plan exceeds byte budget');
  const bytes = await readFile(path);
  assert.ok(bytes.length <= 150_000, 'plan exceeds byte budget');
  const plan = JSON.parse(bytes);
  exactKeys(plan, ['schema', 'inputSha256', 'cases']);
  assert.equal(plan.schema, PLAN_SCHEMA);
  assert.equal(plan.inputSha256, sha256(inputBytes), 'plan must bind exact diagnostics');
  exactKeys(plan.cases, cases.map(item => item.id));
  for (const { id } of cases) {
    const candidates = plan.cases[id];
    assert.ok(Array.isArray(candidates) && candidates.length <= MAX_EVALUATIONS, 'plan exceeds evaluation budget');
  }
  return { plan, planSha256: sha256(bytes) };
}

function operatorEdits(operation, model, refit) {
  if (operation?.op === 'refit') {
    exactKeys(operation, ['op', 'ridge']);
    assert.ok([0.01, 0.1, 1].includes(operation.ridge), 'unrecognized refit ridge');
    return refit(operation.ridge);
  }
  const { op, layer: layerIndex, row } = operation;
  assert.ok(['flip', 'scale', 'clip'].includes(op), 'unrecognized operation');
  exactKeys(operation, ['op', 'layer', 'row', ...({ flip: [], scale: ['factor'], clip: ['bound'] })[op]]);
  assert.ok(Number.isInteger(layerIndex) && layerIndex >= 0 && layerIndex < model.layers.length, 'layer out of range');
  const layer = model.layers[layerIndex];
  assert.ok(Number.isInteger(row) && row >= 0 && row < layer.outputs, 'row out of range');
  if (op === 'scale') assert.ok([0, 0.05, 0.1, 0.25, 0.5, 2].includes(operation.factor), 'invalid scale');
  if (op === 'clip') assert.ok(Number.isFinite(operation.bound) && operation.bound > 0 && operation.bound <= 2, 'invalid clip bound');
  const first = row * layer.inputs;
  return layer.weights.slice(first, first + layer.inputs).flatMap((weight, column) => {
    const value = op === 'clip' ? Math.max(-operation.bound, Math.min(operation.bound, weight)) : weight * (op === 'flip' ? -1 : operation.factor);
    return value === weight ? [] : [{ layer: layerIndex, parameter: 'weights', index: first + column, value }];
  });
}

function rank({ accuracy, mse }) {
  // Across this fixture, the smallest accuracy change is at least 1/20.
  // This bounded MSE tiebreaker cannot outrank it.
  return accuracy - 0.01 * (mse / (1 + mse));
}

async function selectMethod(method, item, baseHash, evaluateGate, gateBaseline, operations, refit) {
  const started = performance.now();
  const { damaged: model, evaluate } = item;
  const developmentBaseline = item.description.development;
  let evaluations = 0, noOps = 0;
  let best = null;
  async function consider(edits) {
    assert.ok(evaluations < MAX_EVALUATIONS, 'candidate budget exceeded');
    evaluations++;
    if (!edits.length) {
      noOps++;
      return null;
    }
    const patch = { schema: 'ruv-fann.weight-patch/1', baseHash, edits };
    const candidate = await applyWeightPatch(model, patch);
    const stats = evaluate(candidate);
    const score = rank(stats);
    if (!best || score > best.score) best = { candidate, patch, stats, score };
    return score;
  }

  if (method === 'clipping') await consider(medianClip(model));
  if (method === 'coordinate') await coordinateSearch(model, consider, MAX_EVALUATIONS);
  if (method === 'refit') await consider(refit(0.1));
  if (method === 'group-search') await groupSearch(model, consider, MAX_EVALUATIONS);
  if (method === 'agent') {
    for (const operation of operations) await consider(operatorEdits(operation, model, refit));
  }

  const gate = best ? evaluateGate(best.candidate) : null;
  const improved = best && best.score > rank(developmentBaseline) + 1e-12;
  const accepted = Boolean(improved && gate.accuracy >= gateBaseline.accuracy && gate.mse <= gateBaseline.mse + 1e-12);
  const selection = { accepted, evaluations, noOps, development: best?.stats ?? null, gate,
    gateBaseline, changedWeights: accepted ? best.patch.edits.length : 0,
    selectionMs: performance.now() - started };
  if (!accepted) return { ...selection, rollback: null, patch: null, hash: baseHash, model };
  const workspace = await createWeightWorkspace(model, { simd: false });
  const data = asDataset(item.gate);
  const proposal = await workspace.propose(best.patch, data, { maxMSE: gate.mse + 1e-12, maxRegressionMSE: 1e-12 });
  const baseRevision = workspace.snapshot().revision;
  const promoted = workspace.promote(proposal.id, baseRevision);
  assert.equal(promoted.hash, await fingerprintModel(best.candidate), 'promotion changed candidate');
  assert.throws(() => workspace.promote(proposal.id, baseRevision), /stale/i);
  const rolled = workspace.rollback(baseRevision, promoted.revision);
  assert.equal(rolled.hash, baseHash, 'rollback must restore the pre-edit checkpoint');
  return { ...selection, rollback: true, patch: best.patch, hash: promoted.hash, model: promoted.model };
}

async function scoreCase(runtime, item, operations) {
  const baseHash = await fingerprintModel(item.damaged);
  const intactHash = await fingerprintModel(item.intact);
  assert.equal(baseHash === intactHash, item.family === 'clean');
  const gate = makeEvaluator(runtime, item.gate);
  const gateBaseline = gate(item.damaged);
  const test = makeEvaluator(runtime, item.test);
  const { inputs, rows, targets } = asDataset(item.development);
  const trace = await traceWeightModel(item.damaged, { inputs, rows }, { simd: false });
  const activations = rowsOf(trace.layers.at(-2).activations, item.damaged.layers.at(-1).inputs);
  const targetRows = rowsOf(targets, item.damaged.layers.at(-1).outputs);
  const refit = ridge => refitReadout(item.damaged, activations, targetRows, ridge);
  const selections = {};
  for (const method of METHODS) selections[method] = await selectMethod(method, item, baseHash, gate, gateBaseline, operations, refit);
  const intactTest = test(item.intact);
  const methods = Object.fromEntries(METHODS.map(method => {
    const { model, ...selection } = selections[method];
    return [method, { ...selection, test: test(model) }];
  }));
  return { id: item.id, dataset: item.dataset, seed: item.seed, family: item.family, splitHash: item.splitHash,
    baseHash, intactHash, intactTest, developmentBaseline: item.description.development, methods };
}

function aggregate(runs) {
  const summarize = cases => Object.fromEntries(METHODS.map(method => {
    const entries = cases.map(item => item.methods[method]);
    const meanAccuracy = entries.reduce((sum, entry) => sum + entry.test.accuracy, 0) / entries.length;
    return [method, { meanAccuracy, accepted: entries.filter(entry => entry.accepted).length,
      candidateEvaluations: entries.reduce((sum, entry) => sum + entry.evaluations, 0),
      medianSelectionMs: entries.map(entry => entry.selectionMs).sort((a, b) => a - b)[Math.floor(entries.length / 2)] }];
  }));
  return Object.fromEntries(['iris', 'wine'].map(dataset => {
    const cases = runs.filter(item => item.dataset === dataset);
    const damage = summarize(cases.filter(item => item.family !== 'clean'));
    const clean = summarize(cases.filter(item => item.family === 'clean'));
    const strongest = METHODS.filter(method => method !== 'agent')
      .toSorted((left, right) => damage[right].meanAccuracy - damage[left].meanAccuracy)[0];
    return [dataset, { damage, clean, byFault: Object.fromEntries(FAMILIES.map(family => [family, summarize(cases.filter(item => item.family === family))])),
      strongest, agentGain: damage.agent.meanAccuracy - damage[strongest].meanAccuracy }];
  }));
}

const { mode, output, plan: planPath } = options(process.argv.slice(2));
try {
  await stat(output);
  throw new Error('output already exists');
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}
const started = performance.now();
const runtimePackage = JSON.parse(await readFile(new URL('../package.json', LOAD_RUNTIME), 'utf8'));
const wasmRoot = new URL('../packages/wsp/node_modules/ruv-fann-wasm/', import.meta.url);
const wasmPackage = JSON.parse(await readFile(new URL('package.json', wasmRoot), 'utf8'));
assert.equal(runtimePackage.name, 'ruv-fann');
assert.equal(runtimePackage.version, '0.1.2');
assert.equal(wasmPackage.name, 'ruv-fann-wasm');
assert.equal(wasmPackage.version, '0.1.2');
const { loadRuntime } = await import(LOAD_RUNTIME);
const runtime = await loadRuntime({ simd: false });
assert.equal(runtime.backend, 'wasm');
const { cases, lockSha256 } = await buildCases(runtime);
const inputBytes = Buffer.from(JSON.stringify(proposerInput(cases), null, 2) + '\n');
if (mode === 'prepare') {
  await writeFile(output, inputBytes, { flag: 'wx' });
  console.log('Prepared %d blind cases (SHA256 %s).', cases.length, sha256(inputBytes));
} else {
  const { plan, planSha256 } = await loadPlan(planPath, inputBytes, cases);
  const runs = [];
  for (const item of cases) runs.push(await scoreCase(runtime, item, plan.cases[item.id]));
  const results = aggregate(runs);
  const gates = {
    twoPointGainOnBothDatasets: Object.values(results).every(dataset => dataset.agentGain >= 0.02),
    cleanAbstention: runs.filter(item => item.family === 'clean').every(item => !item.methods.agent.accepted),
    exactRollback: runs.every(item => METHODS.every(method => !item.methods[method].accepted || item.methods[method].rollback))
  };
  const git = args => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
  const scriptHashes = Object.fromEntries(await Promise.all(['wsp-blind-repair.mjs', 'wsp-blind-repair-controls.mjs', 'wsp-benchmark-lib.mjs']
    .map(async name => [name, sha256(await readFile(new URL(name, import.meta.url)))])));
  const artifacts = new URL('dist/', wasmRoot);
  const wasmHashes = Object.fromEntries(await Promise.all((await readdir(artifacts)).sort()
    .filter(name => name.endsWith('.wasm'))
    .map(async name => [name, sha256(await readFile(new URL(name, artifacts)))])));
  assert.ok(Object.keys(wasmHashes).length > 0, 'expected bundled WASM artifacts');
  const resultDigest = digestJSON(runs.map(run => ({
    ...run,
    methods: Object.fromEntries(Object.entries(run.methods).map(([method, { selectionMs, ...result }]) => [method, result]))
  })));
  const report = {
    schema: 'ruv-fann.blind-repair-report/1',
    acceptance: Object.values(gates).every(Boolean) ? 'pass' : 'hold',
    gates, sourceSha: git(['rev-parse', 'HEAD']), worktreeClean: git(['status', '--porcelain']) === '',
    scriptHashes, datasetLockSha256: lockSha256, inputSha256: sha256(inputBytes), planSha256,
    runtime: { backend: runtime.backend, node: process.version, cpu: cpus()[0]?.model, arch: arch(),
      core: runtimePackage.version, wasm: wasmPackage.version, wasmHashes },
    resultDigest, wallMs: performance.now() - started, results, runs
  };
  await writeFile(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ acceptance: report.acceptance, gates, resultDigest, wallMs: report.wallMs, results }, null, 2));
}
