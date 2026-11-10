#!/usr/bin/env node
// Controlled real-data comparison and workbench repair proof. No SOTA claim.
import assert from 'node:assert/strict';
import { readFile, writeFile, stat } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { cpus, platform, arch, totalmem, release } from 'node:os';
import { performance } from 'node:perf_hooks';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { join, dirname, resolve } from 'node:path';
import { ROOT, DATA_ROOT, SEEDS, sha256, digestJSON, rng, loadDatasets, stratifiedSplit, standardizer, asDataset, metrics, summary, parseOptions } from './wsp-benchmark-lib.mjs';

const options = parseOptions(process.argv.slice(2));
// Refuse existing output before expensive work. Final write remains exclusive;
// a failed run leaves no passing report.
try { await stat(options.output); throw new Error('benchmark output already exists'); } catch (error) { if (error.code !== 'ENOENT') throw error; }
const started = performance.now();
const sourceSha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim();
assert.match(sourceSha, /^[0-9a-f]{40}$/);
const worktreeClean = execFileSync('git', ['status', '--porcelain'], { cwd: ROOT, encoding: 'utf8' }).trim().length === 0;
const wsp = await import(pathToFileURL(options.wspModule));
for (const method of ['inspectWeightModel', 'traceWeightModel', 'analyzeWeightSensitivity', 'createNeuronAblationPatch', 'applyWeightPatch', 'fingerprintModel', 'evaluateWeightPatch', 'createWeightWorkspace'])
  assert.equal(typeof wsp[method], 'function', `workbench API ${method} unavailable`);
// Resolve runtime from the same installed WSP package, not a source checkout shim.
const runtimeModule = fileURLToPath(execFileSync(process.execPath, ['--input-type=module', '-e', "process.stdout.write(import.meta.resolve('ruv-fann'))"],
  { cwd: dirname(options.wspModule), encoding: 'utf8', timeout: 10_000 }).trim());
const wspPackage = JSON.parse(await readFile(join(dirname(dirname(options.wspModule)), 'package.json'), 'utf8'));
const runtimePackage = JSON.parse(await readFile(join(dirname(dirname(runtimeModule)), 'package.json'), 'utf8'));
assert.equal(wspPackage.name, 'ruv-fann-wsp'); assert.equal(runtimePackage.name, 'ruv-fann'); assert.equal(runtimePackage.version, '0.1.2');
const { loadRuntime } = await import(pathToFileURL(runtimeModule));
const runtime = await loadRuntime({ simd: false });
assert.equal(runtime.backend, 'wasm');
const { lock, lockSha256, datasets } = await loadDatasets();
const preparedJobs = datasets.flatMap(({ provenance, rows }) => SEEDS.map(seed => {
  const split = stratifiedSplit(rows, seed), scaler = standardizer(split.train);
  return { id: provenance.id, seed, split, scaler, train: scaler.transform(split.train), validation: scaler.transform(split.validation), test: scaler.transform(split.test) };
}));
const treeInput = { schema: 'ruv-fann.tree-benchmark-input/1', jobs: preparedJobs.map(job => ({ id: job.id, seed: job.seed,
  ...Object.fromEntries(['train', 'validation', 'test'].map(name => [name, { features: job[name].map(row => row.features), labels: job[name].map(row => row.label) }])) })) };
const python = options.python || process.env.WSP_BENCHMARK_PYTHON || 'python3';
assert.ok(python.length <= 4096 && !python.includes('\0'), 'invalid Python executable');
const treeBytes = execFileSync(python, [join(ROOT, 'scripts/wsp-benchmark-tree.py')], { cwd: ROOT,
  input: JSON.stringify(treeInput), encoding: 'utf8', maxBuffer: 2_000_000, timeout: 120_000,
  env: { ...process.env, OMP_NUM_THREADS: '1', OPENBLAS_NUM_THREADS: '1', MKL_NUM_THREADS: '1', PYTHONHASHSEED: '0' } });
const tree = JSON.parse(treeBytes);
assert.equal(tree.schema, 'ruv-fann.tree-benchmark/1'); assert.equal(tree.results.length, 10);

function initialModel(width, hidden, seed) {
  const random = rng(seed ^ 0xa5a5a5a5);
  const layer = (inputs, outputs, activation) => ({ inputs, outputs,
    weights: Array.from({ length: inputs * outputs }, () => (random() * 2 - 1) * Math.sqrt(6 / (inputs + outputs))),
    bias: Array(outputs).fill(0), activation, steepness: 1 });
  return { schema: 'ruv-fann.dense-layers/1', layers: hidden ? [layer(width, 8, 'Tanh'), layer(8, 3, 'Linear')] : [layer(width, 3, 'Linear')] };
}
function infer(model, rows) {
  const prepared = runtime.prepare(model, { maxRows: rows.length });
  try { return Array.from(prepared.predict(Float64Array.from(rows.flatMap(row => row.features)), rows.length)); }
  finally { prepared.dispose(); }
}
function evaluate(model, rows) { const scores = infer(model, rows); return { metrics: metrics(rows.map(row => row.label), scores), scores }; }
function oneHot(labels) { return labels.flatMap(value => [0, 1, 2].map(label => Number(value === label))); }
function baselines(job) {
  const counts = [0, 1, 2].map(label => job.train.filter(row => row.label === label).length);
  const majority = counts.indexOf(Math.max(...counts));
  const centers = [0, 1, 2].map(label => {
    const rows = job.train.filter(row => row.label === label);
    return rows[0].features.map((_, column) => rows.reduce((sum, row) => sum + row.features[column], 0) / rows.length);
  });
  const nearest = job.test.map(row => {
    const distances = centers.map(center => center.reduce((sum, value, column) => sum + (row.features[column] - value) ** 2, 0));
    return distances.indexOf(Math.min(...distances));
  });
  const result = {};
  for (const [name, predicted] of [['majority', job.test.map(() => majority)], ['nearest-centroid', nearest]]) {
    const scores = oneHot(predicted);
    result[name] = { scoreEncoding: 'hard one-hot labels', test: metrics(job.test.map(row => row.label), scores), testScores: scores };
  }
  return result;
}
function trainCandidate(job, hidden) {
  const model = initialModel(job.train[0].features.length, hidden, job.seed), data = asDataset(job.train), candidates = [];
  for (const epochs of [100, 200]) {
    const before = performance.now();
    const trained = runtime.train(model, { inputs: Float64Array.from(data.inputs), targets: Float64Array.from(data.targets), rows: data.rows,
      epochs, learningRate: 0.01, optimizer: 'adamw' });
    candidates.push({ trained, epochs, fitMs: performance.now() - before, validation: evaluate(trained.model, job.validation).metrics });
  }
  const selected = candidates.toSorted((left, right) => right.validation.accuracy - left.validation.accuracy || left.validation.mse - right.validation.mse || left.epochs - right.epochs)[0];
  const result = evaluate(selected.trained.model, job.test), prepared = runtime.prepare(selected.trained.model, { maxRows: job.test.length });
  const inputs = Float64Array.from(job.test.flatMap(row => row.features)), timings = [];
  try {
    for (let iteration = 0; iteration < 10; iteration++) prepared.predict(inputs, job.test.length);
    for (let iteration = 0; iteration < 50; iteration++) { const before = performance.now(); prepared.predict(inputs, job.test.length); timings.push(performance.now() - before); }
  } finally { prepared.dispose(); }
  timings.sort((a, b) => a - b);
  return { model: selected.trained.model, scoreEncoding: 'raw regression scores against one-hot targets', test: result.metrics, testScores: result.scores,
    parameterCount: selected.trained.parameterCount, selectedEpochs: selected.epochs,
    candidates: candidates.map(candidate => ({ epochs: candidate.epochs, fitMs: candidate.fitMs, initialLoss: candidate.trained.initialLoss,
      finalLoss: candidate.trained.finalLoss, validation: candidate.validation })),
    inference: { rows: job.test.length, warmup: 10, samples: 50, medianMs: timings[25], p95Ms: timings[47], scope: 'prepared scalar WASM batch call only; input construction/preparation excluded' } };
}
async function repairProof(model, job) {
  const originalHash = await wsp.fingerprintModel(model), inspection = await wsp.inspectWeightModel(model);
  assert.equal(inspection.baseHash, originalHash);
  const validation = evaluate(model, job.validation).metrics;
  const candidates = [];
  for (let neuron = 0; neuron < 8; neuron++) {
    const patch = await wsp.createNeuronAblationPatch(model, 0, neuron);
    const ablated = await wsp.applyWeightPatch(model, patch);
    candidates.push({ neuron, patch, ablated, validation: evaluate(ablated, job.validation).metrics });
  }
  // Choose intervention using validation only. Test labels never influence this choice.
  const selected = candidates.toSorted((left, right) => left.validation.accuracy - right.validation.accuracy || right.validation.mse - left.validation.mse || left.neuron - right.neuron)[0];
  const repair = { schema: 'ruv-fann.weight-patch/1', baseHash: await wsp.fingerprintModel(selected.ablated),
    edits: selected.patch.edits.map(edit => ({ ...edit, value: model.layers[edit.layer][edit.parameter][edit.index] })) };
  const validationData = asDataset(job.validation), originalValidation = evaluate(model, job.validation).metrics;
  const proof = await wsp.evaluateWeightPatch(selected.ablated, repair, validationData,
    { maxMSE: originalValidation.mse + 1e-12, maxRegressionMSE: 0 }, { simd: false });
  assert.equal(proof.accepted, true, 'exact repair must pass validation policy');
  assert.equal(proof.candidateHash, originalHash, 'repair must recover exact original weights');
  assert.ok(Math.abs(proof.candidate.mse - originalValidation.mse) <= 1e-12, 'independent JS MSE oracle disagrees with WASM repair evaluation');
  const workspace = await wsp.createWeightWorkspace(selected.ablated, { simd: false });
  const proposal = await workspace.propose(repair, validationData, { maxMSE: originalValidation.mse + 1e-12, maxRegressionMSE: 0 });
  const promoted = workspace.promote(proposal.id, workspace.snapshot().revision);
  assert.equal(promoted.hash, originalHash);
  const recovered = evaluate(promoted.model, job.test), original = evaluate(model, job.test), damaged = evaluate(selected.ablated, job.test);
  assert.deepEqual(recovered.scores, original.scores, 'exact repair must reproduce held-out outputs');
  const traceData = asDataset(job.test.slice(0, 3));
  const trace = await wsp.traceWeightModel(model, { inputs: traceData.inputs, rows: traceData.rows }, { simd: false });
  assert.deepEqual(trace.layers.at(-1).activations, infer(model, job.test.slice(0, 3)), 'trace must match full actual WASM model');
  const sensitivity = await wsp.analyzeWeightSensitivity(model, validationData,
    [{ layer: 1, parameter: 'weights', index: selected.neuron }], { simd: false, epsilon: 1e-4 });
  assert.equal(sensitivity.method, 'central-finite-difference');
  const minusModel = structuredClone(model), plusModel = structuredClone(model);
  minusModel.layers[1].weights[selected.neuron] -= 1e-4;
  plusModel.layers[1].weights[selected.neuron] += 1e-4;
  const minusMSE = evaluate(minusModel, job.validation).metrics.mse, plusMSE = evaluate(plusModel, job.validation).metrics.mse;
  const expectedDerivative = (plusMSE - minusMSE) / (plusModel.layers[1].weights[selected.neuron] - minusModel.layers[1].weights[selected.neuron]);
  assert.ok(Math.abs(sensitivity.entries[0].minus.mse - minusMSE) < 1e-12);
  assert.ok(Math.abs(sensitivity.entries[0].plus.mse - plusMSE) < 1e-12);
  assert.ok(Math.abs(sensitivity.entries[0].derivative - expectedDerivative) < 1e-8, 'sensitivity disagrees with independent MSE reduction');
  const rolled = workspace.rollback(1, promoted.revision);
  assert.equal(rolled.hash, repair.baseHash);
  return { selection: 'largest validation accuracy decrease, then MSE increase, then lowest neuron index; no test selection',
    baselineValidation: validation, candidates: candidates.map(candidate => ({ neuron: candidate.neuron, validation: candidate.validation })),
    selectedNeuron: selected.neuron, patch: selected.patch, repairPatch: repair, originalHash, damagedHash: repair.baseHash,
    acceptedOnValidation: proof.accepted, originalTest: original.metrics, ablatedTest: damaged.metrics, repairedTest: recovered.metrics,
    exactWeightHashRestored: true, exactHeldoutOutputsRestored: true, rollbackRestoresAblation: true, traceMatchesFullWasm: true, sensitivityMatchesIndependentMSE: true,
    sensitivity: { method: sensitivity.method, baseHash: sensitivity.baseHash, datasetHash: sensitivity.datasetHash, backend: sensitivity.backend,
      epsilon: sensitivity.epsilon, baseline: sensitivity.baseline,
      parameterValue: model.layers[1].weights[selected.neuron],
      minusValue: minusModel.layers[1].weights[selected.neuron], plusValue: plusModel.layers[1].weights[selected.neuron],
      independentMSE: { minus: minusMSE, plus: plusMSE }, entries: sensitivity.entries },
    limitation: 'Controlled single-neuron damage with known original weights; restoration is not discovery of better weights or general autonomous repair.' };
}
const runs = [];
for (const [index, job] of preparedJobs.entries()) {
  const linear = trainCandidate(job, false), mlp = trainCandidate(job, true), baseline = baselines(job);
  const treeResult = tree.results[index];
  assert.equal(treeResult.id, job.id); assert.equal(treeResult.seed, job.seed);
  const treeMetrics = metrics(job.test.map(row => row.label), treeResult.testScores);
  for (const metric of ['accuracy', 'macroF1', 'mse']) assert.ok(Math.abs(treeMetrics[metric] - treeResult.oracle[metric]) < 1e-12, 'independent sklearn metric oracle mismatch');
  const repair = await repairProof(mlp.model, job);
  const modelHashes = { linear: await wsp.fingerprintModel(linear.model), mlp: await wsp.fingerprintModel(mlp.model) };
  delete linear.model; delete mlp.model;
  runs.push({ dataset: job.id, seed: job.seed, splitHash: job.split.hash, splitRowIds: job.split.ids,
    partitions: Object.fromEntries(['train', 'validation', 'test'].map(name => [name, { rows: job[name].length, classCounts: [0, 1, 2].map(label => job[name].filter(row => row.label === label).length) }])),
    standardization: { fit: 'train only; population SD; zero SD replaced with 1', mean: job.scaler.mean, scale: job.scaler.scale, fittedRowIds: job.scaler.fittedRowIds },
    heldoutLabels: job.test.map(row => row.label), modelHashes, models: { ...baseline, linear, mlp,
      'extra-trees': { scoreEncoding: 'predict_proba', test: treeMetrics, testScores: treeResult.testScores, selected: treeResult.selected,
        candidates: treeResult.candidates, testPredictionMs: treeResult.testPredictionMs } }, repair });
  process.stderr.write(`benchmark ${job.id} seed ${job.seed}: linear=${linear.test.accuracy.toFixed(3)} mlp=${mlp.test.accuracy.toFixed(3)} trees=${treeMetrics.accuracy.toFixed(3)}\n`);
}
const aggregate = datasets.map(({ provenance }) => {
  const selected = runs.filter(run => run.dataset === provenance.id), models = {};
  for (const model of ['majority', 'nearest-centroid', 'linear', 'mlp', 'extra-trees'])
    models[model] = Object.fromEntries(['accuracy', 'macroF1', 'mse'].map(metric => [metric, summary(selected.map(run => run.models[model].test[metric]))]));
  return { dataset: provenance.id, models, pairedAccuracyDeltas: Object.fromEntries(['nearest-centroid', 'linear', 'extra-trees'].map(baseline => [baseline,
    summary(selected.map(run => run.models.mlp.test.accuracy - run.models[baseline].test.accuracy))])) };
});
const cpu = cpus();
const report = { schema: 'ruv-fann.workbench-benchmark/1', status: 'passed', sourceSha, sourceWorktreeClean: worktreeClean,
  package: { name: wspPackage.name, version: wspPackage.version }, runtimePackage: { name: runtimePackage.name, version: runtimePackage.version },
  benchmarkScriptSha256: sha256(await readFile(new URL(import.meta.url))), benchmarkHelpersSha256: sha256(await readFile(join(ROOT, 'scripts/wsp-benchmark-lib.mjs'))),
  moduleEntrySha256: sha256(await readFile(options.wspModule)), runtimeModuleEntrySha256: sha256(await readFile(runtimeModule)),
  datasetLockSha256: lockSha256, dataProvenance: lock,
  tooling: { node: process.version, backend: runtime.backend, python: tree.python, pythonPackages: tree.versions,
    pythonRequirementsSha256: sha256(await readFile(join(DATA_ROOT, 'requirements.lock'))), treeScriptSha256: sha256(await readFile(join(ROOT, 'scripts/wsp-benchmark-tree.py'))) },
  hardware: { platform: platform(), release: release(), arch: arch(), cpuModel: cpu[0]?.model ?? 'unknown', logicalCPUs: cpu.length, totalMemoryBytes: totalmem() },
  plan: { seeds: SEEDS, split: 'stratified per class: floor(0.60*n) train, floor(0.20*n) validation, remainder test',
    primaryMetric: 'classification accuracy; raw argmax, first class wins ties', secondaryMetrics: ['macroF1', 'MSE against one-hot targets'],
    models: { linear: 'D→3 Linear', mlp: 'D→8 Tanh→3 Linear', optimizer: 'single-call full-batch AdamW, learningRate=0.01', epochs: [100, 200],
      tree: 'ExtraTreesClassifier,128 trees,max_depth=[3,None],n_jobs=1' },
    selection: 'Within each model family: validation accuracy, then validation MSE, then lower epochs/earlier config. No refit on validation. Test only after selection.',
    budgetRationale: '200 is the largest shared planned round epoch budget below the published runtime 100M per-call work bound for Wine MLP (105 rows,139 parameters); 300 exceeds it. Chosen by cost formula, not test scores.',
    tuningBudget: 'Two fixed candidates per neural/tree family; compute/objectives differ and this is not an equal-compute comparison.' },
  checks: { rawHashesVerified: true, disjointSplits: true, trainOnlyStandardization: true, fiveSeedsEach: runs.length === 10,
    noTestSelection: true, sklearnMetricOracle: true, exactRepairValidated: runs.every(run => run.repair.exactHeldoutOutputsRestored),
    traceMatchesFullWasm: runs.every(run => run.repair.traceMatchesFullWasm), sensitivityMatchesIndependentMSE: runs.every(run => run.repair.sensitivityMatchesIndependentMSE) },
  aggregate, runs, elapsedMs: performance.now() - started,
  limitations: ['Two small classical UCI datasets, not agent-task or large-model evidence.',
    'Five overlapping stratified splits; descriptive t intervals do not establish independent population significance.',
    'All outcomes retained, including losses to baselines. No SOTA claim or speedup claim.',
    'Timing depends on this host; WASM inference samples exclude preparation and use different call boundaries from sklearn.',
    'MSE score encodings differ across hard-label baselines, probability trees, and neural regression scores.',
    'Validation labels choose epochs and ablation; untouched test labels score final selected candidates only.'] };
assert.ok(Object.values(report.checks).every(value => value === true));
await writeFile(options.output, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx', mode: 0o600 });
console.log(JSON.stringify({ status: report.status, sourceSha, sourceWorktreeClean: worktreeClean, output: options.output, reportSha256: sha256(await readFile(options.output)), runs: runs.length, elapsedMs: report.elapsedMs }));
