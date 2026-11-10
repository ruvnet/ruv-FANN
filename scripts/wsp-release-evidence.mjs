// Semantic revalidation shared by candidate sealing and protected publication.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { ROOT, DATA_ROOT, SEEDS, sha256, loadDatasets, stratifiedSplit, standardizer, metrics, summary, asDataset, digestJSON } from './wsp-benchmark-lib.mjs';

export function validateWorkbenchProof(proof) {
  assert.equal(proof?.workbench?.traceBackend, 'wasm-simd');
  assert.equal(proof.workbench.sensitivityMethod, 'central-finite-difference');
  assert.equal(proof.workbench.ablationHashBound, true);
  assert.equal(proof.workbench.analyticDerivative, true);
  assert.equal(proof.security?.status, 'passed');
  assert.equal(proof.security.failed, 0);
  assert.equal(proof.security.getterCalls, 0);
  assert.ok(Number.isSafeInteger(proof.security.checks) && proof.security.checks >= 19);
}
export function validateSecurity(report, sourceSha, tarballSha256) {
  assert.equal(report.schema, 'ruv-fann-wsp.npm-security/1');
  assert.equal(report.status, 'passed'); assert.equal(report.sourceSha, sourceSha);
  assert.equal(report.tarballSha256, tarballSha256);
  for (const level of ['total', 'info', 'low', 'moderate', 'high', 'critical']) assert.equal(report.vulnerabilities?.[level], 0);
}
export async function validateBenchmark(report, sourceSha, archivePath) {
  assert.equal(report.schema, 'ruv-fann.workbench-benchmark/1');
  assert.equal(report.status, 'passed'); assert.equal(report.sourceSha, sourceSha);
  assert.equal(report.sourceWorktreeClean, true);
  for (const key of ['rawHashesVerified', 'disjointSplits', 'trainOnlyStandardization', 'fiveSeedsEach',
    'noTestSelection', 'sklearnMetricOracle', 'exactRepairValidated', 'traceMatchesFullWasm', 'sensitivityMatchesIndependentMSE']) assert.equal(report.checks?.[key], true, key);
  assert.ok(Object.values(report.checks).every(value => value === true));
  assert.deepEqual(report.package, { name: 'ruv-fann-wsp', version: '0.2.0' });
  assert.deepEqual(report.runtimePackage, { name: 'ruv-fann', version: '0.1.2' });
  assert.equal(report.benchmarkScriptSha256, sha256(await readFile(join(ROOT, 'scripts/wsp-benchmark.mjs'))));
  assert.equal(report.benchmarkHelpersSha256, sha256(await readFile(join(ROOT, 'scripts/wsp-benchmark-lib.mjs'))));
  assert.equal(report.tooling?.backend, 'wasm');
  assert.equal(report.tooling.pythonPackages?.['scikit-learn'], '1.7.2');
  assert.equal(report.tooling.pythonPackages?.numpy, '1.26.4');
  const { datasets, lock, lockSha256 } = await loadDatasets();
  assert.equal(report.datasetLockSha256, lockSha256); assert.deepEqual(report.dataProvenance, lock);
  assert.equal(report.tooling.pythonRequirementsSha256, sha256(await readFile(join(DATA_ROOT, 'requirements.lock'))));
  assert.equal(report.tooling.treeScriptSha256, sha256(await readFile(join(ROOT, 'scripts/wsp-benchmark-tree.py'))));
  const entryBytes = archivePath ? execFileSync('tar', ['-xOzf', archivePath, 'package/src/index.js']) : await readFile(join(ROOT, 'packages/wsp/src/index.js'));
  assert.equal(report.moduleEntrySha256, sha256(entryBytes));
  assert.deepEqual(report.plan.seeds, SEEDS);
  assert.deepEqual(report.plan.models.epochs, [100, 200]);
  assert.equal(report.runs?.length, 10); assert.equal(report.aggregate?.length, 2);
  const keys = new Set();
  for (const run of report.runs) {
    const key = `${run.dataset}:${run.seed}`; assert.ok(!keys.has(key)); keys.add(key);
    const dataset = datasets.find(entry => entry.provenance.id === run.dataset); assert.ok(dataset);
    assert.ok(SEEDS.includes(run.seed));
    const split = stratifiedSplit(dataset.rows, run.seed), scaler = standardizer(split.train);
    assert.equal(run.splitHash, split.hash); assert.deepEqual(run.splitRowIds, split.ids);
    assert.deepEqual(run.standardization.fittedRowIds, scaler.fittedRowIds);
    assert.deepEqual(run.standardization.mean, scaler.mean); assert.deepEqual(run.standardization.scale, scaler.scale);
    assert.deepEqual(run.heldoutLabels, split.test.map(row => row.label));
    for (const name of ['majority', 'nearest-centroid', 'linear', 'mlp', 'extra-trees']) {
      const model = run.models?.[name]; assert.ok(model);
      const measured = metrics(run.heldoutLabels, model.testScores);
      for (const metric of ['accuracy', 'macroF1', 'mse']) assert.ok(Math.abs(measured[metric] - model.test[metric]) <= 1e-12, `${name} ${metric}`);
    }
    for (const name of ['linear', 'mlp']) {
      assert.match(run.modelHashes[name], /^[a-f0-9]{64}$/);
      const model = run.models[name]; assert.deepEqual(model.candidates.map(row => row.epochs), [100, 200]);
      const chosen = model.candidates.toSorted((a,b) => b.validation.accuracy - a.validation.accuracy || a.validation.mse - b.validation.mse || a.epochs - b.epochs)[0];
      assert.equal(model.selectedEpochs, chosen.epochs);
    }
    assert.equal(run.repair.exactWeightHashRestored, true); assert.equal(run.repair.exactHeldoutOutputsRestored, true);
    assert.equal(run.repair.rollbackRestoresAblation, true); assert.equal(run.repair.traceMatchesFullWasm, true);
    assert.equal(run.repair.acceptedOnValidation, true); assert.equal(run.repair.originalHash, run.modelHashes.mlp);
    assert.equal(run.repair.repairPatch.baseHash, run.repair.damagedHash);
    const tree = run.models['extra-trees'];
    assert.deepEqual(tree.candidates.map(row => [row.trees, row.maxDepth]), [[128, 3], [128, null]]);
    const selectedTree = tree.candidates.map((row,index) => ({...row,index})).toSorted((a,b) => b.validationAccuracy-a.validationAccuracy || a.validationMSE-b.validationMSE || a.index-b.index)[0];
    assert.equal(tree.selected, selectedTree.index);
    const repair = run.repair;
    assert.deepEqual(repair.candidates.map(row => row.neuron), [0,1,2,3,4,5,6,7]);
    const selectedNeuron = repair.candidates.toSorted((a,b) => a.validation.accuracy-b.validation.accuracy || b.validation.mse-a.validation.mse || a.neuron-b.neuron)[0];
    assert.equal(repair.selectedNeuron, selectedNeuron.neuron);
    assert.deepEqual(repair.originalTest, run.models.mlp.test); assert.deepEqual(repair.repairedTest, repair.originalTest);
    const sensitivity = repair.sensitivity;
    assert.equal(sensitivity.method, 'central-finite-difference'); assert.equal(sensitivity.backend, 'wasm');
    assert.equal(sensitivity.baseHash, repair.originalHash); assert.equal(sensitivity.epsilon, 1e-4);
    const validation = asDataset(scaler.transform(split.validation));
    assert.equal(sensitivity.datasetHash, digestJSON({schema:'ruv-fann.weight-dataset/1',inputSize:dataset.provenance.features,outputSize:3,
      rows:validation.rows,inputs:validation.inputs,targets:validation.targets}));
    assert.ok(Math.abs(sensitivity.baseline.mse-repair.baselineValidation.mse) <= 1e-12);
    assert.equal(sensitivity.entries.length, 1); const entry=sensitivity.entries[0];
    assert.equal(entry.layer, 1); assert.equal(entry.parameter, 'weights'); assert.equal(entry.index, repair.selectedNeuron);
    const restored = repair.repairPatch.edits.find(edit => edit.layer===1 && edit.parameter==='weights' && edit.index===repair.selectedNeuron);
    assert.ok(restored); assert.equal(sensitivity.parameterValue, restored.value);
    assert.equal(sensitivity.minusValue, sensitivity.parameterValue-sensitivity.epsilon);
    assert.equal(sensitivity.plusValue, sensitivity.parameterValue+sensitivity.epsilon);
    assert.ok(sensitivity.plusValue>sensitivity.minusValue);
    const denominator=sensitivity.plusValue-sensitivity.minusValue;
    assert.ok(Number.isFinite(entry.derivative));
    assert.ok(Math.abs(entry.derivative-(entry.plus.mse-entry.minus.mse)/denominator) <= 1e-12);
    assert.ok(Math.abs(entry.minus.mse-sensitivity.independentMSE.minus) <= 1e-12);
    assert.ok(Math.abs(entry.plus.mse-sensitivity.independentMSE.plus) <= 1e-12);
    assert.ok(Math.abs(entry.derivative-(sensitivity.independentMSE.plus-sensitivity.independentMSE.minus)/denominator) <= 1e-8);
  }
  for (const dataset of datasets) for (const seed of SEEDS) assert.ok(keys.has(`${dataset.provenance.id}:${seed}`));
  for (const aggregate of report.aggregate) {
    const runs = report.runs.filter(run => run.dataset === aggregate.dataset); assert.equal(runs.length, 5);
    for (const model of ['majority', 'nearest-centroid', 'linear', 'mlp', 'extra-trees']) for (const metric of ['accuracy', 'macroF1', 'mse']) {
      assert.deepEqual(aggregate.models[model][metric], summary(runs.map(run => run.models[model].test[metric])));
    }
    for (const baseline of ['nearest-centroid','linear','extra-trees'])
      assert.deepEqual(aggregate.pairedAccuracyDeltas[baseline], summary(runs.map(run => run.models.mlp.test.accuracy-run.models[baseline].test.accuracy)));
  }
}
