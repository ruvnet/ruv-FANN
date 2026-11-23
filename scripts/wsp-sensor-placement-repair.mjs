#!/usr/bin/env node
import assert from 'node:assert/strict';
import { readFile, stat, writeFile } from 'node:fs/promises';
import { cpus } from 'node:os';
import { isAbsolute, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { applyWeightPatch, createWeightWorkspace, fingerprintModel, traceWeightModel } from '../packages/wsp/src/index.js';
import { digestJSON, rng, sha256 } from './wsp-benchmark-lib.mjs';
import { refitReadout } from './wsp-blind-repair-controls.mjs';

const ARCHIVE_SHA = '16ad0ce709f3f00df18f348610d15bce0884b79e2143f57f446493673f02b8e0';
const PLACEMENTS = ['chest', 'wrist', 'ankle'];
const SUBJECTS = Array.from({ length: 10 }, (_, index) => index + 1);
const CLASSES = Array.from({ length: 12 }, (_, index) => index + 1);
const WIDTH = 11;
const HIDDEN = 8;
const RUNTIME = new URL('../packages/wsp/node_modules/ruv-fann/src/index.js', import.meta.url);
const WASM_ROOT = new URL('../packages/wsp/node_modules/ruv-fann-wasm/', import.meta.url);
const RULES = Object.freeze({
  triggerDrop: 0.10,
  gateGain: 0.03,
  lossRecovery: 0.80,
  maximumRetrainingGap: 0.02,
  trainingWindowsPerSubjectActivity: 3,
  epochs: 80,
  sourceLearningRate: 0.01,
  fineTuneLearningRate: 0.005,
  ridge: 0.1
});

export function subjectRoles(test) {
  const offset = SUBJECTS.indexOf(test);
  assert.ok(offset !== -1, 'unknown test subject');
  const ordered = SUBJECTS.map((_, index) => SUBJECTS[(offset + index) % SUBJECTS.length]);
  return { test: ordered[0], trigger: ordered[1], fit: ordered.slice(2, 4), gate: ordered.slice(4, 6), source: ordered.slice(6) };
}

function panel(windows, subjects, parity) {
  const rows = windows.filter(row => subjects.includes(row.subject) && (parity === undefined || row.ordinal % 2 === parity));
  for (const subject of subjects) {
    for (const activity of CLASSES)
      assert.ok(rows.some(row => row.subject === subject && row.activity === activity), 'subject panel is missing a class');
  }
  return rows;
}

function sample(rows) {
  const groups = Map.groupBy(rows, row => row.subject + ':' + row.activity);
  return [...groups.values()].flatMap(group => {
    const count = Math.min(RULES.trainingWindowsPerSubjectActivity, group.length);
    return Array.from({ length: count }, (_, index) => group[Math.floor((index + 0.5) * group.length / count)]);
  });
}

function normalizer(rows) {
  const means = Array.from({ length: WIDTH }, (_, column) => rows.reduce((sum, row) => sum + row.chest[column], 0) / rows.length);
  const scales = means.map((mean, column) => Math.sqrt(rows.reduce((sum, row) => sum + (row.chest[column] - mean) ** 2, 0) / rows.length) || 1);
  return vector => vector.map((value, column) => (value - means[column]) / scales[column]);
}

function batch(rows, placement, normalize) {
  return {
    rows: rows.length,
    inputs: Float64Array.from(rows.flatMap(row => normalize(row[placement]))),
    targets: Float64Array.from(rows.flatMap(row => CLASSES.map(activity => Number(row.activity === activity))))
  };
}

export function evaluateScores(labels, predictions) {
  assert.equal(predictions.length, labels.length * CLASSES.length);
  const counts = Array(CLASSES.length).fill(0);
  const correct = Array(CLASSES.length).fill(0);
  let sumSq = 0;
  for (const [row, label] of labels.entries()) {
    assert.ok(CLASSES.includes(label), 'unknown activity label');
    counts[label - 1]++;
    let winner = 0;
    for (const [column, activity] of CLASSES.entries()) {
      const offset = row * CLASSES.length;
      const score = predictions[offset + column];
      assert.ok(Number.isFinite(score), 'nonfinite score');
      if (score > predictions[offset + winner]) winner = column;
      sumSq += (score - Number(activity === label)) ** 2;
    }
    if (winner + 1 === label) correct[winner]++;
  }
  assert.ok(counts.every(Boolean), 'metric requires every class');
  const recall = correct.map((count, index) => count / counts[index]);
  return { balancedAccuracy: recall.reduce((sum, value) => sum + value, 0) / recall.length,
    accuracy: correct.reduce((sum, value) => sum + value, 0) / labels.length,
    mse: sumSq / predictions.length, classCounts: counts, classCorrect: correct };
}

function scorer(runtime, rows, placement, normalize) {
  const labels = rows.map(row => row.activity);
  const { inputs } = batch(rows, placement, normalize);
  return model => {
    const prepared = runtime.prepare(model, { maxRows: rows.length });
    try {
      return evaluateScores(labels, prepared.predict(inputs, rows.length));
    } finally {
      prepared.dispose();
    }
  };
}

function initialModel(seed) {
  const random = rng(seed);
  function layer(inputs, outputs, activation) {
    const scale = Math.sqrt(6 / (inputs + outputs));
    return { inputs, outputs, activation, steepness: 1, bias: Array(outputs).fill(0),
      weights: Array.from({ length: inputs * outputs }, () => (2 * random() - 1) * scale) };
  }
  return { schema: 'ruv-fann.dense-layers/1', layers: [layer(WIDTH, HIDDEN, 'Tanh'), layer(HIDDEN, CLASSES.length, 'Linear')] };
}

function train(runtime, model, data, learningRate) {
  const started = performance.now();
  const result = runtime.train(model, { ...data, epochs: RULES.epochs, optimizer: 'adamw', learningRate });
  return { model: result.model, ms: performance.now() - started };
}

function rowsOf(values, width) {
  assert.equal(values.length % width, 0);
  return Array.from({ length: values.length / width }, (_, index) =>
    Array.from(values.slice(index * width, (index + 1) * width)));
}

export function passesTrigger(reference, observed) {
  return reference.balancedAccuracy - observed.balancedAccuracy >= RULES.triggerDrop - 1e-12;
}

export function passesGate(previous, candidate) {
  return candidate.balancedAccuracy - previous.balancedAccuracy >= RULES.gateGain - 1e-12 &&
    candidate.mse <= previous.mse + 1e-12;
}

async function evaluatePlacement(runtime, source, modelSeed, roles, calibration, normalize, placement) {
  const { triggerReference, triggerObserved, fit, gate, test } = calibration;
  const sourceHash = await fingerprintModel(source);
  const reference = scorer(runtime, triggerReference, 'chest', normalize)(source);
  const observed = scorer(runtime, triggerObserved, placement, normalize)(source);
  const triggered = passesTrigger(reference, observed);
  const fitBatch = batch(fit, placement, normalize);
  const gateData = batch(gate, placement, normalize);
  const scoreGate = scorer(runtime, gate, placement, normalize);
  const gateBefore = scoreGate(source);
  const started = performance.now();
  // Include an unconditional control even when the simulated production trigger abstains.
  const trace = await traceWeightModel(source, { inputs: Array.from(fitBatch.inputs), rows: fitBatch.rows }, { simd: false });
  const edits = refitReadout(source, rowsOf(trace.layers[0].activations, HIDDEN),
    rowsOf(fitBatch.targets, CLASSES.length), RULES.ridge);
  const patch = { schema: 'ruv-fann.weight-patch/1', baseHash: sourceHash, edits };
  const refit = await applyWeightPatch(source, patch);
  const refitMs = performance.now() - started;
  const gateAfter = scoreGate(refit);
  const gatePassed = passesGate(gateBefore, gateAfter);
  const accepted = triggered && gatePassed;
  let active = source, activeHash = sourceHash, rollBackExact = null;
  if (accepted) {
    const workspace = await createWeightWorkspace(source, { simd: false });
    const data = { rows: gateData.rows, inputs: Array.from(gateData.inputs), targets: Array.from(gateData.targets) };
    const proposal = await workspace.propose(patch, data, { maxMSE: gateAfter.mse + 1e-12, maxRegressionMSE: 1e-12 });
    const initial = workspace.snapshot();
    assert.equal(initial.hash, sourceHash);
    const promoted = workspace.promote(proposal.id, initial.revision);
    assert.throws(() => workspace.promote(proposal.id, initial.revision), /stale/);
    assert.equal(promoted.hash, await fingerprintModel(refit));
    active = promoted.model;
    activeHash = promoted.hash;
    rollBackExact = workspace.rollback(initial.revision, promoted.revision).hash === sourceHash;
    assert.ok(rollBackExact, 'rollback fingerprint mismatch');
  }

  // Both ordinary training baselines see exactly the fit panel used by the refit.
  const fineTune = train(runtime, source, fitBatch, RULES.fineTuneLearningRate);
  const retrain = train(runtime, initialModel(modelSeed), fitBatch, RULES.sourceLearningRate);
  const scoreTest = scorer(runtime, test, placement, normalize);
  const scoreAnchor = scorer(runtime, test, 'chest', normalize);
  assert.equal(await fingerprintModel(source), sourceHash, 'benchmark mutated source checkpoint');
  const restored = scoreTest(source);
  return {
    subject: roles.test, placement, sourceHash, activeHash, triggered, gatePassed, accepted, rollBackExact, refitEdits: edits.length,
    panels: { triggerReference: triggerReference.length, triggerObserved: triggerObserved.length,
      fit: fit.length, gate: gate.length, test: test.length },
    trigger: { reference, observed }, gate: { before: gateBefore, after: gateAfter },
    test: { cleanReference: scoreAnchor(source), activeOnSourcePlacement: scoreAnchor(active),
      unchanged: restored, restored, unconditionalRefit: scoreTest(refit), gatedRefit: scoreTest(active),
      fineTune: scoreTest(fineTune.model), retrain: scoreTest(retrain.model) },
    timing: { refitMs, fineTuneMs: fineTune.ms, retrainMs: retrain.ms }
  };
}

function validateInput(document) {
  assert.equal(document.schema, 'ruv-fann.mhealth-features/1');
  assert.equal(document.archiveSha256, ARCHIVE_SHA);
  assert.ok(Array.isArray(document.windows) && document.windows.length > 0 && document.windows.length <= 5_000);
  const lastSample = new Map();
  for (const row of document.windows) {
    assert.ok(SUBJECTS.includes(row.subject) && CLASSES.includes(row.activity));
    assert.ok(Number.isSafeInteger(row.start) && row.start >= 0 && Number.isSafeInteger(row.ordinal) && row.ordinal >= 0);
    assert.ok(row.start >= (lastSample.get(row.subject) ?? 0), 'overlapping or unordered windows');
    lastSample.set(row.subject, row.start + 100);
    for (const placement of PLACEMENTS) {
      assert.ok(Array.isArray(row[placement]) && row[placement].length === WIDTH && row[placement].every(Number.isFinite));
    }
  }
  assert.equal(lastSample.size, SUBJECTS.length);
  return document.windows;
}

function summarize(cases) {
  const n = cases.length;
  const test = Object.fromEntries(Object.keys(cases[0].test).map(method => [method,
    cases.reduce((sum, result) => sum + result.test[method].balancedAccuracy, 0) / n]));
  const loss = test.cleanReference - test.restored;
  const recovered = test.gatedRefit - test.restored;
  const gainRatio = loss > 0 ? recovered / loss : null;
  const strongestTraining = Math.max(test.fineTune, test.retrain);
  const individual = cases.map(({ test: scores }) => {
    const accuracy = method => scores[method].balancedAccuracy;
    const individualLoss = accuracy('cleanReference') - accuracy('restored');
    const gain = accuracy('gatedRefit') - accuracy('restored');
    return {
      recovered: individualLoss > 0 && gain / individualLoss >= RULES.lossRecovery,
      nearTraining: Math.max(accuracy('fineTune'), accuracy('retrain')) - accuracy('gatedRefit') <= RULES.maximumRetrainingGap
    };
  });
  const medians = Object.fromEntries(Object.keys(cases[0].timing).map(method => {
    const sorted = cases.map(result => result.timing[method]).toSorted((a, b) => a - b);
    return [method, (sorted[Math.floor((n - 1) / 2)] + sorted[Math.floor(n / 2)]) / 2];
  }));
  return { subjects: n, triggered: cases.filter(result => result.triggered).length,
    accepted: cases.filter(result => result.accepted).length, test, loss, recovered, gainRatio,
    gapToTraining: strongestTraining - test.gatedRefit, medianMs: medians,
    subjectsAtRecoveryTarget: individual.filter(result => result.recovered).length,
    subjectsAtBothTargets: individual.filter(result => result.recovered && result.nearTraining).length };
}

export async function runStudy(runtime, windows) {
  const cases = [];
  const participants = SUBJECTS.map(subjectRoles);
  for (const roles of participants) {
    const seed = 10_073 + roles.test * 1_009;
    const sourceRows = sample(panel(windows, roles.source));
    const normalize = normalizer(sourceRows);
    const source = train(runtime, initialModel(seed), batch(sourceRows, 'chest', normalize), RULES.sourceLearningRate).model;
    const calibration = {
      triggerReference: panel(windows, [roles.trigger], 0),
      triggerObserved: panel(windows, [roles.trigger], 1),
      fit: sample(panel(windows, roles.fit)),
      gate: panel(windows, roles.gate, 1),
      test: panel(windows, [roles.test])
    };
    for (const placement of PLACEMENTS) cases.push(await evaluatePlacement(runtime, source, seed, roles, calibration, normalize, placement));
  }
  const results = Object.fromEntries(PLACEMENTS.map(placement => [placement, summarize(cases.filter(row => row.placement === placement))]));
  const shifts = [results.wrist, results.ankle];
  const gates = {
    lossRecovery: shifts.every(result => result.gainRatio !== null && result.gainRatio >= RULES.lossRecovery),
    nearRetraining: shifts.every(result => result.gapToTraining <= RULES.maximumRetrainingGap),
    noCleanEdits: results.chest.accepted === 0,
    exactRollback: cases.some(row => row.accepted) && cases.every(row => !row.accepted || row.rollBackExact)
  };
  const resultDigest = digestJSON(cases.map(({ timing, ...row }) => row));
  return { acceptance: Object.values(gates).every(Boolean) ? 'pass' : 'hold', rules: RULES,
    gates, participants, results, cases, resultDigest };
}

async function main(args) {
  const [features, output, ...extra] = args;
  assert.ok(features && output && extra.length === 0 && isAbsolute(features) && isAbsolute(output), 'usage: provide an absolute feature JSON path and an absolute new report path');
  const input = await stat(features);
  assert.ok(input.isFile() && input.size <= 6_000_000, 'invalid feature file');
  const raw = await readFile(features);
  assert.ok(raw.length <= 6_000_000, 'feature file exceeds byte limit');
  const windows = validateInput(JSON.parse(raw));
  const core = JSON.parse(await readFile(new URL('../package.json', RUNTIME), 'utf8'));
  const wasm = JSON.parse(await readFile(new URL('package.json', WASM_ROOT), 'utf8'));
  assert.equal(core.name, 'ruv-fann');
  assert.equal(wasm.name, 'ruv-fann-wasm');
  assert.equal(core.version, '0.1.2');
  assert.equal(wasm.version, '0.1.2');
  const { loadRuntime } = await import(RUNTIME);
  const runtime = await loadRuntime({ simd: false });
  assert.equal(runtime.backend, 'wasm');
  const started = performance.now();
  const study = await runStudy(runtime, windows);
  const report = { schema: 'ruv-fann.sensor-placement-repair/1', archiveSha256: ARCHIVE_SHA, featureSha256: sha256(raw),
    runnerSha256: sha256(await readFile(new URL(import.meta.url))),
    extractorSha256: sha256(await readFile(new URL('wsp-mhealth-features.py', import.meta.url))),
    scalarWasmSha256: sha256(await readFile(new URL('dist/ruv_swarm_wasm_bg.wasm', WASM_ROOT))),
    runtime: { backend: runtime.backend, node: process.version, cpu: cpus()[0]?.model, core: core.version, wasm: wasm.version },
    wallMs: performance.now() - started, ...study };
  await writeFile(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ acceptance: study.acceptance, gates: study.gates, resultDigest: study.resultDigest, results: study.results }, null, 2));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main(process.argv.slice(2));
