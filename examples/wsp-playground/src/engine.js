import { loadRuntime } from 'ruv-fann';
import { createWeightWorkspace, fingerprintModel, applyWeightPatch, traceWeightModel } from 'ruv-fann-wsp';
import { runRegressionGate } from 'ruv-fann-wsp-lifecycle';
import { checkModelInvariants } from 'ruv-fann-wsp-invariants';
import { searchWeightPatch } from 'ruv-fann-wsp-search';

export const POLICY = Object.freeze({ maxMSE: 0.000001, maxRegressionMSE: 0, maxAbsoluteError: 0.001 });
export const SCENARIOS = {
  robot: {
    title: 'Repair a robot brain', inputs: ['Left obstacle', 'Right obstacle'],
    description: 'Two sensors. One steering decision. Teach a small network to turn away from obstacles.',
    instruction: 'Positive steering turns right. Negative steering turns left. Equal sensor readings mean straight ahead.',
    good: [1, -1], shortcut: [1, 0], initial: [0, 0],
    development: { inputs: [1, 0], targets: [1], rows: 1 },
    holdout: { inputs: [0, 1, 0, 0, 1, 1, 0.2, 0.8, 0.8, 0.2], targets: [-1, 0, 0, -0.6, 0.6], rows: 5 },
    expected: (a, b) => a - b, bounds: [-1, 1],
  },
  room: {
    title: 'Tune an ambient room', inputs: ['Motion evidence', 'Background noise'],
    description: 'Keep a presence signal when motion exceeds background noise. Inspect a simulated RuView decision.',
    instruction: 'The target is max(0, motion minus noise). This is a synthetic signal exercise, not measured WiFi accuracy.',
    good: [1, 0], shortcut: [1, 1], initial: [0.2, 0],
    development: { inputs: [1, 0], targets: [1], rows: 1 },
    holdout: { inputs: [0, 1, 0, 0, 1, 1, 0.2, 0.8, 0.8, 0.2], targets: [0, 0, 0, 0, 0.6], rows: 5 },
    expected: (a, b) => Math.max(0, a - b), bounds: [0, 1],
  },
};

export function makeModel(weights) {
  return { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 2, outputs: 2, weights: [1, -1, -1, 1], bias: [0, 0], activation: 'ReLU', steepness: 1 },
    { inputs: 2, outputs: 1, weights: [...weights], bias: [0], activation: 'Linear', steepness: 1 },
  ] };
}

export async function makePatch(model, weights) {
  if (!Array.isArray(weights) || weights.length !== 2 || !weights.every(v => Number.isFinite(v) && Math.abs(v) <= 2)) {
    throw new RangeError('Choose two finite output weights between -2 and 2');
  }
  return { schema: 'ruv-fann.weight-patch/1', baseHash: await fingerprintModel(model),
    edits: weights.map((value, index) => ({ layer: 1, parameter: 'weights', index, value })) };
}

export function grid(scenario) {
  const inputs = [], expected = [];
  for (let a = 0; a <= 10; a++) for (let b = 0; b <= 10; b++) {
    inputs.push(a / 10, b / 10); expected.push(scenario.expected(a / 10, b / 10));
  }
  return { inputs, expected };
}

export async function createLab(name, simd = 'auto', initialWeights) {
  const scenario = SCENARIOS[name];
  if (!scenario) throw new TypeError('Unknown scenario');
  const runtime = await loadRuntime({ simd });
  const initial = initialWeights ?? scenario.initial;
  await makePatch(makeModel(scenario.initial), initial);
  const workspace = await createWeightWorkspace(makeModel(initial), { simd });
  let validated = null;
  let generation = 0;
  const log = [];
  const record = (action, evidence) => log.push({ action, at: new Date().toISOString(), ...evidence });
  return {
    scenario, runtime, workspace, log,
    snapshot: () => workspace.snapshot(),
    invalidate() { generation++; validated = null; },
    predict(model, inputs) {
      const prepared = runtime.prepare(model, { maxRows: Math.max(1, inputs.length / 2) });
      try { return [...prepared.predict(Float64Array.from(inputs))]; } finally { prepared.dispose(); }
    },
    trace(model, inputs) { return traceWeightModel(model, { inputs, rows: 1 }, { simd }); },
    async candidate(weights) { return applyWeightPatch(workspace.snapshot().model, await makePatch(workspace.snapshot().model, weights)); },
    async evaluate(weights) {
      const ticket = ++generation;
      validated = null;
      const baseline = workspace.snapshot();
      const patch = await makePatch(baseline.model, weights);
      const candidate = await applyWeightPatch(baseline.model, patch);
      const regression = await runRegressionGate({ model: baseline.model, patch, suites: [
        { name: 'development', dataset: scenario.development, policy: POLICY },
        { name: 'independent-cases', dataset: scenario.holdout, policy: POLICY },
      ] }, { simd });
      const domain = grid(scenario);
      const invariant = await checkModelInvariants(candidate, { schema: 'ruv-fann.invariants/1', cases: [{
        name: 'finite-sensor-grid', inputs: domain.inputs, rows: 121,
        expected: domain.expected, tolerance: 0.001,
        minimum: [scenario.bounds[0]], maximum: [scenario.bounds[1]],
      }] }, { simd });
      const proposal = await workspace.propose(patch, scenario.development, POLICY);
      const accepted = regression.accepted && invariant.accepted && proposal.evaluation.accepted &&
        regression.candidateHash === invariant.modelHash && regression.candidateHash === proposal.evaluation.candidateHash;
      if (ticket !== generation || workspace.snapshot().revision !== baseline.revision) throw new Error('Candidate changed while evaluating; evaluate again');
      const evidence = { scenario: name, backend: runtime.backend, baseRevision: baseline.revision,
        baseHash: baseline.hash, candidateHash: regression.candidateHash, weights: [...weights], accepted,
        regression, invariant, policy: POLICY };
      validated = accepted ? { proposal, baseline, evidence } : null;
      record('evaluate', evidence);
      return evidence;
    },
    promote() {
      if (!validated) throw new Error('Evaluate and pass all gates before promotion');
      const next = workspace.promote(validated.proposal.id, validated.baseline.revision);
      record('promote', { revision: next.revision, hash: next.hash });
      validated = null; generation++;
      return next;
    },
    rollback() {
      const next = workspace.rollback(1, workspace.snapshot().revision);
      record('rollback', { revision: next.revision, hash: next.hash });
      validated = null; generation++;
      return next;
    },
    async search() {
      const snapshot = workspace.snapshot();
      const result = await searchWeightPatch(snapshot.model, scenario.development, POLICY, {
        locations: [0, 1].map(index => ({ layer: 1, parameter: 'weights', index, values: [-1, 0, 0.5, 1] })),
        maxEvaluations: 24, maxPasses: 2, simd,
      });
      record('search', { evaluations: result.evaluations, bestScore: result.bestScore, developmentOnly: true });
      if (!result.patch) return { weights: [...snapshot.model.layers[1].weights], result };
      const candidate = await applyWeightPatch(snapshot.model, result.patch);
      return { weights: candidate.layers[1].weights, result };
    },
  };
}

export async function benchmarkRuntime(simd) {
  const runtime = await loadRuntime({ simd });
  const model = makeModel([1, -1]);
  const inputs = Float64Array.from({ length: 512 }, (_, i) => (i * 17 % 101) / 100);
  const prepared = runtime.prepare(model, { maxRows: 256 });
  const samples = []; let output;
  try {
    for (let n = 0; n < 20; n++) prepared.predict(inputs);
    for (let n = 0; n < 30; n++) {
      const start = performance.now();
      for (let j = 0; j < 100; j++) output = prepared.predict(inputs);
      samples.push((performance.now() - start) / 100);
    }
    let maxError = 0;
    output.forEach((v, i) => { maxError = Math.max(maxError, Math.abs(v - (inputs[i * 2] - inputs[i * 2 + 1]))); });
    samples.sort((a, b) => a - b);
    return { backend: runtime.backend, rows: 256, iterations: 3000, medianBatchMs: samples[15],
      p95BatchMs: samples[28], predictionsPerSecond: 256000 / samples[15], maxError,
      note: 'Local browser observation; includes JS/WASM call and output allocation. No network latency.' };
  } finally { prepared.dispose(); }
}
