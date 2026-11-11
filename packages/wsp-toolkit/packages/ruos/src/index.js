import { createWeightWorkspace, traceWeightModel, createNeuronAblationPatch,
  analyzeWeightSensitivity } from 'ruv-fann-wsp';

/** Shell plan only: the caller chooses an observed desktop and invokes its supported executor. */
export function createValidationCommand(directory) {
  if (typeof directory !== 'string' || !directory.startsWith('/') || directory.length > 1024 || /[\x00-\x1f\x7f]/.test(directory))
    throw new TypeError('directory must be a bounded absolute path');
  const quoted = `'${directory.replaceAll("'", "'\\''")}'`;
  return `cd -- ${quoted} && npm ci --ignore-scripts --no-audit --no-fund && npm run validate`;
}

/** Check the host executor's structured completion. Never parse completion from stdout. */
export function assertRuOSCompletion(toolResult, expectedMachineId) {
  if (typeof expectedMachineId !== 'string' || !/^[a-zA-Z0-9_-]{1,128}$/.test(expectedMachineId))
    throw new TypeError('expectedMachineId must identify the selected executor');
  const result = toolResult?.structuredContent;
  if (toolResult?.isError || !result || result.completionVersion !== 1 || result.completionVerified !== true ||
    result.status !== 'ok' || !Number.isInteger(result.exitCode) || result.exitCode !== 0 ||
    result.machineId !== expectedMachineId || result.truncated === true || result.outputTruncated === true ||
    result.transportError || result.signal ||
    ['truncated', 'outputTruncated', 'timedOut', 'timeout'].some(key =>
      result[key] !== undefined && result[key] !== false))
    throw new Error('unverified, failed, mismatched or incomplete ruOS executor completion');
  return Object.freeze({ completionVersion: 1, completionVerified: true, exitCode: 0, machineId: expectedMachineId,
    // Exit success alone does not independently attest every application assertion.
    assertionsIndependentlyVerified: false });
}

/** Exhaustive binary XOR lifecycle canary; throws on any failed assertion. */
export async function runWasmCanary() {
  const assert = (condition, name) => { if (!condition) throw new Error(`WSP canary failed: ${name}`); };
  const model = { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 2, outputs: 2, weights: [1, -1, -1, 1], bias: [0, 0], activation: 'ReLU', steepness: 1 },
    { inputs: 2, outputs: 1, weights: [0, 0], bias: [0], activation: 'Linear', steepness: 1 }
  ] };
  const dataset = { inputs: [0, 0, 0, 1, 1, 0, 1, 1], targets: [0, 1, 1, 0], rows: 4 };
  const policy = { maxMSE: 0, maxRegressionMSE: 0, maxAbsoluteError: 0 };
  const workspace = await createWeightWorkspace(model, { simd: false });
  const baseline = workspace.snapshot();
  const proposal = await workspace.propose({ schema: 'ruv-fann.weight-patch/1', baseHash: baseline.hash,
    edits: [0, 1].map(index => ({ layer: 1, parameter: 'weights', index, value: 1 })) }, dataset, policy);
  assert(proposal.evaluation.backend === 'wasm' && proposal.evaluation.accepted &&
    proposal.evaluation.baseline.mse === 0.5 && proposal.evaluation.candidate.mse === 0, 'real WASM repair');
  const promoted = workspace.promote(proposal.id, baseline.revision);
  const trace = await traceWeightModel(promoted.model, { inputs: dataset.inputs, rows: dataset.rows }, { simd: false });
  assert(trace.layers[1].activations.every((v, i) => v === dataset.targets[i]), 'exhaustive XOR oracle');
  const ablation = await createNeuronAblationPatch(promoted.model, 0, 0);
  const rejected = await workspace.propose(ablation, dataset, policy);
  assert(!rejected.evaluation.accepted && rejected.evaluation.candidate.mse === 0.25, 'regression rejected');
  let denied = false;
  try { workspace.promote(rejected.id, promoted.revision); } catch { denied = true; }
  assert(denied && workspace.snapshot().hash === promoted.hash, 'rejection preserves state');
  const sensitivity = await analyzeWeightSensitivity(promoted.model, dataset,
    [{ layer: 1, parameter: 'weights', index: 0 }], { epsilon: 0.001, simd: false });
  assert(Math.abs(sensitivity.entries[0].derivative) < 1e-12, 'finite difference oracle');
  const restored = workspace.rollback(1, promoted.revision);
  assert(restored.revision === 3 && restored.hash === baseline.hash, 'exact rollback');
  return { schema: 'ruv-fann.wasm-canary/1', backend: proposal.evaluation.backend,
    baselineHash: baseline.hash, repairedHash: promoted.hash, restoredHash: restored.hash,
    baselineMSE: 0.5, repairedMSE: 0, rejectedMSE: 0.25, finalRevision: restored.revision, assertions: 6 };
}
