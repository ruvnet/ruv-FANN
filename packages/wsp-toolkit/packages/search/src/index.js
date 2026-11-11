import { loadRuntime } from 'ruv-fann';
import { WSP_LIMITS, inspectWeightModel, evaluateWeightPatch, applyWeightPatch, traceWeightModel } from 'ruv-fann-wsp';
import { record, integer, finite, array, numbers, modelSnapshot, dataSnapshot, limitsSnapshot, simdOption, clone, deepFreeze } from './data.js';

export const SEARCH_LIMITS = Object.freeze({ maxLocations: 16, maxValues: 32, maxEvaluations: 256, maxPasses: 16, maxWork: WSP_LIMITS.maxWork });
function policySnapshot(value) {
  const raw = record(value, ['maxMSE', 'maxRegressionMSE', 'maxAbsoluteError'], ['maxMSE', 'maxRegressionMSE'], 'policy');
  const policy = { maxMSE: finite(raw.maxMSE, 'maxMSE', 0), maxRegressionMSE: finite(raw.maxRegressionMSE, 'maxRegressionMSE', 0) };
  if (raw.maxAbsoluteError !== undefined) policy.maxAbsoluteError = finite(raw.maxAbsoluteError, 'maxAbsoluteError', 0);
  return policy;
}
function locationsSnapshot(values, model, bounds) {
  const seen = new Set();
  const locations = array(values, SEARCH_LIMITS.maxLocations, 'locations', value => {
    const raw = record(value, ['layer', 'parameter', 'index', 'values'], ['layer', 'parameter', 'index', 'values'], 'location');
    const layer = integer(raw.layer, 0, model.layers.length - 1, 'location layer');
    if (raw.parameter !== 'weights' && raw.parameter !== 'bias') throw new TypeError('location parameter must be weights or bias');
    const index = integer(raw.index, 0, model.layers[layer][raw.parameter].length - 1, 'location index');
    const key = `${layer}:${raw.parameter}:${index}`;
    if (seen.has(key)) throw new TypeError('duplicate location');
    seen.add(key);
    const old = model.layers[layer][raw.parameter][index];
    const candidates = numbers(raw.values, SEARCH_LIMITS.maxValues, 'values');
    if (!candidates.length || new Set(candidates).size !== candidates.length) throw new RangeError('values must be nonempty and unique');
    for (const next of candidates) {
      if (Math.abs(next) > bounds.maxAbsValue || Math.abs(next - old) > bounds.maxAbsDelta) throw new RangeError('candidate exceeds patch limits');
    }
    return { layer, parameter: raw.parameter, index, values: candidates };
  });
  if (!locations.length || locations.length > bounds.maxEdits) throw new RangeError('locations exceed edit limits');
  return locations;
}
function measure(runtime, model, data, objective) {
  const inputSize = model.layers[0].inputs, outputSize = model.layers.at(-1).outputs;
  const maxRows = Math.min(data.rows, 32), prepared = runtime.prepare(model, { maxRows });
  const predictions = new Array(data.rows * outputSize);
  const input = Float64Array.from(data.inputs);
  let mse = 0, maxAbsoluteError = 0, count = 0;
  try {
    for (let first = 0; first < data.rows; first += maxRows) {
      const rows = Math.min(maxRows, data.rows - first);
      const output = prepared.predict(input.subarray(first * inputSize, (first + rows) * inputSize), rows);
      for (let offset = 0; offset < output.length; offset++) {
        const index = first * outputSize + offset;
        predictions[index] = finite(output[offset], 'prediction');
        const error = finite(output[offset] - data.targets[index], 'prediction error');
        const square = finite(error * error, 'squared error');
        mse = finite(mse + (square - mse) / ++count, 'MSE');
        maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(error));
      }
    }
  } finally { prepared.dispose(); }
  const metrics = { mse, maxAbsoluteError };
  let score = mse;
  if (objective) {
    const proposedScore = objective(deepFreeze({ predictions, metrics: { ...metrics }, model: clone(model) }));
    if (proposedScore && typeof proposedScore === 'object') {
      // Async objectives are invalid, but their rejection must not crash the host
      // after it catches our validation error. The intrinsic brand-check refuses
      // ordinary thenables without reading or calling their `then` property.
      try { Promise.prototype.then.call(proposedScore, undefined, () => {}); } catch { /* not a native promise */ }
    }
    score = finite(proposedScore, 'objective score');
  }
  return { score, metrics };
}
function patchFor(base, current, locations, baseHash) {
  const edits = [];
  for (const location of locations) {
    const { layer, parameter, index } = location;
    const value = current.layers[layer][parameter][index];
    if (value !== base.layers[layer][parameter][index]) edits.push({ layer, parameter, index, value });
  }
  return edits.length ? { schema: 'ruv-fann.weight-patch/1', baseHash, edits } : null;
}

/** Deterministic coordinate descent over explicit discrete values. Lower objective scores win; ties preserve the earlier model. */
export async function searchWeightPatch(model, dataset, policy, options) {
  const source = modelSnapshot(model), data = dataSnapshot(dataset, source.model), criteria = policySnapshot(policy);
  const settings = record(options, ['locations', 'maxEvaluations', 'maxPasses', 'maxWork', 'objective', 'simd', 'limits'], ['locations'], 'search options');
  const bounds = limitsSnapshot(settings.limits), locations = locationsSnapshot(settings.locations, source.model, bounds);
  const maxEvaluations = integer(settings.maxEvaluations === undefined ? 64 : settings.maxEvaluations, 1, SEARCH_LIMITS.maxEvaluations, 'maxEvaluations');
  const maxPasses = integer(settings.maxPasses === undefined ? 4 : settings.maxPasses, 1, SEARCH_LIMITS.maxPasses, 'maxPasses');
  const maxWork = integer(settings.maxWork === undefined ? SEARCH_LIMITS.maxWork : settings.maxWork, 1, SEARCH_LIMITS.maxWork, 'maxWork');
  const simd = simdOption(settings.simd);
  const objective = settings.objective;
  if (objective !== undefined && typeof objective !== 'function') throw new TypeError('objective must be a trusted synchronous callback');
  const plannedEvaluations = Math.min(maxEvaluations, maxPasses * locations.reduce((sum, x) => sum + x.values.length, 0));
  // Baseline + candidate screens + the authoritative final baseline/candidate evaluation.
  const reservedWork = (plannedEvaluations + 3) * data.rows * source.parameters;
  if (reservedWork > maxWork) throw new RangeError('search exceeds aggregate work budget');
  const inspection = await inspectWeightModel(source.model);
  const runtime = await loadRuntime({ simd });
  let current = clone(source.model), best = measure(runtime, current, data, objective), evaluations = 0, passes = 0;
  const baselineScore = best.score;
  outer: for (; passes < maxPasses;) {
    passes++;
    let improved = false;
    for (const location of locations) {
      for (const next of location.values) {
        if (next === current.layers[location.layer][location.parameter][location.index]) continue;
        if (evaluations >= maxEvaluations) break outer;
        const candidate = clone(current);
        candidate.layers[location.layer][location.parameter][location.index] = next;
        const result = measure(runtime, candidate, data, objective);
        evaluations++;
        if (result.score < best.score) { current = candidate; best = result; improved = true; }
      }
    }
    if (!improved || (!objective && best.score === 0)) break;
  }
  const patch = patchFor(source.model, current, locations, inspection.baseHash);
  const evaluation = patch ? await evaluateWeightPatch(source.model, patch, data, criteria, { simd, limits: bounds }) : null;
  return { schema: 'ruv-fann.weight-search/1', method: 'discrete-coordinate-search', baseHash: inspection.baseHash,
    objective: objective ? 'custom' : 'mse', baselineScore, bestScore: best.score, patch, evaluation,
    evaluations, inferencePasses: 1 + evaluations + (patch ? 2 : 0), passes,
    budgetExhausted: evaluations >= maxEvaluations, reservedWork, backend: runtime.backend };
}

/** Experimental: compile a complete binary truth table into exact pattern-detector Threshold layers. */
export function compileBooleanTable(table) {
  const raw = record(table, ['inputs', 'outputs', 'targets'], ['inputs', 'outputs', 'targets'], 'boolean table');
  const inputs = integer(raw.inputs, 1, 6, 'boolean inputs'), outputs = integer(raw.outputs, 1, 16, 'boolean outputs');
  const rows = 2 ** inputs;
  const targets = numbers(raw.targets, rows * outputs, 'boolean targets');
  if (targets.length !== rows * outputs || targets.some(x => x !== 0 && x !== 1)) throw new RangeError('targets must contain the complete 0/1 truth table');
  const weights = [], bias = [], outputWeights = [];
  for (let row = 0; row < rows; row++) {
    let ones = 0;
    for (let input = 0; input < inputs; input++) {
      const bit = (row >> (inputs - input - 1)) & 1;
      weights.push(bit ? 1 : -1); ones += bit;
    }
    bias.push(0.5 - ones);
  }
  for (let output = 0; output < outputs; output++) for (let row = 0; row < rows; row++) outputWeights.push(targets[row * outputs + output]);
  return { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs, outputs: rows, weights, bias, activation: 'Threshold', steepness: 1 },
    { inputs: rows, outputs, weights: outputWeights, bias: Array(outputs).fill(-0.5), activation: 'Threshold', steepness: 1 }
  ] };
}

/** Experimental: propose Δw = rate * mean(pre * post) at explicit weight indices, clipped to edit bounds. No training or promotion. */
export async function createHebbianPatch(model, batch, options) {
  const source = modelSnapshot(model), data = dataSnapshot(batch, source.model, false);
  const raw = record(options, ['layer', 'indices', 'learningRate', 'limits', 'simd'], ['layer', 'indices', 'learningRate'], 'Hebbian options');
  const layer = integer(raw.layer, 0, source.model.layers.length - 1, 'Hebbian layer');
  const indices = array(raw.indices, WSP_LIMITS.maxEdits, 'indices', x => integer(x, 0, source.model.layers[layer].weights.length - 1, 'weight index'));
  if (!indices.length || new Set(indices).size !== indices.length) throw new RangeError('indices must be nonempty and unique');
  const rate = finite(raw.learningRate, 'learningRate', Number.MIN_VALUE, 1), bounds = limitsSnapshot(raw.limits), simd = simdOption(raw.simd);
  if (indices.length > bounds.maxEdits) throw new RangeError('indices exceed maxEdits');
  if ((2 * source.parameters + indices.length) * data.rows > WSP_LIMITS.maxWork) throw new RangeError('Hebbian proposal exceeds aggregate work budget');
  // traceWeightModel separately enforces aggregate trace storage and two-pass work limits.
  const trace = await traceWeightModel(source.model, data, { simd });
  const info = source.model.layers[layer], pre = layer === 0 ? data.inputs : trace.layers[layer - 1].activations;
  const post = trace.layers[layer].activations, edits = [];
  for (const index of indices) {
    const input = index % info.inputs, output = Math.floor(index / info.inputs);
    let correlation = 0;
    for (let row = 0; row < data.rows; row++) {
      const product = finite(pre[row * info.inputs + input] * post[row * info.outputs + output], 'Hebbian product');
      correlation = finite(correlation + (product - correlation) / (row + 1), 'Hebbian mean');
    }
    const old = info.weights[index], delta = Math.max(-bounds.maxAbsDelta, Math.min(bounds.maxAbsDelta, finite(rate * correlation, 'Hebbian delta')));
    let next = Math.max(-bounds.maxAbsValue, Math.min(bounds.maxAbsValue, old + delta));
    // Adding the clipped delta may round just outside the strict represented-delta bound.
    // Move a few ULPs toward the source rather than weakening the authoritative patch check.
    for (let attempt = 0; Math.abs(next - old) > bounds.maxAbsDelta && attempt < 4; attempt++) next = nextToward(next, old);
    if (Math.abs(next - old) > bounds.maxAbsDelta) throw new RangeError('source weight is outside admissible Hebbian patch bounds');
    if (next !== old) edits.push({ layer, parameter: 'weights', index, value: next });
  }
  if (!edits.length) throw new RangeError('Hebbian proposal has no representable edits');
  const patch = { schema: 'ruv-fann.weight-patch/1', baseHash: trace.baseHash, edits };
  await applyWeightPatch(source.model, patch, bounds);
  return patch;
}

function nextToward(value, target) {
  if (value === target) return value;
  if (value === 0) return target > 0 ? Number.MIN_VALUE : -Number.MIN_VALUE;
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value);
  const bits = view.getBigUint64(0);
  view.setBigUint64(0, bits + ((value > 0) === (target > value) ? 1n : -1n));
  return view.getFloat64(0);
}
