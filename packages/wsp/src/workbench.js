import { loadRuntime } from 'ruv-fann';
import { WSP_LIMITS, snapshotModel, snapshotDataset, snapshotLimits, record, integer, finite,
  arrayLength, element, patchCandidate, cloneModel } from './validation.js';
import { hashJSON } from './crypto.js';

function options(value = {}, sensitivity = false) {
  const data = record(value, sensitivity ? ['simd', 'epsilon'] : ['simd'], [], 'workbench options');
  const simd = data.simd === undefined ? 'auto' : data.simd;
  if (simd !== true && simd !== false && simd !== 'auto') throw new TypeError('simd must be boolean or auto');
  return { simd, epsilon: sensitivity ? finite(data.epsilon === undefined ? 1e-4 : data.epsilon,
    'epsilon', 1e-8, 1) : undefined };
}
function work(amount) {
  integer(amount, 1, WSP_LIMITS.maxWork, 'aggregate work');
}
function batch(value, source) {
  const data = record(value, ['inputs', 'rows'], ['inputs', 'rows'], 'batch');
  const rows = integer(data.rows, 1, WSP_LIMITS.maxRows, 'rows');
  const count = integer(rows * source.inputSize, 1, WSP_LIMITS.maxElements, 'batch input elements');
  let elements = count;
  for (const layer of source.model.layers) elements += 2 * rows * layer.outputs;
  integer(elements, 1, WSP_LIMITS.maxElements, 'aggregate trace elements');
  work(2 * rows * source.parameters);
  if (arrayLength(data.inputs, WSP_LIMITS.maxElements, 'batch inputs') !== count)
    throw new RangeError('batch inputs length does not match dimensions');
  const inputs = new Array(count);
  for (let index = 0; index < count; index++) inputs[index] = finite(element(data.inputs, index, 'batch inputs'), 'batch input');
  return { rows, inputs, canonical: { schema: 'ruv-fann.weight-input/1', inputSize: source.inputSize, rows, inputs } };
}
function infer(runtime, model, inputs, rows) {
  const inputSize = model.layers[0].inputs, outputSize = model.layers.at(-1).outputs;
  const maxRows = Math.min(rows, 32), prepared = runtime.prepare(model, { maxRows });
  try {
    const result = new Array(rows * outputSize);
    for (let first = 0; first < rows; first += maxRows) {
      const count = Math.min(rows - first, maxRows);
      const values = prepared.predict(Float64Array.from(inputs.slice(first * inputSize, (first + count) * inputSize)), count);
      for (let index = 0; index < values.length; index++) result[first * outputSize + index] = finite(values[index], 'prediction');
    }
    return result;
  } finally { prepared.dispose(); }
}
function measure(runtime, model, data) {
  const values = infer(runtime, model, data.inputs, data.rows);
  let mse = 0, maxAbsoluteError = 0;
  for (let index = 0; index < values.length; index++) {
    const error = finite(values[index] - data.targets[index], 'prediction error');
    const square = finite(error * error, 'squared error');
    mse = finite(mse + (square - mse) / (index + 1), 'MSE');
    maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(error));
  }
  return { mse, maxAbsoluteError };
}

export async function inspectWeightModel(model) {
  const source = snapshotModel(model);
  const layers = source.model.layers.map((layer, index) => ({ layer: index, inputs: layer.inputs,
    outputs: layer.outputs, activation: layer.activation, steepness: layer.steepness,
    weightCount: layer.weights.length, biasCount: layer.bias.length,
    parameterCount: layer.weights.length + layer.bias.length }));
  return { schema: 'ruv-fann.weight-model-inspection/1', baseHash: await hashJSON(source.json),
    parameterCount: source.parameters, inputSize: source.inputSize, outputSize: source.outputSize, layers };
}

export async function traceWeightModel(model, inputBatch, configuration) {
  const source = snapshotModel(model), data = batch(inputBatch, source), settings = options(configuration);
  const [baseHash, inputHash] = await Promise.all([hashJSON(source.json), hashJSON(JSON.stringify(data.canonical))]);
  const runtime = await loadRuntime({ simd: settings.simd });
  const layers = [];
  let inputs = data.inputs;
  for (let index = 0; index < source.model.layers.length; index++) {
    const layer = source.model.layers[index];
    const preactivations = infer(runtime, { schema: source.model.schema,
      layers: [{ ...layer, activation: 'Linear', steepness: 1 }] }, inputs, data.rows);
    const activations = infer(runtime, { schema: source.model.schema, layers: [layer] }, inputs, data.rows);
    layers.push({ layer: index, inputs: layer.inputs, outputs: layer.outputs, preactivations, activations });
    inputs = activations;
  }
  return { schema: 'ruv-fann.weight-trace/1', baseHash, inputHash, backend: runtime.backend, rows: data.rows, layers };
}

export async function analyzeWeightSensitivity(model, dataset, locations, configuration) {
  const source = snapshotModel(model), data = snapshotDataset(dataset, source), settings = options(configuration, true);
  const count = integer(arrayLength(locations, 16, 'locations'), 1, 16, 'locations');
  work((2 * count + 1) * data.rows * source.parameters);
  const probes = [], seen = new Set();
  for (let offset = 0; offset < count; offset++) {
    const location = record(element(locations, offset, 'locations'), ['layer', 'parameter', 'index'],
      ['layer', 'parameter', 'index'], 'location');
    const layer = integer(location.layer, 0, source.model.layers.length - 1, 'location layer');
    const parameter = location.parameter;
    if (parameter !== 'weights' && parameter !== 'bias') throw new TypeError('location parameter must be weights or bias');
    const values = source.model.layers[layer][parameter];
    const index = integer(location.index, 0, values.length - 1, 'location index');
    const key = `${layer}:${parameter}:${index}`;
    if (seen.has(key)) throw new TypeError('duplicate sensitivity location');
    seen.add(key);
    const minus = finite(values[index] - settings.epsilon, 'minus probe', -WSP_LIMITS.maxAbsValue, WSP_LIMITS.maxAbsValue);
    const plus = finite(values[index] + settings.epsilon, 'plus probe', -WSP_LIMITS.maxAbsValue, WSP_LIMITS.maxAbsValue);
    if (minus === values[index] || plus === values[index] || !(plus > minus)) throw new RangeError('epsilon does not produce representable probes');
    probes.push({ layer, parameter, index, minus, plus });
  }
  const [baseHash, datasetHash] = await Promise.all([hashJSON(source.json), hashJSON(JSON.stringify(data.canonical))]);
  const runtime = await loadRuntime({ simd: settings.simd });
  const baseline = measure(runtime, source.model, data), entries = [];
  for (const probe of probes) {
    const candidate = cloneModel(source.model);
    candidate.layers[probe.layer][probe.parameter][probe.index] = probe.minus;
    const minus = measure(runtime, candidate, data);
    candidate.layers[probe.layer][probe.parameter][probe.index] = probe.plus;
    const plus = measure(runtime, candidate, data);
    const derivative = finite((plus.mse - minus.mse) / (probe.plus - probe.minus), 'sensitivity derivative');
    entries.push({ layer: probe.layer, parameter: probe.parameter, index: probe.index, minus, plus, derivative });
  }
  return { schema: 'ruv-fann.weight-sensitivity/1', method: 'central-finite-difference', baseHash, datasetHash,
    backend: runtime.backend, epsilon: settings.epsilon, baseline, entries };
}

export async function createNeuronAblationPatch(model, layerIndex, neuronIndex, limits) {
  const source = snapshotModel(model), bounds = snapshotLimits(limits);
  const layer = integer(layerIndex, 0, source.model.layers.length - 2, 'hidden layer');
  const neuron = integer(neuronIndex, 0, source.model.layers[layer].outputs - 1, 'neuron');
  const next = source.model.layers[layer + 1], edits = [];
  for (let output = 0; output < next.outputs; output++) {
    const index = output * next.inputs + neuron;
    if (next.weights[index] !== 0) edits.push({ layer: layer + 1, parameter: 'weights', index, value: 0 });
    if (edits.length > bounds.maxEdits) throw new RangeError('ablation edits exceed maxEdits');
  }
  if (!edits.length) throw new RangeError('neuron ablation has no nonzero outgoing weights');
  // Validate the complete edit and limits before hashing or any native allocation.
  const prepared = patchCandidate({ schema: 'ruv-fann.weight-patch/1', baseHash: '0'.repeat(64), edits }, source, bounds);
  prepared.patch.baseHash = await hashJSON(source.json);
  return prepared.patch;
}
