import { WSP_LIMITS } from 'ruv-fann-wsp';

export const SWARM_LIMITS = Object.freeze({ maxJobs: 64, maxConcurrency: 4, maxBytes: 10 * 1024 * 1024,
  maxTimeoutMs: 60_000, maxTotalWork: 100_000_000 });
const activationNames = new Set(['Linear', 'Threshold', 'ThresholdSymmetric', 'Sigmoid', 'SigmoidSymmetric',
  'Tanh', 'Gaussian', 'GaussianSymmetric', 'Elliot', 'ElliotSymmetric', 'LinearPiece', 'LinearPieceSymmetric',
  'ReLU', 'ReLULeaky', 'Sin', 'Cos', 'SinSymmetric', 'CosSymmetric', 'Gelu', 'Swish', 'LeakyRelu']);

export function record(value, allowed, required, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError(`${name} must be plain data`);
  const out = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string' || !allowed.includes(key)) throw new TypeError(`unknown ${name} field`);
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError(`${name} accessors are forbidden`);
    out[key] = descriptor.value;
  }
  if (required.some(key => !Object.hasOwn(out, key))) throw new TypeError(`missing ${name} field`);
  return out;
}
export function integer(value, minimum, maximum, name) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new RangeError(`${name} must be in ${minimum}..${maximum}`);
  return value;
}
function finite(value, name, minimum = -Number.MAX_VALUE, maximum = Number.MAX_VALUE) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) throw new RangeError(`${name} must be finite and in bounds`);
  return value === 0 ? 0 : value;
}
function shape(value, allowed, required, name) {
  return record(value, allowed, required, name);
}
function numbers(value, expected, name) {
  if (!Array.isArray(value) || value.length !== expected || value.length > WSP_LIMITS.maxElements)
    throw new RangeError(`${name} dimensions disagree`);
  for (const item of value) finite(item, name);
  return value;
}

/** Bounded deep snapshot; no caller getters, iterators, toJSON or functions are invoked. */
export function snapshotJobs(value) {
  let bytes = 0, nodes = 0;
  const ancestors = new Set();
  const budget = count => { bytes += count; if (bytes > SWARM_LIMITS.maxBytes) throw new RangeError('swarm input exceeds aggregate byte limit'); };
  function string(value) {
    if (value.length > WSP_LIMITS.maxModelBytes) throw new RangeError('swarm input string exceeds byte limit');
    // A string is capped before escaping; this allocation is at most 6 MiB.
    budget(Buffer.byteLength(JSON.stringify(value)));
    return value;
  }
  function copy(item, depth) {
    if (++nodes > 700_000 || depth > 16) throw new RangeError('swarm input exceeds structural limits');
    if (typeof item === 'number') { const result = finite(item, 'input number'); budget(String(result).length); return result; }
    if (typeof item === 'string') return string(item);
    if (item === null || typeof item === 'boolean') { budget(item === null ? 4 : item ? 4 : 5); return item; }
    if (!item || typeof item !== 'object' || ancestors.has(item)) throw new TypeError('swarm jobs must be acyclic JSON data');
    const array = Array.isArray(item);
    if (!(array ? [Array.prototype, null].includes(Object.getPrototypeOf(item)) :
      [Object.prototype, null].includes(Object.getPrototypeOf(item)))) throw new TypeError('swarm jobs must contain plain data');
    ancestors.add(item);
    const keys = Reflect.ownKeys(item);
    let out;
    if (array) {
      const length = integer(Object.getOwnPropertyDescriptor(item, 'length')?.value, 0, WSP_LIMITS.maxElements, 'array length');
      if (keys.length !== length + 1) throw new TypeError('swarm arrays must have dense own elements only');
      out = new Array(length);
      budget(length ? length + 1 : 2);
      for (let index = 0; index < length; index++) {
        const descriptor = Object.getOwnPropertyDescriptor(item, String(index));
        if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError('swarm arrays cannot contain holes or accessors');
        out[index] = copy(descriptor.value, depth + 1);
      }
    } else {
      if (keys.length > 64) throw new RangeError('swarm object has too many fields');
      out = Object.create(null);
      budget(keys.length ? keys.length + 1 : 2);
      for (const key of keys) {
        if (typeof key !== 'string' || key.length > 128 || ['__proto__', 'constructor', 'prototype'].includes(key))
          throw new TypeError('swarm object contains an unsafe field');
        string(key); budget(1);
        const descriptor = Object.getOwnPropertyDescriptor(item, key);
        if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError('swarm accessors are forbidden');
        out[key] = copy(descriptor.value, depth + 1);
      }
    }
    ancestors.delete(item);
    return out;
  }
  if (!Array.isArray(value)) throw new TypeError('jobs must be an array');
  integer(Object.getOwnPropertyDescriptor(value, 'length')?.value, 1, SWARM_LIMITS.maxJobs, 'job count');
  return copy(value, 0);
}

function modelInfo(input) {
  let value = input;
  if (typeof value === 'string') {
    if (Buffer.byteLength(value) > WSP_LIMITS.maxModelBytes) throw new RangeError('model exceeds byte limit');
    value = JSON.parse(value);
  }
  const source = shape(value, ['schema', 'layers'], ['schema', 'layers'], 'model');
  if (source.schema !== 'ruv-fann.dense-layers/1' || !Array.isArray(source.layers)) throw new TypeError('invalid dense model schema');
  integer(source.layers.length, 1, WSP_LIMITS.maxLayers, 'layer count');
  let parameters = 0, previous;
  const layers = source.layers.map(value => {
    const layer = shape(value, ['inputs', 'outputs', 'weights', 'bias', 'activation', 'steepness'],
      ['inputs', 'outputs', 'weights', 'activation', 'steepness'], 'layer');
    const inputs = integer(layer.inputs, 1, WSP_LIMITS.maxDimension, 'inputs');
    const outputs = integer(layer.outputs, 1, WSP_LIMITS.maxDimension, 'outputs');
    if (previous !== undefined && previous !== inputs) throw new RangeError('adjacent model dimensions disagree');
    previous = outputs;
    parameters += inputs * outputs + outputs;
    integer(parameters, 1, WSP_LIMITS.maxParameters, 'parameter count');
    if (!activationNames.has(layer.activation)) throw new TypeError('unsupported activation');
    const weights = numbers(layer.weights, inputs * outputs, 'weights');
    const bias = layer.bias == null ? Array(outputs).fill(0) : numbers(layer.bias, outputs, 'bias');
    return { inputs, outputs, weights, bias, activation: layer.activation, steepness: finite(layer.steepness, 'steepness') };
  });
  const model = { schema: source.schema, layers };
  if (Buffer.byteLength(JSON.stringify(model)) > WSP_LIMITS.maxModelBytes) throw new RangeError('model exceeds canonical byte limit');
  return { model, parameters };
}
function datasetInfo(value, model) {
  const source = shape(value, ['inputs', 'targets', 'rows'], ['inputs', 'targets', 'rows'], 'dataset');
  const rows = integer(source.rows, 1, WSP_LIMITS.maxRows, 'rows');
  numbers(source.inputs, rows * model.layers[0].inputs, 'dataset inputs');
  numbers(source.targets, rows * model.layers.at(-1).outputs, 'dataset targets');
  return source;
}
function policyInfo(value) {
  const source = shape(value, ['maxMSE', 'maxRegressionMSE', 'maxAbsoluteError'], ['maxMSE', 'maxRegressionMSE'], 'policy');
  finite(source.maxMSE, 'maxMSE', 0); finite(source.maxRegressionMSE, 'maxRegressionMSE', 0);
  if (Object.hasOwn(source, 'maxAbsoluteError')) finite(source.maxAbsoluteError, 'maxAbsoluteError', 0);
  return source;
}
function searchInfo(value, model) {
  const source = shape(value, ['locations', 'maxEvaluations', 'maxPasses', 'maxWork', 'simd', 'limits'], ['locations'], 'search options');
  const limits = shape(source.limits === undefined ? {} : source.limits, ['maxEdits', 'maxAbsValue', 'maxAbsDelta'], [], 'limits');
  const maxEdits = limits.maxEdits === undefined ? WSP_LIMITS.maxEdits : integer(limits.maxEdits, 1, WSP_LIMITS.maxEdits, 'maxEdits');
  const maxAbsValue = limits.maxAbsValue === undefined ? WSP_LIMITS.maxAbsValue : finite(limits.maxAbsValue, 'maxAbsValue', 0, WSP_LIMITS.maxAbsValue);
  const maxAbsDelta = limits.maxAbsDelta === undefined ? WSP_LIMITS.maxAbsDelta : finite(limits.maxAbsDelta, 'maxAbsDelta', 0, WSP_LIMITS.maxAbsDelta);
  if (source.simd !== undefined && source.simd !== false && source.simd !== true && source.simd !== 'auto') throw new TypeError('simd must be boolean or auto');
  if (!Array.isArray(source.locations)) throw new TypeError('locations must be an array');
  integer(source.locations.length, 1, Math.min(16, maxEdits), 'location count');
  const seen = new Set(); let choices = 0;
  for (const item of source.locations) {
    const location = shape(item, ['layer', 'parameter', 'index', 'values'], ['layer', 'parameter', 'index', 'values'], 'location');
    const layer = integer(location.layer, 0, model.layers.length - 1, 'location layer');
    if (location.parameter !== 'weights' && location.parameter !== 'bias') throw new TypeError('invalid location parameter');
    const index = integer(location.index, 0, model.layers[layer][location.parameter].length - 1, 'location index');
    const key = `${layer}:${location.parameter}:${index}`;
    if (seen.has(key)) throw new TypeError('duplicate search location');
    seen.add(key);
    if (!Array.isArray(location.values)) throw new TypeError('location values must be an array');
    integer(location.values.length, 1, 32, 'choice count');
    if (new Set(location.values).size !== location.values.length) throw new TypeError('duplicate search value');
    const old = model.layers[layer][location.parameter][index];
    for (const value of location.values) {
      finite(value, 'candidate value', -maxAbsValue, maxAbsValue);
      if (Math.abs(value - old) > maxAbsDelta) throw new RangeError('candidate exceeds maxAbsDelta');
    }
    choices += location.values.length;
  }
  const maxEvaluations = source.maxEvaluations === undefined ? 64 : integer(source.maxEvaluations, 1, 256, 'maxEvaluations');
  const maxPasses = source.maxPasses === undefined ? 4 : integer(source.maxPasses, 1, 16, 'maxPasses');
  const maxWork = source.maxWork === undefined ? WSP_LIMITS.maxWork : integer(source.maxWork, 1, WSP_LIMITS.maxWork, 'search maxWork');
  return { options: source, plannedPasses: Math.min(maxEvaluations, maxPasses * choices) + 3, maxWork };
}
export function patchInfo(value) {
  const patch = shape(value, ['schema', 'baseHash', 'edits'], ['schema', 'baseHash', 'edits'], 'patch');
  if (patch.schema !== 'ruv-fann.weight-patch/1' || typeof patch.baseHash !== 'string' || !/^[a-f0-9]{64}$/.test(patch.baseHash))
    throw new TypeError('invalid patch schema or baseHash');
  if (!Array.isArray(patch.edits)) throw new TypeError('patch edits must be an array');
  integer(patch.edits.length, 1, WSP_LIMITS.maxEdits, 'edit count');
  const seen = new Set();
  for (const item of patch.edits) {
    const edit = shape(item, ['layer', 'parameter', 'index', 'value'], ['layer', 'parameter', 'index', 'value'], 'edit');
    integer(edit.layer, 0, WSP_LIMITS.maxLayers - 1, 'edit layer');
    integer(edit.index, 0, WSP_LIMITS.maxParameters - 1, 'edit index');
    if (edit.parameter !== 'weights' && edit.parameter !== 'bias') throw new TypeError('invalid edit parameter');
    finite(edit.value, 'edit value', -WSP_LIMITS.maxAbsValue, WSP_LIMITS.maxAbsValue);
    const key = `${edit.layer}:${edit.parameter}:${edit.index}`;
    if (seen.has(key)) throw new TypeError('duplicate patch location');
    seen.add(key);
  }
  return patch;
}

/** Called only after snapshotJobs; returns normalized job data and conservative work reservation. */
export function prepareJobs(input) {
  const identifiers = new Set(); let reservedWork = 0;
  const jobs = input.map(value => {
    if (!value || (value.kind !== 'search' && value.kind !== 'patch')) throw new TypeError('job kind must be search or patch');
    const source = value.kind === 'search' ? shape(value,
      ['id', 'kind', 'model', 'dataset', 'policy', 'searchOptions'], ['id', 'kind', 'model', 'dataset', 'policy', 'searchOptions'], 'search job') :
      shape(value, ['id', 'kind', 'patch'], ['id', 'kind', 'patch'], 'patch job');
    if (typeof source.id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(source.id) || identifiers.has(source.id))
      throw new TypeError('job IDs must be unique bounded identifiers');
    identifiers.add(source.id);
    if (source.kind === 'patch') return { id: source.id, kind: source.kind, patch: patchInfo(source.patch) };
    const info = modelInfo(source.model), dataset = datasetInfo(source.dataset, info.model), policy = policyInfo(source.policy);
    const settings = searchInfo(source.searchOptions, info.model);
    const work = settings.plannedPasses * dataset.rows * info.parameters;
    if (work > settings.maxWork) throw new RangeError('search job exceeds aggregate work budget');
    reservedWork += work;
    if (reservedWork > SWARM_LIMITS.maxTotalWork) throw new RangeError('swarm exceeds aggregate work budget');
    return { id: source.id, kind: source.kind, model: info.model, dataset, policy, searchOptions: settings.options };
  });
  return { jobs, reservedWork };
}
