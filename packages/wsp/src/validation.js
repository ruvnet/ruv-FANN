/** All snapshots use own data descriptors; caller getters/iterators/toJSON never run. */
export const WSP_LIMITS = Object.freeze({ maxEdits: 128, maxAbsValue: 1e6, maxAbsDelta: 1000,
  maxRows: 1024, maxElements: 65536, maxParameters: 65536, maxModelBytes: 1024 * 1024,
  maxWork: 10_000_000, maxLayers: 64, maxDimension: 65536, maxSpecificationChars: 8000,
  maxPromptModelBytes: 48 * 1024, maxPromptBytes: 64 * 1024, maxProposals: 16, maxHistory: 32 });
const activations = new Set(['Linear', 'Threshold', 'ThresholdSymmetric', 'Sigmoid',
  'SigmoidSymmetric', 'Tanh', 'Gaussian', 'GaussianSymmetric', 'Elliot', 'ElliotSymmetric',
  'LinearPiece', 'LinearPieceSymmetric', 'ReLU', 'ReLULeaky', 'Sin', 'Cos', 'SinSymmetric',
  'CosSymmetric', 'Gelu', 'Swish', 'LeakyRelu']);
const own = Object.getOwnPropertyDescriptor;
export function record(value, allowed, required, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError(`${name} must be a plain data object`);
  const result = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string' || !allowed.includes(key)) throw new TypeError(`unknown ${name} field`);
    const descriptor = own(value, key);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError(`${name} accessors are not allowed`);
    result[key] = descriptor.value;
  }
  if (required.some(key => !Object.hasOwn(result, key))) throw new TypeError(`missing ${name} field`);
  return result;
}
export function integer(value, minimum, maximum, name) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum)
    throw new RangeError(`${name} must be an integer in ${minimum}..${maximum}`);
  return value;
}
export function finite(value, name, minimum = -Number.MAX_VALUE, maximum = Number.MAX_VALUE) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum)
    throw new RangeError(`${name} must be a finite primitive number in bounds`);
  return value === 0 ? 0 : value;
}
export function arrayLength(value, maximum, name) {
  if (!Array.isArray(value) || ![Array.prototype, null].includes(Object.getPrototypeOf(value)))
    throw new TypeError(`${name} must be an array`);
  const descriptor = own(value, 'length');
  const length = integer(descriptor?.value, 0, maximum, `${name} length`);
  for (const key of Reflect.ownKeys(value)) {
    if (key === 'length') continue;
    if (typeof key !== 'string' || !/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= length)
      throw new TypeError(`${name} contains unknown array fields`);
  }
  return length;
}
export function element(value, index, name) {
  const descriptor = own(value, String(index));
  if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError(`${name} must contain own data elements`);
  return descriptor.value;
}
function numbers(value, expected, maximum, name) {
  if (arrayLength(value, maximum, name) !== expected) throw new RangeError(`${name} length does not match dimensions`);
  const result = new Array(expected);
  for (let index = 0; index < expected; index++) result[index] = finite(element(value, index, name), name);
  return result;
}
export function modelJSON(model) {
  // Compute ASCII canonical length before allocating its JSON string. All names
  // are from fixed schemas/activation lists; finite number tokens are bounded.
  let length = '{"schema":"ruv-fann.dense-layers/1","layers":[]}'.length;
  for (let index = 0; index < model.layers.length; index++) {
    const layer = model.layers[index];
    length += JSON.stringify({ inputs: layer.inputs, outputs: layer.outputs, weights: [], bias: [],
      activation: layer.activation, steepness: layer.steepness }).length + (index ? 1 : 0);
    for (const values of [layer.weights, layer.bias]) {
      length += Math.max(0, values.length - 1);
      for (let offset = 0; offset < values.length; offset++) {
        length += String(values[offset]).length;
        if (length > WSP_LIMITS.maxModelBytes) throw new RangeError('model exceeds canonical JSON byte limit');
      }
    }
  }
  if (length > WSP_LIMITS.maxModelBytes) throw new RangeError('model exceeds canonical JSON byte limit');
  return JSON.stringify(model);
}
export function snapshotModel(value) {
  if (typeof value === 'string') {
    if (value.length > WSP_LIMITS.maxModelBytes || new TextEncoder().encode(value).length > WSP_LIMITS.maxModelBytes)
      throw new RangeError('model exceeds JSON byte limit');
    value = JSON.parse(value);
  }
  const source = record(value, ['schema', 'layers'], ['schema', 'layers'], 'model');
  if (source.schema !== 'ruv-fann.dense-layers/1') throw new TypeError('unsupported dense model schema');
  const count = integer(arrayLength(source.layers, WSP_LIMITS.maxLayers, 'layers'), 1, WSP_LIMITS.maxLayers, 'layers');
  const layers = [];
  let parameters = 0, previous;
  for (let index = 0; index < count; index++) {
    const layer = record(element(source.layers, index, 'layers'),
      ['inputs', 'outputs', 'weights', 'bias', 'activation', 'steepness'],
      ['inputs', 'outputs', 'weights', 'activation', 'steepness'], 'layer');
    const inputs = integer(layer.inputs, 1, WSP_LIMITS.maxDimension, 'inputs');
    const outputs = integer(layer.outputs, 1, WSP_LIMITS.maxDimension, 'outputs');
    if (previous !== undefined && previous !== inputs) throw new RangeError('adjacent layer dimensions disagree');
    previous = outputs;
    // Count canonical bias slots even when absent in the caller's model.
    parameters += inputs * outputs + outputs;
    integer(parameters, 1, WSP_LIMITS.maxParameters, 'parameter count');
    if (!activations.has(layer.activation)) throw new TypeError('unsupported activation');
    const steepness = finite(layer.steepness, 'steepness');
    const weights = numbers(layer.weights, inputs * outputs, WSP_LIMITS.maxParameters, 'weights');
    const bias = layer.bias === undefined || layer.bias === null ? new Array(outputs).fill(0) :
      numbers(layer.bias, outputs, WSP_LIMITS.maxParameters, 'bias');
    layers.push({ inputs, outputs, weights, bias, activation: layer.activation, steepness });
  }
  const model = { schema: 'ruv-fann.dense-layers/1', layers };
  return { model, json: modelJSON(model), parameters, inputSize: layers[0].inputs, outputSize: layers.at(-1).outputs };
}
export function snapshotLimits(value = {}) {
  const source = record(value, ['maxEdits', 'maxAbsValue', 'maxAbsDelta'], [], 'limits');
  return { maxEdits: source.maxEdits === undefined ? WSP_LIMITS.maxEdits : integer(source.maxEdits, 1, WSP_LIMITS.maxEdits, 'maxEdits'),
    maxAbsValue: source.maxAbsValue === undefined ? WSP_LIMITS.maxAbsValue : finite(source.maxAbsValue, 'maxAbsValue', 0, WSP_LIMITS.maxAbsValue),
    maxAbsDelta: source.maxAbsDelta === undefined ? WSP_LIMITS.maxAbsDelta : finite(source.maxAbsDelta, 'maxAbsDelta', 0, WSP_LIMITS.maxAbsDelta) };
}
export function cloneModel(model) {
  return { schema: model.schema, layers: model.layers.map(layer => ({ inputs: layer.inputs, outputs: layer.outputs,
    weights: layer.weights.slice(), bias: layer.bias.slice(), activation: layer.activation, steepness: layer.steepness })) };
}
export function patchCandidate(value, source, limits) {
  const patch = record(value, ['schema', 'baseHash', 'edits'], ['schema', 'baseHash', 'edits'], 'patch');
  if (patch.schema !== 'ruv-fann.weight-patch/1' || typeof patch.baseHash !== 'string' || !/^[a-f0-9]{64}$/.test(patch.baseHash))
    throw new TypeError('invalid patch schema or baseHash');
  const count = integer(arrayLength(patch.edits, limits.maxEdits, 'edits'), 1, limits.maxEdits, 'edits');
  const edits = [], seen = new Set(), candidate = cloneModel(source.model);
  for (let offset = 0; offset < count; offset++) {
    const edit = record(element(patch.edits, offset, 'edits'), ['layer', 'parameter', 'index', 'value'],
      ['layer', 'parameter', 'index', 'value'], 'edit');
    const layer = integer(edit.layer, 0, source.model.layers.length - 1, 'edit layer');
    if (edit.parameter !== 'weights' && edit.parameter !== 'bias') throw new TypeError('edit parameter must be weights or bias');
    const values = source.model.layers[layer][edit.parameter];
    const index = integer(edit.index, 0, values.length - 1, 'edit index');
    const next = finite(edit.value, 'edit value', -limits.maxAbsValue, limits.maxAbsValue);
    const delta = Math.abs(next - values[index]);
    if (!Number.isFinite(delta) || delta > limits.maxAbsDelta) throw new RangeError('edit exceeds maxAbsDelta');
    const location = `${layer}:${edit.parameter}:${index}`;
    if (seen.has(location)) throw new TypeError('duplicate edit location');
    seen.add(location);
    edits.push({ layer, parameter: edit.parameter, index, value: next });
    candidate.layers[layer][edit.parameter][index] = next;
  }
  return { patch: { schema: 'ruv-fann.weight-patch/1', baseHash: patch.baseHash, edits },
    candidate, candidateJSON: modelJSON(candidate) };
}
export function snapshotDataset(value, source) {
  const data = record(value, ['inputs', 'targets', 'rows'], ['inputs', 'targets', 'rows'], 'dataset');
  const rows = integer(data.rows, 1, WSP_LIMITS.maxRows, 'rows');
  if (rows * source.parameters > WSP_LIMITS.maxWork) throw new RangeError('evaluation exceeds work limit');
  const inputCount = integer(rows * source.inputSize, 1, WSP_LIMITS.maxElements, 'input elements');
  const outputCount = integer(rows * source.outputSize, 1, WSP_LIMITS.maxElements, 'target elements');
  const inputs = numbers(data.inputs, inputCount, WSP_LIMITS.maxElements, 'dataset inputs');
  const targets = numbers(data.targets, outputCount, WSP_LIMITS.maxElements, 'dataset targets');
  return { rows, inputs, targets, canonical: { schema: 'ruv-fann.weight-dataset/1', inputSize: source.inputSize,
    outputSize: source.outputSize, rows, inputs, targets } };
}
export function snapshotPolicy(value) {
  const policy = record(value, ['maxMSE', 'maxRegressionMSE', 'maxAbsoluteError'], ['maxMSE', 'maxRegressionMSE'], 'policy');
  const result = { maxMSE: finite(policy.maxMSE, 'maxMSE', 0), maxRegressionMSE: finite(policy.maxRegressionMSE, 'maxRegressionMSE', 0) };
  if (Object.hasOwn(policy, 'maxAbsoluteError')) result.maxAbsoluteError = finite(policy.maxAbsoluteError, 'maxAbsoluteError', 0);
  return result;
}
export function snapshotOptions(value = {}) {
  const options = record(value, ['simd', 'limits'], [], 'options');
  const simd = options.simd === undefined ? 'auto' : options.simd;
  if (simd !== true && simd !== false && simd !== 'auto') throw new TypeError('simd must be boolean or auto');
  return { simd, limits: snapshotLimits(options.limits) };
}
export function owned(value) { return JSON.parse(JSON.stringify(value)); }
