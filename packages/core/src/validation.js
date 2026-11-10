export const LIMITS = Object.freeze({ maxRows: 4096, maxDimension: 65536, maxLayers: 64,
  maxParameters: 1_000_000, maxModelBytes: 4 * 1024 * 1024, maxBytes: 64 * 1024 * 1024,
  maxEpochs: 1000, maxTrainingWork: 100_000_000 });
const typed = Object.getPrototypeOf(Float64Array.prototype);
const tag = Object.getOwnPropertyDescriptor(typed, Symbol.toStringTag).get;
const length = Object.getOwnPropertyDescriptor(typed, 'length').get;
const buffer = Object.getOwnPropertyDescriptor(typed, 'buffer').get;
const bytes = Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, 'byteLength').get;
const values = Float64Array.prototype.values;

export function integer(value, min, max, name) {
  if (!Number.isSafeInteger(value) || value < min || value > max)
    throw new RangeError(`${name} must be an integer from ${min} to ${max}`);
  return value;
}
export function options(value, keys, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      Object.keys(value).some(key => !keys.includes(key))) throw new TypeError(`invalid ${name}`);
}
export function finite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
}
export function array(value, name, memory) {
  let size;
  try {
    if (tag.call(value) !== 'Float64Array') throw new TypeError();
    const backing = buffer.call(value);
    bytes.call(backing);
    values.call(value).next();
    if (backing === memory.buffer) throw new TypeError();
    size = length.call(value);
  } catch { throw new TypeError(`${name} must be a JS-owned, nonshared, nondetached Float64Array`); }
  return size;
}
export function inputs(value, dimension, rows, maxRows, name, memory) {
  const size = array(value, name, memory);
  const count = integer(rows === undefined ? size / dimension : rows, 0, maxRows, 'rows');
  if (size !== count * dimension) throw new RangeError(`${name} length does not match rows`);
  for (let index = 0; index < size; index++) finite(value[index], name);
  return count;
}
function numericArray(value, count, name) {
  if (!Array.isArray(value) || value.length !== count) throw new RangeError(`invalid ${name} shape`);
  const copy = new Array(count);
  for (let index = 0; index < count; index++) copy[index] = finite(value[index], name);
  return copy;
}
export function modelDocument(model) {
  if (typeof model === 'string') {
    if (model.length > LIMITS.maxModelBytes || new TextEncoder().encode(model).length > LIMITS.maxModelBytes)
      throw new RangeError('model exceeds JSON byte budget');
    model = JSON.parse(model);
  }
  options(model, ['schema', 'layers'], 'model');
  if (model.schema !== 'ruv-fann.dense-layers/1') throw new TypeError('unsupported model schema');
  if (!Array.isArray(model.layers)) throw new TypeError('layers must be an array');
  integer(model.layers.length, 1, LIMITS.maxLayers, 'layers');
  let previous, parameterCount = 0;
  const layers = [];
  for (const layer of model.layers) {
    options(layer, ['inputs', 'outputs', 'weights', 'bias', 'activation', 'steepness'], 'layer');
    const input = integer(layer.inputs, 1, LIMITS.maxDimension, 'layer inputs');
    const output = integer(layer.outputs, 1, LIMITS.maxDimension, 'layer outputs');
    if (previous !== undefined && input !== previous) throw new RangeError('layers must have adjacent dimensions');
    previous = output;
    const count = input * output;
    parameterCount += count + (layer.bias == null ? 0 : output);
    integer(parameterCount, 1, LIMITS.maxParameters, 'parameter count');
    if (typeof layer.activation !== 'string') throw new TypeError('activation must be a string');
    layers.push({ inputs: input, outputs: output,
      weights: numericArray(layer.weights, count, 'weights'),
      bias: layer.bias == null ? null : numericArray(layer.bias, output, 'bias'),
      activation: layer.activation, steepness: finite(layer.steepness, 'steepness') });
  }
  const json = JSON.stringify({ schema: 'ruv-fann.dense-layers/1', layers });
  if (new TextEncoder().encode(json).length > LIMITS.maxModelBytes) throw new RangeError('model exceeds JSON byte budget');
  return { json, parameterCount, inputSize: layers[0].inputs, outputSize: layers.at(-1).outputs };
}
