import { WSP_LIMITS } from 'ruv-fann-wsp';

export function record(value, allowed, required, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError(`${name} must be a plain data object`);
  const result = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string' || !allowed.includes(key)) throw new TypeError(`unknown ${name} field`);
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError(`${name} accessors are forbidden`);
    result[key] = descriptor.value;
  }
  if (required.some(key => !Object.hasOwn(result, key))) throw new TypeError(`missing ${name} field`);
  return result;
}
export function integer(value, minimum, maximum, name) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new RangeError(`${name} must be an integer in ${minimum}..${maximum}`);
  return value;
}
export function finite(value, name, minimum = -Number.MAX_VALUE, maximum = Number.MAX_VALUE) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum)
    throw new RangeError(`${name} must be a finite number in bounds`);
  return value === 0 ? 0 : value;
}
export function arrayLength(value, maximum, name) {
  if (!Array.isArray(value) || ![Array.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError(`${name} must be an array`);
  const length = integer(Object.getOwnPropertyDescriptor(value, 'length')?.value, 0, maximum, `${name} length`);
  for (const key of Reflect.ownKeys(value)) {
    if (key !== 'length' && (typeof key !== 'string' || !/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= length))
      throw new TypeError(`${name} contains unknown array properties`);
  }
  return length;
}
export function element(value, index, name) {
  const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
  if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError(`${name} requires own dense data elements`);
  return descriptor.value;
}
export function numbers(value, length, maximum, name, budget = () => {}) {
  if (arrayLength(value, maximum, name) !== length) throw new RangeError(`${name} length disagrees with dimensions`);
  const output = new Array(length);
  for (let index = 0; index < length; index++) {
    output[index] = finite(element(value, index, name), name);
    budget(String(output[index]).length + 1);
  }
  return output;
}
const ACTIVATIONS = new Set(['Linear', 'Threshold', 'ThresholdSymmetric', 'Sigmoid',
  'SigmoidSymmetric', 'Tanh', 'Gaussian', 'GaussianSymmetric', 'Elliot', 'ElliotSymmetric',
  'LinearPiece', 'LinearPieceSymmetric', 'ReLU', 'ReLULeaky', 'Sin', 'Cos', 'SinSymmetric',
  'CosSymmetric', 'Gelu', 'Swish', 'LeakyRelu']);
export function modelSnapshot(value) {
  if (typeof value === 'string') {
    if (value.length > WSP_LIMITS.maxModelBytes || new TextEncoder().encode(value).length > WSP_LIMITS.maxModelBytes)
      throw new RangeError('model exceeds byte limit');
    value = JSON.parse(value);
  }
  const raw = record(value, ['schema', 'layers'], ['schema', 'layers'], 'model');
  if (raw.schema !== 'ruv-fann.dense-layers/1') throw new TypeError('unsupported model schema');
  const count = integer(arrayLength(raw.layers, WSP_LIMITS.maxLayers, 'layers'), 1, WSP_LIMITS.maxLayers, 'layers');
  const layers = [];
  let parameters = 0, previous, bytes = 48;
  const budget = size => { bytes += size; if (bytes > WSP_LIMITS.maxModelBytes) throw new RangeError('model exceeds byte limit'); };
  for (let index = 0; index < count; index++) {
    const layer = record(element(raw.layers, index, 'layers'), ['inputs', 'outputs', 'weights', 'bias', 'activation', 'steepness'],
      ['inputs', 'outputs', 'weights', 'activation', 'steepness'], 'layer');
    const inputs = integer(layer.inputs, 1, WSP_LIMITS.maxDimension, 'inputs');
    const outputs = integer(layer.outputs, 1, WSP_LIMITS.maxDimension, 'outputs');
    if (previous !== undefined && previous !== inputs) throw new RangeError('adjacent layer dimensions disagree');
    previous = outputs;
    parameters += inputs * outputs + outputs;
    integer(parameters, 1, WSP_LIMITS.maxParameters, 'parameters');
    if (!ACTIVATIONS.has(layer.activation)) throw new TypeError('unsupported activation');
    const steepness = finite(layer.steepness, 'steepness');
    budget(160 + layer.activation.length + String(steepness).length);
    const weights = numbers(layer.weights, inputs * outputs, WSP_LIMITS.maxParameters, 'weights', budget);
    const bias = layer.bias === undefined || layer.bias === null ? Array(outputs).fill(0) :
      numbers(layer.bias, outputs, WSP_LIMITS.maxParameters, 'bias', budget);
    if (layer.bias === undefined || layer.bias === null) budget(outputs * 2);
    layers.push({ inputs, outputs, weights, bias, activation: layer.activation, steepness });
  }
  const model = { schema: raw.schema, layers };
  if (new TextEncoder().encode(JSON.stringify(model)).length > WSP_LIMITS.maxModelBytes) throw new RangeError('model exceeds canonical byte limit');
  return { model, parameters, inputSize: layers[0].inputs, outputSize: layers.at(-1).outputs };
}
export function simdOptions(value = {}) {
  const options = record(value, ['simd'], [], 'invariant options');
  const simd = options.simd === undefined ? 'auto' : options.simd;
  if (simd !== true && simd !== false && simd !== 'auto') throw new TypeError('simd must be boolean or auto');
  return { simd };
}
