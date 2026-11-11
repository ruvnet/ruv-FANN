import { WSP_LIMITS } from 'ruv-fann-wsp';

export function record(value, allowed, required, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError(`${name} must be a plain data object`);
  const result = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string' || !allowed.includes(key)) throw new TypeError(`unknown ${name} field`);
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError(`${name} accessors are not allowed`);
    result[key] = descriptor.value;
  }
  if (required.some(key => !Object.hasOwn(result, key))) throw new TypeError(`missing ${name} field`);
  return result;
}
export function integer(value, min, max, name) {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new RangeError(`${name} must be an integer in ${min}..${max}`);
  return value;
}
export function finite(value, name, min = -Number.MAX_VALUE, max = Number.MAX_VALUE) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) throw new RangeError(`${name} must be finite and in bounds`);
  return value === 0 ? 0 : value;
}
export function array(value, max, name, item = x => x) {
  if (!Array.isArray(value) || ![Array.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError(`${name} must be an array`);
  const length = integer(Object.getOwnPropertyDescriptor(value, 'length')?.value, 0, max, `${name} length`);
  for (const key of Reflect.ownKeys(value)) {
    if (key !== 'length' && (typeof key !== 'string' || !/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= length))
      throw new TypeError(`${name} contains unknown array fields`);
  }
  const result = new Array(length);
  for (let index = 0; index < length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError(`${name} requires own data elements`);
    result[index] = item(descriptor.value, index);
  }
  return result;
}
export const numbers = (value, max, name) => array(value, max, name, x => finite(x, name));
export function modelSnapshot(value) {
  if (typeof value === 'string') {
    if (value.length > WSP_LIMITS.maxModelBytes || new TextEncoder().encode(value).length > WSP_LIMITS.maxModelBytes) throw new RangeError('model exceeds byte limit');
    value = JSON.parse(value);
  }
  const source = record(value, ['schema', 'layers'], ['schema', 'layers'], 'model');
  if (source.schema !== 'ruv-fann.dense-layers/1') throw new TypeError('unsupported model schema');
  let parameters = 0;
  const layers = array(source.layers, WSP_LIMITS.maxLayers, 'layers', value => {
    const layer = record(value, ['inputs', 'outputs', 'weights', 'bias', 'activation', 'steepness'],
      ['inputs', 'outputs', 'weights', 'activation', 'steepness'], 'layer');
    const inputs = integer(layer.inputs, 1, WSP_LIMITS.maxDimension, 'inputs');
    const outputs = integer(layer.outputs, 1, WSP_LIMITS.maxDimension, 'outputs');
    parameters += inputs * outputs + outputs;
    integer(parameters, 1, WSP_LIMITS.maxParameters, 'parameter count');
    const weights = numbers(layer.weights, inputs * outputs, 'weights');
    const bias = layer.bias === undefined || layer.bias === null ? Array(outputs).fill(0) : numbers(layer.bias, outputs, 'bias');
    if (weights.length !== inputs * outputs || bias.length !== outputs) throw new RangeError('layer dimensions disagree');
    // Authoritative WSP inspection checks activation names, adjacent shapes and canonical size before execution.
    if (typeof layer.activation !== 'string' || layer.activation.length > 32) throw new TypeError('invalid activation');
    return { inputs, outputs, weights, bias, activation: layer.activation, steepness: finite(layer.steepness, 'steepness') };
  });
  if (!layers.length) throw new RangeError('model requires layers');
  return { model: { schema: source.schema, layers }, parameters };
}
export function dataSnapshot(value, model, targets = true) {
  const source = record(value, targets ? ['inputs', 'targets', 'rows'] : ['inputs', 'rows'],
    targets ? ['inputs', 'targets', 'rows'] : ['inputs', 'rows'], 'dataset');
  const rows = integer(source.rows, 1, WSP_LIMITS.maxRows, 'rows');
  const inputs = numbers(source.inputs, WSP_LIMITS.maxElements, 'inputs');
  if (inputs.length !== rows * model.layers[0].inputs) throw new RangeError('input dimensions disagree');
  if (!targets) return { inputs, rows };
  const output = numbers(source.targets, WSP_LIMITS.maxElements, 'targets');
  if (output.length !== rows * model.layers.at(-1).outputs) throw new RangeError('target dimensions disagree');
  return { inputs, targets: output, rows };
}
export function limitsSnapshot(value = {}) {
  const raw = record(value, ['maxEdits', 'maxAbsValue', 'maxAbsDelta'], [], 'limits');
  return {
    maxEdits: raw.maxEdits === undefined ? WSP_LIMITS.maxEdits : integer(raw.maxEdits, 1, WSP_LIMITS.maxEdits, 'maxEdits'),
    maxAbsValue: finite(raw.maxAbsValue === undefined ? WSP_LIMITS.maxAbsValue : raw.maxAbsValue, 'maxAbsValue', 0, WSP_LIMITS.maxAbsValue),
    maxAbsDelta: finite(raw.maxAbsDelta === undefined ? WSP_LIMITS.maxAbsDelta : raw.maxAbsDelta, 'maxAbsDelta', 0, WSP_LIMITS.maxAbsDelta)
  };
}
export function simdOption(value = 'auto') {
  if (value !== true && value !== false && value !== 'auto') throw new TypeError('simd must be boolean or auto');
  return value;
}
export function deepFreeze(value) {
  for (const child of Object.values(value)) if (child && typeof child === 'object') deepFreeze(child);
  return Object.freeze(value);
}
export const clone = value => JSON.parse(JSON.stringify(value));
