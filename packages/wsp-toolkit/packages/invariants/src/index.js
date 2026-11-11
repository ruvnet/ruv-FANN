import { loadRuntime } from 'ruv-fann';
import { inspectWeightModel, WSP_LIMITS } from 'ruv-fann-wsp';
import { record, integer, finite, arrayLength, element, numbers, modelSnapshot, simdOptions } from './data.js';

export const INVARIANT_LIMITS = Object.freeze({ maxCases: 16, maxRows: 1024, maxElements: 131072,
  maxWork: 10_000_000, maxSuiteBytes: 2 * 1024 * 1024, maxViolations: 128 });

function suiteSnapshot(value, source) {
  const raw = record(value, ['schema', 'cases'], ['schema', 'cases'], 'invariant suite');
  if (raw.schema !== 'ruv-fann.invariants/1') throw new TypeError('unsupported invariant suite schema');
  const count = integer(arrayLength(raw.cases, INVARIANT_LIMITS.maxCases, 'cases'), 1, INVARIANT_LIMITS.maxCases, 'cases');
  const pending = [], names = new Set();
  let rows = 0, elements = 0, work = 0;
  // Reserve all aggregate work and scalar storage before copying case arrays.
  for (let index = 0; index < count; index++) {
    const item = record(element(raw.cases, index, 'cases'), ['name', 'inputs', 'rows', 'minimum', 'maximum', 'expected', 'tolerance'],
      ['name', 'inputs', 'rows'], 'invariant case');
    if (typeof item.name !== 'string' || !/^[A-Za-z0-9_.-]{1,64}$/.test(item.name) || names.has(item.name))
      throw new TypeError('invariant names must be unique bounded identifiers');
    names.add(item.name);
    const caseRows = integer(item.rows, 1, INVARIANT_LIMITS.maxRows, 'case rows');
    rows += caseRows;
    if (rows > INVARIANT_LIMITS.maxRows) throw new RangeError('invariant suite exceeds aggregate row limit');
    work += caseRows * source.parameters;
    if (work > INVARIANT_LIMITS.maxWork) throw new RangeError('invariant suite exceeds aggregate work limit');
    const constraints = ['minimum', 'maximum', 'expected'].filter(key => Object.hasOwn(item, key));
    if (!constraints.length) throw new TypeError('each invariant case requires at least one constraint');
    if (Object.hasOwn(item, 'tolerance') && !Object.hasOwn(item, 'expected')) throw new TypeError('tolerance requires expected outputs');
    const inputCount = caseRows * source.inputSize, outputCount = caseRows * source.outputSize;
    elements += inputCount + (Object.hasOwn(item, 'expected') ? outputCount : 0) +
      (Object.hasOwn(item, 'minimum') ? source.outputSize : 0) + (Object.hasOwn(item, 'maximum') ? source.outputSize : 0);
    if (elements > INVARIANT_LIMITS.maxElements) throw new RangeError('invariant suite exceeds aggregate element limit');
    // Each output is materialized in a bounded batch and never retained in the receipt.
    if (inputCount > WSP_LIMITS.maxElements || outputCount > WSP_LIMITS.maxElements)
      throw new RangeError('invariant case exceeds scalar element limit');
    pending.push({ item, rows: caseRows, inputCount, outputCount });
  }
  let bytes = 128;
  const budget = size => { bytes += size; if (bytes > INVARIANT_LIMITS.maxSuiteBytes) throw new RangeError('invariant suite exceeds byte limit'); };
  const cases = pending.map(({ item, rows, inputCount, outputCount }) => {
    budget(item.name.length + 128);
    const result = { name: item.name, inputs: numbers(item.inputs, inputCount, WSP_LIMITS.maxElements, 'case inputs', budget), rows };
    for (const key of ['minimum', 'maximum', 'expected']) {
      if (Object.hasOwn(item, key)) result[key] = numbers(item[key], key === 'expected' ? outputCount : source.outputSize,
        WSP_LIMITS.maxElements, key, budget);
    }
    if (result.minimum && result.maximum && result.minimum.some((value, i) => value > result.maximum[i]))
      throw new RangeError('minimum must not exceed maximum');
    if (result.expected) {
      result.tolerance = item.tolerance === undefined ? 0 : finite(item.tolerance, 'tolerance', 0);
      budget(String(result.tolerance).length);
    }
    return result;
  });
  const canonical = { schema: raw.schema, inputSize: source.inputSize, outputSize: source.outputSize, cases };
  const json = JSON.stringify(canonical);
  if (new TextEncoder().encode(json).length > INVARIANT_LIMITS.maxSuiteBytes) throw new RangeError('invariant suite exceeds canonical byte limit');
  return { cases, work, json };
}
async function hashJSON(json) {
  let provider = globalThis.crypto;
  if (!provider?.subtle && typeof process === 'object' && process.versions?.node) {
    const moduleName = 'node:crypto'; provider = (await import(moduleName)).webcrypto;
  }
  if (!provider?.subtle) throw new Error('secure-context WebCrypto is required');
  const bytes = await provider.subtle.digest('SHA-256', new TextEncoder().encode(json));
  return Array.from(new Uint8Array(bytes), byte => byte.toString(16).padStart(2, '0')).join('');
}

/** Pure synchronous preflight. Returns a work reservation after complete model/suite validation. */
export function estimateInvariantWork(model, suite) {
  const source = modelSnapshot(model);
  return suiteSnapshot(suite, source).work;
}

/** Finite-domain evidence only. This does not authorize promotion or prove arbitrary-input safety. */
export async function checkModelInvariants(model, suite, options) {
  // All caller-owned model/suite/configuration data is copied before yielding.
  const source = modelSnapshot(model), preparedSuite = suiteSnapshot(suite, source), configuration = simdOptions(options);
  const [inspection, suiteHash] = await Promise.all([inspectWeightModel(source.model), hashJSON(preparedSuite.json)]);
  const runtime = await loadRuntime(configuration);
  const violations = [];
  let checks = 0, violationCount = 0;
  const compare = (item, row, output, constraint, actual, bound, accepted) => {
    checks++;
    if (accepted) return;
    violationCount++;
    if (violations.length < INVARIANT_LIMITS.maxViolations) violations.push({ case: item.name, row, output, constraint, actual, bound });
  };
  for (const item of preparedSuite.cases) {
    const batchRows = Math.min(32, item.rows);
    const prepared = runtime.prepare(source.model, { maxRows: batchRows });
    try {
      const inputs = Float64Array.from(item.inputs);
      for (let first = 0; first < item.rows; first += batchRows) {
        const rows = Math.min(batchRows, item.rows - first);
        const predictions = prepared.predict(inputs.subarray(first * source.inputSize, (first + rows) * source.inputSize), rows);
        for (let index = 0; index < predictions.length; index++) {
          const actual = finite(predictions[index], 'invariant prediction');
          const row = first + Math.floor(index / source.outputSize), output = index % source.outputSize;
          if (item.minimum) compare(item, row, output, 'minimum', actual, item.minimum[output], actual >= item.minimum[output]);
          if (item.maximum) compare(item, row, output, 'maximum', actual, item.maximum[output], actual <= item.maximum[output]);
          if (item.expected) {
            const expected = item.expected[row * source.outputSize + output];
            const error = finite(Math.abs(actual - expected), 'invariant absolute error');
            compare(item, row, output, 'expected', actual, expected, error <= item.tolerance);
          }
        }
      }
    } finally { prepared.dispose(); }
  }
  return { schema: 'ruv-fann.invariant-check/1', modelHash: inspection.baseHash, suiteHash, accepted: violationCount === 0,
    checks, violations, violationCount, truncatedViolations: violationCount > violations.length, backend: runtime.backend, work: preparedSuite.work };
}

/** Exhaustively enumerate 1..6 input bits in most-significant-bit-first order. */
export function createBooleanInvariant(table) {
  const raw = record(table, ['inputs', 'outputs', 'targets', 'tolerance'], ['inputs', 'outputs', 'targets'], 'Boolean invariant');
  const inputs = integer(raw.inputs, 1, 6, 'Boolean inputs'), outputs = integer(raw.outputs, 1, 16, 'Boolean outputs');
  const rows = 2 ** inputs;
  const expected = numbers(raw.targets, rows * outputs, rows * outputs, 'Boolean targets');
  if (expected.some(value => value !== 0 && value !== 1)) throw new RangeError('Boolean targets must contain only numeric 0 or 1');
  const tolerance = raw.tolerance === undefined ? 0 : finite(raw.tolerance, 'Boolean tolerance', 0);
  const values = [];
  for (let row = 0; row < rows; row++) for (let bit = 0; bit < inputs; bit++) values.push((row >> (inputs - bit - 1)) & 1);
  return { schema: 'ruv-fann.invariants/1', cases: [{ name: 'boolean-domain', inputs: values, rows, expected, tolerance }] };
}
