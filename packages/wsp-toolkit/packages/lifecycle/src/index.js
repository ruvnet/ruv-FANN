import { inspectWeightModel, evaluateWeightPatch, WSP_LIMITS } from 'ruv-fann-wsp';

export const LIFECYCLE_LIMITS = Object.freeze({ maxSuites: 16, maxBranches: 16, maxBytes: 2 * 1024 * 1024,
  maxTotalWork: 10_000_000, maxTotalElements: 262144 });

// Copy only own data. Refuse getters, toJSON hooks, cycles and excessive trees.
function snapshot(value) {
  let elements = 0, chars = 0;
  const ancestors = new Set();
  function copy(item, depth) {
    if (++elements > LIFECYCLE_LIMITS.maxTotalElements || depth > 16) throw new RangeError('input tree exceeds limits');
    if (item === null || typeof item === 'boolean') return item;
    if (typeof item === 'number') {
      if (!Number.isFinite(item)) throw new TypeError('input numbers must be finite');
      return item;
    }
    if (typeof item === 'string') {
      chars += item.length;
      if (chars > LIFECYCLE_LIMITS.maxBytes) throw new RangeError('input text exceeds limits');
      return item;
    }
    if (!item || typeof item !== 'object' || ancestors.has(item)) throw new TypeError('input must be an acyclic JSON data tree');
    const array = Array.isArray(item);
    if (!(array ? Object.getPrototypeOf(item) === Array.prototype :
      [Object.prototype, null].includes(Object.getPrototypeOf(item)))) throw new TypeError('input must contain plain data');
    ancestors.add(item);
    const out = array ? [] : Object.create(null);
    if (array && item.length > LIFECYCLE_LIMITS.maxTotalElements) throw new RangeError('array exceeds limits');
    for (const key of Reflect.ownKeys(item)) {
      if (array && key === 'length') continue;
      if (typeof key !== 'string' || key === '__proto__' || (array && !/^(0|[1-9][0-9]*)$/.test(key)))
        throw new TypeError('unknown input property');
      const descriptor = Object.getOwnPropertyDescriptor(item, key);
      if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError('input accessors are forbidden');
      chars += key.length;
      if (chars > LIFECYCLE_LIMITS.maxBytes) throw new RangeError('input text exceeds limits');
      out[key] = copy(descriptor.value, depth + 1);
    }
    if (array && (out.length !== item.length || Object.keys(out).length !== item.length)) throw new TypeError('sparse input arrays are forbidden');
    ancestors.delete(item);
    return out;
  }
  const result = copy(value, 0);
  if (new TextEncoder().encode(JSON.stringify(result)).length > LIFECYCLE_LIMITS.maxBytes) throw new RangeError('input exceeds byte limit');
  return result;
}
function fields(value, allowed, required, label) {
  if (!value || Array.isArray(value) || typeof value !== 'object' || Object.keys(value).some(k => !allowed.includes(k)) ||
    required.some(k => !Object.hasOwn(value, k))) throw new TypeError(`invalid ${label} fields`);
}
function count(value, max, label) {
  if (!Array.isArray(value) || value.length < 1 || value.length > max) throw new RangeError(`invalid ${label} count`);
}
function budget(info, datasets, multiplicity = 1) {
  let work = 0;
  for (const dataset of datasets) {
    if (!dataset || !Number.isSafeInteger(dataset.rows) || dataset.rows < 1 || dataset.rows > WSP_LIMITS.maxRows)
      throw new RangeError('invalid dataset rows');
    work += 2 * info.parameterCount * dataset.rows * multiplicity;
  }
  if (work > LIFECYCLE_LIMITS.maxTotalWork) throw new RangeError('aggregate evaluation work exceeds limit');
}
function receipt(evaluation) {
  const { model: _model, patch: _patch, ...evidence } = evaluation;
  return evidence;
}

/** Check every named suite. This evidence never authorizes promotion. */
export async function runRegressionGate(bundle, options) {
  const input = snapshot(bundle), configuration = options === undefined ? undefined : snapshot(options);
  fields(input, ['model', 'patch', 'suites'], ['model', 'patch', 'suites'], 'gate');
  count(input.suites, LIFECYCLE_LIMITS.maxSuites, 'suite');
  const names = new Set();
  for (const suite of input.suites) {
    fields(suite, ['name', 'dataset', 'policy'], ['name', 'dataset', 'policy'], 'suite');
    if (typeof suite.name !== 'string' || !/^[a-zA-Z0-9_.-]{1,64}$/.test(suite.name) || names.has(suite.name))
      throw new TypeError('suite names must be unique bounded identifiers');
    names.add(suite.name);
  }
  const info = await inspectWeightModel(input.model);
  budget(info, input.suites.map(s => s.dataset));
  const suites = [];
  for (const suite of input.suites) suites.push({ name: suite.name,
    evaluation: receipt(await evaluateWeightPatch(input.model, input.patch, suite.dataset, suite.policy, configuration)) });
  return { schema: 'ruv-fann.regression-gate/1', baseHash: info.baseHash,
    candidateHash: suites[0].evaluation.candidateHash, accepted: suites.every(s => s.evaluation.accepted), suites };
}

/** Candidate branches share the same immutable base and cannot mutate a live workspace. */
export async function evaluatePatchBranches(model, patches, dataset, policy, options) {
  const input = snapshot({ model, patches, dataset, policy });
  const configuration = options === undefined ? undefined : snapshot(options);
  count(input.patches, LIFECYCLE_LIMITS.maxBranches, 'branch');
  const info = await inspectWeightModel(input.model);
  budget(info, [input.dataset], input.patches.length);
  const branches = [];
  // Sequential allocation keeps WASM memory bounded. This is not a security sandbox.
  for (let index = 0; index < input.patches.length; index++) branches.push({ index,
    evaluation: await evaluateWeightPatch(input.model, input.patches[index], input.dataset, input.policy, configuration) });
  return { schema: 'ruv-fann.patch-branches/1', baseHash: info.baseHash, branches };
}
