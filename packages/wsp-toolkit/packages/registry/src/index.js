/** Hard bounds apply to each copied argument/result and to queued operations. */
export const REGISTRY_LIMITS = Object.freeze({ maxPending: 16, maxBytes: 2 * 1024 * 1024,
  maxNodes: 262144, maxDepth: 16 });

function identifier(value, name = 'id') {
  if (typeof value !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(value))
    throw new TypeError(`${name} must contain 1..128 safe ASCII characters`);
  return value;
}
function integer(value, maximum, name) {
  if (!Number.isSafeInteger(value) || value < 1 || value > maximum) throw new RangeError(`invalid ${name}`);
  return value;
}
function snapshot(value) {
  let nodes = 0, stringChars = 0;
  const ancestors = new Set();
  function copy(item, depth) {
    if (++nodes > REGISTRY_LIMITS.maxNodes || depth > REGISTRY_LIMITS.maxDepth)
      throw new RangeError('registry data exceeds structural bounds');
    if (item === null || typeof item === 'boolean') return item;
    if (typeof item === 'string') {
      stringChars += item.length;
      if (stringChars > REGISTRY_LIMITS.maxBytes) throw new RangeError('registry text exceeds byte bound');
      return item;
    }
    if (typeof item === 'number' && Number.isFinite(item)) return item === 0 ? 0 : item;
    if (!item || typeof item !== 'object' || ancestors.has(item)) throw new TypeError('registry data must be an acyclic finite JSON tree');
    const isArray = Array.isArray(item);
    if (!(isArray ? Object.getPrototypeOf(item) === Array.prototype :
      [Object.prototype, null].includes(Object.getPrototypeOf(item)))) throw new TypeError('registry data must contain plain records and arrays');
    if (isArray && item.length > REGISTRY_LIMITS.maxNodes) throw new RangeError('registry array exceeds bounds');
    ancestors.add(item);
    const result = isArray ? [] : Object.create(null);
    let arrayItems = 0;
    for (const key of Reflect.ownKeys(item)) {
      if (isArray && key === 'length') continue;
      if (typeof key !== 'string' || ['__proto__', 'constructor', 'prototype'].includes(key) ||
        (isArray && (!/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= item.length)))
        throw new TypeError('registry data contains unsafe or unknown properties');
      stringChars += key.length;
      if (stringChars > REGISTRY_LIMITS.maxBytes) throw new RangeError('registry text exceeds byte bound');
      const descriptor = Object.getOwnPropertyDescriptor(item, key);
      if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new TypeError('registry data accessors are forbidden');
      result[key] = copy(descriptor.value, depth + 1);
      arrayItems++;
    }
    if (isArray && (arrayItems !== item.length || result.length !== item.length)) throw new TypeError('registry arrays must be dense');
    ancestors.delete(item);
    return result;
  }
  const owned = copy(value, 0);
  if (new TextEncoder().encode(JSON.stringify(owned)).length > REGISTRY_LIMITS.maxBytes)
    throw new RangeError('registry data exceeds byte bound');
  return owned;
}
function options(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value)))
    throw new TypeError('registry options must be a plain record');
  const owned = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !['flush', 'close', 'maxPending'].includes(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw new TypeError('unknown registry option or accessor');
    owned[key] = descriptor.value;
  }
  for (const key of ['flush', 'close']) if (owned[key] !== undefined && typeof owned[key] !== 'function')
    throw new TypeError(`${key} must be a trusted callback`);
  return owned;
}
function capability(target, key) {
  let owner = target;
  while (owner && owner !== Object.prototype) {
    const descriptor = Object.getOwnPropertyDescriptor(owner, key);
    if (descriptor) {
      if (!Object.hasOwn(descriptor, 'value') || typeof descriptor.value !== 'function')
        throw new TypeError(`WeightPrograms ${key} must be a data method`);
      return descriptor.value.bind(target);
    }
    owner = Object.getPrototypeOf(owner);
  }
  throw new TypeError(`WeightPrograms capability lacks ${key}`);
}
function arity(args, expected) {
  if (args.length !== expected) throw new TypeError(`expected ${expected} registry arguments`);
}

export class RegistryPersistenceError extends Error {
  constructor(cause) {
    super('registry persistence failed; state is uncertain, close and reopen before continuing', { cause });
    this.name = 'RegistryPersistenceError';
    this.code = 'WSP_REGISTRY_PERSISTENCE';
  }
}

export class RegistryStaleError extends Error {
  constructor(cause) {
    super('registry revision changed; candidate is stale', { cause });
    this.name = 'RegistryStaleError';
    this.code = 'WSP_REGISTRY_STALE';
  }
}

/**
 * Adapt a caller-owned real AgentDB WeightPrograms controller. The capability and
 * persistence callbacks are trusted local code, not a security sandbox.
 */
export function createAgentDBRegistry(programs, configuration = {}) {
  if (!programs || typeof programs !== 'object') throw new TypeError('a WeightPrograms capability is required');
  const methods = Object.fromEntries(['create', 'get', 'list', 'propose', 'promote', 'rollback'].map(key => [key, capability(programs, key)]));
  const config = options(configuration);
  const maximum = config.maxPending === undefined ? REGISTRY_LIMITS.maxPending : integer(config.maxPending, REGISTRY_LIMITS.maxPending, 'maxPending');
  const flush = config.flush, cleanup = config.close;
  let tail = Promise.resolve(), closing = false, closed = false, poisoned = false, pending = 0, closePromise;
  function healthy() {
    if (poisoned) throw new Error('registry is poisoned after a persistence failure; close and reopen');
    if (closed) throw new Error('registry is closed');
  }
  function enqueue(operation, write) {
    healthy();
    if (closing) throw new Error('registry is closing');
    if (pending >= maximum) throw new RangeError('registry operation queue is full');
    pending++;
    const result = tail.then(async () => {
      healthy();
      let value;
      try { value = await operation(); }
      catch (error) {
        if (error instanceof Error && /^(?:Program revision changed; stale (?:promotion|rollback)|Program changed during evaluation; proposal is stale|Program changed during update; retry with latest revision|Proposal is stale)$/.test(error.message))
          throw new RegistryStaleError(error);
        throw error;
      }
      // Copy while still owning the queue, before callbacks can yield/mutate results.
      let owned;
      try { owned = snapshot(value); }
      catch (error) {
        if (write) { poisoned = true; throw new RegistryPersistenceError(error); }
        throw error;
      }
      if (write && flush) {
        try { await flush(); }
        catch (error) { poisoned = true; throw new RegistryPersistenceError(error); }
      }
      return owned;
    });
    const settled = result.finally(() => { pending--; });
    tail = settled.then(() => undefined, () => undefined);
    return settled;
  }
  return Object.freeze({
    async create(...args) {
      arity(args, 2); const id = identifier(args[0]), model = snapshot(args[1]);
      return enqueue(() => methods.create(id, model), true);
    },
    async get(...args) {
      arity(args, 1); const id = identifier(args[0]);
      return enqueue(() => methods.get(id), false);
    },
    async list(...args) {
      if (args.length > 1) throw new TypeError('list accepts at most one argument');
      const limit = args[0] === undefined ? 50 : integer(args[0], 100, 'list limit');
      return enqueue(() => methods.list(limit), false);
    },
    async propose(...args) {
      arity(args, 4); const id = identifier(args[0]);
      const patch = snapshot(args[1]), dataset = snapshot(args[2]), policy = snapshot(args[3]);
      return enqueue(() => methods.propose(id, patch, dataset, policy), true);
    },
    async promote(...args) {
      arity(args, 3); const id = identifier(args[0]), proposalId = identifier(args[1], 'proposalId');
      const revision = integer(args[2], Number.MAX_SAFE_INTEGER, 'expectedRevision');
      return enqueue(() => methods.promote(id, proposalId, revision), true);
    },
    async rollback(...args) {
      arity(args, 3); const id = identifier(args[0]);
      const target = integer(args[1], Number.MAX_SAFE_INTEGER, 'targetRevision'), expected = integer(args[2], Number.MAX_SAFE_INTEGER, 'expectedRevision');
      return enqueue(() => methods.rollback(id, target, expected), true);
    },
    close(...args) {
      try { arity(args, 0); } catch (error) { return Promise.reject(error); }
      if (closePromise) return closePromise;
      closing = true;
      closePromise = tail.then(async () => {
        try { if (cleanup) await cleanup(Object.freeze({ poisoned })); }
        finally { closed = true; }
      });
      return closePromise;
    },
    status() { return Object.freeze({ pending, closing, closed, poisoned, durability: flush ? 'caller-flush' : 'memory-only' }); }
  });
}
