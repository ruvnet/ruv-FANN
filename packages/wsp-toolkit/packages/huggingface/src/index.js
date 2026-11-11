import { inspectWeightModel, WSP_LIMITS } from 'ruv-fann-wsp';

const ORIGIN = 'https://huggingface.co';
const MAX_REDIRECTS = 3;
const MAX_TIMEOUT_MS = 60_000;

function record(value, allowed, required, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value)))
    throw new TypeError(`${name} must be a plain data object`);
  const result = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !allowed.includes(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw new TypeError(`${name} contains unknown fields or accessors`);
    result[key] = descriptor.value;
  }
  if (required.some(key => !Object.hasOwn(result, key))) throw new TypeError(`missing ${name} field`);
  return result;
}

function snapshotSource(value) {
  const source = record(value, ['repo', 'revision', 'path', 'sha256', 'maxBytes'],
    ['repo', 'revision', 'path', 'sha256'], 'Hub source');
  if (typeof source.repo !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/.test(source.repo))
    throw new TypeError('repo must be a Hugging Face owner/name');
  if (typeof source.revision !== 'string' || !/^[a-f0-9]{40}$/.test(source.revision))
    throw new TypeError('revision must be an immutable lowercase 40-character commit SHA');
  if (typeof source.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(source.sha256))
    throw new TypeError('sha256 must be a lowercase 64-character raw-file SHA-256');
  if (typeof source.path !== 'string' || source.path.length > 1024 || !source.path.endsWith('.json') ||
      source.path.split('/').some(part => !/^[A-Za-z0-9_.-]+$/.test(part) || part === '.' || part === '..'))
    throw new TypeError('path must be a relative JSON file path without traversal or URL syntax');
  const maxBytes = source.maxBytes === undefined ? WSP_LIMITS.maxModelBytes : source.maxBytes;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > WSP_LIMITS.maxModelBytes)
    throw new RangeError(`maxBytes must be in 1..${WSP_LIMITS.maxModelBytes}`);
  return { repo: source.repo, revision: source.revision, path: source.path, sha256: source.sha256, maxBytes };
}

async function fileHash(bytes) {
  let provider = globalThis.crypto;
  if (!provider?.subtle && typeof process === 'object' && process.versions?.node) {
    const moduleName = 'node:crypto';
    provider = (await import(moduleName)).webcrypto;
  }
  if (!provider?.subtle) throw new Error('WebCrypto is required in a secure context');
  const digest = await provider.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}

// Enforce our deadline even when a caller's trusted transport does not honor
// AbortSignal. Cancellation is cooperative; the caller-facing promise is bounded.
function abortable(promise, signal) {
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise((resolve, reject) => {
    const cleanup = () => signal.removeEventListener('abort', abort);
    const abort = () => { cleanup(); reject(signal.reason); };
    signal.addEventListener('abort', abort, { once: true });
    Promise.resolve(promise).then(value => { cleanup(); resolve(value); }, error => { cleanup(); reject(error); });
  });
}

async function readBytes(response, maximum, signal) {
  if (!response.body || typeof response.body.getReader !== 'function') throw new TypeError('Hub response requires a readable stream');
  const reader = response.body.getReader();
  try {
    const length = response.headers.get('content-length');
    if (length !== null && (!/^\d+$/.test(length) || Number(length) > maximum))
      throw new RangeError('Hub response Content-Length exceeds the byte limit or is invalid');
    const bytes = new Uint8Array(maximum);
    let count = 0;
    for (;;) {
      const { done, value } = await abortable(reader.read(), signal);
      if (done) break;
      if (!(value instanceof Uint8Array)) throw new TypeError('Hub response stream must contain bytes');
      if (value.byteLength > maximum - count) throw new RangeError('Hub response exceeds the streamed byte limit');
      bytes.set(value, count);
      count += value.byteLength;
    }
    return bytes.subarray(0, count);
  } catch (error) {
    // An injected transport may also ignore cancellation; do not await it.
    try { Promise.resolve(reader.cancel()).catch(() => {}); } catch { /* preserve failure */ }
    throw error;
  } finally { try { reader.releaseLock(); } catch { /* an aborted custom read may still be pending */ } }
}

function redirectTarget(location, current) {
  let target;
  try { target = new URL(location, current); } catch { throw new TypeError('invalid Hub redirect'); }
  if (target.origin !== ORIGIN || target.username || target.password || target.hash)
    throw new TypeError('Hub redirects must stay on https://huggingface.co without credentials');
  return target.href;
}

/** Common pinned public-JSON transport. No provider authentication or executable formats. */
async function loadPinnedJSON(source, options = {}) {
  // Snapshot every source/configuration field before the first await.
  const pinned = snapshotSource(source);
  const configuration = record(options, ['fetch', 'signal', 'timeoutMs'], [], 'Hub options');
  const fetcher = configuration.fetch === undefined ? globalThis.fetch : configuration.fetch;
  if (typeof fetcher !== 'function') throw new TypeError('fetch must be a function');
  const timeoutMs = configuration.timeoutMs === undefined ? 15_000 : configuration.timeoutMs;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > MAX_TIMEOUT_MS)
    throw new RangeError(`timeoutMs must be in 1..${MAX_TIMEOUT_MS}`);
  const signal = configuration.signal;
  if (signal !== undefined && !(signal instanceof AbortSignal)) throw new TypeError('signal must be an AbortSignal');
  const controller = new AbortController();
  const abort = () => controller.abort(signal.reason);
  if (signal?.aborted) abort();
  else signal?.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(() => controller.abort(new Error('Hub download timeout')), timeoutMs);
  const url = `${ORIGIN}/${pinned.repo}/resolve/${pinned.revision}/${pinned.path.split('/').map(encodeURIComponent).join('/')}`;
  let current = url;
  try {
    let response;
    for (let redirects = 0; ; redirects++) {
      controller.signal.throwIfAborted();
      response = await abortable(fetcher(current, { redirect: 'manual', credentials: 'omit', referrerPolicy: 'no-referrer', signal: controller.signal }), controller.signal);
      if (!response || !response.headers || response.redirected || response.type === 'opaqueredirect')
        throw new TypeError('fetch must expose manual redirect responses');
      if (response.url && new URL(response.url).origin !== ORIGIN)
        throw new TypeError('unexpected cross-origin Hub response');
      if (response.status >= 300 && response.status < 400) {
        await abortable(response.body?.cancel(), controller.signal);
        const location = response.headers.get('location');
        if (!location || redirects >= MAX_REDIRECTS) throw new Error('Hub redirect is missing a location or exceeds the redirect limit');
        current = redirectTarget(location, current);
        continue;
      }
      if (response.status !== 200) {
        await abortable(response.body?.cancel(), controller.signal);
        throw new Error(`Hub download failed with HTTP ${response.status}`);
      }
      break;
    }
    const bytes = await readBytes(response, pinned.maxBytes, controller.signal);
    controller.signal.throwIfAborted();
    const sha256 = await fileHash(bytes);
    if (sha256 !== pinned.sha256) throw new Error('Hub file SHA-256 mismatch');
    let text;
    try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes); }
    catch { throw new TypeError('Hub file must contain valid UTF-8'); }
    const document = JSON.parse(text);
    controller.signal.throwIfAborted();
    return { document,
      provenance: { repo: pinned.repo, revision: pinned.revision, path: pinned.path, sha256, bytes: bytes.length, url } };
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}


/** Fetch and validate an already explicit ruv-fann.dense-layers/1 document. */
export async function loadHubModel(source, options) {
  const { document: model, provenance } = await loadPinnedJSON(source, options);
  const inspection = await inspectWeightModel(model);
  return { schema: 'ruv-fann.hub-model/1', model, inspection, provenance };
}

/** Explicit affine interpretation only: sum(weights[i]*input[i])+bias. */
export async function loadHubLinearHead(source, options) {
  const { document, provenance } = await loadPinnedJSON(source, options);
  const head = record(document, ['weights', 'bias'], ['weights', 'bias'], 'linear head');
  if (!Array.isArray(head.weights) || head.weights.length < 1 || head.weights.length > WSP_LIMITS.maxParameters - 1)
    throw new RangeError(`linear head weights must contain 1..${WSP_LIMITS.maxParameters - 1} values`);
  const finite = value => {
    if (typeof value !== 'number' || !Number.isFinite(value)) throw new TypeError('linear head parameters must be finite numbers');
    return value === 0 ? 0 : value;
  };
  const weights = head.weights.map(finite), bias = finite(head.bias);
  const model = { schema: 'ruv-fann.dense-layers/1', layers: [{
    inputs: weights.length, outputs: 1, weights, bias: [bias], activation: 'Linear', steepness: 1
  }] };
  const inspection = await inspectWeightModel(model);
  return { schema: 'ruv-fann.hub-linear-head/1', interpretation: 'affine-projection', model, inspection, provenance };
}
