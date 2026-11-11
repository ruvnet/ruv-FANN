import {
  evaluateWeightPatch, traceWeightModel, analyzeWeightSensitivity, createNeuronAblationPatch
} from 'ruv-fann-wsp';

export const MAX_REQUEST_BYTES = 1024 * 1024;

class HTTPError extends Error {
  constructor(status, code, message) { super(message); this.status = status; this.code = code; }
}

function plain(value, allowed, required, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value)))
    throw new TypeError(`${name} must be a plain object`);
  const output = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !allowed.includes(key) || !Object.hasOwn(descriptor, 'value'))
      throw new TypeError(`unknown or accessor ${name} field`);
    output[key] = descriptor.value;
  }
  if (required.some(key => !Object.hasOwn(output, key))) throw new TypeError(`missing ${name} field`);
  return output;
}

function response(body, status = 200, extra = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff', ...extra }
  });
}

// Equal-length token comparison without an early exit on a mismatched character.
function tokenMatches(value, token) {
  if (typeof value !== 'string' || value.length !== token.length) return false;
  let different = 0;
  for (let index = 0; index < token.length; index++) different |= value.charCodeAt(index) ^ token.charCodeAt(index);
  return different === 0;
}

async function readJSON(request, limit) {
  const type = request.headers.get('content-type');
  if (!type || !/^application\/json(?:\s*;\s*charset\s*=\s*(?:utf-8|"utf-8"))?\s*$/i.test(type))
    throw new HTTPError(415, 'unsupported_media_type', 'Content-Type must be application/json with UTF-8 encoding');
  const declared = request.headers.get('content-length');
  if (declared !== null && (!/^(0|[1-9][0-9]*)$/.test(declared) || Number(declared) > limit))
    throw new HTTPError(413, 'body_too_large', 'Request body exceeds byte limit');
  if (!request.body) throw new HTTPError(400, 'invalid_json', 'A JSON body is required');
  const reader = request.body.getReader();
  // Geometric storage bounds overhead even if a client fragments into one-byte chunks.
  let bytes = new Uint8Array(Math.min(limit, 8192));
  let size = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) {
        await reader.cancel('request body exceeds byte limit');
        throw new HTTPError(413, 'body_too_large', 'Request body exceeds byte limit');
      }
      if (size > bytes.length) {
        const grown = new Uint8Array(Math.min(limit, Math.max(size, bytes.length * 2)));
        grown.set(bytes);
        bytes = grown;
      }
      bytes.set(value, size - value.byteLength);
    }
  } finally { reader.releaseLock(); }
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(0, size))); }
  catch { throw new HTTPError(400, 'invalid_json', 'Request must contain valid UTF-8 JSON'); }
}

const shapes = Object.freeze({
  '/evaluate': ['model', 'patch', 'dataset', 'policy', 'options'],
  '/trace': ['model', 'batch', 'options'],
  '/sensitivity': ['model', 'dataset', 'locations', 'options'],
  '/ablation': ['model', 'layer', 'neuron', 'limits']
});
const required = Object.freeze({
  '/evaluate': ['model', 'patch', 'dataset', 'policy'],
  '/trace': ['model', 'batch'],
  '/sensitivity': ['model', 'dataset', 'locations'],
  '/ablation': ['model', 'layer', 'neuron']
});

/** Stateless adapter. Authentication defaults to unavailable until a token is configured. */
export function createWorkerHandler(options = {}) {
  const config = plain(options, ['bearerToken', 'allowUnauthenticated', 'maxRequestBytes'], [], 'handler options');
  if (config.bearerToken !== undefined &&
      (typeof config.bearerToken !== 'string' || config.bearerToken.length < 24 || config.bearerToken.length > 1024 ||
       !/^[\x21-\x7e]+$/.test(config.bearerToken)))
    throw new TypeError('bearerToken must be 24..1024 printable non-space ASCII characters');
  if (config.allowUnauthenticated !== undefined && typeof config.allowUnauthenticated !== 'boolean')
    throw new TypeError('allowUnauthenticated must be boolean');
  const maxBytes = config.maxRequestBytes ?? 512 * 1024;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > MAX_REQUEST_BYTES)
    throw new RangeError(`maxRequestBytes must be an integer in 1..${MAX_REQUEST_BYTES}`);
  const token = config.bearerToken;
  const allowUnauthenticated = config.allowUnauthenticated === true;
  return Object.freeze({
    async fetch(request) {
      try {
        if (!(request instanceof Request)) throw new TypeError('request must be a Request');
        const url = new URL(request.url);
        if (url.search) throw new HTTPError(400, 'invalid_query', 'Query parameters are not supported');
        if (url.pathname === '/health') {
          if (request.method !== 'GET') return response({ error: 'method_not_allowed' }, 405, { Allow: 'GET' });
          return response({ schema: 'ruv-fann.wsp-worker-health/1', stateless: true,
            authenticatedOperations: Boolean(token), maxRequestBytes: maxBytes });
        }
        if (!Object.hasOwn(shapes, url.pathname)) throw new HTTPError(404, 'not_found', 'Unknown operation');
        if (request.method !== 'POST') return response({ error: 'method_not_allowed' }, 405, { Allow: 'POST' });
        if (!token && !allowUnauthenticated)
          throw new HTTPError(503, 'authentication_unconfigured', 'Configure WSP_BEARER_TOKEN before using operations');
        if (token && !tokenMatches(request.headers.get('authorization'), `Bearer ${token}`))
          throw new HTTPError(401, 'unauthorized', 'A valid bearer token is required');
        const body = plain(await readJSON(request, maxBytes), shapes[url.pathname], required[url.pathname], 'request');
        switch (url.pathname) {
          case '/evaluate': return response(await evaluateWeightPatch(body.model, body.patch, body.dataset, body.policy, body.options));
          case '/trace': return response(await traceWeightModel(body.model, body.batch, body.options));
          case '/sensitivity': return response(await analyzeWeightSensitivity(body.model, body.dataset, body.locations, body.options));
          case '/ablation': return response(await createNeuronAblationPatch(body.model, body.layer, body.neuron, body.limits));
        }
      } catch (error) {
        if (error instanceof HTTPError) return response({ error: error.code, message: error.message }, error.status);
        if (error instanceof TypeError || error instanceof RangeError ||
            /^patch baseHash does not match source model$/.test(error?.message))
          return response({ error: 'invalid_input', message: error.message }, 422);
        // Do not return stack traces or unexpected runtime details over the network.
        return response({ error: 'execution_failed', message: 'WSP execution failed' }, 500);
      }
    }
  });
}
