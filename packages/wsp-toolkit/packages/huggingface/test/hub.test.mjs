import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { loadHubModel, loadHubLinearHead } from '../src/index.js';
import { traceWeightModel } from 'ruv-fann-wsp';

const model = { schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 1, outputs: 1, weights: [2], bias: [0], activation: 'Linear', steepness: 1 }
] };
const encode = value => new TextEncoder().encode(value);
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const payload = encode(JSON.stringify(model, null, 2));
const source = (bytes = payload) => ({ repo: 'example/dense', revision: 'a'.repeat(40), path: 'checkpoints/model.json', sha256: digest(bytes) });
const response = bytes => new Response(bytes, { status: 200, headers: { 'content-type': 'text/plain' } });

// These are contract tests against an injectable transport, not a live Hub receipt.
test('loads raw-byte pinned JSON, validates model and records independent hashes', async () => {
  let calls = 0;
  const result = await loadHubModel(source(), { fetch: async (url, init) => {
    calls++;
    assert.equal(url, `https://huggingface.co/example/dense/resolve/${'a'.repeat(40)}/checkpoints/model.json`);
    assert.equal(init.redirect, 'manual');
    assert.equal(init.credentials, 'omit');
    assert.equal(init.referrerPolicy, 'no-referrer');
    assert.ok(init.signal instanceof AbortSignal);
    assert.equal(init.headers, undefined);
    return response(payload);
  } });
  assert.equal(calls, 1);
  assert.deepEqual(result.model, model);
  assert.equal(result.schema, 'ruv-fann.hub-model/1');
  assert.equal(result.provenance.sha256, digest(payload));
  assert.equal(result.provenance.bytes, payload.length);
  assert.notEqual(result.provenance.sha256, result.inspection.baseHash);
  assert.equal(result.inspection.parameterCount, 2);
});

test('source and options are snapshotted before fetch yields', async () => {
  const mutable = source(), options = { timeoutMs: 1000 };
  options.fetch = async () => {
    mutable.sha256 = 'f'.repeat(64); mutable.repo = 'evil/replacement'; mutable.revision = 'b'.repeat(40);
    options.timeoutMs = 0;
    return response(payload);
  };
  const result = await loadHubModel(mutable, options);
  assert.equal(result.provenance.repo, 'example/dense');
  assert.equal(result.provenance.revision, 'a'.repeat(40));
  assert.equal(result.provenance.sha256, digest(payload));
});

test('rejects mutable revisions, traversal, URL syntax, credentials and raised limits before networking', async () => {
  const mutations = [
    { revision: 'main' }, { revision: 'a'.repeat(39) }, { revision: 'A'.repeat(40) },
    { path: '../model.json' }, { path: 'a/../model.json' }, { path: '/model.json' },
    { path: 'a//model.json' }, { path: 'a\\model.json' }, { path: '%2e%2e/model.json' },
    { path: 'model.json?token=secret' }, { path: 'model.json#fragment' }, { path: 'model.safetensors' },
    { repo: 'https://evil.test/model' }, { repo: 'user/repo/extra' }, { repo: 'a@b/c' },
    { sha256: '' }, { sha256: 'A'.repeat(64) }, { maxBytes: 0 }, { maxBytes: 1024 * 1024 + 1 },
    { maxBytes: Infinity }, { headers: { Authorization: 'unused' } }
  ];
  let called = false;
  for (const fields of mutations) await assert.rejects(loadHubModel({ ...source(), ...fields }, {
    fetch: async () => { called = true; return response(payload); }
  }));
  assert.equal(called, false);
});

test('rejects accessors and symbols without invoking application getters', async () => {
  let reads = 0;
  const spec = source();
  Object.defineProperty(spec, 'path', { get() { reads++; return 'model.json'; } });
  await assert.rejects(loadHubModel(spec), /accessors/);
  await assert.rejects(loadHubModel(source(), { get fetch() { reads++; return async () => response(payload); } }), /accessors/);
  const symbol = source(); symbol[Symbol('hidden')] = true;
  await assert.rejects(loadHubModel(symbol), /unknown/);
  assert.equal(reads, 0);
});

test('detects digest mismatch before decoding or parsing', async () => {
  const invalid = encode('this is not JSON');
  await assert.rejects(loadHubModel(source(), { fetch: async () => response(invalid) }), /SHA-256 mismatch/);
});

test('rejects invalid UTF-8, malformed JSON and unsupported checkpoints after matching bytes', async () => {
  const cases = [
    [new Uint8Array([0xc3, 0x28]), /valid UTF-8/],
    [encode('{'), /JSON|position|property|end/i],
    [encode('{"schema":"pytorch","state_dict":{}}'), /unknown|schema|missing/],
    [encode(JSON.stringify({ ...model, ignored: true })), /unknown/],
    [encode(JSON.stringify({ ...model, layers: [{ ...model.layers[0], weights: ['2'] }] })), /finite/]
  ];
  for (const [bytes, pattern] of cases)
    await assert.rejects(loadHubModel(source(bytes), { fetch: async () => response(bytes) }), pattern);
});

test('bounds declared Content-Length and cancels before reading', async () => {
  let canceled = false;
  const body = new ReadableStream({ cancel() { canceled = true; } });
  await assert.rejects(loadHubModel({ ...source(), maxBytes: 10 }, {
    fetch: async () => new Response(body, { headers: { 'content-length': '1000000000' } })
  }), /Content-Length/);
  assert.equal(canceled, true);
});

test('bounds streamed bytes with missing or misleading Content-Length', async () => {
  for (const length of [undefined, '1']) {
    let canceled = false;
    const body = new ReadableStream({ start(controller) {
      controller.enqueue(new Uint8Array(4)); controller.enqueue(new Uint8Array(7));
    }, cancel() { canceled = true; } });
    await assert.rejects(loadHubModel({ ...source(), maxBytes: 10 }, {
      fetch: async () => new Response(body, { headers: length === undefined ? {} : { 'content-length': length } })
    }), /streamed byte limit/);
    assert.equal(canceled, true);
  }
});

test('permits bounded same-origin resolve-cache redirects and preserves immutable source URL', async () => {
  const calls = [];
  const result = await loadHubModel(source(), { fetch: async (url, init) => {
    calls.push(url); assert.equal(init.credentials, 'omit');
    return calls.length === 1 ? new Response(null, { status: 307, headers: { location: '/api/resolve-cache/models/example/dense/file' } }) : response(payload);
  } });
  assert.equal(calls.length, 2);
  assert.match(calls[1], /^https:\/\/huggingface.co\/api\/resolve-cache\//);
  assert.equal(result.provenance.url, calls[0]);
});

test('refuses cross-origin, credentialed, fragment and opaque redirects', async () => {
  for (const location of ['https://cdn-lfs.hf.co/model.json', 'http://huggingface.co/model.json',
    'https://user:secret@huggingface.co/model.json', '/model.json#fragment']) {
    let count = 0;
    await assert.rejects(loadHubModel(source(), { fetch: async () => {
      count++; return new Response(null, { status: 302, headers: { location } });
    } }), /redirects must stay/);
    assert.equal(count, 1);
  }
  const followed = response(payload); Object.defineProperty(followed, 'redirected', { value: true });
  await assert.rejects(loadHubModel(source(), { fetch: async () => followed }), /manual redirect/);
});

test('rejects redirect loops, partial responses and failed fetches', async () => {
  let calls = 0;
  await assert.rejects(loadHubModel(source(), { fetch: async () => {
    calls++; return new Response(null, { status: 307, headers: { location: '/loop' } });
  } }), /redirect limit/);
  assert.equal(calls, 4);
  for (const status of [206, 401, 404, 500]) await assert.rejects(loadHubModel(source(), {
    fetch: async () => new Response(null, { status })
  }), new RegExp(`HTTP ${status}`));
  await assert.rejects(loadHubModel(source(), { fetch: async () => { throw new Error('offline'); } }), /offline/);
});

test('supports pre-abort and enforces a finite download deadline', async () => {
  const controller = new AbortController(); controller.abort(new Error('cancelled'));
  let called = false;
  await assert.rejects(loadHubModel(source(), { signal: controller.signal, fetch: async () => {
    called = true; return response(payload);
  } }), /cancelled/);
  assert.equal(called, false);
  await assert.rejects(loadHubModel(source(), { timeoutMs: 5, fetch: async (_url, { signal }) =>
    new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(signal.reason), { once: true }))
  }), /timeout/);
  for (const timeoutMs of [0, 60_001, Infinity, 1.1])
    await assert.rejects(loadHubModel(source(), { timeoutMs }), /timeoutMs/);
});


test('explicit linear-head adapter matches independent affine oracle using real WASM', async () => {
  const bytes = encode(JSON.stringify({ weights: [2, -3, 0.5], bias: 4 }));
  const result = await loadHubLinearHead(source(bytes), { fetch: async () => response(bytes) });
  assert.equal(result.schema, 'ruv-fann.hub-linear-head/1');
  assert.equal(result.interpretation, 'affine-projection');
  assert.equal(result.provenance.sha256, digest(bytes));
  assert.equal(result.inspection.inputSize, 3);
  assert.equal(result.inspection.parameterCount, 4);
  assert.deepEqual(result.model.layers[0], { inputs: 3, outputs: 1, weights: [2, -3, 0.5], bias: [4], activation: 'Linear', steepness: 1 });
  const inputs = [1, 2, 3, -2, 4, 0, 0, 0, 0];
  const trace = await traceWeightModel(result.model, { inputs, rows: 3 }, { simd: false });
  assert.equal(trace.backend, 'wasm');
  const oracle = [2 * 1 - 3 * 2 + 0.5 * 3 + 4, 2 * -2 - 3 * 4 + 0.5 * 0 + 4, 4];
  assert.deepEqual(trace.layers[0].preactivations, oracle);
  assert.deepEqual(trace.layers[0].activations, oracle);
  // The generic dense loader never silently infers this different document format.
  await assert.rejects(loadHubModel(source(bytes), { fetch: async () => response(bytes) }), /unknown|missing/);
});

test('linear-head adapter rejects unknown fields, nonfinite parameters and invalid lengths', async () => {
  const invalid = [
    '{"weights":[1],"bias":0,"activation":"Sigmoid"}',
    '{"weights":[1]}', '{"weights":[],"bias":0}',
    '{"weights":["1"],"bias":0}', '{"weights":[1e309],"bias":0}',
    '{"weights":[1],"bias":1e309}', '{"weights":[1],"bias":null}',
    '{"weights":null,"bias":0}', '{"weights":[1],"bias":[0]}',
    JSON.stringify({ weights: Array(65536).fill(0), bias: 0 })
  ];
  for (const text of invalid) {
    const bytes = encode(text);
    await assert.rejects(loadHubLinearHead(source(bytes), { fetch: async () => response(bytes) }));
  }
});

test('linear-head adapter shares digest, redirect and byte-limit enforcement', async () => {
  const bytes = encode('{"weights":[1],"bias":0}');
  await assert.rejects(loadHubLinearHead(source(bytes), { fetch: async () => response(payload) }), /SHA-256 mismatch/);
  await assert.rejects(loadHubLinearHead({ ...source(bytes), maxBytes: 1 }, { fetch: async () => response(bytes) }), /byte limit/);
  await assert.rejects(loadHubLinearHead(source(bytes), { fetch: async () => new Response(null, {
    status: 307, headers: { location: 'https://other.example/model.json' }
  }) }), /redirects must stay/);
});
