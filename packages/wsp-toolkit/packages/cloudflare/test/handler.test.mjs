import test from 'node:test';
import assert from 'node:assert/strict';
import { createWorkerHandler, MAX_REQUEST_BYTES } from '../src/index.js';
import { fingerprintModel } from 'ruv-fann-wsp';

const token = 'test-secret-do-not-use-in-production';
const model = () => ({ schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 1, outputs: 1, weights: [0], bias: [0], activation: 'Linear', steepness: 1 }
] });
const data = { inputs: [1, 2], targets: [2, 4], rows: 2 };
const request = (path, body, headers = {}) => new Request(`https://worker.test${path}`, {
  method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${token}`, ...headers },
  body: typeof body === 'string' ? body : JSON.stringify(body)
});

test('handler requires configured authentication and never evaluates unauthorized requests', async () => {
  assert.equal((await createWorkerHandler().fetch(request('/evaluate', {}))).status, 503);
  const handler = createWorkerHandler({ bearerToken: token });
  assert.equal((await handler.fetch(request('/evaluate', {}, { authorization: 'Bearer wrong' }))).status, 401);
  assert.equal((await handler.fetch(request('/evaluate', {}, { authorization: '' }))).status, 401);
  const health = await handler.fetch(new Request('https://worker.test/health'));
  assert.equal(health.status, 200);
  assert.equal((await health.json()).authenticatedOperations, true);
  assert.equal(health.headers.get('cache-control'), 'no-store');
  assert.throws(() => createWorkerHandler({ bearerToken: 'short' }), /24/);
  assert.throws(() => createWorkerHandler({ bearerToken: 'x'.repeat(23) + '\n' }), /ASCII/);
  assert.throws(() => createWorkerHandler({ allowUnauthenticated: 'yes' }), /boolean/);
  assert.throws(() => createWorkerHandler({ maxRequestBytes: MAX_REQUEST_BYTES + 1 }), /maxRequestBytes/);
  let getterCalls = 0;
  assert.throws(() => createWorkerHandler({ get bearerToken() { getterCalls++; return token; } }), /accessor/);
  assert.equal(getterCalls, 0);
});

test('HTTP routing, JSON encoding and unknown fields fail before WASM', async () => {
  const handler = createWorkerHandler({ bearerToken: token });
  assert.equal((await handler.fetch(new Request('https://worker.test/evaluate'))).status, 405);
  assert.equal((await handler.fetch(request('/promote', {}))).status, 404);
  assert.equal((await handler.fetch(request('/evaluate?url=https://elsewhere.test', {}))).status, 400);
  assert.equal((await handler.fetch(request('/evaluate', '{}', { 'content-type': 'text/plain' }))).status, 415);
  assert.equal((await handler.fetch(request('/evaluate', 'not-json'))).status, 400);
  assert.equal((await handler.fetch(request('/evaluate', { unknown: true }))).status, 422);
  const invalidUTF8 = new Request('https://worker.test/trace', {
    method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${token}` },
    body: Uint8Array.of(0xff)
  });
  assert.equal((await handler.fetch(invalidUTF8)).status, 400);
});

test('streamed requests are bounded independently of content-length', async () => {
  let canceled = false;
  const handler = createWorkerHandler({ bearerToken: token, maxRequestBytes: 20 });
  const stream = new ReadableStream({
    start(controller) { controller.enqueue(new Uint8Array(21)); },
    cancel() { canceled = true; }
  });
  const body = new Request('https://worker.test/trace', {
    method: 'POST', duplex: 'half', headers: { 'content-type': 'application/json', authorization: `Bearer ${token}` }, body: stream
  });
  assert.equal((await handler.fetch(body)).status, 413);
  assert.equal(canceled, true);
  assert.equal((await handler.fetch(request('/trace', '{}', { 'content-length': '21' }))).status, 413);
});

test('fragmented body grows bounded storage without retaining per-chunk objects', async () => {
  const handler = createWorkerHandler({ bearerToken: token });
  const bytes = new TextEncoder().encode(JSON.stringify({ model: model(), batch: { inputs: [1], rows: 1 } }) + ' '.repeat(9000));
  let offset = 0;
  const stream = new ReadableStream({
    pull(controller) {
      if (offset === bytes.length) controller.close();
      else controller.enqueue(bytes.slice(offset, ++offset));
    }
  });
  const request = new Request('https://worker.test/trace', { method: 'POST', duplex: 'half', body: stream,
    headers: { 'content-type': 'application/json', authorization: `Bearer ${token}` } });
  const result = await handler.fetch(request);
  assert.equal(result.status, 200);
  assert.deepEqual((await result.json()).layers[0].activations, [0]);
});

test('real WASM evaluation distinguishes acceptance and regression without state mutation', async () => {
  const source = model();
  const baseHash = await fingerprintModel(source);
  const patch = { schema: 'ruv-fann.weight-patch/1', baseHash,
    edits: [{ layer: 0, parameter: 'weights', index: 0, value: 2 }] };
  const handler = createWorkerHandler({ bearerToken: token });
  const result = await handler.fetch(request('/evaluate', { model: source, patch, dataset: data,
    policy: { maxMSE: 0, maxRegressionMSE: 0 }, options: { simd: false } }));
  assert.equal(result.status, 200);
  const accepted = await result.json();
  assert.equal(accepted.accepted, true);
  assert.equal(accepted.backend, 'wasm');
  assert.equal(accepted.candidate.mse, 0);
  const rejected = await handler.fetch(request('/evaluate', { model: source, patch, dataset: { ...data, targets: [0, 0] },
    policy: { maxMSE: 100, maxRegressionMSE: 0 } }));
  assert.equal((await rejected.json()).accepted, false);
  assert.equal(source.layers[0].weights[0], 0);
  const stale = await handler.fetch(request('/evaluate', { model: source, patch: { ...patch, baseHash: '0'.repeat(64) },
    dataset: data, policy: { maxMSE: 0, maxRegressionMSE: 0 } }));
  assert.equal(stale.status, 422);
});
