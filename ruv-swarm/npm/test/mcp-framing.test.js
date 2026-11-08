import assert from 'node:assert/strict';
import { test } from 'node:test';
import { PassThrough, Writable } from 'node:stream';
import { createBoundedLineDecoder } from '../src/bounded-lines.js';
import { createStdioRpcServer } from '../src/stdio-rpc.js';

function decoder(maxBytes = 32) {
  const lines = [];
  const errors = [];
  return { lines, errors, reader: createBoundedLineDecoder({
    maxBytes, onLine: line => lines.push(line), onError: error => errors.push(error),
  }) };
}

test('frames survive chunk boundaries and CRLF JSON parsing', () => {
  const { reader, lines, errors } = decoder();
  reader.write('{"id":');
  reader.write('1}\r\n{"id":2}\n');
  assert.deepEqual(lines.map(JSON.parse), [{ id: 1 }, { id: 2 }]);
  assert.equal(errors.length, 0);
});

test('UTF-8 byte limit is enforced before parsing a complete frame', () => {
  const { reader, lines, errors } = decoder(8);
  reader.write('ééé');
  reader.write('é\n');
  reader.write('ééééé\n');
  reader.write('{}\n');
  assert.deepEqual(lines, ['éééé', '{}']);
  assert.equal(errors.length, 1);
});

test('oversized partial frames are discarded until newline and recover', () => {
  const { reader, lines, errors } = decoder(16);
  for (let i = 0; i < 2000; i++) {
    reader.write('A'.repeat(1024));
  }
  assert.equal(errors.length, 1);
  reader.write('\n{"id":1}\n');
  assert.deepEqual(lines, ['{"id":1}']);
});

test('multiple oversized frames in one chunk each produce one error', () => {
  const { reader, lines, errors } = decoder(8);
  reader.write(`${'x'.repeat(100) }\n${ 'y'.repeat(100) }\n{}\n`);
  assert.equal(errors.length, 2);
  assert.deepEqual(lines, ['{}']);
});

test('EOF discards an unfinished frame', () => {
  const { reader, lines } = decoder();
  reader.write('{"secret":"partial"}');
  reader.end();
  reader.write('{}\n');
  assert.deepEqual(lines, ['{}']);
});

test('invalid limits are rejected', () => {
  for (const maxBytes of [0, -1, NaN, Infinity, 1.5]) {
    assert.throws(() => decoder(maxBytes), RangeError);
  }
});

test('stdio drains pending handlers and slow output at EOF without replying to notifications', async() => {
  const input = new PassThrough();
  const frames = [];
  let release;
  const gate = new Promise(resolve => {
    release = resolve;
  });
  let calls = 0;
  const output = new Writable({ write(chunk, _encoding, callback) {
    frames.push(JSON.parse(chunk.toString()));
    setTimeout(callback, 5);
  } });
  const server = createStdioRpcServer({ input, output, async handle(request) {
    calls++; await gate;
    return { jsonrpc: '2.0', id: request.id, result: request.method };
  } });
  input.end(`${[
    { jsonrpc: '2.0', method: 'notifications/initialized' },
    { jsonrpc: '2.0', id: 1, method: 'ping' },
    { jsonrpc: '2.0', method: 'ping' },
    { jsonrpc: '2.0', id: 2, method: 'tools/list' },
  ].map(JSON.stringify).join('\n') }\n`);
  let done = false;
  server.done.then(() => {
    done = true;
  });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(done, false);
  release();
  await server.done;
  assert.equal(calls, 3);
  assert.deepEqual(frames.map(frame => frame.id), [1, 2]);
});

test('stdio bounds outgoing responses and validates envelope IDs', async() => {
  const input = new PassThrough();
  const frames = [];
  const output = new Writable({ write(chunk, _encoding, callback) {
    frames.push(JSON.parse(chunk.toString())); callback();
  } });
  const server = createStdioRpcServer({ input, output, maxFrameBytes: 1024,
    handle: request => ({ jsonrpc: '2.0', id: request.id, result: 'x'.repeat(2048) }) });
  input.end(`${[
    JSON.stringify({ jsonrpc: '2.0', id: 'x'.repeat(257), method: 'ping' }),
    JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'ping' }),
    '{bad json}',
  ].join('\n') }\n`);
  await server.done;
  assert.equal(frames.find(frame => frame.id === 2).error.code, -32001);
  assert.ok(frames.some(frame => frame.error.code === -32600 && frame.id === null));
  assert.ok(frames.some(frame => frame.error.code === -32700));
  assert.ok(frames.every(frame => Buffer.byteLength(JSON.stringify(frame)) < 1024));
});

test('stdio fails closed when a blocked writer exhausts the bounded output queue', async() => {
  const input = new PassThrough();
  let release;
  const output = new Writable({ write(_chunk, _encoding, callback) {
    release = callback;
  } });
  const server = createStdioRpcServer({ input, output, maxFrameBytes: 256, maxQueuedBytes: 400,
    handle: request => ({ jsonrpc: '2.0', id: request.id, result: null }) });
  const first = server.send({ jsonrpc: '2.0', id: 1, result: 'x'.repeat(120) });
  const second = server.send({ jsonrpc: '2.0', id: 2, result: 'x'.repeat(120) });
  const rejectedSecond = assert.rejects(second, /queue limit/);
  await assert.rejects(server.send({ jsonrpc: '2.0', id: 3, result: 'x'.repeat(120) }), /queue limit/);
  await assert.rejects(server.done, /queue limit/);
  await rejectedSecond;
  assert.equal(input.isPaused(), true);
  release();
  await first;
});
