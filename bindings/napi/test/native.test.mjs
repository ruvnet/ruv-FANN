/** These tests intentionally bypass every JS loader and call the real .node exports. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { Worker } from 'node:worker_threads';
import vm from 'node:vm';
import { dense, network, expected } from './fixtures.mjs';

const binary = fileURLToPath(new URL('../ruv_fann.node', import.meta.url));
const { NativePreparedNetwork } = createRequire(import.meta.url)(binary);
const model = () => NativePreparedNetwork.from_dense_layers_json(dense());

test('actual dense/native JSON imports preserve deterministic f64 inference including sparse/duplicate order', () => {
  for (const prepared of [model(), NativePreparedNetwork.from_network_json(network()),
    NativePreparedNetwork.from_network_json(network({ sparse: true }))]) {
    try {
      assert.equal(prepared.input_size, 2); assert.equal(prepared.output_size, 2);
      assert.ok(prepared.estimated_bytes > 0 && prepared.estimated_bytes <= 64 * 1024 * 1024);
      assert.deepEqual([...prepared.run(new Float64Array([1, 2]))], expected([1, 2]));
      assert.ok(prepared.run(new Float64Array([2, -1])) instanceof Float64Array);
    } finally { prepared.free(); }
  }
});

test('output staging preserves caller memory on shape, nonfinite and overflow errors', () => {
  const prepared = model(), output = new Float64Array([77, 88]);
  try {
    for (const input of [new Float64Array(3), new Float64Array([NaN, 1]), new Float64Array([Infinity, 2])]) {
      assert.throws(() => prepared.run_into(input, output)); assert.deepEqual([...output], [77, 88]);
    }
    assert.throws(() => prepared.run_into(new Float64Array([1, 2]), new Float64Array(1)));
    const overflow = NativePreparedNetwork.from_dense_layers_json(dense({ weights: [1e308, 1e308, 1, 3] }));
    try { assert.throws(() => overflow.run_into(new Float64Array([2, 2]), output)); assert.deepEqual([...output], [77, 88]); }
    finally { overflow.free(); }
  } finally { prepared.free(); }
});

test('input/output aliases and partially overlapping views are supported without Rust slice aliasing', () => {
  const prepared = model();
  try {
    const alias = new Float64Array([1, 2]); prepared.run_into(alias, alias); assert.deepEqual([...alias], expected([1, 2]));
    const buffer = new Float64Array([1, 2, 0]);
    prepared.run_into(buffer.subarray(0, 2), buffer.subarray(1, 3));
    assert.deepEqual([...buffer], [1, ...expected([1, 2])]);
  } finally { prepared.free(); }
});

test('raw addon rejects wrong-type/proxy/tag-spoof/shared/detached/resizable arrays before unsafe copying', () => {
  const prepared = model(), valid = new Float64Array([1, 2]), output = new Float64Array([77, 88]);
  try {
    const detached = new Float64Array(2); structuredClone(detached.buffer, { transfer: [detached.buffer] });
    const resizable = new ArrayBuffer(16, { maxByteLength: 32 });
    Object.defineProperty(resizable, 'resizable', { value: false });
    const bad = [new Float32Array(2), [1, 2], { length: 2, [Symbol.toStringTag]: 'Float64Array' },
      Object.create(Float64Array.prototype), new Proxy(valid, {}), new Float64Array(new SharedArrayBuffer(16)),
      detached, new Float64Array(resizable), new Float64Array(64 * 1024 * 1024 / 8 + 1)];
    for (const input of bad) {
      assert.throws(() => prepared.run(input));
      assert.throws(() => prepared.run_into(input, output)); assert.deepEqual([...output], [77, 88]);
    }
    for (const destination of bad) assert.throws(() => prepared.run_into(valid, destination));
    const spoof = new Float64Array([1, 2]);
    Object.defineProperty(spoof, 'buffer', { get() { throw new Error('instance buffer getter must not run'); } });
    Object.defineProperty(spoof, 'length', { get() { throw new Error('instance length getter must not run'); } });
    Object.defineProperty(spoof, Symbol.toStringTag, { value: 'Float32Array' });
    assert.deepEqual([...prepared.run(spoof)], expected([1, 2]));
  } finally { prepared.free(); }
});

test('cross-realm genuine fixed Float64Arrays pass intrinsic Node-API checks', () => {
  const prepared = model();
  try { assert.deepEqual([...prepared.run(vm.runInNewContext('new Float64Array([1,2])'))], expected([1, 2])); }
  finally { prepared.free(); }
});

test('JSON input is a primitive bounded UTF8 string, never a coercion or early Rust String copy', () => {
  for (const value of [undefined, null, {}, new String(dense()), { toString() { throw new Error('coercion'); } },
    ' '.repeat(4 * 1024 * 1024 + 1), 'é'.repeat(2 * 1024 * 1024 + 1), '{}', '[]'])
    assert.throws(() => NativePreparedNetwork.from_dense_layers_json(value));
  for (const text of [dense({ weights: [1] }), dense({ inputs: -1 }), dense({ activation: 'FutureActivation' }),
    dense({ unexpected: 1 }), dense({ weights: [Infinity, 1, 1, 3] })])
    assert.throws(() => NativePreparedNetwork.from_dense_layers_json(text));
});

test('batch inherits immutable snapshot after parent free and handles exact tails/zero/aliases', () => {
  const prepared = model(), batch = prepared.create_batch(3), other = prepared.create_batch(1);
  assert.equal(batch.input_size, 2); assert.equal(batch.output_size, 2); assert.equal(batch.max_rows, 3);
  prepared.free(); prepared.free(); assert.throws(() => prepared.run(new Float64Array(2)), /freed/);
  try {
    const input = new Float64Array([1, 2, 3, 4]), output = new Float64Array(4);
    batch.run_into(input, output, 2); assert.deepEqual([...output], [...expected([1, 2]), ...expected([3, 4])]);
    batch.run_into(input, input, 2); assert.deepEqual([...input], [...expected([1, 2]), ...expected([3, 4])]);
    batch.run_into(new Float64Array(), new Float64Array(), 0);
    const one = new Float64Array(2); other.run_into(new Float64Array([2, 3]), one, 1); assert.deepEqual([...one], expected([2, 3]));
    const sentinel = new Float64Array([77, 88, 99, 100]);
    assert.throws(() => batch.run_into(new Float64Array([1, 2, NaN, 4]), sentinel, 2));
    assert.deepEqual([...sentinel], [77, 88, 99, 100]);
    const emptyDetached = new Float64Array(); structuredClone(emptyDetached.buffer, { transfer: [emptyDetached.buffer] });
    assert.throws(() => batch.run_into(emptyDetached, new Float64Array(), 0));
  } finally { batch.free(); other.free(); }
  batch.free(); assert.throws(() => batch.run_into(new Float64Array(), new Float64Array(), 0), /freed/);
});

test('capacity/row primitives are strictly bounded inside direct native methods', () => {
  const prepared = model(), batch = prepared.create_batch(2);
  try {
    for (const value of [0, -1, 4097, NaN, Infinity, 1.5, '2', 2n, new Number(2), undefined])
      assert.throws(() => prepared.create_batch(value));
    for (const value of [-1, 3, NaN, Infinity, 0.5, '1', 1n, new Number(1), undefined])
      assert.throws(() => batch.run_into(new Float64Array(2), new Float64Array(2), value));
  } finally { batch.free(); prepared.free(); }
});

test('reentrant cached getter free/detach happens before native mutable state borrowing', () => {
  const script = `
    const assert=require('node:assert/strict');
    const original=Object.getOwnPropertyDescriptor(ArrayBuffer.prototype,'resizable'); let callback;
    Object.defineProperty(ArrayBuffer.prototype,'resizable',{...original,get(){const fn=callback;callback=null;if(fn)fn();return original.get.call(this)}});
    const {NativePreparedNetwork}=require(${JSON.stringify(binary)});
    const model=()=>NativePreparedNetwork.from_dense_layers_json(${JSON.stringify(dense())});
    let prepared=model(), output=new Float64Array([77,88]);
    callback=()=>prepared.free();
    assert.throws(()=>prepared.run_into(new Float64Array([1,2]),output),/freed/);assert.deepEqual([...output],[77,88]);
    prepared=model();const input=new Float64Array([1,2]);callback=()=>structuredClone(input.buffer,{transfer:[input.buffer]});
    assert.throws(()=>prepared.run_into(input,output));assert.deepEqual([...output],[77,88]);prepared.free();
    console.log('reentrant-free-detach-passed');
  `;
  const child = spawnSync(process.execPath, ['-e', script], { encoding: 'utf8', timeout: 15000 });
  assert.equal(child.status, 0, child.stderr); assert.match(child.stdout, /passed/);
});

test('each Node Worker owns its getter references and teardown leaves no cross-Env pointers', async () => {
  for (let index = 0; index < 4; index++) {
    const answer = await new Promise((resolve, reject) => {
      const worker = new Worker(`const{parentPort}=require('node:worker_threads');const{NativePreparedNetwork}=require(${JSON.stringify(binary)});const p=NativePreparedNetwork.from_dense_layers_json(${JSON.stringify(dense())});parentPort.postMessage([...p.run(new Float64Array([1,2]))]);p.free();`, { eval: true });
      let message;
      worker.on('message', value => { message = value; }); worker.on('error', reject);
      worker.on('exit', code => code === 0 ? resolve(message) : reject(new Error('Worker exit ' + code)));
    });
    assert.deepEqual(answer, expected([1, 2]));
  }
});
