import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { RuvSwarm } from '../src/index.js';

const activations = ['Linear', 'Threshold', 'ThresholdSymmetric', 'Sigmoid', 'SigmoidSymmetric',
  'Tanh', 'Gaussian', 'GaussianSymmetric', 'Elliot', 'ElliotSymmetric', 'LinearPiece',
  'LinearPieceSymmetric', 'ReLU', 'ReLULeaky', 'Sin', 'Cos', 'SinSymmetric', 'CosSymmetric',
  'Gelu', 'Swish', 'LeakyRelu'];

function expectedRows(prepared, input, rows) {
  const expected = [];
  for (let row = 0; row < rows; row++) {
    expected.push(...prepared.run(input.subarray(row * prepared.input_size, (row + 1) * prepared.input_size)));
  }
  return expected;
}

function close(actual, expected) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) <= 1e-12 * Math.max(1, Math.abs(expected[i])), `Output ${i}`);
  }
}

for (const useSIMD of [false, true]) {
  test(`batch shares its snapshot across source frees and varying active rows (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const source = runtime.createNeuralNetwork([2, 1], 'Linear');
    source.set_weights(Float64Array.of(2, -1));
    const prepared = source.prepare();
    const batch = prepared.create_batch(7);
    const other = prepared.create_batch(3);
    assert.equal(batch.input_size, 2);
    assert.equal(batch.output_size, 1);
    assert.equal(batch.max_rows, 7);
    assert.ok(Number.isSafeInteger(batch.estimated_bytes) && batch.estimated_bytes <= 64 * 1024 * 1024);
    source.set_weights(Float64Array.of(0, 0)); source.free(); prepared.free();
    try {
      for (const rows of [7, 1, 5, 0, 2, 7]) {
        const input = Float64Array.from({ length: rows * 2 }, (_, i) => i % 2 === 0 ? 3 + i / 2 : 4);
        const output = new Float64Array(rows);
        batch.run_into(input, output, rows);
        assert.deepEqual(Array.from(output), Array.from({ length: rows }, (_, i) => 2 + 2 * i));
      }
    } finally {
      batch.free();
    }
    try {
      const output = new Float64Array(2);
      other.run_into(Float64Array.of(3, 4, 5, 6), output, 2);
      assert.deepEqual(Array.from(output), [2, 4]);
    } finally {
      other.free();
    }
  });

  test(`batch matches single inference for all21 native dense/sparse activations and odd rows (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    for (const name of ['dense', 'sparse']) {
      const fixture = JSON.parse(await readFile(new URL(`../../crates/ruv-swarm-wasm/tests/fixtures/prepared-${name}.json`, import.meta.url), 'utf8'));
      for (const activation of activations) {
        const graph = structuredClone(fixture.network);
        for (const layer of graph.layers.slice(1)) {
          for (const neuron of layer.neurons) {
            if (!neuron.is_bias) {
              neuron.activation_function = activation;
            }
          }
        }
        const prepared = runtime.createPreparedNetwork(JSON.stringify(graph));
        const batch = prepared.create_batch(7);
        try {
          for (const rows of [1, 2, 3, 5, 7, 1]) {
            const input = Float64Array.from({ length: rows * prepared.input_size }, (_, i) => fixture.input[i % prepared.input_size] + Math.floor(i / prepared.input_size) / 100);
            const expected = expectedRows(prepared, input, rows);
            const output = new Float64Array(rows * prepared.output_size);
            batch.run_into(input, output, rows);
            close(output, expected);
          }
        } finally {
          batch.free(); prepared.free();
        }
      }
    }
  });

  test(`batch keeps entire output atomic on later-row failures and supports aliases (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const source = runtime.createNeuralNetwork([1, 1], 'Linear');
    source.set_weights(Float64Array.of(Number.MAX_VALUE));
    const prepared = source.prepare(); source.free();
    const batch = prepared.create_batch(5); prepared.free();
    try {
      for (const last of [2, NaN, Infinity]) {
        const output = Float64Array.of(11, 12, 13, 14, 15);
        assert.throws(() => batch.run_into(Float64Array.of(0, 0.25, 0.5, 0.125, last), output, 5));
        assert.deepEqual(Array.from(output), [11, 12, 13, 14, 15]);
      }
      const output = new Float64Array(3);
      batch.run_into(Float64Array.of(0.5, 0.25, 0), output, 3);
      assert.deepEqual(Array.from(output), [Number.MAX_VALUE * 0.5, Number.MAX_VALUE * 0.25, 0]);
    } finally {
      batch.free();
    }
    const square = runtime.createNeuralNetwork([2, 2], 'Linear');
    square.set_weights(Float64Array.of(1, 2, 3, 4));
    const reference = square.prepare(); square.free();
    const aliases = reference.create_batch(3);
    try {
      const inPlace = Float64Array.of(3, 4, 1, 2, -1, 0.5);
      const expected = expectedRows(reference, inPlace, 3);
      aliases.run_into(inPlace, inPlace, 3);
      close(inPlace, expected);
      const storage = Float64Array.of(3, 4, 1, 2, -1, 0.5, 999);
      aliases.run_into(storage.subarray(0, 6), storage.subarray(1, 7), 3);
      close(storage, [3, ...expected]);
    } finally {
      aliases.free(); reference.free();
    }
  });

  test(`batch retains intrinsic array guards including empty views and memory growth (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const source = runtime.createNeuralNetwork([2, 1], 'Linear');
    source.set_weights(Float64Array.of(2, -1));
    const prepared = source.prepare(); source.free();
    const batch = prepared.create_batch(3); prepared.free();
    try {
      const output = Float64Array.of(123, 456);
      const detached = new Float64Array(4);
      structuredClone(detached.buffer, { transfer: [detached.buffer] });
      const disguised = new SharedArrayBuffer(32);
      Object.setPrototypeOf(disguised, ArrayBuffer.prototype);
      const { memory } = await runtime._bindings.default();
      for (const input of [[3, 4, 5, 6], Float32Array.of(3, 4, 5, 6), new BigInt64Array(4),
        new DataView(new ArrayBuffer(32)), Object.create(Float64Array.prototype),
        Object.setPrototypeOf(new Float32Array(4), Float64Array.prototype),
        new Proxy(new Float64Array(4), {}), detached, new Float64Array(new SharedArrayBuffer(32)),
        new Float64Array(disguised), new Float64Array(memory.buffer, 0, 4), new Float64Array(3)]) {
        assert.throws(() => batch.run_into(input, output, 2));
        assert.deepEqual(Array.from(output), [123, 456]);
      }
      for (const destination of [Float32Array.of(123, 456), Float64Array.of(123),
        new Float64Array(new SharedArrayBuffer(16)), new Float64Array(memory.buffer, 0, 2)]) {
        const bytes = new Uint8Array(destination.buffer, destination.byteOffset, destination.byteLength);
        const before = bytes.slice();
        assert.throws(() => batch.run_into(Float64Array.of(3, 4, 5, 6), destination, 2));
        assert.deepEqual(bytes, before);
      }
      const input = Float64Array.of(3, 4, 5, 6);
      const validOutput = new Float64Array(2);
      for (const array of [input, validOutput]) {
        for (const field of ['length', 'buffer', 'byteOffset', 'set']) {
          Object.defineProperty(array, field, { get() {
            throw new Error('Caller-controlled metadata was accessed');
          } });
        }
      }
      batch.run_into(input, validOutput, 2);
      assert.deepEqual(Array.from(validOutput), [2, 4]);
      memory.grow(1);
      assert.throws(() => batch.run_into(new Float64Array(memory.buffer, 0, 4), output, 2));
      assert.deepEqual(Array.from(output), [123, 456]);
      batch.run_into(new Float64Array(0), new Float64Array(0), 0);
      const emptyDetached = new Float64Array(0);
      structuredClone(emptyDetached.buffer, { transfer: [emptyDetached.buffer] });
      for (const empty of [emptyDetached, new Float64Array(new SharedArrayBuffer(0)), Object.create(Float64Array.prototype)]) {
        assert.throws(() => batch.run_into(empty, new Float64Array(0), 0));
        assert.throws(() => batch.run_into(new Float64Array(0), empty, 0));
      }
    } finally {
      batch.free();
    }
  });

  test(`batch validates numeric capacities and work/memory limits before allocation (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const source = runtime.createNeuralNetwork([1, 1], 'Linear');
    source.set_weights(Float64Array.of(1));
    const prepared = source.prepare(); source.free();
    try {
      for (const capacity of [0, -1, 1.5, 4097, 2 ** 32, Number.MAX_SAFE_INTEGER + 1,
        NaN, Infinity, '2', true, null, undefined, 2n, Object(2)]) {
        assert.throws(() => prepared.create_batch(capacity));
      }
      const maximum = prepared.create_batch(4096); maximum.free();
      const batch = prepared.create_batch(3);
      try {
        const output = Float64Array.of(123);
        for (const rows of [-1, 1.5, 4, 2 ** 32, Number.MAX_SAFE_INTEGER + 1, NaN, Infinity,
          '1', true, null, undefined, 1n, Object(1)]) {
          assert.throws(() => batch.run_into(Float64Array.of(3), output, rows));
          assert.deepEqual(Array.from(output), [123]);
        }
        assert.throws(() => batch.run_into(Float64Array.of(3), output, 0));
        batch.run_into(Float64Array.of(3), output, 1);
        assert.deepEqual(Array.from(output), [3]);
      } finally {
        batch.free();
      }
    } finally {
      prepared.free();
    }
    for (const [shape, message, capacity] of [[[160, 160], /work/i, 4096], [[15000, 1], /64 MiB/i, 1024]]) {
      const network = runtime.createNeuralNetwork(shape, 'Linear');
      const snapshot = network.prepare(); network.free();
      try {
        assert.throws(() => snapshot.create_batch(capacity), error => message.test(String(error)));
        const small = snapshot.create_batch(1); small.free();
      } finally {
        snapshot.free();
      }
    }
    // Reuse an actual Rust serde neuron shape. A zero-edge graph still performs
    // 30,000 layer/neuron updates per sample; edge-only work accounting missed it.
    const fixture = JSON.parse(await readFile(new URL('../../crates/ruv-swarm-wasm/tests/fixtures/prepared-sparse.json', import.meta.url), 'utf8'));
    const neuron = structuredClone(fixture.network.layers[0].neurons[0]);
    Object.assign(neuron, { activation_function: 'Sin', activation_steepness: 1, connections: [], is_bias: false, sum: 0, value: 0 });
    const json = JSON.stringify({ connection_rate: 0, layers: Array.from({ length: 30000 }, () => ({ neurons: [neuron] })) });
    assert.ok(Buffer.byteLength(json, 'utf8') <= 4 * 1024 * 1024);
    const deep = runtime.createPreparedNetwork(json);
    try {
      assert.throws(() => deep.create_batch(4096), error => /work/i.test(String(error)));
      const bounded = deep.create_batch(1);
      try {
        const output = new Float64Array(1);
        bounded.run_into(Float64Array.of(0), output, 1);
        assert.equal(output[0], 0.5);
      } finally {
        bounded.free();
      }
    } finally {
      deep.free();
    }
  });
}
