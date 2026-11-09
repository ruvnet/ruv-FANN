import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { RuvSwarm } from '../src/index.js';

const simple = (weights = [0], bias = [0], activation = 'Linear') => JSON.stringify({
  schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 1, weights, bias, activation, steepness: 1 },
  ],
});
const flat = json => JSON.parse(json).layers.flatMap(l => [...l.weights, ...(l.bias ?? [])]);
function close(actual, expected, tolerance = 1e-12) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) <= tolerance * (1 + Math.abs(expected[i])),
      `index ${i}: ${actual[i]} != ${expected[i]}`);
  }
}

for (const useSIMD of [false, true]) {
  test(`real reverse MSE gradients, owned copies, SGD and independent freeze (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const train = runtime.createTrainableNetwork(simple(), 2);
    const input = Float64Array.of(1, 2), target = Float64Array.of(2, 4);
    assert.equal(train.parameter_count, 2);
    assert.equal(train.max_rows, 2);
    assert.ok(train.estimated_bytes <= 64 * 1024 * 1024);
    assert.deepEqual(Array.from(train.parameter_gradients()), []);
    assert.throws(() => train.sgd_step(0.1));
    assert.equal(train.mse_backward(input, target, 2), 10);
    close(train.parameter_gradients(), [-10, -6]);
    close(train.input_gradients(), [0, 0]);
    close(train.predictions(), [0, 0]);
    train.parameter_gradients()[0] = 999;
    close(train.parameter_gradients(), [-10, -6]);
    const frozen = train.prepare();
    train.sgd_step(0.1);
    close(flat(train.to_dense_layers_json()), [1, 0.6]);
    assert.throws(() => train.sgd_step(0.1));
    assert.ok(train.mse_backward(input, target, 2) < 10);
    const checkpoint = train.to_dense_layers_json();
    const reload = runtime.createTrainableNetwork(checkpoint, 2);
    close(flat(reload.to_dense_layers_json()), flat(checkpoint));
    reload.free(); train.free();
    close(frozen.run(Float64Array.of(2)), [0]);
    const batch = frozen.create_batch(2); frozen.free();
    const out = new Float64Array(2);
    batch.run_into(input, out, 2); close(out, [0, 0]); batch.free();
  });

  test(`SGD and AdamW learn a dataset with fresh backward passes (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const input = Float64Array.of(-2, -1, 0, 1, 2), target = Float64Array.of(-3, -1, 1, 3, 5);
    for (const optimizer of ['sgd', 'adamw']) {
      const train = runtime.createTrainableNetwork(simple(), 5);
      if (optimizer === 'adamw') {
        train.configure_adamw(0.05, 0.9, 0.999, 1e-8, 0);
      }
      const first = train.mse_backward(input, target, 5);
      let last = first;
      for (let step = 0; step < 500; step++) {
        if (optimizer === 'sgd') {
          train.sgd_step(0.05);
        } else {
          train.adamw_step();
        }
        last = train.mse_backward(input, target, 5);
      }
      assert.ok(last < 1e-8 && last < first / 1000, `${optimizer}: ${first} -> ${last}`);
      close(flat(train.to_dense_layers_json()), [2, 1], 1e-4);
      train.free();
    }
  });

  test(`training validates intrinsic arrays, scalar options, limits and atomic errors (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const raw = runtime._bindings.WasmTrainableNetwork;
    for (const bad of [0, -1, 1.5, 4097, NaN, Infinity, '1', Object(1), null, undefined, 1n]) {
      assert.throws(() => raw.from_dense_layers_json(simple(), bad));
    }
    assert.throws(() => raw.from_dense_layers_json({}, 1));
    assert.throws(() => raw.from_dense_layers_json(simple([0], [0], 'Gaussian'), 1));
    const wide = JSON.stringify({ schema: 'ruv-fann.dense-layers/1', layers: [
      { inputs: 1000, outputs: 1, weights: Array(1000).fill(0), bias: null, activation: 'Linear', steepness: 1 },
    ] });
    assert.throws(() => raw.from_dense_layers_json(wide, 4096)); // work cap before tape allocation
    const train = raw.from_dense_layers_json(simple([1]), 2);
    const input = Float64Array.of(1, 2), targets = Float64Array.of(2, 4);
    const { memory } = await runtime._bindings.default();
    const detached = new Float64Array(2);
    structuredClone(detached.buffer, { transfer: [detached.buffer] });
    const before = train.to_dense_layers_json();
    for (const invalid of [[1, 2], Float32Array.of(1, 2), new BigInt64Array(2),
      Object.create(Float64Array.prototype), new Proxy(input, {}), detached,
      new Float64Array(new SharedArrayBuffer(16)), new Float64Array(memory.buffer, 0, 2),
      Float64Array.of(1), Float64Array.of(1, NaN)]) {
      assert.throws(() => train.mse_backward(invalid, targets, 2));
      assert.throws(() => train.mse_backward(input, invalid, 2));
      assert.equal(train.to_dense_layers_json(), before);
    }
    for (const bad of [0, -1, 1.5, 3, NaN, Infinity, '1', null, Object(1)]) {
      assert.throws(() => train.mse_backward(input, targets, bad));
    }
    train.mse_backward(input, targets, 2);
    for (const bad of [-1, 0, NaN, Infinity, '0.1', Object(0.1), null]) {
      assert.throws(() => train.sgd_step(bad));
      assert.equal(train.to_dense_layers_json(), before);
    }
    assert.throws(() => train.configure_adamw(0.01, 1, 0.999, 1e-8, 0));
    train.configure_adamw(0.01, 0.9, 0.999, 1e-8, 0);
    train.adamw_step();
    assert.throws(() => train.adamw_step());
    train.mse_backward(input, targets, 2); train.sgd_step(0.01);
    train.mse_backward(input, targets, 2);
    assert.throws(() => train.adamw_step()); // optimizer predates the SGD update
    train.configure_adamw(0.01, 0.9, 0.999, 1e-8, 0); train.adamw_step();
    // Caller-defined metadata getters cannot run inside the array boundary.
    for (const array of [input, targets]) {
      for (const key of ['length', 'buffer', 'byteOffset', 'set']) {
        Object.defineProperty(array, key, { get() {
          throw new Error('Metadata getter was read');
        } });
      }
    }
    assert.ok(Number.isFinite(train.mse_backward(input, targets, 2)));
    memory.grow(1);
    assert.throws(() => train.mse_backward(new Float64Array(memory.buffer, 0, 2), targets, 2));
    train.free();
    assert.throws(() => train.to_dense_layers_json());
  });

  if (process.env.RUV_FANN_CANDLE_FIXTURE) {
    test(`actual Candle GradStore and three optimizer updates match WASM (SIMD=${useSIMD})`, async() => {
      const f = JSON.parse(await readFile(process.env.RUV_FANN_CANDLE_FIXTURE, 'utf8'));
      assert.match(f.oracle, /candle-core\/candle-nn 0\.11\.1 CPU f64/);
      const runtime = await RuvSwarm.initialize({ useSIMD });
      for (const optimizer of ['sgd', 'adamw']) {
        const train = runtime.createTrainableNetwork(f.dense_json, 1);
        if (optimizer === 'adamw') {
          const c = f.adamw_config;
          train.configure_adamw(c.learning_rate, c.beta1, c.beta2, c.epsilon, c.weight_decay);
        }
        assert.equal(f[`${optimizer}_steps`].length, 3);
        for (const expected of f[`${optimizer}_steps`]) {
          close([train.mse_backward(Float64Array.from(f.input), Float64Array.from(f.training_target), 1)], [expected.loss]);
          close(train.predictions(), expected.predictions);
          close(train.parameter_gradients(), expected.parameter_gradients);
          close(train.input_gradients(), expected.input_gradients);
          if (optimizer === 'sgd') {
            train.sgd_step(f.sgd_learning_rate);
          } else {
            train.adamw_step();
          }
          close(flat(train.to_dense_layers_json()), expected.parameters);
        }
        train.free();
      }
    });
  }
}
