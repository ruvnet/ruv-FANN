import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { RuvSwarm } from '../src/index.js';

const model = (layers) => JSON.stringify({ schema: 'ruv-fann.dense-layers/1', layers });
const layer = { inputs: 2, outputs: 2, weights: [1, 2, 3, 4], bias: [0.25, -0.5],
  activation: 'Linear', steepness: 1 };

for (const useSIMD of [false, true]) {
  test(`canonical dense [out,in] import, mixed bias and owned snapshot (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const prepared = runtime.createDensePreparedNetwork(model([layer,
      { inputs: 2, outputs: 1, weights: [5, 6], bias: null, activation: 'Linear', steepness: 1 }]));
    const batch = prepared.create_batch(3);
    assert.deepEqual(Array.from(prepared.run(Float64Array.of(1, 2))), [89.25]);
    assert.ok(prepared.estimated_bytes <= 64 * 1024 * 1024);
    prepared.free();
    const output = new Float64Array(3);
    batch.run_into(Float64Array.of(1, 2, 1, 2, 1, 2), output, 3);
    assert.deepEqual(Array.from(output), [89.25, 89.25, 89.25]);
    batch.free();
  });

  test(`dense import validates primitive strings, schemas, shapes and limits directly (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const raw = runtime._bindings.WasmPreparedNetwork;
    const poison = { toString() {
      throw new Error('Coercion must not run');
    } };
    for (const bad of [null, {}, poison, Object(model([layer])), 1, [],
      ' '.repeat(4 * 1024 * 1024 + 1), `"${ 'é'.repeat(3 * 1024 * 1024) }"`,
      '{}', model([]), model([{ ...layer, weights: [1] }]), model([{ ...layer, bias: [1] }]),
      model([{ ...layer, steepness: null }]), model([{ ...layer, extra: 1 }]),
      model([{ ...layer, outputs: 0 }]), model([{ ...layer, inputs: Number.MAX_SAFE_INTEGER }]),
      model([layer, { ...layer, inputs: 3 }]), model(Array(65).fill(layer)),
      model([layer]).replace('dense-layers/1', 'dense-layers/999')]) {
      assert.throws(() => raw.from_dense_layers_json(bad));
    }
    // Bias is appended after weights, including signed zero, in this schema.
    const cancellation = raw.from_dense_layers_json(model([{ inputs: 2, outputs: 1,
      weights: [1e20, -1e20], bias: [1], activation: 'Linear', steepness: 1 }]));
    assert.deepEqual(Array.from(cancellation.run(Float64Array.of(1, 1))), [1]);
    cancellation.free();
  });

  if (process.env.RUV_FANN_CANDLE_FIXTURE) {
    test(`actual Candle CPU f64 exported weights reproduce predictions (SIMD=${useSIMD})`, async() => {
      const fixture = JSON.parse(await readFile(process.env.RUV_FANN_CANDLE_FIXTURE, 'utf8'));
      assert.match(fixture.oracle, /candle-core\/candle-nn 0\.11\.1 CPU f64/);
      const runtime = await RuvSwarm.initialize({ useSIMD });
      const prepared = runtime.createDensePreparedNetwork(fixture.dense_json);
      const actual = prepared.run(Float64Array.from(fixture.input));
      for (let i = 0; i < actual.length; i++) {
        assert.ok(Math.abs(actual[i] - fixture.expected[i]) <= fixture.absolute_tolerance
          + fixture.relative_tolerance * Math.abs(fixture.expected[i]));
      }
      prepared.free();
    });
  }
}
