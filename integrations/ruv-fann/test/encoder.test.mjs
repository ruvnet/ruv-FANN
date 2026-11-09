/** Boundary fixtures plus one real production prepared-network execution. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createEncoder, checkedInputs, preparedDimensions } from '../src/encoder.mjs';

function fixture({ failAt = 0, nonfinite = false } = {}) {
  const state = { calls: [], batches: 0, freed: 0, preparedFreed: 0 };
  const prepared = { input_size: 2, output_size: 2, estimated_bytes: 256,
    run(input) { return new Float64Array([input[0] * 2 + 1, input[1] * 3 - 1]); },
    free() { state.preparedFreed++; },
    create_batch(capacity) {
      state.batches++;
      return { input_size: 2, output_size: 2, max_rows: capacity, estimated_bytes: 1024,
        run_into(input, output, rows) {
          assert.equal(input.length, rows * 2); assert.equal(output.length, rows * 2);
          assert.ok(rows <= capacity); state.calls.push(rows);
          if (state.calls.length === failAt) throw new Error('boundary fixture backend failure');
          for (let row = 0; row < rows; row++) output.set(prepared.run(input.subarray(row * 2, row * 2 + 2)), row * 2);
          if (nonfinite) output[output.length - 1] = Infinity;
        }, free() { state.freed++; } };
    } };
  return { prepared, state };
}

test('exact rows and odd tail use actual configured capacity without changing caller input', () => {
  const { prepared, state } = fixture(); const encoder = createEncoder(prepared, { batchRows: 2, maxRows: 5 });
  const input = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), copy = input.slice();
  const output = encoder.encode(input);
  assert.deepEqual([...output], [3, 5, 7, 11, 11, 17, 15, 23, 19, 29]);
  assert.deepEqual(state.calls, [2, 2, 1]); assert.deepEqual(input, copy);
  assert.equal(encoder.inputSize, 2); assert.equal(encoder.outputSize, 2); assert.equal(encoder.batchRows, 2);
  encoder.close(); encoder.close(); assert.equal(state.freed, 1); assert.equal(state.preparedFreed, 0);
  assert.throws(() => encoder.encode(input), /closed/);
});

test('all input values are validated before any inference, including late rows', () => {
  const { prepared, state } = fixture(); const encoder = createEncoder(prepared, { batchRows: 1 });
  assert.throws(() => encoder.encode(new Float64Array([1, 2, 3, NaN])), /finite/);
  assert.equal(state.calls.length, 0);
  assert.throws(() => encoder.encode(new Float64Array(3)), /exact row/);
  assert.throws(() => encoder.encode(new Float64Array(2 * 4097)), /budget/);
  assert.equal(encoder.encode(new Float64Array()).length, 0); encoder.close();
});

test('spoofed, wrong-type, shared and detached typed arrays are rejected', () => {
  const { prepared } = fixture(); const encoder = createEncoder(prepared);
  for (const input of [Object.create(Float64Array.prototype), new Float32Array(2),
    { length: 2, [Symbol.toStringTag]: 'Float64Array' }, new Float64Array(new SharedArrayBuffer(16))])
    assert.throws(() => encoder.encode(input), /Float64Array/);
  const detached = new Float64Array(2); structuredClone(detached.buffer, { transfer: [detached.buffer] });
  assert.throws(() => encoder.encode(detached), /nondetached/); encoder.close();
});

test('caller iterators, subarray overrides and species are never invoked', () => {
  const { prepared } = fixture(); const encoder = createEncoder(prepared, { batchRows: 1 });
  const input = new Float64Array([2, 3]);
  input[Symbol.iterator] = () => { throw new Error('caller iterator'); };
  input.subarray = () => { throw new Error('caller subarray'); };
  input.constructor = { [Symbol.species]: class { constructor() { throw new Error('caller species'); } } };
  assert.deepEqual([...encoder.encode(input)], [5, 8]); encoder.close();
});

test('invalid configuration and resource dimensions fail before batch construction', () => {
  const { prepared, state } = fixture();
  for (const options of [{ batchRows: 0 }, { batchRows: 129 }, { maxRows: 4097 }, { maxRows: Infinity }, { typo: 1 }])
    assert.throws(() => createEncoder(prepared, options));
  assert.equal(state.batches, 0);
  assert.throws(() => preparedDimensions({ ...prepared, input_size: 65537 }));
  assert.throws(() => checkedInputs(new Float64Array(129), 1, 65536, 129), /byte budget/);
});

test('a failed or nonfinite backend never exposes partial output and remains closable', () => {
  for (const settings of [{ failAt: 2 }, { nonfinite: true }]) {
    const { prepared, state } = fixture(settings); const encoder = createEncoder(prepared, { batchRows: 1 });
    assert.throws(() => encoder.encode(new Float64Array([1, 2, 3, 4])));
    encoder.close(); assert.equal(state.freed, 1); assert.equal(state.preparedFreed, 0);
  }
});

test('incompatible batch construction releases the owned handle', () => {
  const { prepared, state } = fixture(); const create = prepared.create_batch;
  prepared.create_batch = rows => ({ ...create(rows), input_size: 3 });
  assert.throws(() => createEncoder(prepared), /incompatible/); assert.equal(state.freed, 1);
});

test('real FANN WASM prepared batch agrees exactly with scalar inference across tails', async () => {
  const module = await import('../../../ruv-swarm/npm/wasm/ruv_swarm_wasm.js');
  module.initSync({ module: await readFile(new URL('../../../ruv-swarm/npm/wasm/ruv_swarm_wasm_bg.wasm', import.meta.url)) });
  const network = new module.WasmNeuralNetwork(new Uint32Array([2, 3, 2]), module.ActivationFunction.Linear);
  network.set_weights(Float64Array.from(network.get_weights(), (_, index) => (index % 7 - 3) / 8));
  const prepared = network.prepare(); network.free();
  const encoder = createEncoder(prepared, { batchRows: 3, maxRows: 7 });
  try {
    const input = Float64Array.from({ length: 14 }, (_, index) => (index - 5) / 9), expected = [];
    for (let first = 0; first < input.length; first += 2) expected.push(...prepared.run(input.subarray(first, first + 2)));
    assert.deepEqual([...encoder.encode(input)], expected);
    encoder.close(); assert.equal(prepared.run(new Float64Array([1, 2])).length, 2);
  } finally { encoder.close(); prepared.free(); }
});
