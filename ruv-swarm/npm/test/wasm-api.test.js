import assert from 'node:assert/strict';
import { test } from 'node:test';
import { RuvSwarm } from '../src/index.js';
import { WasmBindingsLoader } from '../wasm/wasm-bindings-loader.mjs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

function calibrationConfig() {
  const records = [];
  for (const [series, seriesScale] of [['a', 1], ['b', 5]]) {
    for (let window = 0; window < 3; window++) {
      for (let step = 1; step <= 2; step++) {
        for (const [model, modelScale] of [['linear', 1], ['mean', 10]]) {
          const actual = step * 10 * seriesScale;
          records.push({ series, model, cutoff: window * 100, timestamp: window * 100 + step * 10,
            actual, prediction: actual - (window + 1) * step * seriesScale * modelScale });
        }
      }
    }
  }
  return { as_of: 1000, horizon: 2, coverages: [0.75, 0.5], records };
}

function forecastRecords() {
  const records = [];
  for (const [series, scale] of [['a', 1], ['b', 2]]) {
    for (const [model, offset] of [['linear', 0], ['mean', 20]]) {
      for (let step = 1; step <= 2; step++) {
        records.push({ series, model, origin: 1000, timestamp: 1000 + step * 10,
          prediction: scale * 100 + offset + step });
      }
    }
  }
  return records;
}

test('bindings loader uses actual module memory and never substitutes predictions on failure', async t => {
  const loader = await new WasmBindingsLoader().initialize();
  assert.ok(loader.memory instanceof WebAssembly.Memory);
  assert.equal(loader.getTotalMemoryUsage(), loader.memory.buffer.byteLength);
  assert.equal(loader.isPlaceholder, false);
  const directory = await mkdtemp(path.join(os.tmpdir(), 'ruv-invalid-wasm-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const missing = new WasmBindingsLoader(directory);
  await assert.rejects(missing.initialize(), /ENOENT/);
  assert.equal(missing.initialized, false);
  assert.equal(missing.create_neural_network, undefined);
  await writeFile(path.join(directory, 'ruv_swarm_wasm_bg.wasm'), Buffer.from('invalid WASM'));
  await writeFile(path.join(directory, 'ruv_swarm_wasm.js'), 'export default async ({module_or_path}) => WebAssembly.instantiate(module_or_path)');
  await assert.rejects(missing.initialize(), WebAssembly.CompileError);
  assert.equal(missing.initialized, false);
});

for (const useSIMD of [false, true]) {
  test(`generated WASM performs actual inference and statistical forecasts (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    assert.match(RuvSwarm.getVersion(), /^\d+\.\d+\.\d+/);
    assert.ok(RuvSwarm.getMemoryUsage() > 0);
    assert.equal(typeof RuvSwarm.getRuntimeFeatures().threadsAvailable, 'boolean');
    const network = runtime.createNeuralNetwork([2, 1], 'Linear');
    try {
      network.set_weights(Float64Array.of(2, -1));
      assert.deepEqual(Array.from(network.run(Float64Array.of(3, 4))), [2]);
      assert.throws(() => network.run(Float64Array.of(3)));
      assert.throws(() => network.set_weights(Float64Array.of(NaN, 1)));
    } finally {
      network.free();
    }
    for (const [name, expected] of [['linear', 7], ['mean', 3], ['naive', 5]]) {
      const model = runtime.createForecastingModel(name);
      try {
        assert.deepEqual(Array.from(model.predict(Float64Array.of(1, 3, 5))), [expected]);
      } finally {
        model.free();
      }
    }
    assert.throws(() => runtime.createForecastingModel('lstm'));
    assert.throws(() => runtime.createNeuralNetwork([2.5, 1]));
  });

  test(`prepared inference matches every legacy activation and keeps an independent snapshot (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const activations = ['Linear', 'Sigmoid', 'SymmetricSigmoid', 'Tanh', 'ReLU', 'LeakyReLU', 'Swish',
      'Gaussian', 'Elliot', 'SymmetricElliot', 'Sine', 'Cosine', 'SinSymmetric', 'CosSymmetric',
      'ThresholdSymmetric', 'Threshold', 'StepSymmetric', 'Step'];
    for (const activation of activations) {
      const network = runtime.createNeuralNetwork([2, 3, 2], activation);
      let prepared;
      try {
        network.set_weights(Float64Array.from({ length: 12 }, (_, i) => (i - 6) / 10));
        prepared = network.prepare();
        const input = Float64Array.of(0.3, -0.7);
        const expected = network.run(input);
        assert.equal(prepared.input_size, 2);
        assert.equal(prepared.output_size, 2);
        assert.ok(Number.isSafeInteger(prepared.estimated_bytes) && prepared.estimated_bytes > 0);
        const actual = prepared.run(input);
        for (let i = 0; i < expected.length; i++) {
          assert.ok(Math.abs(actual[i] - expected[i]) <= 1e-12, activation);
        }
      } finally {
        prepared?.free(); network.free();
      }
    }
    const network = runtime.createNeuralNetwork([2, 1], 'Linear');
    network.set_weights(Float64Array.of(2, -1));
    const prepared = network.prepare();
    network.set_weights(Float64Array.of(0, 0));
    assert.deepEqual(Array.from(network.run(Float64Array.of(3, 4))), [0]);
    network.free();
    try {
      assert.deepEqual(Array.from(prepared.run(Float64Array.of(3, 4))), [2]);
      const output = Float64Array.of(123);
      const input = Float64Array.of(3, 4);
      for (let i = 0; i < 1000; i++) {
        prepared.run_into(input, output);
      }
      assert.deepEqual(Array.from(output), [2]);
    } finally {
      prepared.free();
    }
  });

  test(`prepared run_into rejects unsafe views atomically and recovers after numeric failures (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const network = runtime.createNeuralNetwork([2, 1], 'Linear');
    network.set_weights(Float64Array.of(2, -1));
    const prepared = network.prepare();
    network.free();
    try {
      const output = Float64Array.of(123);
      const spoof = Object.setPrototypeOf(Float32Array.of(3, 4), Float64Array.prototype);
      const detached = Float64Array.of(3, 4);
      structuredClone(detached.buffer, { transfer: [detached.buffer] });
      const shared = new Float64Array(new SharedArrayBuffer(16));
      const disguisedShared = new SharedArrayBuffer(16);
      Object.setPrototypeOf(disguisedShared, ArrayBuffer.prototype);
      const disguisedSharedInput = new Float64Array(disguisedShared);
      const { memory } = await runtime._bindings.default();
      const wasmInput = new Float64Array(memory.buffer, 0, 2);
      for (const input of [[3, 4], Float32Array.of(3, 4), BigInt64Array.of(3n, 4n), spoof,
        Float64Array.of(3), Float64Array.of(3, 4, 5), Float64Array.of(NaN, 4),
        Float64Array.of(Infinity, 4), detached, shared, disguisedSharedInput, wasmInput]) {
        assert.throws(() => prepared.run_into(input, output));
        assert.deepEqual(Array.from(output), [123]);
      }
      for (const destination of [Float32Array.of(123), Float64Array.of(123, 456),
        new Float64Array(new SharedArrayBuffer(8)), new Float64Array(memory.buffer, 0, 1)]) {
        const bytes = new Uint8Array(destination.buffer, destination.byteOffset, destination.byteLength);
        const before = bytes.slice();
        assert.throws(() => prepared.run_into(Float64Array.of(3, 4), destination));
        assert.deepEqual(bytes, before);
      }
      memory.grow(1);
      const grownInput = new Float64Array(memory.buffer, 0, 2);
      assert.throws(() => prepared.run_into(grownInput, output));
      assert.deepEqual(Array.from(output), [123]);
      const grownOutput = new Float64Array(memory.buffer, 0, 1);
      const beforeGrowthOutput = grownOutput.slice();
      assert.throws(() => prepared.run_into(Float64Array.of(3, 4), grownOutput));
      assert.deepEqual(grownOutput, beforeGrowthOutput);
      prepared.run_into(Float64Array.of(3, 4), output);
      assert.deepEqual(Array.from(output), [2]);
    } finally {
      prepared.free();
    }
    const square = runtime.createNeuralNetwork([2, 2], 'Linear');
    square.set_weights(Float64Array.of(1, 2, 3, 4));
    const expected = Array.from(square.run(Float64Array.of(3, 4)));
    const alias = square.prepare(); square.free();
    try {
      const inPlace = Float64Array.of(3, 4);
      alias.run_into(inPlace, inPlace);
      assert.deepEqual(Array.from(inPlace), expected);
      const backing = Float64Array.of(3, 4, 999);
      alias.run_into(backing.subarray(0, 2), backing.subarray(1, 3));
      assert.deepEqual(Array.from(backing), [3, ...expected]);
    } finally {
      alias.free();
    }
    const overflowing = runtime.createNeuralNetwork([2, 1], 'Linear');
    overflowing.set_weights(Float64Array.of(Number.MAX_VALUE, 0));
    const checked = overflowing.prepare(); overflowing.free();
    try {
      const output = Float64Array.of(123);
      assert.throws(() => checked.run_into(Float64Array.of(2, 0), output));
      assert.deepEqual(Array.from(output), [123]);
      checked.run_into(Float64Array.of(0.5, 0), output);
      assert.equal(output[0], Number.MAX_VALUE * 0.5);
    } finally {
      checked.free();
    }
  });

  test(`prepared imports reproduce actual native trained and sparse serde graphs (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    // Generated by the Rust crate's native_export_fixtures_are_reproducible test:
    // trained40epochs from fixed weights; sparse has shuffled/duplicate Swish connections.
    for (const name of ['dense', 'sparse']) {
      const fixture = JSON.parse(await readFile(new URL(`../../crates/ruv-swarm-wasm/tests/fixtures/prepared-${name}.json`, import.meta.url), 'utf8'));
      const prepared = runtime.createPreparedNetwork(JSON.stringify(fixture.network));
      try {
        assert.equal(prepared.input_size, fixture.input.length);
        assert.equal(prepared.output_size, fixture.output.length);
        const input = Float64Array.from(fixture.input);
        const output = new Float64Array(fixture.output.length);
        prepared.run_into(input, output);
        const owned = prepared.run(input);
        for (let i = 0; i < fixture.output.length; i++) {
          assert.ok(Math.abs(output[i] - fixture.output[i]) <= 1e-12, `${name} native parity`);
          assert.equal(owned[i], output[i]);
        }
      } finally {
        prepared.free();
      }
    }
    for (const value of [{}, null, 1, 1n]) {
      assert.throws(() => runtime.createPreparedNetwork(value), TypeError);
    }
    for (const value of ['{', '{}', 'null', '{"layers":[]}']) {
      assert.throws(() => runtime.createPreparedNetwork(value));
    }
    assert.throws(() => runtime.createPreparedNetwork(' '.repeat(4 * 1024 * 1024 + 1)), /4 MiB/);
    assert.throws(() => runtime.createPreparedNetwork('é'.repeat(2 * 1024 * 1024 + 1)), /4 MiB/);
    assert.throws(() => runtime._bindings.WasmPreparedNetwork.from_network_json(' '.repeat(4 * 1024 * 1024 + 1)));
  });

  test(`conformal intervals use exact finite-sample quantiles per series/model/horizon (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const config = calibrationConfig();
    Object.defineProperty(config, 'toJSON', { value: () => {
      throw new Error('Untrusted toJSON invoked');
    } });
    const calibrator = runtime.createConformalCalibrator(config);
    try {
      const groups = calibrator.groups();
      assert.equal(groups.length, 8);
      const group = groups.find(g => g.series === 'a' && g.model === 'linear' && g.horizon === 1);
      assert.deepEqual(group, { series: 'a', model: 'linear', horizon: 1, samples: 3,
        first_outcome: 10, last_outcome: 210, radii: [2, 3] });
      const rows = calibrator.apply({ records: forecastRecords() });
      assert.equal(rows.length, 16);
      for (const row of rows) {
        const scale = (row.series === 'a' ? 1 : 5) * (row.model === 'linear' ? 1 : 10);
        const radius = (row.coverage === 0.5 ? 2 : 3) * row.horizon * scale;
        assert.equal(row.lower, row.point - radius);
        assert.equal(row.upper, row.point + radius);
        assert.equal(row.calibration_samples, 3);
        assert.equal(row.forecast_origin, 1000);
        assert.equal(row.calibration_as_of, 1000);
        assert.equal(row.timestamp, 1000 + row.horizon * 10);
      }
      assert.deepEqual(config.coverages, [0.75, 0.5]);
    } finally {
      calibrator.free();
    }
    calibrator.free();
    assert.throws(() => calibrator.groups(), /freed/);
    assert.throws(() => calibrator.apply({ records: forecastRecords() }), /freed/);
    const magnitude = 2 ** 98;
    const exactConfig = { as_of: 20, horizon: 1, coverages: [0.5],
      records: [{ series: 's', model: 'm', cutoff: 0, timestamp: 10, actual: magnitude, prediction: 0 }] };
    const exact = runtime.createConformalCalibrator(exactConfig);
    try {
      assert.equal(exact.groups()[0].radii[0], magnitude);
      assert.equal(exact.apply({ records: [{ series: 's', model: 'm', origin: 20, timestamp: 30, prediction: 0 }] })[0].upper, magnitude);
    } finally {
      exact.free();
    }
    const integerJson = JSON.stringify(exactConfig).replace(magnitude.toString(), (2n ** 98n).toString());
    const exponentJson = JSON.stringify(exactConfig).replace(magnitude.toString(), '3.1691265005705735e29');
    for (const json of [integerJson, exponentJson]) {
      const rawExact = new runtime._bindings.WasmConformalCalibrator(json);
      try {
        assert.equal(JSON.parse(rawExact.groups())[0].radii[0], magnitude);
      } finally {
        rawExact.free();
      }
    }
  });

  test(`conformal serializes only the once-validated array lengths (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    const config = calibrationConfig();
    let levelReads = 0, recordReads = 0;
    config.coverages = new Proxy(Array.from({ length: 17 }, (_, i) => (i + 1) / 100), {
      get(target, key, receiver) {
        if (key === 'length') {
          return ++levelReads <= 2 ? 1 : 17;
        }
        return Reflect.get(target, key, receiver);
      },
    });
    const count = config.records.length;
    config.records = new Proxy(config.records, { get(target, key, receiver) {
      if (key === 'length') {
        return ++recordReads === 1 ? count : 100000;
      }
      return Reflect.get(target, key, receiver);
    } });
    const calibrator = runtime.createConformalCalibrator(config);
    try {
      assert.equal(levelReads, 1);
      assert.equal(recordReads, 1);
      assert.equal(calibrator.apply({ records: forecastRecords() }).length, 8);
    } finally {
      calibrator.free();
    }
  });

  test(`conformal checks chronology, precision, finite values and complete model windows (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    for (const as_of of [NaN, Infinity, 1.5, Number.MAX_SAFE_INTEGER + 1, 1000n]) {
      assert.throws(() => runtime.createConformalCalibrator({ ...calibrationConfig(), as_of }));
    }
    for (const coverages of [[], [0], [1], [NaN], [0.5, 0.5], Array(17).fill(0.5), [0.9]]) {
      assert.throws(() => runtime.createConformalCalibrator({ ...calibrationConfig(), coverages }));
    }
    for (const change of [{ actual: NaN }, { prediction: Infinity }, { cutoff: 10 }, { timestamp: 1001 },
      { timestamp: 10.5 }, { series: '' }, { model: 'é'.repeat(129) }, { extra: 'unsupported' }]) {
      const config = calibrationConfig(); Object.assign(config.records[0], change);
      assert.throws(() => runtime.createConformalCalibrator(config));
    }
    for (const edit of [records => records.slice(1), records => [...records, records[0]],
      records => records.map((row, i) => i === 1 ? { ...row, actual: 999 } : row)]) {
      const config = calibrationConfig(); config.records = edit(config.records);
      assert.throws(() => runtime.createConformalCalibrator(config));
    }
    const raw = runtime._bindings.WasmConformalCalibrator;
    for (const value of [undefined, null, {}, [], 1, 1n, Object('{}')]) {
      assert.throws(() => new raw(value));
    }
    assert.throws(() => new raw(' '.repeat(4 * 1024 * 1024 + 1)));
    assert.throws(() => new raw('é'.repeat(2 * 1024 * 1024 + 1)));
    assert.throws(() => new raw(JSON.stringify({ ...calibrationConfig(), as_of: Number.MAX_SAFE_INTEGER + 1 })));
    assert.throws(() => new raw(JSON.stringify(calibrationConfig()).replace('"actual":10', '"actual":1e309')));
    const direct = new raw(JSON.stringify(calibrationConfig()));
    try {
      for (const value of [undefined, null, {}, [], 1, 1n, Object('{}'), ' '.repeat(4 * 1024 * 1024 + 1)]) {
        assert.throws(() => direct.apply(value));
      }
      const records = forecastRecords(); records[0].timestamp = 1010.5;
      assert.throws(() => direct.apply(JSON.stringify({ records })));
      assert.equal(JSON.parse(direct.apply(JSON.stringify({ records: forecastRecords() }))).length, 16);
    } finally {
      direct.free();
    }
    const calibrator = runtime.createConformalCalibrator(calibrationConfig());
    try {
      for (const change of [{ origin: 999 }, { timestamp: 1000 }, { timestamp: 1010.5 },
        { timestamp: Number.MAX_SAFE_INTEGER + 1 }, { prediction: NaN }, { prediction: 1n },
        { model: 'unknown' }, { series: 'missing' }, { origin: 1001 }, { timestamp: 1011 }]) {
        const records = forecastRecords(); Object.assign(records[0], change);
        assert.throws(() => calibrator.apply({ records }));
      }
      assert.throws(() => calibrator.apply({ records: forecastRecords().slice(1) }));
      assert.throws(() => calibrator.apply({ records: [...forecastRecords(), forecastRecords()[0]] }));
      assert.equal(calibrator.apply({ records: forecastRecords() }).length, 16);
    } finally {
      calibrator.free();
    }
    const extremes = { as_of: 100, horizon: 1, coverages: [0.5], records: [0, 10, 20].map(cutoff => ({
      series: 's', model: 'm', cutoff, timestamp: cutoff + 5, actual: Number.MAX_VALUE, prediction: -Number.MAX_VALUE,
    })) };
    assert.throws(() => runtime.createConformalCalibrator(extremes));
    extremes.records = extremes.records.map(row => ({ ...row, prediction: 0 }));
    const largeRadius = runtime.createConformalCalibrator(extremes);
    try {
      const record = { series: 's', model: 'm', origin: 100, timestamp: 110, prediction: Number.MAX_VALUE };
      assert.throws(() => largeRadius.apply({ records: [record] }));
      const [interval] = largeRadius.apply({ records: [{ ...record, prediction: 0 }] });
      assert.equal(interval.lower, -Number.MAX_VALUE);
      assert.equal(interval.upper, Number.MAX_VALUE);
    } finally {
      largeRadius.free();
    }
  });

  test(`conformal resource limits reject oversized work and preserve a fitted calibrator (SIMD=${useSIMD})`, async() => {
    const runtime = await RuvSwarm.initialize({ useSIMD });
    assert.throws(() => runtime.createConformalCalibrator({ ...calibrationConfig(), horizon: 1001 }));
    assert.throws(() => runtime.createConformalCalibrator({ ...calibrationConfig(), records: Array(10001).fill(calibrationConfig().records[0]) }));
    const escaped = { series: '\0'.repeat(256), model: '\0'.repeat(256), cutoff: 0, timestamp: 1, actual: 1, prediction: 0 };
    assert.throws(() => runtime.createConformalCalibrator({ as_of: 1, horizon: 1, coverages: [0.5], records: Array(2000).fill(escaped) }), /4 MiB/);
    const names = Array.from({ length: 1000 }, (_, i) => 's'.repeat(252) + String(i).padStart(4, '0'));
    const model = 'm'.repeat(256);
    const config = { as_of: 10, horizon: 1, coverages: Array.from({ length: 16 }, (_, i) => (i + 1) / 100),
      records: names.map(series => ({ series, model, cutoff: 0, timestamp: 10, actual: 1, prediction: 0 })) };
    assert.throws(() => runtime.createConformalCalibrator({ ...config, records: [...config.records,
      { ...config.records[0], series: 'over-budget' }] }));
    const calibrator = runtime.createConformalCalibrator(config);
    try {
      const records = names.map(series => ({ series, model, origin: 10, timestamp: 20, prediction: 100 }));
      assert.throws(() => calibrator.apply({ records }), /output|serializ|8 MiB|budget|limit/i);
      assert.equal(calibrator.groups().length, 1000);
      assert.equal(calibrator.apply({ records: records.slice(0, 1) }).length, 16);
    } finally {
      calibrator.free();
    }
  });
}

test('published facade preserves caller metadata and actual capability assignments', async() => {
  const runtime = await RuvSwarm.initialize();
  const swarm = await runtime.createSwarm({ name: 'actual', topology: 'ring', maxAgents: 1 });
  try {
    const agent = await swarm.spawn({ id: 'coder-1', name: 'Actual coder', type: 'coder', capabilities: ['code'] });
    assert.equal(agent.id, 'coder-1');
    assert.equal(agent.agentType, 'coder');
    assert.deepEqual(agent.getCapabilities(), ['code']);
    assert.equal(agent.status, 'idle');
    assert.deepEqual(swarm.getAgents(), ['coder-1']);
    const task = await swarm.orchestrate({ description: 'Compile the actual input', priority: 'high', requiredCapabilities: ['code'] });
    assert.equal(task.description, 'Compile the actual input');
    assert.equal(task.priority, 'high');
    assert.equal(task.status, 'assigned');
    assert.deepEqual(task.assignedAgents, ['coder-1']);
    assert.equal(agent.status, 'busy');
    agent.reset();
    assert.equal(agent.status, 'idle');
    const unmatched = await swarm.orchestrate({ description: 'Requires a different skill', requiredCapabilities: ['vision'] });
    assert.equal(unmatched.status, 'unassigned');
    assert.deepEqual(unmatched.assignedAgents, []);
    const status = swarm.getStatus();
    assert.equal(status.name, 'actual');
    assert.equal(status.topology, 'ring');
    assert.equal(status.agentCount, 1);
    await assert.rejects(swarm.spawn({ type: 'coder' }), /agent limit/);
    await assert.rejects(agent.execute({ description: 'No configured executor' }), /external task execution/);
    await assert.rejects(swarm.orchestrate({ description: 'Unsupported DAG', dependencies: ['previous'] }), /Dependency scheduling/);
  } finally {
    swarm.free();
  }
});
