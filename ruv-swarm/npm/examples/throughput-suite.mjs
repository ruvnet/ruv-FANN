// Browser-safe fixtures and measurements shared by the Node and Chromium gates.
// All reported ns values are derived from performance.now(), not ns resolution.
const now = () => performance.now();
const finite = value => { if (!Number.isFinite(value)) throw new Error('Non-finite benchmark checksum'); };

function mean(values) { return values.reduce((sum, value) => sum + value, 0) / values.length; }
function stats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return {mean_ns: mean(values), median_ns: (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.ceil((sorted.length - 1) / 2)]) / 2,
    min_ns: sorted[0], max_ns: sorted.at(-1), samples_ns: values};
}

function outputError(actual, expected) {
  if (actual.length !== expected.length) throw new Error('Benchmark output shape mismatch');
  let maximum = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) throw new Error('Non-finite benchmark output');
    maximum = Math.max(maximum, Math.abs(actual[i] - expected[i]));
  }
  if (maximum > 1e-12) throw new Error(`Batch numerical mismatch: ${maximum}`);
  return maximum;
}

function timed(call, iterations) {
  let checksum = 0;
  const start = now();
  for (let i = 0; i < iterations; i++) checksum += call();
  const elapsed = now() - start;
  finite(checksum);
  return {ns: elapsed * 1e6 / iterations, checksum};
}

function iterationsFor(call, maximum = 10000) {
  // Warm the exact call before choosing a bounded ~3 ms sample block.
  let iterations = 1;
  for (let attempt = 0; attempt < 5; attempt++) {
    const probe = timed(call, iterations);
    if (probe.ns * iterations >= 3e6 || iterations === maximum) {
      return Math.max(1, Math.min(maximum, Math.ceil(3e6 / Math.max(probe.ns, 1))));
    }
    iterations = Math.min(maximum, iterations * 8);
  }
  return iterations;
}

export function measureCalls(calls, samples = 20) {
  const names = Object.keys(calls);
  const iterations = Object.fromEntries(names.map(name => [name, iterationsFor(calls[name])]));
  const raw = Object.fromEntries(names.map(name => [name, []]));
  let checksum = 0;
  for (let sample = 0; sample < samples; sample++) {
    // Rotate order to reduce systematic drift; collect paired sample blocks.
    for (let offset = 0; offset < names.length; offset++) {
      const name = names[(sample + offset) % names.length];
      const measured = timed(calls[name], iterations[name]);
      raw[name].push(measured.ns);
      checksum += measured.checksum;
    }
  }
  finite(checksum);
  return {iterations, checksum, timings: Object.fromEntries(names.map(name => [name, stats(raw[name])]))};
}

export function inferenceSuite(bindings, {samples = 20, batchSizes = [1, 3, 8, 32, 128], shapes = [[5, 7, 3], [100, 50, 25, 10], [784, 128, 64, 10]]} = {}) {
  const results = [];
  for (const activation of ['Linear', 'Sigmoid']) {
    for (const layers of shapes) {
      const parameters = layers.slice(1).reduce((count, size, i) => count + layers[i] * size, 0);
      const network = new bindings.WasmNeuralNetwork(Uint32Array.from(layers), bindings.ActivationFunction[activation]);
      network.set_weights(Float64Array.from({length: parameters}, (_, i) => (i * 17 % 31 - 15) / 64));
      const prepared = network.prepare();
      try {
        for (const rows of batchSizes) {
          const inputs = Float64Array.from({length: rows * layers[0]}, (_, i) =>
            ((i % layers[0]) * 7 % 13 - 6) / 8 + (Math.floor(i / layers[0]) % 5) / 64);
          const rowInputs = Array.from({length: rows}, (_, row) => inputs.subarray(row * layers[0], (row + 1) * layers[0]));
          const singleOutput = new Float64Array(rows * layers.at(-1));
          const rowOutputs = Array.from({length: rows}, (_, row) => singleOutput.subarray(row * layers.at(-1), (row + 1) * layers.at(-1)));
          const expected = Float64Array.from(rowInputs.flatMap(input => Array.from(network.run(input))));
          const calls = {
            legacy_loop: () => { let sum = 0; for (const input of rowInputs) sum += network.run(input)[0]; return sum; },
            prepared_single_loop: () => { let sum = 0; for (let row = 0; row < rows; row++) {
              prepared.run_into(rowInputs[row], rowOutputs[row]); sum += rowOutputs[row][0];
            } return sum; },
          };
          calls.prepared_single_loop();
          let batch;
          let setupNs = null;
          let estimate = null;
          let maximumError = 0;
          try {
            if (typeof prepared.create_batch === 'function') {
              const start = now();
              batch = prepared.create_batch(rows);
              setupNs = (now() - start) * 1e6;
              estimate = batch.estimated_bytes;
              const output = new Float64Array(expected.length);
              batch.run_into(inputs, output, rows);
              maximumError = outputError(output, expected);
              calls.prepared_batch = () => { batch.run_into(inputs, output, rows); let sum = 0;
                for (let row = 0; row < rows; row++) sum += output[row * layers.at(-1)]; return sum; };
            }
            maximumError = Math.max(maximumError, outputError(singleOutput, expected));
            const measured = measureCalls(calls, samples);
            for (const timing of Object.values(measured.timings)) {
              timing.mean_ns_per_row = timing.mean_ns / rows;
              timing.rows_per_second = rows * 1e9 / timing.mean_ns;
            }
            results.push({activation, layers, parameters, rows, precision: 'f64', maximum_absolute_error: maximumError,
              batch_creation_once_ns: setupNs, batch_estimated_bytes: estimate, ...measured});
          } finally { if (batch) batch.free(); }
        }
      } finally { prepared.free(); network.free(); }
    }
  }
  return results;
}

function calibrationFixture(kind, levels) {
  const dimensions = {one_long: [1, 1, 1, 10000], few_long: [2, 2, 12, 200], few_long_shuffled: [2, 2, 12, 200], many_series: [80, 2, 3, 20]}[kind];
  const [seriesCount, models, horizon, windows] = dimensions;
  const records = [];
  const forecasts = [];
  for (let series = 0; series < seriesCount; series++) {
    for (let window = 0; window < windows; window++) {
      for (let step = 1; step <= horizon; step++) {
        const actual = 100 + series / 2 + window + step;
        for (let model = 0; model < models; model++) {
          records.push({series: `s${series}`, model: `m${model}`, cutoff: window * horizon * 4,
            timestamp: window * horizon * 4 + step, actual,
            prediction: actual - ((window * 17 + step * 3 + series) % 97 + 1) * (model + 1) / 8});
        }
      }
    }
    for (let model = 0; model < models; model++) {
      for (let step = 1; step <= horizon; step++) {
        forecasts.push({series: `s${series}`, model: `m${model}`, origin: 100000, timestamp: 100000 + step, prediction: 100 + step});
      }
    }
  }
  const coverages = Array.from({length: levels}, (_, i) => 0.1 + i * 0.8 / Math.max(1, levels - 1));
  if (kind.endsWith('_shuffled')) {
    let state = 42;
    for (const values of [records, forecasts]) {
      for (let i = values.length - 1; i > 0; i--) {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        const j = state % (i + 1);
        [values[i], values[j]] = [values[j], values[i]];
      }
    }
  }
  return {fit: JSON.stringify({as_of: 100000, horizon, coverages, records}),
    apply: JSON.stringify({records: forecasts}), records: records.length, forecast_rows: forecasts.length};
}

export function calibrationSuite(bindings, {samples = 12, validate = () => false} = {}) {
  const results = [];
  for (const kind of ['one_long', 'few_long', 'few_long_shuffled', 'many_series']) {
    for (const levels of [1, 4, 16]) {
      const fixture = calibrationFixture(kind, levels);
      const calibrator = new bindings.WasmConformalCalibrator(fixture.fit);
      try {
        const output = calibrator.apply(fixture.apply);
        const groups = calibrator.groups();
        const exactComparison = validate(`${kind}/${levels}`, groups, output) === true;
        const measured = measureCalls({
          fit_and_free: () => { const value = new bindings.WasmConformalCalibrator(fixture.fit); value.free(); return 1; },
          apply_json: () => calibrator.apply(fixture.apply).length,
        }, samples);
        results.push({kind, levels, input_records: fixture.records, forecast_rows: fixture.forecast_rows,
          output_rows: JSON.parse(output).length, output_json_code_units: output.length,
          fit_json_code_units: fixture.fit.length, exact_before_after_output_validated: exactComparison, ...measured});
      } finally { calibrator.free(); }
    }
  }
  return results;
}
