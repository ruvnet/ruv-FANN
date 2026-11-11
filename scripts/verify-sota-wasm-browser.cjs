'use strict';
// Test the actual published core and SIMD artifacts in Chromium over HTTP.
// Prerequisites: Playwright and Chromium. No Rust toolchain is needed to run.
// CHROMIUM_BIN=/usr/bin/chromium node scripts/verify-sota-wasm-browser.cjs
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const crypto = require('node:crypto');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');
const root = path.resolve(__dirname, '..');
const npmRoot = path.resolve(process.argv[2] || path.join(root, 'ruv-swarm/npm'));
const probeDir = process.env.BATCH_SIMD_PROBE_DIR ? path.resolve(process.env.BATCH_SIMD_PROBE_DIR) : null;
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

(async () => {
  const {verifyWasmBuild} = await import(pathToFileURL(path.join(npmRoot, 'scripts/wasm-provenance.js')).href);
  // Historical optional binaries can remain in a source checkout. Exercise
  // the two binaries explicitly shipped by this package, plus receipted glue.
  await verifyWasmBuild(npmRoot, null, ['ruv_swarm_wasm_bg.wasm', 'ruv_swarm_simd.wasm']);
  const routes = new Map([
    ['/bindings.js', {file: path.join(npmRoot, 'wasm/ruv_swarm_wasm.js'), type: 'text/javascript'}],
    ['/core.wasm', {file: path.join(npmRoot, 'wasm/ruv_swarm_wasm_bg.wasm'), type: 'application/wasm'}],
    ['/simd.wasm', {file: path.join(npmRoot, 'wasm/ruv_swarm_simd.wasm'), type: 'application/wasm'}],
    ...['dense', 'sparse'].map(kind => [`/model-${kind}.json`, {
      file: path.join(root, `ruv-swarm/crates/ruv-swarm-wasm/tests/fixtures/prepared-${kind}.json`),
      type: 'application/json',
    }]),
    ['/examples/wasm-prepared-conformal.html', {
      file: path.join(root, 'ruv-swarm/npm/examples/wasm-prepared-conformal.html'), type: 'text/html',
    }],
    ['/wasm/ruv_swarm_wasm.js', {file: path.join(npmRoot, 'wasm/ruv_swarm_wasm.js'), type: 'text/javascript'}],
    ['/wasm/ruv_swarm_wasm_bg.wasm', {file: path.join(npmRoot, 'wasm/ruv_swarm_wasm_bg.wasm'), type: 'application/wasm'}],
  ]);
  const hashes = {};
  if (probeDir) {
    routes.set('/probe.js', {file: path.join(probeDir, 'batch_probe.js'), type: 'text/javascript'});
    routes.set('/probe.wasm', {file: path.join(probeDir, 'batch_probe_bg.wasm'), type: 'application/wasm'});
  }
  for (const [name, artifact] of routes) {
    artifact.bytes = fs.readFileSync(artifact.file);
    hashes[name] = hash(artifact.bytes);
  }
  const server = http.createServer((request, response) => {
    response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    response.setHeader('Cache-Control', 'no-store');
    const pathname = new URL(request.url, 'http://localhost').pathname;
    const artifact = routes.get(pathname);
    if (artifact) {
      response.setHeader('Content-Type', artifact.type);
      response.end(artifact.bytes);
    } else if (pathname === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><title>Prepared inference and calibration WASM gate</title>');
    } else {
      response.writeHead(404);
      response.end();
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({
      ...(process.env.CHROMIUM_BIN ? {executablePath: process.env.CHROMIUM_BIN} : {}),
      headless: true,
    });
    const page = await browser.newPage();
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(String(error)));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const variants = await page.evaluate(async () => {
      const ensure = (condition, message) => { if (!condition) throw new Error(message); };
      const same = (a, b, message) => ensure(JSON.stringify(a) === JSON.stringify(b), message);
      const results = [];
      for (const variant of ['core', 'simd']) {
        const bindings = await import(`/bindings.js?variant=${variant}`);
        const wasm = await bindings.default({module_or_path: fetch(`/${variant}.wasm`)});
        const features = JSON.parse(bindings.get_features());
        ensure(features.prepared_inference && features.conformal_intervals, 'Feature declarations missing');
        ensure(features.simd_support === (variant === 'simd'), 'Unexpected core/SIMD build selection');
        ensure(features.prepared_batching && features.prepared_batch_simd === (variant === 'simd'), 'Batch capability declaration');
        let rejections = 0;
        const reject = operation => {
          let rejected = false;
          try { operation(); } catch { rejected = true; }
          ensure(rejected, `${variant}: invalid operation succeeded`);
          rejections++;
        };
        const rejectBeforeCopy = operation => {
          const before = wasm.memory.buffer.byteLength;
          reject(operation);
          ensure(wasm.memory.buffer.byteLength - before < 1024 * 1024, 'Oversized JSON was copied before rejection');
        };
        const network = new bindings.WasmNeuralNetwork(new Uint32Array([2, 2, 1]), bindings.ActivationFunction.Linear);
        let networkFreed = false;
        let prepared;
        let calibrator;
        try {
          network.set_weights(new Float64Array([1, 2, 3, 4, 5, 6]));
          const input = new Float64Array([1, 2]);
          same(Array.from(network.run(input)), [95], 'Legacy weight layout');
          prepared = network.prepare();
          ensure(prepared.input_size === 2 && prepared.output_size === 1 && prepared.estimated_bytes > 0, 'Prepared metadata');
          same(Array.from(prepared.run(input)), [95], 'Prepared numerical result');
          network.set_weights(new Float64Array(6));
          same(Array.from(network.run(input)), [0], 'Source edit did not take effect');
          network.free();
          networkFreed = true;
          const output = new Float64Array([12345]);
          prepared.run_into(input, output);
          same(Array.from(output), [95], 'Snapshot depends on edited/freed source');

          for (const bad of [new Float64Array([1]), new Float64Array([NaN, 2]),
            new Float64Array([Infinity, 2]), new Float32Array([1, 2]), [1, 2],
            Object.create(Float64Array.prototype)]) {
            output[0] = 12345;
            reject(() => prepared.run_into(bad, output));
            same(Array.from(output), [12345], 'Invalid input changed caller output');
          }
          reject(() => prepared.run_into(input, new Float64Array(2)));
          reject(() => prepared.run_into(input, new Float32Array(1)));
          const fake = new Int32Array([1, 2]);
          Object.setPrototypeOf(fake, Float64Array.prototype);
          reject(() => prepared.run_into(fake, output));
          const guarded = new Float64Array([1, 2]);
          Object.defineProperty(guarded, 'length', {get() { throw new Error('Caller length getter invoked'); }});
          Object.defineProperty(guarded, 'buffer', {get() { throw new Error('Caller buffer getter invoked'); }});
          prepared.run_into(guarded, output);
          same(Array.from(output), [95], 'Intrinsic metadata checks');
          wasm.memory.grow(1);
          const moduleView = new Float64Array(wasm.memory.buffer, 0, 2);
          reject(() => prepared.run_into(moduleView, output));
          reject(() => prepared.run_into(input, new Float64Array(wasm.memory.buffer, 0, 1)));
          let sharedChecked = false;
          if (typeof SharedArrayBuffer === 'function') {
            reject(() => prepared.run_into(new Float64Array(new SharedArrayBuffer(16)), output));
            reject(() => prepared.run_into(input, new Float64Array(new SharedArrayBuffer(8))));
            const shared = new SharedArrayBuffer(16);
            const sharedView = new Float64Array(shared);
            Object.setPrototypeOf(shared, ArrayBuffer.prototype);
            reject(() => prepared.run_into(sharedView, output));
            sharedChecked = true;
          }
          const huge = new Float64Array(1000001);
          const beforeReject = wasm.memory.buffer.byteLength;
          reject(() => prepared.run_into(huge, output));
          ensure(wasm.memory.buffer.byteLength - beforeReject < 1024 * 1024, 'Oversized input was copied before rejection');
          prepared.run_into(input, output);
          const steadyBytes = wasm.memory.buffer.byteLength;
          let checksum = 0;
          for (let i = 0; i < 1000; i++) {
            prepared.run_into(input, output);
            checksum += output[0];
          }
          ensure(checksum === 95000, 'Repeated prepared calls');
          ensure(wasm.memory.buffer.byteLength === steadyBytes, 'Steady prepared workload grew WASM memory');
          for (const kind of ['dense', 'sparse']) {
            const fixture = await (await fetch(`/model-${kind}.json`)).json();
            const imported = bindings.WasmPreparedNetwork.from_network_json(JSON.stringify(fixture.network));
            try {
              const actual = imported.run(new Float64Array(fixture.input));
              ensure(actual.length === fixture.output.length, 'Imported native output shape');
              actual.forEach((value, i) => ensure(Math.abs(value - fixture.output[i]) < 1e-12, `Native ${kind} parity`));
              const batch = imported.create_batch(17);
              try {
                for (const rows of [0, 1, 2, 3, 8, 17]) {
                  const inputs = Float64Array.from({length: rows * fixture.input.length}, (_, i) => fixture.input[i % fixture.input.length] + Math.floor(i / fixture.input.length) / 128);
                  const expected = [];
                  for (let row = 0; row < rows; row++) expected.push(...imported.run(inputs.subarray(row * fixture.input.length, (row + 1) * fixture.input.length)));
                  const output = new Float64Array(expected.length);
                  batch.run_into(inputs, output, rows);
                  output.forEach((value, i) => ensure(Math.abs(value - expected[i]) < 1e-12, `${kind} batch/tail parity`));
                }
              } finally { batch.free(); }
            } finally {
              imported.free();
            }
          }
          reject(() => bindings.WasmPreparedNetwork.from_network_json('{}'));
          rejectBeforeCopy(() => bindings.WasmPreparedNetwork.from_network_json(' '.repeat(4 * 1024 * 1024 + 1)));
          const batch = prepared.create_batch(17);
          try {
            const inputs = Float64Array.from({length: 34}, (_, i) => i % 2 ? 2 : Math.floor(i / 2) + 1);
            const output = new Float64Array(17);
            batch.run_into(inputs, output, 17);
            same(Array.from(output), Array.from({length: 17}, (_, i) => 95 + 17 * i), 'Batched row-major output');
            const invalid = inputs.slice(); invalid[32] = NaN;
            output.fill(12345);
            reject(() => batch.run_into(invalid, output, 17));
            ensure(output.every(value => value === 12345), 'Later invalid row changed output');
            for (const rows of [-1, 1.5, NaN, Infinity, 2 ** 32 + 1, '17', new Number(17), 17n, 18]) {
              reject(() => batch.run_into(inputs, output, rows));
            }
            reject(() => batch.run_into(new Float32Array(34), output, 17));
            reject(() => batch.run_into(new Float64Array(wasm.memory.buffer, 0, 34), output, 17));
            if (typeof SharedArrayBuffer === 'function') reject(() => batch.run_into(new Float64Array(new SharedArrayBuffer(34 * 8)), output, 17));
            prepared.free(); prepared = undefined;
            batch.run_into(inputs, output, 17);
            ensure(output[0] === 95 && output[16] === 367, 'Batch depends on freed prepared handle');
            const steady = wasm.memory.buffer.byteLength;
            for (let i = 0; i < 1000; i++) batch.run_into(inputs, output, 17);
            ensure(wasm.memory.buffer.byteLength === steady, 'Reused batch grew WASM memory');
          } finally { batch.free(); }

          const records = [];
          for (const [series, scale] of [['alpha', 1], ['beta', 2]]) {
            for (let window = 0; window < 5; window++) {
              for (let step = 1; step <= 2; step++) {
                const actual = 100 + window + step;
                for (const [model, factor] of [['base', 1], ['other', 10]]) {
                  records.push({series, model, cutoff: window * 10, timestamp: window * 10 + step,
                    actual, prediction: actual - scale * factor * (window + 1) * step});
                }
              }
            }
          }
          const config = {as_of: 100, horizon: 2, coverages: [0.8, 0.5], records};
          calibrator = new bindings.WasmConformalCalibrator(JSON.stringify(config));
          const groups = JSON.parse(calibrator.groups());
          ensure(groups.length === 8 && groups.every(g => g.samples === 5), 'Calibration group isolation');
          const forecasts = [];
          for (const series of ['alpha', 'beta']) {
            for (const model of ['base', 'other']) {
              for (let step = 1; step <= 2; step++) {
                forecasts.push({series, model, origin: 200, timestamp: 200 + step, prediction: 20});
              }
            }
          }
          const request = {records: forecasts};
          const intervals = JSON.parse(calibrator.apply(JSON.stringify(request)));
          ensure(intervals.length === 16, 'Interval output shape');
          const first = intervals.find(r => r.series === 'alpha' && r.model === 'base' && r.horizon === 1 && r.coverage === 0.5);
          ensure(first.lower === 17 && first.upper === 23 && first.point === 20 && first.calibration_samples === 5, 'Corrected rank');
          const isolated = intervals.find(r => r.series === 'beta' && r.model === 'other' && r.horizon === 2 && r.coverage === 0.8);
          ensure(isolated.lower === -180 && isolated.upper === 220, 'Series/model/horizon separation');
          ensure(intervals.every(r => r.forecast_origin === 200 && r.calibration_as_of === 100 && r.timestamp > 200), 'Interval provenance');
          reject(() => new bindings.WasmConformalCalibrator(JSON.stringify({...config, coverages: [0.9]})));
          reject(() => new bindings.WasmConformalCalibrator(JSON.stringify({...config, as_of: 20})));
          reject(() => new bindings.WasmConformalCalibrator(JSON.stringify({...config, as_of: 2 ** 53})));
          reject(() => new bindings.WasmConformalCalibrator(JSON.stringify({...config, records: [...records, records[0]]})));
          rejectBeforeCopy(() => new bindings.WasmConformalCalibrator(' '.repeat(4 * 1024 * 1024 + 1)));
          rejectBeforeCopy(() => calibrator.apply(' '.repeat(4 * 1024 * 1024 + 1)));
          reject(() => new bindings.WasmConformalCalibrator(42));
          reject(() => calibrator.apply({records: forecasts}));
          reject(() => calibrator.apply(JSON.stringify({records: forecasts.map(r => ({...r, origin: 99}))})));
          reject(() => calibrator.apply(JSON.stringify({records: forecasts.map(r => ({...r, timestamp: 199}))})));
          reject(() => calibrator.apply(JSON.stringify({records: forecasts.slice(1)})));
          reject(() => calibrator.apply(JSON.stringify({records: forecasts.map(r => ({...r, prediction: NaN}))})));
          same(JSON.parse(calibrator.apply(JSON.stringify(request))), intervals, 'Rejected work mutated calibration');
          for (const actual of ['316912650057057350374175801344', '3.1691265005705735e29']) {
            const precise = new bindings.WasmConformalCalibrator(`{"as_of":1,"horizon":1,"coverages":[0.5],"records":[{"series":"p","model":"m","cutoff":0,"timestamp":1,"actual":${actual},"prediction":0}]}`);
            try {
              ensure(JSON.parse(precise.groups())[0].radii[0] === 2 ** 98, 'JSON Float64 parsing changed an exact value');
            } finally {
              precise.free();
            }
          }
          reject(() => new bindings.WasmConformalCalibrator('{"as_of":1,"horizon":1,"coverages":[0.5],"records":[{"series":"p","model":"m","cutoff":0,"timestamp":1,"actual":9007199254740993,"prediction":9007199254740992}]}'));
          const longName = 'x'.repeat(250);
          const largeRecords = [];
          for (let window = 0; window < 5; window++) {
            for (let step = 1; step <= 1000; step++) {
              largeRecords.push({series: longName, model: longName, cutoff: window * 2000,
                timestamp: window * 2000 + step, actual: 0, prediction: -window - 1});
            }
          }
          const large = new bindings.WasmConformalCalibrator(JSON.stringify({as_of: 10000, horizon: 1000,
            coverages: Array.from({length: 16}, (_, i) => (i + 1) / 20), records: largeRecords}));
          try {
            const request = {records: Array.from({length: 1000}, (_, i) => ({series: longName,
              model: longName, origin: 11000, timestamp: 12001 + i, prediction: 20}))};
            reject(() => large.apply(JSON.stringify(request)));
          } finally {
            large.free();
          }
          results.push({variant, prepared_calls: 1000, checksum, wasm_memory_bytes: steadyBytes,
            memory_growth_during_loop: 0, shared_buffers_checked: sharedChecked,
            module_memory_growth_checked: true,
            calibration_records: records.length, groups: groups.length, intervals: intervals.length,
            native_model_imports: 2, conservative_output_budget_checked: true,
            batched_rows_per_call: 17, reused_batch_calls: 1000,
            exact_json_float64_cases: 2,
            rejected_invalid_operations: rejections});
        } finally {
          if (!networkFreed) network.free();
          if (prepared) prepared.free();
          if (calibrator) calibrator.free();
        }
      }
      return results;
    });
    let nativeSimdProbe = null;
    if (probeDir) {
      nativeSimdProbe = await page.evaluate(async () => {
        const module = await import('/probe.js');
        await module.default({module_or_path: fetch('/probe.wasm')});
        const result = JSON.parse(module.verify_prepared_batches());
        if (!result.passed || !result.simd128 || !result.bitwise_scalar_equivalence || result.float32_rows !== 2310 || result.float64_rows !== 2310) {
          throw new Error('Native f32/f64 SIMD probe failed');
        }
        if (result.wide_single_input_width !== 65 || result.wide_single_float32_comparisons !== 105 ||
            result.wide_single_float64_comparisons !== 105 || result.wide_single_float32_overflow_cases !== 3 ||
            result.wide_single_float64_overflow_cases !== 3 ||
            JSON.stringify(result.wide_single_topologies) !== JSON.stringify(['canonical', 'mixed', 'biasless'])) {
          throw new Error('Actual wide65 single inference parity, overflow atomicity or recovery proof missing');
        }
        return result;
      });
    }
    await page.goto(`http://127.0.0.1:${server.address().port}/examples/wasm-prepared-conformal.html`);
    await page.waitForFunction(() => {
      const text = document.getElementById('result')?.textContent || '';
      return text.startsWith('{') || text.startsWith('Could not run');
    });
    const example = JSON.parse(await page.locator('#result').textContent());
    if (example.neural_output[0] !== 2 || example.intervals[0].lower !== 11 || example.intervals[0].upper !== 29) {
      throw new Error('Documented browser example returned incorrect results');
    }
    if (pageErrors.length) throw new Error(pageErrors.join('\n'));
    console.log(JSON.stringify({browser: await browser.version(), artifact_sha256: hashes, variants, native_simd_probe: nativeSimdProbe,
      documented_browser_example: {passed: true, neural_output: 2, interval: [11, 29]}}, null, 2));
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
