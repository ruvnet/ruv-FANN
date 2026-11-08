'use strict';
// Real Rust/WASM bridge gate. Prerequisites: wasm-pack, Playwright and Chromium.
// Build and test: CHROMIUM_BIN=/usr/bin/chromium node scripts/verify-rust-browser-gpu.cjs --build /tmp/fann-browser-pkg
// Test an existing package: node scripts/verify-rust-browser-gpu.cjs /tmp/fann-browser-pkg
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const crypto = require('node:crypto');
const {execFileSync} = require('node:child_process');
const {chromium} = require('playwright');
const root = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
const pkg = path.resolve(args.find(arg => !arg.startsWith('--')) || process.env.ROOT_BROWSER_GPU_PKG || path.join(root, 'target/browser-gpu-pkg'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

(async () => {
  if (args.includes('--build')) {
    execFileSync(process.env.WASM_PACK || 'wasm-pack', ['build', root, '--dev', '--target', 'web', '--out-dir', pkg,
      '--', '--locked', '--no-default-features', '--features', 'webgpu'], {
      cwd: root, stdio: 'inherit', env: {...process.env, CARGO_BUILD_JOBS: '1', CARGO_PROFILE_DEV_DEBUG: '0', CARGO_INCREMENTAL: '0'},
    });
  }
  const artifacts = new Map([
    ['/ruv_fann.js', {file: path.join(pkg, 'ruv_fann.js'), type: 'text/javascript'}],
    ['/ruv_fann_bg.wasm', {file: path.join(pkg, 'ruv_fann_bg.wasm'), type: 'application/wasm'}],
  ]);
  for (const {file} of artifacts.values()) if (!fs.statSync(file).isFile()) throw new Error(`Missing artifact ${file}`);
  const server = http.createServer((request, response) => {
    response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    const artifact = artifacts.get(request.url);
    if (artifact) {
      response.setHeader('Content-Type', artifact.type);
      response.end(fs.readFileSync(artifact.file));
    } else if (request.url === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><title>Rust browser GPU bridge gate</title>');
    } else { response.writeHead(404); response.end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({
      ...(process.env.CHROMIUM_BIN ? {executablePath: process.env.CHROMIUM_BIN} : {}), headless: true,
      args: ['--enable-unsafe-webgpu', '--use-vulkan=swiftshader', '--enable-features=Vulkan', '--disable-vulkan-surface'],
    });
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async () => {
      const ensure = (condition, message) => { if (!condition) throw new Error(message); };
      const module = await import('/ruv_fann.js');
      const wasm = await module.default();
      ensure(typeof module.WasmGpuBridge === 'function', 'Built Rust class missing');
      // Capture real device objects and preserve all native calls. This permits
      // a genuine device-loss test without substituting compute or readback.
      const nativeRequest = GPUAdapter.prototype.requestDevice;
      const nativeSubmit = GPUQueue.prototype.submit;
      let activeDevice;
      let loseOnSubmit = false;
      GPUAdapter.prototype.requestDevice = async function (...args) {
        activeDevice = await nativeRequest.apply(this, args);
        return activeDevice;
      };
      GPUQueue.prototype.submit = function (...args) {
        const result = nativeSubmit.apply(this, args);
        if (loseOnSubmit) { loseOnSubmit = false; activeDevice.destroy(); }
        return result;
      };
      const bridge = new module.WasmGpuBridge();
      let rejections = 0;
      async function reject(operation, pattern) {
        let rejected = false;
        try { await operation(); } catch (error) {
          rejected = true;
          if (pattern) ensure(pattern.test(String(error)), `Unexpected rejection: ${error}`);
        }
        ensure(rejected, 'Invalid operation succeeded');
        rejections++;
      }
      const initial = bridge.check_feature_support();
      ensure(initial.navigator_gpu && !initial.initialized && initial.max_buffer_bytes === null, 'Uninitialized feature report');
      ensure(bridge.get_performance_metrics().completed_dispatches === 0, 'False initial metrics');
      await reject(() => bridge.gpu_matrix_multiply(new Float32Array([1]), new Float32Array([2]), 1, 1, 1), /not initialized/);
      await reject(() => bridge.initialize_webgpu('canvas'), /Canvas rendering is unsupported/);
      await bridge.initialize_webgpu();
      await bridge.initialize_webgpu();
      const support = bridge.check_feature_support();
      ensure(support.initialized && support.supports_f64 === false && support.max_buffer_bytes > 0, 'Initialized feature report');
      let dispatches = 0;
      let returnedBytes = 0;
      let matrixResults = 0;
      for (const [rows, inner, cols] of [[1, 1, 1], [7, 13, 3], [37, 19, 23], [129, 79, 101]]) {
        const a = Float32Array.from({length: rows * inner}, (_, i) => (i % 17 - 8) / 8);
        const b = Float32Array.from({length: inner * cols}, (_, i) => (i % 11 - 5) / 4);
        const actual = await bridge.gpu_matrix_multiply(a, b, rows, inner, cols);
        ensure(actual instanceof Float32Array && actual.length === rows * cols, 'Matrix output type/shape');
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
          let expected = 0;
          for (let k = 0; k < inner; k++) expected += a[r * inner + k] * b[k * cols + c];
          ensure(Math.abs(actual[r * cols + c] - expected) < 1e-4, `Matrix ${rows}/${inner}/${cols}/${r}/${c}`);
          matrixResults++;
        }
        dispatches++; returnedBytes += actual.byteLength;
      }
      const steepness = Math.fround(0.7);
      const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
      const variants = [
        ['Linear', x => x * steepness], ['Threshold', x => x < 0 ? 0 : 1],
        ['ThresholdSymmetric', x => x < 0 ? -1 : 1], ['Sigmoid', x => sigmoid(x * steepness)],
        ['SigmoidSymmetric', x => Math.tanh(x * steepness)], ['Tanh', x => Math.tanh(x * steepness)],
        ['Gaussian', x => Math.exp(-((x * steepness) ** 2))],
        ['GaussianSymmetric', x => 2 * Math.exp(-((x * steepness) ** 2)) - 1],
        ['Elliot', x => x * steepness / (2 * (1 + Math.abs(x * steepness))) + 0.5],
        ['ElliotSymmetric', x => x * steepness / (1 + Math.abs(x * steepness))],
        ['LinearPiece', x => Math.max(0, Math.min(1, x * steepness))],
        ['LinearPieceSymmetric', x => Math.max(-1, Math.min(1, x * steepness))],
        ['ReLU', x => Math.max(0, x)], ['ReLULeaky', x => x > 0 ? x : 0.01 * x],
        ['LeakyRelu', x => x > 0 ? x : 0.01 * x],
        ['Sin', x => Math.sin(x * steepness) / 2 + 0.5], ['Cos', x => Math.cos(x * steepness) / 2 + 0.5],
        ['SinSymmetric', x => Math.sin(x * steepness)], ['CosSymmetric', x => Math.cos(x * steepness)],
        ['Gelu', x => x * 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))],
        ['Swish', x => x * sigmoid(x)],
      ];
      const inputs = Float32Array.from({length: 1027}, (_, i) => (i % 81 - 40) / 2);
      let activationResults = 0;
      for (const [name, reference] of variants) {
        const actual = await bridge.gpu_apply_activation(inputs, name, steepness);
        ensure(actual.length === inputs.length, `${name} output length`);
        const tolerance = /^(Sin|Cos)/.test(name) ? 1e-3 : 4e-5;
        actual.forEach((value, i) => {
          const expected = reference(inputs[i]);
          ensure(Math.abs(value - expected) <= tolerance * (1 + Math.abs(expected)), `${name}/${i}: ${value} vs ${expected}`);
          activationResults++;
        });
        dispatches++; returnedBytes += actual.byteLength;
      }
      const before = bridge.get_performance_metrics();
      const one = new Float32Array([1]);
      for (const dimension of [0, -1, 1.5, 2 ** 32 + 1, -(2 ** 32) + 1, Infinity, NaN, '1', null]) {
        await reject(() => bridge.gpu_matrix_multiply(one, one, dimension, 1, 1), /dimensions/);
      }
      await reject(() => bridge.gpu_matrix_multiply(new Float32Array([1, 2]), one, 1, 1, 1), /lengths/);
      await reject(() => bridge.gpu_matrix_multiply(new Float32Array(512 * 1024), new Float32Array(1024 * 512), 512, 1024, 512), /multiply-add/);
      await reject(() => bridge.gpu_apply_activation(new Float32Array(), 'Linear', 1), /positive/);
      await reject(() => bridge.gpu_apply_activation(new Float32Array([NaN]), 'Linear', 1), /finite/);
      await reject(() => bridge.gpu_apply_activation(one, 'Linear', Infinity), /finite/);
      await reject(() => bridge.gpu_apply_activation(one, 'unknown', 1), /Unsupported activation/);
      await reject(() => bridge.gpu_apply_activation(one, 'x'.repeat(1024 * 1024), 1), /Unsupported activation/);
      await reject(() => bridge.gpu_apply_activation([1], 'Linear', 1), /Float32Array/);
      await reject(() => bridge.gpu_apply_activation(new Float64Array([1]), 'Linear', 1), /Float32Array/);
      await reject(() => bridge.gpu_apply_activation(Object.create(Float32Array.prototype), 'Linear', 1), /Float32Array/);
      for (const wrong of [new Float64Array([1]), new Int32Array([1])]) {
        Object.setPrototypeOf(wrong, Float32Array.prototype);
        await reject(() => bridge.gpu_apply_activation(wrong, 'Linear', 1), /Float32Array/);
      }
      // Reject oversized typed-array handles before allocating a Rust Vec copy.
      const huge = new Float32Array(support.max_buffer_bytes / 4 + 1);
      const heapBefore = wasm.memory.buffer.byteLength;
      await reject(() => bridge.gpu_apply_activation(huge, 'Linear', 1), /limits/);
      ensure(wasm.memory.buffer.byteLength - heapBefore < 1024 * 1024, 'Oversized input was copied into WASM before rejection');
      const groupOverflow = support.max_workgroups_per_dimension * 8 + 1;
      if (groupOverflow * 4 <= support.max_buffer_bytes) {
        await reject(() => bridge.gpu_matrix_multiply(one, new Float32Array(groupOverflow), 1, 1, groupOverflow), /workgroup/);
      }
      const after = bridge.get_performance_metrics();
      ensure(after.completed_dispatches === before.completed_dispatches && after.returned_bytes === before.returned_bytes && !after.in_flight, 'Rejected work changed completed metrics or left bridge busy');
      // Growable shared storage may change while an async operation runs.
      // Intrinsic metadata ignores caller getters; a fixed snapshot retains eight.
      let sharedGrowthChecked = false;
      if (typeof SharedArrayBuffer === 'function' && typeof SharedArrayBuffer.prototype.grow === 'function') {
        const shared = new SharedArrayBuffer(32, {maxByteLength: 64});
        const tracking = new Float32Array(shared);
        tracking.set([1, -2, 3, -4, 5, -6, 7, -8]);
        Object.defineProperty(tracking, 'buffer', {get() { throw new Error('Caller metadata getter was invoked'); }});
        const pending = bridge.gpu_apply_activation(tracking, 'ReLU', 1);
        await new Promise(resolve => setTimeout(resolve, 0));
        // Grow the real shared backing buffer while the operation owns its snapshot.
        shared.grow(64);
        const actual = await pending;
        ensure(tracking.length === 16 && actual.length === 8, 'Growable input escaped its checked view');
        ensure(Array.from(actual).join(',') === '1,0,3,0,5,0,7,0', 'Shared buffer GPU results');
        dispatches++; returnedBytes += actual.byteLength; sharedGrowthChecked = true;
      }
      // A caller may pass a view of this very WASM memory. Rust Vec allocation
      // must not detach it before the JS-owned snapshot is copied.
      const selfLength = 2 * 1024 * 1024;
      const offset = wasm.memory.buffer.byteLength;
      wasm.memory.grow(selfLength * 4 / 65536);
      const selfBuffer = wasm.memory.buffer;
      const selfInput = new Float32Array(selfBuffer, offset, selfLength);
      selfInput.fill(0.25); selfInput[0] = -0.5; selfInput[selfLength - 1] = 4;
      const selfResult = await bridge.gpu_apply_activation(selfInput, 'Linear', 2);
      ensure(selfResult.length === selfLength && selfResult[0] === -1 && selfResult[selfLength - 1] === 8, 'WASM-owned input edge values');
      ensure(selfResult.subarray(1, -1).every(value => value === 0.5), 'WASM-owned input lost values after memory growth');
      ensure(wasm.memory.buffer !== selfBuffer && selfInput.byteLength === 0, 'Test did not force input detachment during Rust allocation');
      dispatches++; returnedBytes += selfResult.byteLength;
      // Fail concurrent work explicitly instead of corrupting per-device error scopes.
      const pending = bridge.gpu_apply_activation(inputs, 'ReLU', 1);
      await reject(() => bridge.gpu_apply_activation(inputs, 'Linear', 1), /already in progress/);
      const actual = await pending;
      dispatches++; returnedBytes += actual.byteLength;
      ensure(actual[0] === 0 && actual[80] === 20, 'Concurrent operation corrupted output');
      const metrics = bridge.get_performance_metrics();
      ensure(metrics.completed_dispatches === dispatches && metrics.returned_bytes === returnedBytes && !metrics.in_flight, 'Completed metrics differ from actual readbacks');
      loseOnSubmit = true;
      await reject(() => bridge.gpu_apply_activation(inputs, 'ReLU', 1), /Abort|lost|destroy|invalid/i);
      ensure(!bridge.get_performance_metrics().in_flight, 'Device loss left the operation guard busy');
      bridge.destroy();
      ensure(!bridge.check_feature_support().initialized, 'Destroy did not release device state');
      await reject(() => bridge.gpu_apply_activation(one, 'Linear', 1), /not initialized/);
      await bridge.initialize_webgpu();
      const resumed = await bridge.gpu_matrix_multiply(new Float32Array([2]), new Float32Array([3]), 1, 1, 1);
      ensure(resumed.length === 1 && resumed[0] === 6, 'Reinitialized GPU result');
      bridge.destroy(); bridge.free();
      GPUAdapter.prototype.requestDevice = nativeRequest;
      GPUQueue.prototype.submit = nativeSubmit;
      return {matrixResults, activationVariants: variants.length, activationResults, rejectedRequests: rejections,
        sharedGrowthChecked, completedBeforeReinitialization: metrics,
        maxBufferBytes: support.max_buffer_bytes, maxWorkgroupsPerDimension: support.max_workgroups_per_dimension};
    });
    if (errors.length) throw new Error(errors.join('\n'));
    console.log(JSON.stringify({...result, wasmSha256: hash(path.join(pkg, 'ruv_fann_bg.wasm')),
      bindingsSha256: hash(path.join(pkg, 'ruv_fann.js')), lockSha256: hash(path.join(root, 'Cargo.lock'))}));
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
