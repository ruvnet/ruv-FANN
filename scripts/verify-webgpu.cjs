'use strict';
// Executes repository WGSL in a real browser. Requires Playwright and Chromium.
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {chromium} = require('playwright');

(async () => {
  const directory = path.join(__dirname, '../src/webgpu/shaders');
  const sources = Object.fromEntries(fs.readdirSync(directory)
    .filter(name => name.endsWith('.wgsl'))
    .map(name => [name, fs.readFileSync(path.join(directory, name), 'utf8')]));
  const server = http.createServer((request, response) => {
    response.setHeader('Content-Type', 'text/html');
    response.end('<!doctype html><title>FANN WebGPU release checks</title>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({
      ...(process.env.CHROMIUM_BIN ? {executablePath: process.env.CHROMIUM_BIN} : {}),
      headless: true,
      args: ['--enable-unsafe-webgpu', '--use-vulkan=swiftshader',
        '--enable-features=Vulkan', '--disable-vulkan-surface'],
    });
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async sources => {
      if (!navigator.gpu) throw new Error('WebGPU unavailable');
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No compute adapter');
      const device = await adapter.requestDevice();
      const failures = [];
      device.addEventListener('uncapturederror', event => failures.push(event.error.message));
      const modules = {};
      try {
        const compilationErrors = [];
        for (const [name, code] of Object.entries(sources)) {
          const module = device.createShaderModule({code, label: name});
          const info = await module.getCompilationInfo();
          const errors = info.messages.filter(message => message.type === 'error');
          if (errors.length) compilationErrors.push(`${name}: ${errors.map(error => `${error.lineNum}:${error.linePos} ${error.message}`).join('; ')}`);
          modules[name] = module;
        }
        if (compilationErrors.length) throw new Error(compilationErrors.join('\n'));
        const ensure = (condition, message) => { if (!condition) throw new Error(message); };
        async function dispatch(module, entryPoint, inputs, uniforms, length, groups) {
          device.pushErrorScope('validation');
          const owned = [];
          function upload(data, usage) {
            const buffer = device.createBuffer({size: data.byteLength,
              usage: usage | GPUBufferUsage.COPY_DST});
            device.queue.writeBuffer(buffer, 0, data);
            owned.push(buffer);
            return buffer;
          }
          let staging;
          try {
            const pipeline = await device.createComputePipelineAsync({layout: 'auto',
              compute: {module, entryPoint}});
            const buffers = inputs.map(data => upload(data, GPUBufferUsage.STORAGE));
            const output = device.createBuffer({size: length * 4,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
            owned.push(output);
            buffers.push(output, upload(uniforms, GPUBufferUsage.UNIFORM));
            const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
              entries: buffers.map((buffer, binding) => ({binding, resource: {buffer}}))});
            staging = device.createBuffer({size: length * 4,
              usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
            owned.push(staging);
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, group);
            pass.dispatchWorkgroups(...groups);
            pass.end();
            encoder.copyBufferToBuffer(output, 0, staging, 0, length * 4);
            device.queue.submit([encoder.finish()]);
            await staging.mapAsync(GPUMapMode.READ);
            const values = new Float32Array(staging.getMappedRange()).slice();
            staging.unmap();
            const error = await device.popErrorScope();
            if (error) throw new Error(error.message);
            return values;
          } finally { for (const buffer of owned) buffer.destroy(); }
        }
        const [rows, cols, batches] = [129, 79, 101];
        const matrix = Float32Array.from({length: rows * cols}, (_, i) => (i % 13) / 8 - 0.75);
        const vectors = Float32Array.from({length: cols * batches}, (_, i) =>
          ((i % cols + Math.floor(i / cols)) % 11) / 4 - 1);
        function verifyMatrix(values, count) {
          ensure(values.length === rows * count, 'Output length');
          for (let b = 0; b < count; b++) {
            for (let r = 0; r < rows; r++) {
              let expected = 0;
              for (let c = 0; c < cols; c++) expected += matrix[r * cols + c] * vectors[b * cols + c];
              ensure(Math.abs(values[b * rows + r] - expected) < 1e-4, `Matrix result ${b}/${r}`);
            }
          }
        }
        verifyMatrix(await dispatch(modules['matrix_vector_multiply.wgsl'], 'main',
          [matrix, vectors.slice(0, cols)], new Uint32Array([rows, cols, 0, 0]), rows,
          [Math.ceil(rows / 32)]), 1);
        verifyMatrix(await dispatch(modules['batch_matrix_vector_multiply.wgsl'], 'main',
          [matrix, vectors], new Uint32Array([rows, cols, batches, 0]), rows * batches,
          [Math.ceil(rows / 32), batches]), batches);
        const inputs = Float32Array.from({length: 1027}, (_, i) => ((i % 81) - 40) * 0.5);
        const steepness = Math.fround(0.7);
        const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
        const variants = [
          ['linear_main', x => x * steepness], ['threshold_main', x => x < 0 ? 0 : 1],
          ['threshold_symmetric_main', x => x < 0 ? -1 : 1], ['sigmoid_main', x => sigmoid(x * steepness)],
          ['tanh_main', x => Math.tanh(x * steepness)], ['tanh_main', x => Math.tanh(x * steepness)],
          ['gaussian_main', x => Math.exp(-((x * steepness) ** 2))],
          ['gaussian_symmetric_main', x => 2 * Math.exp(-((x * steepness) ** 2)) - 1],
          ['elliott_main', x => (x * steepness / 2) / (1 + Math.abs(x * steepness)) + 0.5],
          ['elliott_symmetric_main', x => x * steepness / (1 + Math.abs(x * steepness))],
          ['linear_piece_main', x => Math.max(0, Math.min(1, x * steepness))],
          ['linear_piece_symmetric_main', x => Math.max(-1, Math.min(1, x * steepness))],
          ['relu_main', x => Math.max(0, x)], ['leaky_relu_main', x => x > 0 ? x : 0.01 * x],
          ['leaky_relu_main', x => x > 0 ? x : 0.01 * x],
          ['sin_main', x => Math.sin(x * steepness) / 2 + 0.5],
          ['cos_main', x => Math.cos(x * steepness) / 2 + 0.5],
          ['sin_symmetric_main', x => Math.sin(x * steepness)],
          ['cos_symmetric_main', x => Math.cos(x * steepness)],
          ['gelu_main', x => x * 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))],
          ['swish_main', x => x * sigmoid(x)],
        ];
        const uniform = new ArrayBuffer(16);
        new DataView(uniform).setUint32(0, inputs.length, true);
        new DataView(uniform).setFloat32(4, steepness, true);
        new DataView(uniform).setFloat32(8, 0.01, true);
        for (const [entry, reference] of variants) {
          const values = await dispatch(modules['activation_functions.wgsl'], entry,
            [inputs], new Uint8Array(uniform), inputs.length, [Math.ceil(inputs.length / 256)]);
          values.forEach((value, i) => {
            const expected = reference(inputs[i]);
            // WGSL transcendental accuracy depends on the backend; SwiftShader
            // trigonometry has a larger error than the native Mesa path.
            const tolerance = /^(sin|cos)/.test(entry) ? 1e-3 : 4e-5;
            ensure(Math.abs(value - expected) <= tolerance * (1 + Math.abs(expected)), `${entry}/${i}: ${value} != ${expected}`);
          });
        }
        // Verify actual optimizer shader writes, including tail invocations and
        // multiple bias-correction steps, against scalar Adam/AdamW equations.
        let optimizerResults = 0;
        for (const entryPoint of ['adam_update', 'adamw_update']) {
          const count = 67;
          const original = Float32Array.from({length: count}, (_, i) => 0.2 + i / 100);
          const gradients = Float32Array.from({length: count}, (_, i) => (i % 9 - 4) / 10);
          const expected = Array.from(original);
          const moments = new Float64Array(count);
          const variance = new Float64Array(count);
          const buffers = [];
          const pipeline = await device.createComputePipelineAsync({layout: 'auto',
            compute: {module: modules['adam_optimizer.wgsl'], entryPoint}});
          for (const data of [original, gradients, new Float32Array(count), new Float32Array(count)]) {
            const buffer = device.createBuffer({size: data.byteLength,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
            device.queue.writeBuffer(buffer, 0, data);
            buffers.push(buffer);
          }
          const parameters = device.createBuffer({size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
          buffers.push(parameters);
          const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
            entries: buffers.map((buffer, binding) => ({binding, resource: {buffer}}))});
          const staging = device.createBuffer({size: count * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
          try {
            for (let step = 1; step <= 5; step++) {
              const uniform = new ArrayBuffer(32);
              new Float32Array(uniform).set([0.01, 0.9, 0.999, 1e-8, 0.02]);
              new DataView(uniform).setUint32(entryPoint === 'adam_update' ? 16 : 20, step, true);
              device.queue.writeBuffer(parameters, 0, uniform);
              const encoder = device.createCommandEncoder();
              const pass = encoder.beginComputePass();
              pass.setPipeline(pipeline); pass.setBindGroup(0, group);
              pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
              encoder.copyBufferToBuffer(buffers[0], 0, staging, 0, count * 4);
              device.queue.submit([encoder.finish()]);
              await staging.mapAsync(GPUMapMode.READ);
              const actual = new Float32Array(staging.getMappedRange());
              for (let i = 0; i < count; i++) {
                moments[i] = 0.9 * moments[i] + 0.1 * gradients[i];
                variance[i] = 0.999 * variance[i] + 0.001 * gradients[i] ** 2;
                const update = 0.01 * (moments[i] / (1 - 0.9 ** step)) /
                  (Math.sqrt(variance[i] / (1 - 0.999 ** step)) + 1e-8);
                expected[i] -= update + (entryPoint === 'adamw_update' ? 0.01 * 0.02 * expected[i] : 0);
                ensure(Math.abs(actual[i] - expected[i]) < 2e-5, `${entryPoint}/${step}/${i}`);
                optimizerResults++;
              }
              staging.unmap();
            }
          } finally { staging.destroy(); for (const buffer of buffers) buffer.destroy(); }
        }
        ensure(failures.length === 0, failures.join('; '));
        return {shaderModules: Object.keys(modules).length, matrixResults: rows * (batches + 1),
          activationVariants: variants.length, activationResults: variants.length * inputs.length,
          optimizerResults,
          activationRelativeTolerance: 4e-5, trigonometricRelativeTolerance: 1e-3,
          adapter: {vendor: adapter.info?.vendor, architecture: adapter.info?.architecture,
            isFallbackAdapter: adapter.info?.isFallbackAdapter}};
      } finally { device.destroy(); }
    }, sources);
    console.log(JSON.stringify(result));
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
