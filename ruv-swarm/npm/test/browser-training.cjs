/** Real Chromium acceptance. Playwright is supplied by the development host. */
const { chromium } = require('playwright');
const playwrightVersion = require('playwright/package.json').version;
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');

function checkedChromiumVersion(value, label) {
  if (typeof value !== 'string' || value.length > 64 ||
    !/^[1-9]\d*\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$/.test(value) ||
    value.split('.').some(part => !Number.isSafeInteger(Number(part)))) {
    throw new Error(label + ' must be a full Chromium MAJOR.MINOR.BUILD.PATCH version');
  }
  return value;
}

async function browserProof() {
  const check = (condition, label) => { if (!condition) throw new Error(label); };
  const read = async url => {
    const response = await fetch(url);
    check(response.ok, 'Failed local asset: ' + url);
    return response;
  };
  const fixture = await (await read('/candle-fixture.json')).json();
  check(fixture.oracle === 'candle-core/candle-nn 0.11.1 CPU f64', 'Actual Candle fixture required');
  check(fixture.absolute_tolerance === 1e-12 && fixture.relative_tolerance === 1e-12, 'Candle tolerance');
  check(crossOriginIsolated && typeof SharedArrayBuffer === 'function', 'SharedArrayBuffer isolation');
  const { createNeuralIndex } = await import('/integrations/ruv-fann/src/index.mjs');
  const { createRuvectorStore } = await import('/integrations/ruv-fann/src/ruvector.mjs');
  const vectorBytes = new Uint8Array(await (await read(
    '/integrations/ruv-fann/node_modules/@ruvector/wasm/pkg/ruvector_wasm_bg.wasm'
  )).arrayBuffer());
  const flat = json => JSON.parse(json).layers.flatMap(layer => [...layer.weights, ...(layer.bias ?? [])]);
  const simple = () => JSON.stringify({ schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 1, weights: [0], bias: [0], activation: 'Linear', steepness: 1 }
  ] });
  const cases = [];
  for (const variant of ['core', 'simd']) {
    const module = await import('/ruv-swarm/npm/wasm/ruv_swarm_wasm.js?browser-training=' + variant);
    const binary = variant === 'simd' ? 'ruv_swarm_simd.wasm' : 'ruv_swarm_wasm_bg.wasm';
    const wasm = await module.default({ module_or_path: await (await read('/ruv-swarm/npm/wasm/' + binary)).arrayBuffer() });
    const simdSupport = JSON.parse(module.get_features()).simd_support;
    check(simdSupport === (variant === 'simd'), 'Incorrect WASM variant');
    const Train = module.WasmTrainableNetwork;
    check(typeof Train?.from_dense_layers_json === 'function', 'Actual WASM training export');
    const close = (actual, expected, label, tolerance = 1e-12) => {
      check(actual.length === expected.length, label + ' length');
      let maximumAbsoluteError = 0;
      for (let i = 0; i < actual.length; i++) {
        const error = Math.abs(actual[i] - expected[i]);
        check(Number.isFinite(actual[i]) && Number.isFinite(expected[i]) &&
          error <= tolerance * (1 + Math.abs(expected[i])), label + '[' + i + ']');
        maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
      }
      return maximumAbsoluteError;
    };
    const reject = (action, label) => {
      let rejected = false, value;
      try { value = action(); }
      catch { rejected = true; }
      if (!rejected && typeof value?.free === 'function') value.free();
      check(rejected, 'Expected rejection: ' + label);
    };
    const optimizers = [];
    for (const optimizer of ['sgd', 'adamw']) {
      const expectedSteps = fixture[optimizer + '_steps'];
      check(Array.isArray(expectedSteps) && expectedSteps.length === 3, 'Three Candle update references');
      const train = Train.from_dense_layers_json(fixture.dense_json, 1);
      let frozen, beforeFree, maximumAbsoluteError = 0;
      try {
        check(train.input_size === 3 && train.output_size === 2 && train.parameter_count === 24, 'Candle architecture');
        if (optimizer === 'adamw') {
          const config = fixture.adamw_config;
          train.configure_adamw(config.learning_rate, config.beta1, config.beta2, config.epsilon, config.weight_decay);
        }
        for (let step = 0; step < expectedSteps.length; step++) {
          const expected = expectedSteps[step];
          const loss = train.mse_backward(Float64Array.from(fixture.input), Float64Array.from(fixture.training_target), 1);
          for (const [actual, reference, label] of [
            [[loss], [expected.loss], 'Candle MSE'],
            [train.predictions(), expected.predictions, 'Candle predictions'],
            [train.parameter_gradients(), expected.parameter_gradients, 'Candle parameter gradients'],
            [train.input_gradients(), expected.input_gradients, 'Candle input gradients']
          ]) maximumAbsoluteError = Math.max(maximumAbsoluteError, close(actual, reference, label));
          if (step === 0) {
            close([loss], [fixture.training_loss], 'Initial Candle loss');
            close(train.predictions(), fixture.expected, 'Initial Candle prediction');
            close(train.parameter_gradients(), fixture.parameter_gradients, 'Initial parameter gradients');
            close(train.input_gradients(), fixture.input_gradients, 'Initial input gradients');
            train.parameter_gradients()[0] = 999;
            close(train.parameter_gradients(), expected.parameter_gradients, 'Owned gradient copy');
          }
          if (optimizer === 'sgd') train.sgd_step(fixture.sgd_learning_rate); else train.adamw_step();
          maximumAbsoluteError = Math.max(maximumAbsoluteError,
            close(flat(train.to_dense_layers_json()), expected.parameters, 'Candle updated parameters'));
          if (step === 0) close(flat(train.to_dense_layers_json()),
            fixture[optimizer + '_expected_parameters'], 'First-step compatibility');
        }
        train.mse_backward(Float64Array.from(fixture.input), Float64Array.from(fixture.training_target), 1);
        beforeFree = train.predictions();
        frozen = train.prepare();
        close(frozen.run(Float64Array.from(fixture.input)), beforeFree, 'Frozen current weights');
      } finally { train.free(); }
      try {
        const afterFree = frozen.run(Float64Array.from(fixture.input));
        close(afterFree, beforeFree, 'Frozen snapshot after trainer free');
        const store = await createRuvectorStore({
          wasmBytes: vectorBytes, dimensions: frozen.output_size,
          metric: 'euclidean', useHnsw: true, maxEntries: 32
        });
        let index;
        try {
          index = createNeuralIndex({
            prepared: frozen, store, modelId: 'browser-trained-' + variant + '-' + optimizer,
            batchRows: 7, maxEntries: 32
          });
          check(index.indexType === 'hnsw', 'Actual RuVector HNSW backend');
          const records = Array.from({ length: 17 }, (_, row) => Float64Array.from(
            fixture.input, (value, axis) => value + (row - 8) * (axis + 1) / 32
          ));
          records.forEach((input, row) => index.add({ id: 'sample-' + row, input, metadata: { row, optimizer } }));
          const found = index.search(records[7], { k: 3 });
          check(found.length === 3 && found[0].id === 'sample-7' && found[0].distance === 0, 'Trained HNSW self retrieval');
          check(found[0].metadata.row === 7 && found[0].metadata.optimizer === optimizer, 'Trained HNSW metadata');
          check(index.delete('sample-7') && !index.search(records[7], { k: 3 }).some(x => x.id === 'sample-7'), 'Trained HNSW delete');
          optimizers.push({
            optimizer, steps: 3, backwardPasses: 3, parameterUpdates: 3,
            maximumAbsoluteError, freezeAfterTrainerFree: true,
            hnsw: { indexType: index.indexType, records: 17, actualId: found[0].id, distance: found[0].distance, deleted: true }
          });
        } finally { if (index) index.close(); else store.close(); }
        close(frozen.run(Float64Array.from(fixture.input)), afterFree, 'Snapshot ownership after SDK close');
      } finally { frozen.free(); }
    }

    const learner = Train.from_dense_layers_json(simple(), 5);
    let learning;
    try {
      const input = Float64Array.of(-2, -1, 0, 1, 2);
      const target = Float64Array.of(-3, -1, 1, 3, 5);
      const initialLoss = learner.mse_backward(input, target, 5);
      let finalLoss = initialLoss;
      for (let step = 0; step < 200; step++) {
        learner.sgd_step(0.05);
        finalLoss = learner.mse_backward(input, target, 5);
      }
      check(finalLoss < 1e-12 && finalLoss < initialLoss / 1000, 'SGD learns y=2x+1');
      close(flat(learner.to_dense_layers_json()), [2, 1], 'Learned affine parameters', 1e-8);
      const snapshot = learner.prepare();
      try { close(snapshot.run(Float64Array.of(2)), [5], 'Learned prediction', 1e-8); }
      finally { snapshot.free(); }
      learning = { optimizer: 'sgd', steps: 200, initialLoss, finalLoss };
    } finally { learner.free(); }

    const validJson = simple(), parsed = JSON.parse(validJson);
    const invalidJson = [
      {}, null, '{}', JSON.stringify({ ...parsed, unknown: true }),
      JSON.stringify({ ...parsed, schema: 'unknown' }),
      JSON.stringify({ ...parsed, layers: [{ ...parsed.layers[0], weights: [0, 0] }] }),
      JSON.stringify({ ...parsed, layers: [{ ...parsed.layers[0], activation: 'Gaussian' }] }),
      ' '.repeat(4 * 1024 * 1024 + 1)
    ];
    invalidJson.forEach(value => reject(() => Train.from_dense_layers_json(value, 2), 'dense JSON'));
    const badRows = [0, -1, 1.5, 4097, NaN, Infinity, '1', Object(1), undefined];
    badRows.forEach(value => reject(() => Train.from_dense_layers_json(validJson, value), 'constructor rows'));
    const guarded = Train.from_dense_layers_json(validJson, 2);
    let arrayRejections = 0, scalarRejections = badRows.length, metadataGetterReads = 0;
    try {
      const input = Float64Array.of(1, 2), target = Float64Array.of(2, 4);
      const detached = new Float64Array(2);
      structuredClone(detached.buffer, { transfer: [detached.buffer] });
      const shared = new Float64Array(new SharedArrayBuffer(16));
      Object.defineProperty(shared, 'buffer', { value: new ArrayBuffer(16) });
      const state = guarded.to_dense_layers_json();
      const invalidArrays = [
        [1, 2], Float32Array.of(1, 2), new BigInt64Array(2),
        Object.create(Float64Array.prototype), new Proxy(input, {}), detached,
        shared, new Float64Array(wasm.memory.buffer, 0, 2),
        Float64Array.of(1), Float64Array.of(1, NaN), Float64Array.of(1, Infinity)
      ];
      for (const invalid of invalidArrays) {
        reject(() => guarded.mse_backward(invalid, target, 2), 'input array');
        reject(() => guarded.mse_backward(input, invalid, 2), 'target array');
        check(guarded.to_dense_layers_json() === state, 'Rejected arrays preserve parameters');
        arrayRejections += 2;
      }
      for (const rows of [0, -1, 1.5, 3, NaN, Infinity, '1', null]) {
        reject(() => guarded.mse_backward(input, target, rows), 'active rows');
        scalarRejections++;
      }
      guarded.mse_backward(input, target, 2);
      for (const rate of [-1, 0, NaN, Infinity, '0.1', Object(0.1), null]) {
        reject(() => guarded.sgd_step(rate), 'learning rate');
        check(guarded.to_dense_layers_json() === state, 'Rejected scalar preserves parameters');
        scalarRejections++;
      }
      reject(() => guarded.configure_adamw(0.01, 1, 0.99, 1e-8, 0), 'AdamW beta');
      scalarRejections++;
      for (const array of [input, target]) {
        for (const key of ['length', 'byteLength', 'buffer', 'byteOffset', 'set', 'subarray', Symbol.iterator, Symbol.toStringTag]) {
          Object.defineProperty(array, key, { get() { metadataGetterReads++; throw new Error('Poisoned metadata executed'); } });
        }
      }
      check(Number.isFinite(guarded.mse_backward(input, target, 2)), 'Valid arrays ignore poisoned metadata');
      check(metadataGetterReads === 0, 'No caller metadata callbacks');
      wasm.memory.grow(1);
      reject(() => guarded.mse_backward(new Float64Array(wasm.memory.buffer, 0, 2), target, 2), 'grown WASM memory');
      arrayRejections++;
      check(guarded.to_dense_layers_json() === state, 'WASM alias rejection preserves parameters');
    } finally { guarded.free(); }
    cases.push({
      variant, simdSupport, candle: { oracle: fixture.oracle, optimizers }, learning,
      boundaries: {
        arrayRejections, scalarRejections, invalidJsonRejections: invalidJson.length,
        metadataGetterReads, sharedArrayBufferTested: true, detachedTested: true, wasmMemoryTested: true,
        atomicParametersPreserved: true, poisonedMetadataIgnored: true
      }
    });
  }
  return {
    schema: 'ruv-fann.browser-training/1', status: 'passed', actualWasmTraining: true,
    actualCandle: true, actualRuVectorHnsw: true, coreAndSimd: true,
    crossOriginIsolated, cases
  };
}

(async () => {
  if (process.argv.length !== 4 || process.argv[2] !== '--output') throw new Error('Use --output NEW_EXTERNAL_JSON');
  if (playwrightVersion !== '1.62.1') throw new Error('Playwright 1.62.1 is required');
  if (!process.env.RUV_FANN_CANDLE_FIXTURE) throw new Error('RUV_FANN_CANDLE_FIXTURE is required; no skipped oracle');
  const expectedChromiumVersion = Object.hasOwn(process.env, 'EXPECTED_CHROMIUM_VERSION')
    ? checkedChromiumVersion(process.env.EXPECTED_CHROMIUM_VERSION, 'EXPECTED_CHROMIUM_VERSION') : null;
  const chromiumVersionSource = expectedChromiumVersion === null
    ? 'actual-browser' : process.env.CHROMIUM_VERSION_SOURCE ?? 'explicit-environment';
  if ((expectedChromiumVersion === null && Object.hasOwn(process.env, 'CHROMIUM_VERSION_SOURCE')) ||
    !['actual-browser', 'explicit-environment', 'playwright-core/browsers.json'].includes(chromiumVersionSource) ||
    (expectedChromiumVersion !== null && chromiumVersionSource === 'actual-browser')) {
    throw new Error('CHROMIUM_VERSION_SOURCE must identify the supplied expected version');
  }
  const helper = await import('../../../integrations/ruv-fann/test/e2e/fixture.mjs');
  const output = await helper.externalEvidencePath(process.argv[3]);
  const fixturePath = await fs.realpath(process.env.RUV_FANN_CANDLE_FIXTURE);
  const fixtureStat = await fs.stat(fixturePath);
  if (!fixtureStat.isFile() || fixtureStat.size > 1024 * 1024) throw new Error('Candle fixture must be a file under 1 MiB');
  const fixture = await fs.readFile(fixturePath);
  JSON.parse(fixture);
  const snapshot = async () => ({
    ...await helper.sourceSnapshot(),
    candleFixture: helper.sha(await fs.readFile(fixturePath)),
    browserDriver: helper.sha(await fs.readFile(__filename)),
    ...Object.fromEntries(await Promise.all([
      'src/index.mjs', 'src/encoder.mjs', 'src/neural-index.mjs', 'src/ruvector.mjs',
      'node_modules/@ruvector/wasm/pkg/ruvector_wasm.js',
      'node_modules/@ruvector/wasm/pkg/ruvector_wasm_bg.wasm'
    ].map(async file => [file, helper.sha(await fs.readFile(path.join(helper.packageRoot, file)))])))
  });
  const before = await snapshot();
  const html = '<!doctype html><meta charset="utf-8"><title>ruv-FANN training acceptance</title>' +
    '<script type="importmap">{"imports":{"@ruvector/wasm":"/integrations/ruv-fann/node_modules/@ruvector/wasm/pkg/ruvector_wasm.js"}}</script>';
  const allowed = new Set([
    '/integrations/ruv-fann/src/index.mjs', '/integrations/ruv-fann/src/encoder.mjs',
    '/integrations/ruv-fann/src/neural-index.mjs', '/integrations/ruv-fann/src/ruvector.mjs',
    '/integrations/ruv-fann/node_modules/@ruvector/wasm/pkg/ruvector_wasm.js',
    '/integrations/ruv-fann/node_modules/@ruvector/wasm/pkg/ruvector_wasm_bg.wasm',
    '/ruv-swarm/npm/wasm/ruv_swarm_wasm.js', '/ruv-swarm/npm/wasm/ruv_swarm_wasm_bg.wasm',
    '/ruv-swarm/npm/wasm/ruv_swarm_simd.wasm'
  ]);
  const server = http.createServer(async (req, res) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Cache-Control', 'no-store');
    try {
      if (req.method !== 'GET' || req.headers.host !== '127.0.0.1:' + server.address().port) throw new Error('Local GET only');
      const name = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname);
      if (name === '/') { res.setHeader('Content-Type', 'text/html'); res.end(html); return; }
      if (name === '/candle-fixture.json') { res.setHeader('Content-Type', 'application/json'); res.end(fixture); return; }
      if (!allowed.has(name)) throw new Error('Asset is not allowlisted');
      const target = await fs.realpath(path.join(helper.repo, name));
      const relative = path.relative(helper.repo, target);
      const stat = await fs.stat(target);
      if (path.isAbsolute(relative) || relative === '..' || relative.startsWith('..' + path.sep) ||
        !stat.isFile() || stat.size > 8 * 1024 * 1024) throw new Error('Invalid local asset');
      res.setHeader('Content-Type', name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');
      res.end(await fs.readFile(target));
    } catch { res.writeHead(404); res.end(); }
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  let browser, timer;
  try {
    browser = await chromium.launch({
      executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium', headless: true,
      args: ['--no-sandbox', '--disable-dev-shm-usage']
    });
    const actualChromiumVersion = checkedChromiumVersion(browser.version(), 'Actual browser version');
    if (expectedChromiumVersion !== null && actualChromiumVersion !== expectedChromiumVersion) {
      throw new Error('Chromium version mismatch: expected ' + expectedChromiumVersion + ', got ' + actualChromiumVersion);
    }
    const context = await browser.newContext({ serviceWorkers: 'block' });
    const page = await context.newPage(), errors = [], externalRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    const origin = 'http://127.0.0.1:' + server.address().port;
    await context.route('**/*', route => {
      if (new URL(route.request().url()).origin === origin) return route.continue();
      externalRequests.push(route.request().url()); return route.abort();
    });
    await page.goto(origin, { timeout: 30000 });
    const result = await Promise.race([
      page.evaluate(browserProof),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Browser proof exceeded 60 seconds')), 60000); })
    ]);
    clearTimeout(timer);
    if (errors.length) throw new Error(errors.join('\n'));
    if (externalRequests.length) throw new Error('Unexpected external browser request');
    const after = await snapshot();
    if (JSON.stringify(before) !== JSON.stringify(after)) throw new Error('Source changed during browser proof');
    Object.assign(result, {
      playwright: playwrightVersion, chromium: actualChromiumVersion,
      expectedChromiumVersion, chromiumVersionSource, node: process.version,
      fixtureSha256: helper.sha(fixture), inputsBefore: before, inputsAfter: after,
      unchanged: true, externalRequests: externalRequests.length
    });
    await fs.writeFile(output, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
    console.log(JSON.stringify({ status: result.status, chromium: result.chromium, cases: result.cases.length, output }));
  } finally {
    clearTimeout(timer);
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
