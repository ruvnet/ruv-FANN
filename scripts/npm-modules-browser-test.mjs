/** Real Chromium proof; Playwright is supplied by the development host, not runtime deps. */
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, extname, sep } from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.RUV_FANN_PLAYWRIGHT ?? 'playwright');
const root = resolve(fileURLToPath(new URL('../', import.meta.url)));
const server = createServer(async (request, response) => {
  try {
    if (request.url === '/') {
      response.setHeader('Content-Type', 'text/html');
      return response.end(`<script type="importmap">{"imports":{"ruv-fann-wasm":"/packages/wasm/src/index.js"}}</script>`);
    }
    const pathname = new URL(request.url, 'http://localhost').pathname;
    const file = resolve(root, `.${decodeURIComponent(pathname)}`);
    if (!file.startsWith(root + sep)) { response.writeHead(403); return response.end(); }
    response.setHeader('Content-Type', extname(file) === '.wasm' ? 'application/wasm' : 'text/javascript');
    response.end(await readFile(file));
  } catch { response.writeHead(404); response.end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ headless: true,
    ...(process.env.RUV_FANN_CHROME ? { executablePath: process.env.RUV_FANN_CHROME } : {}),
    args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const proof = await page.evaluate(async () => {
    const check = (condition, label) => { if (!condition) throw new Error(label); };
    check(typeof process === 'undefined', 'browser must not use Node branch');
    const { loadRuntime } = await import('/packages/core/src/index.js');
    const { loadWasm, supportsSIMD } = await import('/packages/wasm/src/index.js');
    const model = { schema: 'ruv-fann.dense-layers/1', layers: [{inputs:1,outputs:1,weights:[2],bias:[1],activation:'Linear',steepness:1}] };
    const results = [];
    for (const simd of [false, ...(supportsSIMD() ? [true] : [])]) {
      const [runtime, concurrent] = await Promise.all([loadRuntime({simd}), loadRuntime({simd})]);
      check(runtime === concurrent, 'runtime initialization identity');
      const prepared = runtime.prepare(model, {maxRows:2});
      check(JSON.stringify([...prepared.predict(Float64Array.of(1,2))]) === '[3,5]', 'real batch browser inference');
      const output = Float64Array.of(123);
      try { prepared.predictInto(Float64Array.of(NaN),output); throw new Error('invalid input accepted'); }
      catch(error) { check(error.message !== 'invalid input accepted' && output[0] === 123,'atomic browser output'); }
      const training = runtime.train(model,{inputs:Float64Array.of(1,2),targets:Float64Array.of(2,4),rows:2,epochs:500,learningRate:.05,optimizer:'adamw'});
      check(training.finalLoss < training.initialLoss / 1000, 'actual browser AdamW decrease');
      prepared.dispose(); prepared.dispose();
      let rejected = false;
      try { prepared.predict(Float64Array.of(1)); } catch { rejected = true; }
      check(rejected, 'browser disposal');
      results.push({ backend: runtime.backend, initialLoss: training.initialLoss, finalLoss: training.finalLoss });
    }
    if (supportsSIMD()) check((await loadWasm({simd:false})).memory !== (await loadWasm({simd:true})).memory, 'separate browser memory');
    return results;
  });
  if (errors.length) throw new Error(errors.join('\n'));
  console.log(JSON.stringify({ browser: await browser.version(), proof }));
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
