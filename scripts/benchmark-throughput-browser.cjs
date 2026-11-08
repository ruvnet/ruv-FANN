'use strict';
// Same fixtures/timing module as the Node benchmark; actual HTTP-served WASM.
// CHROMIUM_BIN=/usr/bin/chromium node scripts/benchmark-throughput-browser.cjs [--baseline /previous/wasm]
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {createHash} = require('node:crypto');
const {chromium} = require('playwright');
const root = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
const baselineIndex = args.indexOf('--baseline');
const baseline = baselineIndex < 0 ? null : path.resolve(args[baselineIndex + 1]);
const onlyCalibration = args.includes('--only-calibration');
const directoryIndex = args.indexOf('--wasm-dir');
const current = directoryIndex >= 0 ? path.resolve(args[directoryIndex + 1]) : path.join(root, 'ruv-swarm/npm/wasm');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

(async () => {
  const routes = new Map();
  const artifacts = [];
  for (const [revision, directory] of [...(baseline ? [['before', baseline]] : []), ['current', current]]) {
    for (const variant of ['core', 'simd']) {
      const prefix = `/${revision}/${variant}`;
      const binary = fs.readFileSync(path.join(directory, variant === 'core' ? 'ruv_swarm_wasm_bg.wasm' : 'ruv_swarm_simd.wasm'));
      routes.set(`${prefix}/bindings.js`, {bytes: fs.readFileSync(path.join(directory, 'ruv_swarm_wasm.js')), type: 'text/javascript'});
      routes.set(`${prefix}/module.wasm`, {bytes: binary, type: 'application/wasm'});
      artifacts.push({revision, variant, prefix, sha256: hash(binary)});
    }
  }
  routes.set('/suite.mjs', {bytes: fs.readFileSync(path.join(root, 'ruv-swarm/npm/examples/throughput-suite.mjs')), type: 'text/javascript'});
  const server = http.createServer((request, response) => {
    response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    response.setHeader('Cache-Control', 'no-store');
    const pathname = new URL(request.url, 'http://localhost').pathname;
    const file = routes.get(pathname);
    if (file) { response.setHeader('Content-Type', file.type); response.end(file.bytes); }
    else if (pathname === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><title>WASM throughput measurements</title>'); }
    else { response.writeHead(404); response.end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({headless: true,
      ...(process.env.CHROMIUM_BIN ? {executablePath: process.env.CHROMIUM_BIN} : {})});
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const measured = await page.evaluate(async ({artifacts, onlyCalibration}) => {
      const {inferenceSuite, calibrationSuite} = await import('/suite.mjs');
      const references = new Map();
      const results = [];
      let exactComparisons = 0;
      for (const artifact of artifacts) {
        const bindings = await import(`${artifact.prefix}/bindings.js`);
        await bindings.default({module_or_path: fetch(`${artifact.prefix}/module.wasm`)});
        const features = JSON.parse(bindings.get_features());
        if (features.simd_support !== (artifact.variant === 'simd')) throw new Error('Wrong WASM variant');
        const calibration = calibrationSuite(bindings, {validate(key, groups, output) {
          const id = `${artifact.variant}/${key}`;
          if (artifact.revision === 'before') references.set(id, [groups, output]);
          else if (references.has(id)) {
            const expected = references.get(id);
            if (expected[0] !== groups || expected[1] !== output) throw new Error(`Calibration JSON changed: ${id}`);
            exactComparisons++;
            return true;
          }
          return false;
        }});
        results.push({...artifact, features, calibration,
          inference: onlyCalibration ? [] : inferenceSuite(bindings)});
      }
      return {results, exact_before_after_calibration_cases: exactComparisons,
        cross_origin_isolated: crossOriginIsolated, user_agent: navigator.userAgent};
    }, {artifacts, onlyCalibration});
    if (errors.length) throw new Error(errors.join('\n'));
    console.log(JSON.stringify({schema_version: 1, date_utc: new Date().toISOString(), runtime: 'Chromium',
      browser: await browser.version(), precision: 'f64', clock: 'performance.now(), ns units derived from ms; no claim of ns resolution',
      method: 'Same shared fixtures/rotating sample blocks as Node. Fixture construction/verification excluded; public boundary checks/copies and result allocation included.',
      ...measured}, null, 2));
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
