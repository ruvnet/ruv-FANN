/** Qualify the already-installed exact core archive, never a source/workspace link. */
import assert from 'node:assert/strict';
import { readFile, writeFile, lstat, realpath, rm } from 'node:fs/promises';
import { join, resolve, dirname, extname, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createServer } from 'node:http';
import { execFileSync } from 'node:child_process';
import {qualifyAgenticow} from './agenticow-scenario.mjs';

if (process.argv[3]) assert.match(process.env.EXPECTED_SHA ?? '', /^[a-f0-9]{40}$/, 'receipts require the full exact source SHA');
const directory = resolve(process.argv[2] ?? '');
assert.ok(process.argv[2], 'pass the fresh archive installation directory');
const coreDirectory = join(directory, 'node_modules/ruv-fann');
assert.equal((await lstat(coreDirectory)).isSymbolicLink(), false, 'core must be an installed archive');
const metadata = JSON.parse(await readFile(join(coreDirectory, 'package.json'), 'utf8'));
assert.equal(metadata.version, '0.3.0');
const mappings = {
  wsp: 'ruv-fann-wsp', lifecycle: 'ruv-fann-wsp-lifecycle', search: 'ruv-fann-wsp-search',
  huggingface: 'ruv-fann-wsp-huggingface', cloudflare: 'ruv-fann-wsp-cloudflare',
  'cloudflare/build': 'ruv-fann-wsp-cloudflare/build', ruos: 'ruv-fann-wsp-ruos',
  invariants: 'ruv-fann-wsp-invariants', swarm: 'ruv-fann-wsp-swarm', registry: 'ruv-fann-wsp-registry',
  loop: 'ruv-fann-wsp-loop', constellation: 'ruvnet-constellation', contracts: 'ruvnet-constellation-contracts',
  telemetry: 'ruvnet-constellation-telemetry', agenticow: 'agenticow'
};
assert.equal(Object.keys(metadata.dependencies).length, 15);
for (const [subpath, dependency] of Object.entries(mappings)) {
  const name = dependency.split('/')[0];
  assert.equal(metadata.dependencies[name], name === 'agenticow' ? '0.3.0' : name === 'ruv-fann-wsp' ? '0.2.0' : '0.1.0');
  assert.deepEqual(Object.keys(metadata.exports[`./${subpath}`]).sort(), ['import', 'types']);
}

// The same independent numerical oracle runs under Node and real Chromium.
async function qualify(core, wsp, search, invariants, lifecycle, simd) {
  let checks = 0;
  const check = (ok, message) => { if (!ok) throw new Error(message); checks++; };
  const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  check(equal(Object.keys(core).sort(), ['LIMITS', 'loadRuntime']), 'root remains an inference-only import');
  const runtime = await core.loadRuntime({ simd });
  check(runtime.backend === (simd ? 'wasm-simd' : 'wasm'), 'requested real WASM backend');
  const linear = { schema: 'ruv-fann.dense-layers/1', layers: [{inputs: 1, outputs: 1, weights: [2], bias: [1], activation: 'Linear', steepness: 1}] };
  const prepared = runtime.prepare(linear, {maxRows: 3});
  check(equal([...prepared.predict(Float64Array.of(1, 2, 3))], [3, 5, 7]), 'core archive exact linear oracle');
  prepared.dispose();
  const trained = runtime.train({...linear, layers: [{...linear.layers[0], weights: [0], bias: [0]}]}, {
    inputs: Float64Array.of(0, 1, 2), targets: Float64Array.of(1, 3, 5), rows: 3,
    epochs: 300, learningRate: 0.05, optimizer: 'adamw'
  });
  check(trained.finalLoss < trained.initialLoss / 1000, 'core training preserves actual loss reduction');
  const model = {schema: 'ruv-fann.dense-layers/1', layers: [
    {inputs: 2, outputs: 2, weights: [1, -1, -1, 1], bias: [0, 0], activation: 'ReLU', steepness: 1},
    {inputs: 2, outputs: 1, weights: [0, 0], bias: [0], activation: 'Linear', steepness: 1}
  ]};
  const dataset = {inputs: [0, 0, 0, 1, 1, 0, 1, 1], targets: [0, 1, 1, 0], rows: 4};
  const policy = {maxMSE: 0, maxRegressionMSE: 0, maxAbsoluteError: 0};
  const found = await search.searchWeightPatch(model, dataset, policy, {simd, locations: [0, 1].map(index => ({layer: 1, parameter: 'weights', index, values: [0, 1]})), maxEvaluations: 8, maxPasses: 2});
  check(found.patch !== null && found.evaluation.accepted && found.evaluation.candidate.mse === 0, 'search subpath repairs XOR with actual inference');
  const gate = await lifecycle.runRegressionGate({model, patch: found.patch, suites: [{name: 'complete-xor', dataset, policy}]}, {simd});
  check(gate.accepted, 'lifecycle subpath independently accepts strict complete-XOR gate');
  const workspace = await wsp.createWeightWorkspace(model, {simd});
  const initial = workspace.snapshot();
  const proposal = await workspace.propose(found.patch, dataset, policy);
  check(workspace.snapshot().hash === initial.hash, 'accepted proposal never implicitly promotes');
  const repaired = workspace.promote(proposal.id, 1);
  const trace = await wsp.traceWeightModel(repaired.model, {inputs: dataset.inputs, rows: 4}, {simd});
  check(equal(trace.layers.at(-1).activations, dataset.targets), 'weight subpath trace matches independent XOR oracle');
  const invariant = invariants.createBooleanInvariant({inputs: 2, outputs: 1, targets: dataset.targets});
  const bounded = await invariants.checkModelInvariants(repaired.model, invariant, {simd});
  check(bounded.accepted && bounded.checks === 4, 'invariants subpath checks all finite-domain rows');
  const bad = await wsp.createNeuronAblationPatch(repaired.model, 0, 0);
  const rejected = await workspace.propose(bad, dataset, policy);
  check(!rejected.evaluation.accepted, 'ablation regression rejected');
  let denied = false;
  try { workspace.promote(rejected.id, 2); } catch { denied = true; }
  check(denied && workspace.snapshot().hash === repaired.hash, 'rejected proposal cannot change live weights');
  const restored = workspace.rollback(1, 2);
  check(restored.revision === 3 && restored.hash === initial.hash, 'explicit rollback restores original model');
  return {backend: runtime.backend, checks, inference: true, training: true, search: true, gate: true, invariants: true, rejectedPromotion: true, rollbackRevision: restored.revision};
}

const token = `core-facade-${process.pid}-${Date.now()}`;
const consumerFile = join(directory, `.${token}.mjs`);
const typeFile = join(directory, `.${token}.ts`);
const browserImports = ['wsp', 'search', 'lifecycle', 'invariants', 'registry', 'huggingface', 'cloudflare', 'ruos', 'contracts', 'telemetry'];
let server, browser;
try {
  await writeFile(consumerFile, `import assert from 'node:assert/strict';
import * as core from 'ruv-fann';
import * as wsp from 'ruv-fann/wsp';
import * as search from 'ruv-fann/search';
import * as invariants from 'ruv-fann/invariants';
import * as lifecycle from 'ruv-fann/lifecycle';
const mappings = ${JSON.stringify(mappings)};
for (const [subpath, dependency] of Object.entries(mappings)) {
  const facade = await import('ruv-fann/' + subpath), satellite = await import(dependency);
  assert.deepEqual(Object.keys(facade), Object.keys(satellite), 'complete export surface: ' + subpath);
  for (const key of Object.keys(satellite)) assert.equal(facade[key], satellite[key], 'direct usable export: ' + subpath + '/' + key);
}
const qualify = ${qualify.toString()};
console.log('CORE_FACADE_RESULT ' + JSON.stringify(await Promise.all([false, true].map(simd => qualify(core, wsp, search, invariants, lifecycle, simd)))));
const qualifyAgenticow = ${qualifyAgenticow.toString()};
const memory = await import('ruv-fann/agenticow');
console.log('CORE_AGENTICOW_RESULT ' + JSON.stringify(await qualifyAgenticow({memory,wsp,lifecycle,core},${JSON.stringify(join(directory, token + '-memory'))})));
`);
  const stdout = execFileSync(process.execPath, [consumerFile], {cwd: directory, encoding: 'utf8'});
  const backendChecks = JSON.parse(stdout.split('\n').find(line => line.startsWith('CORE_FACADE_RESULT ')).slice('CORE_FACADE_RESULT '.length));
  const agenticow = JSON.parse(stdout.split('\n').find(line => line.startsWith('CORE_AGENTICOW_RESULT ')).slice('CORE_AGENTICOW_RESULT '.length));
  assert.deepEqual(backendChecks.map(row => row.backend), ['wasm', 'wasm-simd']);

  // Verify the actual nested dependency that WSP resolves, rather than globally remapping it.
  const findPackage = async (name, start) => {
    let parent = start;
    for (;;) {
      const candidate = join(parent, 'node_modules', name, 'package.json');
      try { return {directory: dirname(candidate), metadata: JSON.parse(await readFile(candidate, 'utf8'))}; }
      catch (error) { if (error.code !== 'ENOENT') throw error; }
      const next = dirname(parent); if (next === parent) throw new Error(`missing installed dependency ${name}`); parent = next;
    }
  };
  const wspPackage = await findPackage('ruv-fann-wsp', coreDirectory);
  const legacyCore = await findPackage('ruv-fann', wspPackage.directory);
  assert.equal(legacyCore.metadata.version, '0.1.2', 'satellites intentionally use published legacy runtime');
  assert.deepEqual(legacyCore.metadata.dependencies, {'ruv-fann-wasm': '0.1.2'}, 'finite version graph terminates at WASM');
  const wasm = await findPackage('ruv-fann-wasm', coreDirectory);
  assert.equal(wasm.metadata.version, '0.1.2');
  const graph = {core: metadata.version, legacyCore: legacyCore.metadata.version, finite: true};

  const typeImports = Object.keys(mappings).map((subpath, index) => `import * as capability${index} from 'ruv-fann/${subpath}';\nvoid capability${index};`).join('\n');
  await writeFile(typeFile, `import {loadRuntime, type DenseModel} from 'ruv-fann';
import {createWeightWorkspace, type WeightPatch} from 'ruv-fann/wsp';
import {searchWeightPatch, type SearchOptions} from 'ruv-fann/search';
import {checkModelInvariants, type InvariantSuite} from 'ruv-fann/invariants';
import memoryDefault, {AgenticMemory, openBase, type OpenOptions} from 'ruv-fann/agenticow';
${typeImports}
const model: DenseModel = {schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[1],activation:'Linear',steepness:1}]};
const patch: WeightPatch = {schema:'ruv-fann.weight-patch/1',baseHash:'0'.repeat(64),edits:[{layer:0,parameter:'weights',index:0,value:2}]};
const search: SearchOptions = {locations:[{layer:0,parameter:'weights',index:0,values:[1,2]}]};
const suite: InvariantSuite = {schema:'ruv-fann.invariants/1',cases:[{name:'identity',inputs:[1],rows:1,expected:[1]}]};
void loadRuntime().then(runtime => runtime.prepare(model)); void createWeightWorkspace(model); void patch; void searchWeightPatch; void search; void checkModelInvariants(model,suite);
// @ts-expect-error explicit finite numerical inputs remain required
const invalid: DenseModel = {schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:['bad'],activation:'Linear',steepness:1}]};
void invalid;
const memoryOptions: OpenOptions = {dimension:2,metric:'l2'};
const memoryOpen: typeof AgenticMemory.open = openBase;
void memoryDefault; void memoryOptions; void memoryOpen;
`);
  const tsc = process.env.RUV_FANN_TSC;
  assert.ok(tsc, 'RUV_FANN_TSC must identify the pinned TypeScript compiler');
  execFileSync(process.execPath, [tsc, '--noEmit', '--strict', '--skipLibCheck', 'false', '--target', 'ES2022', '--module', 'NodeNext', '--moduleResolution', 'NodeNext', '--lib', 'ES2022,DOM', typeFile], {cwd: directory, encoding: 'utf8'});

  const toWeb = path => '/' + path.slice(directory.length + 1).split(sep).join('/');
  const imports = {'ruv-fann': toWeb(join(coreDirectory, 'src/index.js')), 'ruv-fann-wasm': toWeb(join(wasm.directory, 'src/index.js'))};
  const scopes = {};
  const legacyWeb = toWeb(join(legacyCore.directory, 'src/index.js'));
  for (const [subpath, dependency] of Object.entries(mappings)) {
    imports[`ruv-fann/${subpath}`] = toWeb(join(coreDirectory, metadata.exports[`./${subpath}`].import));
    const name = dependency.split('/')[0];
    const pkg = await findPackage(name, coreDirectory);
    assert.equal((await lstat(pkg.directory)).isSymbolicLink(), false, 'capability must be an installed archive: ' + name);
    assert.ok((await realpath(pkg.directory)).startsWith(directory + sep), 'capability must be inside the fresh installation: ' + name);
    const suffix = dependency.includes('/') ? './' + dependency.split('/').slice(1).join('/') : '.';
    const entry = pkg.metadata.exports?.[suffix];
    if (typeof entry === 'object') imports[dependency] = toWeb(join(pkg.directory, entry.import));
    scopes[toWeb(pkg.directory) + '/'] = {'ruv-fann': legacyWeb};
  }
  // The WASM URL inside both core generations must resolve to the bundled archive bytes.
  scopes[toWeb(legacyCore.directory) + '/'] = {'ruv-fann-wasm': imports['ruv-fann-wasm']};
  const importmap = {imports, scopes};
  const errors = [];
  const mime = {'.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json'};
  server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://localhost');
      if (url.pathname === '/') {res.setHeader('Content-Type', 'text/html'); return res.end(`<script type="importmap">${JSON.stringify(importmap)}</script><title>Installed core consumer</title>`);}
      const file = resolve(directory, '.' + decodeURIComponent(url.pathname));
      if (!file.startsWith(directory + sep)) {res.writeHead(403); return res.end();}
      const canonical = await realpath(file);
      if (!canonical.startsWith(directory + sep)) {res.writeHead(403); return res.end();}
      res.setHeader('Content-Type', mime[extname(file)] ?? 'application/octet-stream');
      res.end(await readFile(file));
    } catch {res.writeHead(404); res.end();}
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const playwrightPath = process.env.RUV_FANN_PLAYWRIGHT;
  assert.ok(playwrightPath, 'RUV_FANN_PLAYWRIGHT must identify pinned Playwright');
  const playwright = await import(pathToFileURL(join(playwrightPath, 'index.mjs')).href);
  browser = await playwright.chromium.launch({headless: true});
  const page = await browser.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const browserChecks = await page.evaluate(async ({fn, subpaths}) => {
    const modules = await Promise.all(['ruv-fann', 'ruv-fann/wsp', 'ruv-fann/search', 'ruv-fann/invariants', 'ruv-fann/lifecycle'].map(name => import(name)));
    for (const subpath of subpaths) await import('ruv-fann/' + subpath);
    const qualify = (0, eval)('(' + fn + ')');
    return await Promise.all([false, true].map(simd => qualify(...modules, simd)));
  }, {fn: qualify.toString(), subpaths: browserImports});
  assert.deepEqual(browserChecks.map(row => row.backend), ['wasm', 'wasm-simd']);
  assert.deepEqual(errors, []);
  const receipt = {schema: 'ruv-fann.core-consumer/1', status: 'passed', sourceSha: process.env.EXPECTED_SHA ?? null,
    testCount: backendChecks.reduce((sum, row) => sum + row.checks, 0) + browserChecks.reduce((sum, row) => sum + row.checks, 0) + Object.keys(mappings).length + 4,
    skipped: 0, todo: 0, runtime: process.version, backendChecks, browser: {engine: 'Chromium', realWasm: true, checks: browserChecks, importedSubpaths: browserImports, pageErrors: errors},
    types: {strict: true, subpaths: Object.keys(mappings).length}, graph, agenticow};
  if (process.argv[3]) await writeFile(resolve(process.argv[3]), JSON.stringify(receipt, null, 2) + '\n');
  console.log(JSON.stringify(receipt, null, 2));
} finally {
  await browser?.close();
  if (server) await new Promise(resolve => server.close(resolve));
  await rm(consumerFile, {force: true}); await rm(typeFile, {force: true});
  await rm(join(directory, token + '-memory'), {recursive:true,force:true});
}
