#!/usr/bin/env node
// Build, test, seal and publish only the exact ruv-fann-wsp npm archive.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, readdir, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, basename, dirname, extname, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { runWorkbenchProbe, runWorkbenchSecurityProbe } from './wsp-workbench-probes.mjs';
import { validateWorkbenchProof, validateSecurity, validateBenchmark } from './wsp-release-evidence.mjs';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const argv = process.argv.slice(2);
const mode = argv.shift();
const options = Object.create(null);
for (let i = 0; i < argv.length; i++) {
  const key = argv[i];
  const normalized = key.startsWith('--') ? key.slice(2) : '';
  if (!normalized || i + 1 >= argv.length || normalized in options) throw new Error(`Invalid argument: ${key}`);
  options[normalized] = argv[++i];
}
const required = key => { if (!options[key]) throw new Error(`Missing --${key}`); return options[key]; };
const exec = (command, args, cwd = root, env = process.env) => execFileSync(command, args, {
  cwd, env, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'], timeout: 30 * 60_000
}).trim();
const sha256 = data => createHash('sha256').update(data).digest('hex');
const sha512Integrity = data => `sha512-${createHash('sha512').update(data).digest('base64')}`;
const json = async path => JSON.parse(await readFile(path, 'utf8'));
const saveNew = async (path, value) => writeFile(path, `${JSON.stringify(value, null, 2)}\n`, { flag: 'wx' });
const gitSha = () => exec('git', ['rev-parse', 'HEAD']);
const assertClean = () => { if (exec('git', ['status', '--porcelain'])) throw new Error('Source tree is dirty'); };
const identity = { name: 'ruv-fann-wsp', version: '0.2.0', dependency: { 'ruv-fann': '0.1.2' }, tag: 'ruv-fann-wsp-v0.2.0' };
const archiveInfo = async path => JSON.parse(exec('tar', ['-xOzf', path, 'package/package.json']));

async function pack() {
  assertClean();
  const output = resolve(required('output'));
  await mkdir(output, { recursive: false });
  const manifest = await json(join(root, 'packages/wsp/package.json'));
  if (manifest.name !== identity.name || manifest.version !== identity.version || manifest.private ||
      JSON.stringify(manifest.dependencies) !== JSON.stringify(identity.dependency))
    throw new Error('WSP manifest must publish ruv-fann-wsp@0.2.0 with exact ruv-fann@0.1.2 dependency');
  const packed = JSON.parse(exec('npm', ['pack', '--ignore-scripts', '--json', '--pack-destination', output], join(root, 'packages/wsp')));
  if (packed.length !== 1 || packed[0].name !== identity.name || packed[0].version !== identity.version ||
      packed[0].files.some(file => /(^|\/)(test|tests|node_modules|scripts)(\/|$)/.test(file.path)))
    throw new Error('Unexpected WSP npm archive content or identity');
  const archive = join(output, packed[0].filename);
  const metadata = await archiveInfo(archive);
  if (metadata.dependencies?.['ruv-fann'] !== '0.1.2' || metadata.private) throw new Error('Packed WSP dependency/publication metadata changed');
  const bytes = await readFile(archive);
  await saveNew(join(output, 'package-manifest.json'), { schema: 'ruv-fann-wsp.npm-candidate/1', sourceSha: gitSha(),
    name: identity.name, version: identity.version, tarball: basename(archive), sha256: sha256(bytes), integrity: sha512Integrity(bytes) });
  console.log(JSON.stringify({ name: identity.name, version: identity.version, filename: basename(archive), sha256: sha256(bytes) }));
}

const model = () => ({ schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 1, outputs: 1, weights: [0], bias: [0], activation: 'Linear', steepness: 1 }
] });
const data = () => ({ inputs: [0, 1, 2], targets: [1, 3, 5], rows: 3 });
const policy = () => ({ maxMSE: 0.001, maxRegressionMSE: 0.01, maxAbsoluteError: 0.1 });

function nodeIntegrationSource() {
  return `import assert from 'node:assert/strict';
import { fingerprintModel, applyWeightPatch, evaluateWeightPatch, createWeightWorkspace } from 'ruv-fann-wsp';
import { loadRuntime } from 'ruv-fann';
import { supportsSIMD } from 'ruv-fann-wasm';
if (!supportsSIMD()) throw new Error('Required x64 runner lacks WebAssembly SIMD');
const model={schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[0],bias:[0],activation:'Linear',steepness:1}]};
const dataset={inputs:[0,1,2],targets:[1,3,5],rows:3};
const policy={maxMSE:0.001,maxRegressionMSE:0.01,maxAbsoluteError:0.1};
const base=await fingerprintModel(model); assert.match(base,/^[0-9a-f]{64}$/);
const patch={schema:'ruv-fann.weight-patch/1',baseHash:base,edits:[{layer:0,parameter:'weights',index:0,value:2},{layer:0,parameter:'bias',index:0,value:1}]};
const sourceCopy=JSON.stringify(model); const patched=await applyWeightPatch(model,patch);
assert.equal(JSON.stringify(model),sourceCopy,'patch must not mutate caller model');
assert.deepEqual(patched.layers[0].weights,[2]); assert.deepEqual(patched.layers[0].bias,[1]);
const evaluation=await evaluateWeightPatch(model,patch,dataset,policy,{simd:false});
assert.equal(evaluation.schema,'ruv-fann.weight-evaluation/1'); assert.equal(evaluation.accepted,true);
assert.ok(evaluation.candidate.mse < evaluation.baseline.mse); assert.equal(evaluation.backend,'wasm');
const badPatch={schema:'ruv-fann.weight-patch/1',baseHash:base,edits:[{layer:0,parameter:'bias',index:0,value:100}]};
const rejected=await evaluateWeightPatch(model,badPatch,dataset,policy,{simd:true});
assert.equal(rejected.backend,'wasm-simd'); assert.equal(rejected.accepted,false);
assert.ok(rejected.reasons.length>0);
const workspace=await createWeightWorkspace(model,{simd:false});
const mutablePatch={schema:'ruv-fann.weight-patch/1',baseHash:base,edits:[{layer:0,parameter:'weights',index:0,value:2},{layer:0,parameter:'bias',index:0,value:1}]};
const proposalPromise=workspace.propose(mutablePatch,dataset,policy);
mutablePatch.edits[0].value=999; dataset.targets[0]=999;
const proposal=await proposalPromise; assert.equal(proposal.revision,1); assert.equal(proposal.evaluation.accepted,true);
proposal.evaluation.model.layers[0].weights[0]=777;
const promoted=workspace.promote(proposal.id,1); assert.equal(promoted.revision,2);
assert.deepEqual(promoted.model.layers[0].weights,[2]); assert.deepEqual(promoted.model.layers[0].bias,[1]);
const initial=workspace.rollback(1,2); assert.equal(initial.revision,3); assert.deepEqual(initial.model.layers[0].weights,[0]);
assert.throws(() => workspace.promote(proposal.id,3));
assert.throws(() => workspace.promote({id:proposal.id,evaluation:{accepted:true}},3));
const after=workspace.snapshot(); assert.equal(after.revision,3); assert.deepEqual(after.model.layers[0].weights,[0]);
const raw=await loadRuntime({simd:false}); const prepared=raw.prepare(promoted.model);
assert.deepEqual([...prepared.predict(Float64Array.of(0,1,2))],[1,3,5]); prepared.dispose();
const workbench=await (${runWorkbenchProbe.toString()})();
const security=await (${runWorkbenchSecurityProbe.toString()})();
console.log(JSON.stringify({workbench,security,status:'passed',backends:['wasm','wasm-simd'],accepted:true,rejected:true,promotedRevision:2,rollbackRevision:3}));
`;
}

async function installArchive(candidatePath, temp, env) {
  const candidate = await json(candidatePath);
  if (candidate.name !== identity.name || candidate.version !== identity.version || candidate.sourceSha !== gitSha())
    throw new Error('Candidate does not bind the exact WSP source SHA and version');
  const archive = join(dirname(candidatePath), candidate.tarball);
  const bytes = await readFile(archive);
  if (sha256(bytes) !== candidate.sha256 || sha512Integrity(bytes) !== candidate.integrity) throw new Error('Candidate tarball digest mismatch');
  await writeFile(join(temp, 'package.json'), JSON.stringify({ name: 'wsp-consumer', version: '1.0.0', private: true, type: 'module' }));
  await writeFile(join(temp, '.npmrc'), 'registry=https://registry.npmjs.org\n');
  env.NPM_CONFIG_USERCONFIG = join(temp, '.npmrc');
  env.NPM_CONFIG_CACHE = join(temp, 'npm-cache');
  exec('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', archive], temp, env);
  const installed = await json(join(temp, 'node_modules/ruv-fann-wsp/package.json'));
  const core = await json(join(temp, 'node_modules/ruv-fann/package.json'));
  if (installed.version !== identity.version || installed.dependencies?.['ruv-fann'] !== '0.1.2' || core.version !== '0.1.2')
    throw new Error('Installed package dependency tree does not match the contract');
  return { candidate, archive, bytes };
}

async function consumer() {
  const candidatePath = resolve(required('candidate'));
  const receiptPath = resolve(required('output'));
  const temp = await mkdtemp(join(tmpdir(), 'ruv-fann-wsp-consumer-'));
  const env = { ...process.env };
  for (const key of ['NPM_TOKEN', 'NODE_AUTH_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN', 'CARGO_REGISTRY_TOKEN']) delete env[key];
  try {
    const { candidate, bytes } = await installArchive(candidatePath, temp, env);
    await writeFile(join(temp, 'integration.mjs'), nodeIntegrationSource());
    const integrationOutput = exec(process.execPath, ['integration.mjs'], temp, env);
    console.log(integrationOutput);
    const integrationProof = JSON.parse(integrationOutput.split('\n').at(-1));
    validateWorkbenchProof(integrationProof);
    const tsc = resolve(required('tsc'));
    await writeFile(join(temp, 'types.mts'), `
import { fingerprintModel, applyWeightPatch, evaluateWeightPatch, createWeightWorkspace,
  inspectWeightModel, traceWeightModel, analyzeWeightSensitivity, createNeuronAblationPatch,
  type WeightModelInspection, type WeightTrace, type WeightSensitivity,
  type DenseModel, type WeightPatch, type EvaluationDataset, type EvaluationPolicy, type WeightEvaluation } from 'ruv-fann-wsp';
const model: DenseModel = {schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[0],activation:'Linear',steepness:1}]};
const patch: WeightPatch = {schema:'ruv-fann.weight-patch/1',baseHash:'0'.repeat(64),edits:[{layer:0,parameter:'weights',index:0,value:1}]};
const dataset: EvaluationDataset = {inputs:[0,1],targets:[0,1],rows:2};
const policy: EvaluationPolicy = {maxMSE:1,maxRegressionMSE:1};
const hash: Promise<string> = fingerprintModel(model);
const changed: Promise<DenseModel> = applyWeightPatch(model,patch);
const result: Promise<WeightEvaluation> = evaluateWeightPatch(model,patch,dataset,policy);
const workspace = await createWeightWorkspace(model);
const snapshot = workspace.snapshot();
const proposal = await workspace.propose(patch,dataset,policy);
const promoted = workspace.promote(proposal.id,proposal.revision);
const restored = workspace.rollback(snapshot.revision,promoted.revision);
const history = workspace.history();
if (promoted.revision !== 2 || restored.revision !== 3 || history.length < 3) throw new Error('workspace declarations do not support the published lifecycle');
const inspected: WeightModelInspection = await inspectWeightModel(model);
const traced: WeightTrace = await traceWeightModel(model,{inputs:[0,1],rows:2});
const probed: WeightSensitivity = await analyzeWeightSensitivity(model,dataset,[{layer:0,parameter:'weights',index:0}]);
const ablation: Promise<WeightPatch> = createNeuronAblationPatch(model,0,0); void ablation.catch(() => {});
void inspected; void traced; void probed; void hash; void changed; void result; void snapshot;
`);
    exec(process.execPath, [tsc, '--noEmit', '--strict', '--target', 'ES2022', '--module', 'NodeNext', '--moduleResolution', 'NodeNext', '--lib', 'ESNext,DOM,ESNext.Disposable', 'types.mts'], temp, env);
    await mkdir(dirname(receiptPath), { recursive: true });
    await saveNew(receiptPath, { schema: 'ruv-fann-wsp.npm-consumer/1', sourceSha: candidate.sourceSha,
      package: identity.name, version: identity.version, tarballSha256: sha256(bytes), runtime: process.version,
      status: 'passed', workbench: integrationProof.workbench, security: integrationProof.security, mutationChecks: ['caller patch and dataset snapshot before await', 'returned proposal mutation isolated from workspace state',
        'stale revision rejected', 'forged proposal receipt rejected'],
      scenarios: ['installed archive', 'patch and fingerprint', 'real WASM evaluation', 'workspace propose/promote/rollback', 'strict installed TypeScript declarations'] });
  } finally { await rm(temp, { recursive: true, force: true }); }
}

async function unit() {
  const output = resolve(required('output'));
  const testDir = join(root, 'packages/wsp/test');
  const tests = (await readdir(testDir)).filter(name => name.endsWith('.test.mjs')).sort().map(name => `test/${name}`);
  const scriptTests = (await readdir(join(root, 'scripts'))).filter(name => /^wsp-.*\.test\.mjs$/.test(name)).map(name => join(root, 'scripts', name));
  tests.push(...scriptTests);
  if (!tests.length) throw new Error('No WSP unit tests found');
  const reporter = exec(process.execPath, ['--test', '--test-reporter=tap', ...tests], join(root, 'packages/wsp'));
  const count = name => Number(new RegExp(`^# ${name} (\\d+)$`, 'm').exec(reporter)?.[1] ?? NaN);
  const passed = count('pass'), failed = count('fail'), skipped = count('skipped'), todo = count('todo');
  if (passed < 1 || failed !== 0 || skipped !== 0 || todo !== 0) throw new Error(`WSP unit tests require nonzero passes and zero failures/skips/todos: ${JSON.stringify({ passed, failed, skipped, todo })}`);
  await mkdir(dirname(output), { recursive: true });
  await saveNew(output, { schema: 'ruv-fann-wsp.npm-unit/1', sourceSha: gitSha(), runtime: process.version, passed, failed, skipped, todo, status: 'passed' });
  console.log(JSON.stringify({ passed, failed, skipped, todo, runtime: process.version }));
}

async function browser() {
  const candidatePath = resolve(required('candidate'));
  const output = resolve(required('output'));
  const playwright = await import(pathToFileURL(resolve(required('playwright'))).href);
  const temp = await mkdtemp(join(tmpdir(), 'ruv-fann-wsp-browser-'));
  const env = { ...process.env };
  for (const key of ['NPM_TOKEN', 'NODE_AUTH_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN', 'CARGO_REGISTRY_TOKEN']) delete env[key];
  let server, browserInstance;
  try {
    const { candidate, bytes } = await installArchive(candidatePath, temp, env);
    const rootPath = resolve(temp);
    server = createServer(async (request, response) => {
      try {
        const pathname = new URL(request.url, 'http://localhost').pathname;
        if (pathname === '/') {
          response.setHeader('content-type', 'text/html');
          return response.end(`<script type="importmap">{"imports":{"ruv-fann-wsp":"/node_modules/ruv-fann-wsp/src/index.js","ruv-fann":"/node_modules/ruv-fann/src/index.js","ruv-fann-wasm":"/node_modules/ruv-fann-wasm/src/index.js"}}</script><body>WSP browser test</body>`);
        }
        const file = resolve(rootPath, `.${decodeURIComponent(pathname)}`);
        if (!file.startsWith(rootPath + sep)) { response.writeHead(403); return response.end(); }
        const extension = extname(file);
        response.setHeader('content-type', extension === '.wasm' ? 'application/wasm' : extension === '.js' || extension === '.mjs' ? 'text/javascript' : 'application/octet-stream');
        response.end(await readFile(file));
      } catch { response.writeHead(404); response.end(); }
    });
    await new Promise(resolveListen => server.listen(0, '127.0.0.1', resolveListen));
    const launch = { headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] };
    if (process.env.RUV_FANN_CHROME) launch.executablePath = process.env.RUV_FANN_CHROME;
    browserInstance = await playwright.chromium.launch(launch);
    const page = await browserInstance.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const proof = await page.evaluate(async () => {
      const assert = (value, message) => { if (!value) throw new Error(message); };
      const wsp = await import('ruv-fann-wsp');
      const wasm = await import('ruv-fann-wasm');
      assert(wasm.supportsSIMD(), 'Chromium x64 must expose SIMD');
      const model={schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[0],bias:[0],activation:'Linear',steepness:1}]};
      const dataset={inputs:[0,1,2],targets:[1,3,5],rows:3};
      const policy={maxMSE:0.001,maxRegressionMSE:0.01,maxAbsoluteError:0.1};
      const baseHash=await wsp.fingerprintModel(model);
      const patch={schema:'ruv-fann.weight-patch/1',baseHash,edits:[{layer:0,parameter:'weights',index:0,value:2},{layer:0,parameter:'bias',index:0,value:1}]};
      const evaluation=await wsp.evaluateWeightPatch(model,patch,dataset,policy,{simd:true});
      assert(evaluation.accepted && evaluation.backend==='wasm-simd','browser real SIMD evaluation');
      const workspace=await wsp.createWeightWorkspace(model,{simd:false});
      const proposal=await workspace.propose(patch,dataset,policy);
      const promoted=workspace.promote(proposal.id,1);
      assert(promoted.revision===2 && promoted.model.layers[0].weights[0]===2,'browser promotion');
      const restored=workspace.rollback(1,2);
      assert(restored.revision===3 && restored.model.layers[0].weights[0]===0,'browser rollback');
      return {evaluationBackend:evaluation.backend,revision:promoted.revision,rollbackRevision:restored.revision};
    });
    const workbench = await page.evaluate(runWorkbenchProbe);
    const security = await page.evaluate(runWorkbenchSecurityProbe);
    validateWorkbenchProof({workbench,security});
    proof.workbench = workbench; proof.security = security;
    if (errors.length) throw new Error(errors.join('\n'));
    await mkdir(dirname(output), { recursive: true });
    await saveNew(output, { schema: 'ruv-fann-wsp.npm-browser/1', sourceSha: candidate.sourceSha,
      tarballSha256: sha256(bytes), browser: await browserInstance.version(), proof, status: 'passed' });
  } finally { await browserInstance?.close(); if (server) await new Promise(resolveClose => server.close(resolveClose)); await rm(temp, { recursive: true, force: true }); }
}

async function audit() {
  const candidatePath = resolve(required('candidate'));
  const temp = await mkdtemp(join(tmpdir(), 'ruv-fann-wsp-audit-'));
  const env = { ...process.env };
  for (const key of ['NPM_TOKEN', 'NODE_AUTH_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN', 'CARGO_REGISTRY_TOKEN']) delete env[key];
  try {
    const { candidate, bytes } = await installArchive(candidatePath, temp, env);
    const report = JSON.parse(exec('npm', ['audit', '--omit=dev', '--json', '--registry=https://registry.npmjs.org'], temp, env));
    const vulnerabilities = report.metadata?.vulnerabilities;
    if (!vulnerabilities || vulnerabilities.total !== 0 ||
        ['info', 'low', 'moderate', 'high', 'critical'].some(level => vulnerabilities[level] !== 0))
      throw new Error('Published dependency tree must have zero known npm advisories');
    const output = resolve(required('output')); await mkdir(dirname(output), { recursive: true });
    await saveNew(output, { schema: 'ruv-fann-wsp.npm-security/1', status: 'passed', sourceSha: candidate.sourceSha,
      tarballSha256: sha256(bytes), npmVersion: exec('npm', ['--version']), vulnerabilities,
      dependencyAudit: report.metadata.dependencies, checkedAt: new Date().toISOString(),
      scope: 'Known npm advisories for the installed production tree; adversarial API probes are required in both Node consumer and browser receipts.' });
  } finally { await rm(temp, { recursive: true, force: true }); }
}

async function benchmark() {
  const candidatePath = resolve(required('candidate'));
  const temp = await mkdtemp(join(tmpdir(), 'ruv-fann-wsp-benchmark-'));
  const env = { ...process.env };
  for (const key of ['NPM_TOKEN', 'NODE_AUTH_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN', 'CARGO_REGISTRY_TOKEN']) delete env[key];
  try {
    const { candidate, archive } = await installArchive(candidatePath, temp, env);
    const output = resolve(required('output')); await mkdir(dirname(output), { recursive: true });
    const args = [join(root, 'scripts/wsp-benchmark.mjs'), '--wsp-module', join(temp, 'node_modules/ruv-fann-wsp/src/index.js'), '--output', output];
    if (options.python) args.push('--python', resolve(options.python));
    console.log(exec(process.execPath, args, root, env));
    const report = await json(output); await validateBenchmark(report, candidate.sourceSha, archive);
  } finally { await rm(temp, { recursive: true, force: true }); }
}

async function seal() {
  const sourceSha = required('expected-sha');
  const releaseTag = required('release-tag');
  if (!/^[0-9a-f]{40}$/.test(sourceSha) || gitSha() !== sourceSha) throw new Error('Seal source SHA does not match checkout');
  if (releaseTag !== identity.tag) throw new Error(`Release tag must be ${identity.tag}`);
  assertClean();
  const directory = resolve(required('directory'));
  const candidate = await json(join(directory, 'package-manifest.json'));
  const archive = join(directory, candidate.tarball);
  const bytes = await readFile(archive);
  if (candidate.sourceSha !== sourceSha || candidate.name !== identity.name || candidate.version !== identity.version ||
      sha256(bytes) !== candidate.sha256 || sha512Integrity(bytes) !== candidate.integrity) throw new Error('Packed archive is not bound to the requested source/version/hash');
  const consumerDir = resolve(required('consumer-receipts'));
  const allReceiptFiles = (await readdir(consumerDir)).filter(name => name.endsWith('.json'));
  const receipts = (await Promise.all(allReceiptFiles.map(name => json(join(consumerDir, name)))))
    .filter(item => item.schema === 'ruv-fann-wsp.npm-consumer/1');
  const majors = new Set(receipts.map(item => Number(/^v(\d+)/.exec(item.runtime)?.[1])));
  if (receipts.length !== 2 || !majors.has(22) || !majors.has(24) || receipts.some(item => item.status !== 'passed' ||
      item.sourceSha !== sourceSha || item.tarballSha256 !== candidate.sha256)) throw new Error('Exact packed archive must pass both Node 22 and Node 24 consumers');
  for (const receipt of receipts) validateWorkbenchProof(receipt);
  const unitDir = resolve(required('unit-receipts'));
  const unitFiles = (await readdir(unitDir)).filter(name => name.endsWith('.json'));
  const units = (await Promise.all(unitFiles.map(name => json(join(unitDir, name)))))
    .filter(item => item.schema === 'ruv-fann-wsp.npm-unit/1');
  const unitMajors = new Set(units.map(item => Number(/^v(\d+)/.exec(item.runtime)?.[1])));
  if (units.length !== 2 || !unitMajors.has(22) || !unitMajors.has(24) || units.some(item => item.status !== 'passed' || item.sourceSha !== sourceSha ||
      item.passed < 1 || item.failed !== 0 || item.skipped !== 0 || item.todo !== 0)) throw new Error('WSP unit tests require nonzero passes and zero failures/skips/todos on Node 22 and 24');
  const browser = await json(resolve(required('browser-receipt')));
  if (browser.status !== 'passed' || browser.sourceSha !== sourceSha || browser.tarballSha256 !== candidate.sha256 ||
      browser.proof?.evaluationBackend !== 'wasm-simd') throw new Error('Exact packed archive browser receipt is missing or failed');
  validateWorkbenchProof(browser.proof);
  const benchmarkBytes = await readFile(resolve(required('benchmark-receipt')));
  const securityBytes = await readFile(resolve(required('security-receipt')));
  await validateBenchmark(JSON.parse(benchmarkBytes), sourceSha, archive);
  validateSecurity(JSON.parse(securityBytes), sourceSha, candidate.sha256);
  await writeFile(join(directory, 'benchmark.json'), benchmarkBytes, { flag: 'wx' });
  await writeFile(join(directory, 'security.json'), securityBytes, { flag: 'wx' });
  await saveNew(join(directory, 'candidate-seal.json'), { schema: 'ruv-fann-wsp.npm-seal/1', status: 'passed', sourceSha, releaseTag,
    name: identity.name, version: identity.version, tarball: candidate.tarball, sha256: candidate.sha256, integrity: candidate.integrity,
    unitReceiptSha256: await Promise.all(unitFiles.filter(name => (name.startsWith('node-') && name.includes('unit'))).map(async name => sha256(await readFile(join(unitDir, name))))),
    consumerReceiptSha256: await Promise.all(allReceiptFiles.filter(name => name.includes('consumer')).map(async name => sha256(await readFile(join(consumerDir, name))))),
    browserReceiptSha256: sha256(await readFile(resolve(options['browser-receipt']))),
    benchmarkReceiptSha256: sha256(benchmarkBytes), securityReceiptSha256: sha256(securityBytes),
    packageManifestSha256: sha256(await readFile(join(directory, 'package-manifest.json'))) });
}

function protectedContext(sourceSha) {
  const expectedWorkflowRef = `${process.env.GITHUB_REPOSITORY}/.github/workflows/protected-wsp-release.yml@refs/heads/main`;
  if (process.env.GITHUB_ACTIONS !== 'true' || process.env.GITHUB_WORKFLOW !== 'protected-release' ||
      process.env.GITHUB_WORKFLOW_REF !== expectedWorkflowRef || process.env.GITHUB_REPOSITORY !== 'ruvnet/ruv-FANN' ||
      process.env.GITHUB_REF !== 'refs/heads/main' || process.env.GITHUB_SHA !== sourceSha || gitSha() !== sourceSha)
    throw new Error('WSP publication requires the exact protected WSP workflow on reviewed main SHA');
  if (exec('gh', ['api', 'repos/ruvnet/ruv-FANN/commits/main', '--jq', '.sha']) !== sourceSha)
    throw new Error('origin/main advanced beyond sealed WSP source SHA');
  if (exec('gh', ['api', 'repos/ruvnet/ruv-FANN/issues?labels=release-blocker&state=open&per_page=1', '--jq', 'length']) !== '0')
    throw new Error('Open release-blocker issues prevent WSP publication');
}
const npmTarballUrl = seal => `https://registry.npmjs.org/${encodeURIComponent(seal.name)}/-/${encodeURIComponent(seal.name)}-${seal.version}.tgz`;
async function publicTarball(seal) {
  const response = await fetch(npmTarballUrl(seal), { redirect: 'error' });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`npm registry returned ${response.status} for ${seal.name}@${seal.version}`);
  return Buffer.from(await response.arrayBuffer());
}
async function registryPackage(name, version) {
  const url = `https://registry.npmjs.org/${encodeURIComponent(name)}/-/${encodeURIComponent(name)}-${version}.tgz`;
  const response = await fetch(url, { redirect: 'error' });
  if (!response.ok) throw new Error(`Could not fetch public dependency ${name}@${version}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  const temp = await mkdtemp(join(tmpdir(), 'wsp-dependency-check-'));
  let metadata;
  try { const file = join(temp, 'package.tgz'); await writeFile(file, bytes); metadata = await archiveInfo(file); }
  finally { await rm(temp, { recursive: true, force: true }); }
  if (metadata.name !== name || metadata.version !== version) throw new Error(`Registry dependency metadata mismatch for ${name}@${version}`);
  return { url, integrity: sha512Integrity(bytes), metadata };
}
async function awaitPublic(seal, timeout = 16 * 60_000) {
  const until = Date.now() + timeout;
  while (Date.now() < until) {
    const bytes = await publicTarball(seal);
    if (bytes) {
      const digest = sha256(bytes);
      if (digest !== seal.sha256) throw new Error(`Public tarball mismatch for ${seal.name}@${seal.version}: expected ${seal.sha256}, received ${digest}`);
      return bytes;
    }
    await new Promise(resolveWait => setTimeout(resolveWait, 10_000));
  }
  throw new Error(`npm registry still has no tarball after 16 minutes for ${seal.name}@${seal.version}`);
}

async function verifySeal(seal, directory) {
  if (seal.schema !== 'ruv-fann-wsp.npm-seal/1' || seal.status !== 'passed' || seal.sourceSha !== gitSha() ||
      seal.name !== identity.name || seal.version !== identity.version || seal.releaseTag !== identity.tag)
    throw new Error('WSP seal schema, status, source or release identity is invalid');
  const manifestPath = join(directory, 'package-manifest.json');
  const manifestBytes = await readFile(manifestPath);
  const manifest = JSON.parse(manifestBytes);
  if (sha256(manifestBytes) !== seal.packageManifestSha256 || manifest.sourceSha !== seal.sourceSha || manifest.name !== identity.name ||
      manifest.version !== identity.version || manifest.sha256 !== seal.sha256 || manifest.integrity !== seal.integrity)
    throw new Error('WSP candidate manifest is not bound by the seal');
  const archive = await readFile(join(directory, seal.tarball));
  if (sha256(archive) !== seal.sha256 || sha512Integrity(archive) !== seal.integrity) throw new Error('Sealed WSP tarball bytes changed');
  const files = await readdir(directory);
  const units = [], consumers = [];
  for (const name of files.filter(name => name.endsWith('.json') && name !== 'candidate-seal.json' && name !== 'package-manifest.json')) {
    const bytes = await readFile(join(directory, name));
    let receipt;
    try { receipt = JSON.parse(bytes); } catch { continue; }
    if (receipt.schema === 'ruv-fann-wsp.npm-unit/1') units.push({ name, bytes, receipt });
    if (receipt.schema === 'ruv-fann-wsp.npm-consumer/1') consumers.push({ name, bytes, receipt });
  }
  const digestSet = rows => rows.map(row => sha256(row.bytes)).sort();
  if (JSON.stringify(digestSet(units)) !== JSON.stringify([...seal.unitReceiptSha256].sort()) ||
      JSON.stringify(digestSet(consumers)) !== JSON.stringify([...seal.consumerReceiptSha256].sort()))
    throw new Error('WSP qualification receipt hashes do not match the seal');
  const unitMajors = new Set(units.map(({ receipt }) => Number(/^v(\d+)/.exec(receipt.runtime)?.[1])));
  const consumerMajors = new Set(consumers.map(({ receipt }) => Number(/^v(\d+)/.exec(receipt.runtime)?.[1])));
  if (units.length !== 2 || consumers.length !== 2 || !unitMajors.has(22) || !unitMajors.has(24) || !consumerMajors.has(22) || !consumerMajors.has(24) ||
      units.some(({ receipt }) => !(receipt.passed > 0) || receipt.failed !== 0 || receipt.skipped !== 0 || receipt.todo !== 0))
    throw new Error('WSP seal lacks nonzero Node 22/24 test and consumer results with zero skips/todos');
  for (const { receipt } of consumers) validateWorkbenchProof(receipt);
  const requiredMutations = ['caller patch and dataset snapshot before await', 'returned proposal mutation isolated from workspace state',
    'stale revision rejected', 'forged proposal receipt rejected'];
  for (const { receipt } of [...units, ...consumers]) if (receipt.sourceSha !== seal.sourceSha || receipt.status !== 'passed')
    throw new Error('A WSP qualification receipt is not bound to the sealed source');
  if (consumers.some(({ receipt }) => receipt.tarballSha256 !== seal.sha256 || requiredMutations.some(item => !receipt.mutationChecks?.includes(item))))
    throw new Error('WSP consumer receipts lack the required mutation and forgery probes');
  const browser = files.find(name => name.endsWith('browser.json'));
  const browserReceipt = browser ? await json(join(directory, browser)) : null;
  if (!browser || sha256(await readFile(join(directory, browser))) !== seal.browserReceiptSha256 || browserReceipt.status !== 'passed' ||
      browserReceipt.sourceSha !== seal.sourceSha || browserReceipt.tarballSha256 !== seal.sha256 || browserReceipt.proof?.evaluationBackend !== 'wasm-simd')
    throw new Error('WSP browser receipt hash or runtime proof does not match the seal');
  validateWorkbenchProof(browserReceipt.proof);
  const benchmarkBytes = await readFile(join(directory, 'benchmark.json'));
  const securityBytes = await readFile(join(directory, 'security.json'));
  if (sha256(benchmarkBytes) !== seal.benchmarkReceiptSha256 || sha256(securityBytes) !== seal.securityReceiptSha256)
    throw new Error('Benchmark/security evidence bytes differ from the seal');
  await validateBenchmark(JSON.parse(benchmarkBytes), seal.sourceSha, join(directory, seal.tarball));
  validateSecurity(JSON.parse(securityBytes), seal.sourceSha, seal.sha256);
  return archive;
}

async function publish() {
  const sealPath = resolve(required('seal'));
  const seal = await json(sealPath);
  protectedContext(seal.sourceSha);
  if (seal.name !== identity.name || seal.version !== identity.version || seal.releaseTag !== identity.tag || !process.env.NODE_AUTH_TOKEN)
    throw new Error('Missing NPM_TOKEN or invalid WSP sealed identity');
  const archivePath = join(dirname(sealPath), seal.tarball);
  const archiveBytes = await verifySeal(seal, dirname(sealPath));
  const existing = await publicTarball(seal);
  if (existing) {
    if (sha256(existing) !== seal.sha256) throw new Error('The public WSP version already exists with different bytes');
  } else {
    try { exec('npm', ['publish', archivePath, '--ignore-scripts', '--access', 'public', '--provenance', '--registry=https://registry.npmjs.org'], root); }
    catch (error) {
      // An earlier protected attempt may have completed its upload; accept it only after exact-byte proof.
      try { await awaitPublic(seal); } catch { throw error; }
    }
  }
  await awaitPublic(seal);
  const output = resolve(required('output'));
  await mkdir(dirname(output), { recursive: true });
  await saveNew(output, { schema: 'ruv-fann-wsp.npm-publication/1', sourceSha: seal.sourceSha,
    releaseTag: seal.releaseTag, package: seal.name, version: seal.version, sha256: seal.sha256, status: 'public-byte-verified' });
}

async function publicConsumer() {
  const seal = await json(resolve(required('seal')));
  const publication = await json(resolve(required('publication')));
  if (publication.sourceSha !== seal.sourceSha || publication.sha256 !== seal.sha256 || publication.status !== 'public-byte-verified')
    throw new Error('WSP publication receipt does not match candidate seal');
  const publicBytes = await awaitPublic(seal);
  const temp = await mkdtemp(join(tmpdir(), 'ruv-fann-wsp-public-'));
  const env = { ...process.env };
  for (const key of ['NPM_TOKEN', 'NODE_AUTH_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN', 'CARGO_REGISTRY_TOKEN']) delete env[key];
  try {
    await writeFile(join(temp, 'package.json'), JSON.stringify({ name: 'wsp-public-consumer', version: '1.0.0', private: true, type: 'module', dependencies: { [seal.name]: seal.version } }));
    await writeFile(join(temp, '.npmrc'), 'registry=https://registry.npmjs.org\n');
    env.NPM_CONFIG_USERCONFIG = join(temp, '.npmrc'); env.NPM_CONFIG_CACHE = join(temp, 'fresh-npm-cache');
    const core = await registryPackage('ruv-fann', '0.1.2');
    const wasm = await registryPackage('ruv-fann-wasm', '0.1.2');
    if (core.metadata.dependencies?.['ruv-fann-wasm'] !== '0.1.2') throw new Error('Published core does not declare the exact WASM dependency');
    const lock = { name: 'wsp-public-consumer', version: '1.0.0', lockfileVersion: 3, requires: true, packages: {
      '': { name: 'wsp-public-consumer', version: '1.0.0', dependencies: { [seal.name]: seal.version } },
      [`node_modules/${seal.name}`]: { version: seal.version, resolved: npmTarballUrl(seal), integrity: sha512Integrity(publicBytes), dependencies: { 'ruv-fann': '0.1.2' } },
      'node_modules/ruv-fann': { version: '0.1.2', resolved: core.url, integrity: core.integrity, dependencies: { 'ruv-fann-wasm': '0.1.2' } },
      'node_modules/ruv-fann-wasm': { version: '0.1.2', resolved: wasm.url, integrity: wasm.integrity }
    } };
    await writeFile(join(temp, 'package-lock.json'), JSON.stringify(lock));
    exec('npm', ['ci', '--ignore-scripts', '--no-audit', '--no-fund', '--registry=https://registry.npmjs.org'], temp, env);
    await writeFile(join(temp, 'integration.mjs'), nodeIntegrationSource());
    const runtimeOutput = exec(process.execPath, ['integration.mjs'], temp, env);
    console.log(runtimeOutput);
    const integrationProof = JSON.parse(runtimeOutput.split('\n').at(-1));
    validateWorkbenchProof(integrationProof);
    await mkdir(dirname(resolve(required('output'))), { recursive: true });
    await saveNew(resolve(required('output')), { schema: 'ruv-fann-wsp.npm-public-consumer/1', sourceSha: seal.sourceSha,
      releaseTag: seal.releaseTag, package: seal.name, version: seal.version, tarballSha256: sha256(publicBytes), status: 'passed', runtime: process.version,
      workbench: integrationProof.workbench, security: integrationProof.security,
      scenarios: ['fresh-cache public npm ci with SHA512 integrity', 'real WASM patch evaluation', 'workspace propose/promote/rollback'] });
  } finally { await rm(temp, { recursive: true, force: true }); }
}

function tagCommit(tag) {
  let result = exec('gh', ['api', `repos/ruvnet/ruv-FANN/git/ref/tags/${tag}`, '--jq', '.object | [.type, .sha] | @tsv']);
  for (let depth = 0; depth < 5; depth++) {
    const [type, sha] = result.split('\t');
    if (type === 'commit') return sha;
    if (type !== 'tag' || !/^[0-9a-f]{40}$/.test(sha)) throw new Error('WSP release tag does not resolve to a commit');
    result = exec('gh', ['api', `repos/ruvnet/ruv-FANN/git/tags/${sha}`, '--jq', '.object | [.type, .sha] | @tsv']);
  }
  throw new Error('WSP release tag nesting exceeds safety limit');
}

async function githubRelease() {
  const sealPath = resolve(required('seal'));
  const seal = await json(sealPath);
  protectedContext(seal.sourceSha);
  await verifySeal(seal, dirname(sealPath));
  const publicationPath = resolve(required('publication'));
  const consumerPath = resolve(required('public-consumer'));
  const publication = await json(publicationPath), consumerReceipt = await json(consumerPath);
  if (publication.sourceSha !== seal.sourceSha || publication.sha256 !== seal.sha256 || consumerReceipt.sourceSha !== seal.sourceSha ||
      consumerReceipt.tarballSha256 !== seal.sha256 || consumerReceipt.status !== 'passed') throw new Error('WSP public evidence is inconsistent');
  const receiptPath = resolve(required('release-receipt'));
  await mkdir(dirname(receiptPath), { recursive: true });
  await saveNew(receiptPath, { schema: 'ruv-fann-wsp.npm-release/1', sourceSha: seal.sourceSha, releaseTag: seal.releaseTag,
    package: seal.name, version: seal.version, tarballSha256: seal.sha256, integrity: seal.integrity,
    publicationReceiptSha256: sha256(await readFile(publicationPath)), publicConsumerReceiptSha256: sha256(await readFile(consumerPath)) });
  const assets = [join(dirname(sealPath), seal.tarball), sealPath, publicationPath, consumerPath, receiptPath,
    join(dirname(sealPath), 'benchmark.json'), join(dirname(sealPath), 'security.json')];
  let existing;
  try { existing = JSON.parse(exec('gh', ['release', 'view', seal.releaseTag, '--json', 'assets'])); }
  catch {
    exec('gh', ['release', 'create', seal.releaseTag, ...assets, '--target', seal.sourceSha, '--title', seal.releaseTag,
      '--notes', `ruv-fann-wsp@${seal.version} from source ${seal.sourceSha}; bounded real-WASM traces, neuron ablation, numerical sensitivity and public-data evaluation. Attached archive, benchmark, dependency audit and exact-byte receipts. The benchmark retains all outcomes and makes no SOTA claim.`]);
    existing = null;
  }
  if (tagCommit(seal.releaseTag) !== seal.sourceSha) throw new Error('WSP GitHub release tag does not point to the exact sealed source SHA');
  if (existing) {
    const expectedNames = assets.map(path => basename(path)).sort();
    const actualNames = existing.assets.map(item => item.name).sort();
    if (JSON.stringify(actualNames) !== JSON.stringify(expectedNames)) throw new Error('Existing WSP release asset set differs from the exact sealed release');
  }
  const download = await mkdtemp(join(tmpdir(), 'ruv-fann-wsp-gh-release-'));
  try {
    exec('gh', ['release', 'download', seal.releaseTag, '--dir', download]);
    for (const [index, path] of assets.entries()) {
      const expected = await readFile(path), actual = await readFile(join(download, basename(path)));
      if (sha256(actual) !== sha256(expected)) throw new Error(`WSP GitHub release asset byte mismatch: ${basename(path)}`);
    }
  } finally { await rm(download, { recursive: true, force: true }); }
}

async function verify() { const path = resolve(required('seal')); await verifySeal(await json(path), dirname(path)); console.log('Exact WSP candidate evidence verified'); }

const modes = { verify, pack, unit, consumer, browser, audit, benchmark, seal, publish, 'public-consumer': publicConsumer, 'github-release': githubRelease };
if (!modes[mode]) throw new Error(`Unknown WSP release mode: ${mode}`);
await modes[mode]();
