#!/usr/bin/env node
// Source-bound pack, consumer, publication and release receipts for standalone npm modules.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, readdir, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, basename, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const args = process.argv.slice(2);
const mode = args.shift();
const opts = Object.create(null);
for (let i = 0; i < args.length; i++) {
  const key = args[i];
  if (!key.startsWith('--') || i + 1 >= args.length || key in opts) throw new Error(`Invalid argument: ${key}`);
  opts[key.slice(2)] = args[++i];
}
const need = key => { if (!opts[key]) throw new Error(`Missing --${key}`); return opts[key]; };
const run = (bin, argv, cwd = root, env = process.env) => execFileSync(bin, argv, { cwd, env, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'], timeout: 30 * 60_000 }).trim();
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const readJson = async path => JSON.parse(await readFile(path, 'utf8'));
const saveNew = async (path, data) => {
  await writeFile(path, `${JSON.stringify(data, null, 2)}\n`, { flag: 'wx' });
};
const currentSha = () => run('git', ['rev-parse', 'HEAD']);
const cleanSource = () => {
  if (run('git', ['status', '--porcelain']).length) throw new Error('Release source tree is not clean');
};
const stripCredentials = () => Object.fromEntries(Object.entries(process.env).filter(([key]) =>
  !/^(NPM_TOKEN|NODE_AUTH_TOKEN|GH_TOKEN|GITHUB_TOKEN|CARGO_REGISTRY_TOKEN)$/i.test(key)));
const packageRows = async manifestPath => {
  const manifest = await readJson(manifestPath);
  if (!Array.isArray(manifest.packages) || manifest.packages.length < 1) throw new Error('Manifest must contain ordered packages');
  const names = new Set();
  for (const row of manifest.packages) {
    if (!row || typeof row.name !== 'string' || typeof row.version !== 'string' || typeof row.tarball !== 'string' || names.has(row.name))
      throw new Error('Each package needs unique name, version and tarball');
    names.add(row.name);
    row.tarball = resolve(dirname(manifestPath), row.tarball);
  }
  return manifest;
};
const archiveMetadata = async tarball => {
  const stdout = run('tar', ['-xOzf', tarball, 'package/package.json']);
  return JSON.parse(stdout);
};
const packageConfig = JSON.parse('{"packages":[{"name":"ruv-fann-wasm","version":"0.1.0","directory":"packages/wasm"},{"name":"ruv-fann","version":"0.1.0","directory":"packages/core"}]}');

async function pack() {
  const out = resolve(need('output'));
  const tarballDir = resolve(need('tarballs'));
  cleanSource();
  const rows = [];
  for (const spec of packageConfig.packages) {
    const entries = (await readdir(tarballDir)).filter(name => name === `${spec.name}-${spec.version}.tgz`);
    if (entries.length !== 1) throw new Error(`Expected the already-tested one-time pack for ${spec.name}`);
    const tarball = join(tarballDir, entries[0]);
    const metadata = await archiveMetadata(tarball);
    if (metadata.name !== spec.name || metadata.version !== spec.version || metadata.private)
      throw new Error(`Packed manifest identity mismatch for ${spec.name}`);
    rows.push({ name: spec.name, version: spec.version, tarball: basename(tarball), sha256: hash(await readFile(tarball)) });
  }
  if (rows[1].name !== 'ruv-fann' || rows[1].version !== '0.1.0') throw new Error('Core package identity mismatch');
  const core = await archiveMetadata(join(tarballDir, rows[1].tarball));
  if (core.dependencies?.['ruv-fann-wasm'] !== rows[0].version) throw new Error('Core must depend on the exact WASM release');
  await mkdir(out, { recursive: true });
  await saveNew(join(out, 'package-manifest.json'), { schema: 'ruv-fann.npm-candidate/1', packages: rows });
  console.log(JSON.stringify(rows.map(({ name, version, sha256 }) => ({ name, version, sha256 })), null, 2));
}

async function consumer() {
  const manifest = await packageRows(resolve(need('manifest')));
  const out = resolve(need('output'));
  const temp = await mkdtemp(join(tmpdir(), 'ruv-fann-npm-consumer-'));
  const env = stripCredentials();
  try {
    await writeFile(join(temp, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
    await writeFile(join(temp, '.npmrc'), 'registry=https://registry.npmjs.org\n');
    env.NPM_CONFIG_USERCONFIG = join(temp, '.npmrc');
    const specs = manifest.packages.map(row => resolve(row.tarball));
    run('npm', ['install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund', ...specs], temp, env);
    const script = opts['consumer-script'] ? await readFile(resolve(opts['consumer-script']), 'utf8') : `import assert from 'node:assert/strict';\nimport {loadRuntime} from 'ruv-fann';\nimport {loadWasm,supportsSIMD} from 'ruv-fann-wasm';\nif(!supportsSIMD()) throw new Error('SIMD unavailable on required x64 runner');\nconst cases=[false,true];\nfor(const simd of cases){const raw=await loadWasm({simd});assert.equal(raw.backend,simd?'wasm-simd':'wasm');const runtime=await loadRuntime({simd});const model={schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[0],bias:[0],activation:'Linear',steepness:1}]};const prepared=runtime.prepare(model,{maxRows:2});assert.deepEqual([...prepared.predict(Float64Array.of(1,2))],[0,0]);prepared.dispose();const result=runtime.train(model,{inputs:Float64Array.of(0,1,2),targets:Float64Array.of(1,3,5),rows:3,epochs:300,learningRate:0.05,optimizer:'adamw'});assert.ok(result.finalLoss<result.initialLoss/1000);assert.ok(Math.abs(result.model.layers[0].weights[0]-2)<0.01);assert.ok(Math.abs(result.model.layers[0].bias[0]-1)<0.01);}\nconsole.log('packed real WASM inference and training passed');\n`;
    await writeFile(join(temp, 'verify.mjs'), script);
    run(process.execPath, ['verify.mjs'], temp, env);
    const manifestWithDigests = await Promise.all(manifest.packages.map(async row => ({
      name: row.name, version: row.version, sha256: hash(await readFile(resolve(row.tarball)))
    })));
    await mkdir(dirname(out), { recursive: true });
    await saveNew(out, { schema: 'ruv-fann.npm-consumer/1', runtime: process.version, status: 'passed', packageDigests: manifestWithDigests, command: 'fresh npm install of exact tarballs; real WASM inference and AdamW training' });
  } finally { await rm(temp, { recursive: true, force: true }); }
}

async function seal() {
  const expectedSha = need('expected-sha');
  const releaseTag = need('release-tag');
  if (!/^[0-9a-f]{40}$/.test(expectedSha) || currentSha() !== expectedSha) throw new Error('Expected source SHA does not match checkout');
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{1,100}$/.test(releaseTag)) throw new Error('Invalid GitHub release tag');
  cleanSource();
  const base = resolve(need('directory'));
  const manifest = await packageRows(join(base, 'package-manifest.json'));
  const receiptPaths = opts['consumer-receipts']
    ? (await readdir(resolve(opts['consumer-receipts']))).filter(name => name.endsWith('.json')).map(name => join(resolve(opts['consumer-receipts']), name))
    : [resolve(need('consumer-receipt'))];
  const receipts = await Promise.all(receiptPaths.map(readJson));
  if (receipts.length < 2 || receipts.some(receipt => receipt.schema !== 'ruv-fann.npm-consumer/1' || receipt.status !== 'passed' || !receipt.packageDigests?.length))
    throw new Error('Packed consumer receipts are absent, incomplete or failed');
  const runtimes = new Set(receipts.map(receipt => Number(/^v(\d+)/.exec(receipt.runtime)?.[1])));
  if (!runtimes.has(22) || !runtimes.has(24)) throw new Error('Packed consumers must pass on Node 22 and Node 24');
  const rows = [];
  for (const row of manifest.packages) {
    const path = resolve(row.tarball);
    const digest = hash(await readFile(path));
    const metadata = await archiveMetadata(path);
    if (metadata.name !== row.name || metadata.version !== row.version || digest !== row.sha256) throw new Error(`Package archive changed: ${row.name}`);
    for (const receipt of receipts) if (!receipt.packageDigests.some(item => item.name === row.name && item.version === row.version && item.sha256 === digest))
      throw new Error(`Packed consumer did not test exact archive on ${receipt.runtime}: ${row.name}`);
    rows.push({ name: row.name, version: row.version, tarball: basename(path), filename: basename(path), sha256: digest });
  }
  if (rows.length === 2 && (rows[0].name !== 'ruv-fann-wasm' || rows[1].name !== 'ruv-fann')) throw new Error('Packages must publish in WASM then core order');
  const receiptHashes = await Promise.all(receiptPaths.map(async path => hash(await readFile(path))));
  await saveNew(join(base, 'candidate-seal.json'), { schema: 'ruv-fann.npm-seal/1', sourceSha: expectedSha, releaseTag,
    consumerReceiptSha256: receiptHashes, packages: rows });
}

function protectedContext(expectedSha) {
  if (process.env.GITHUB_ACTIONS !== 'true' || process.env.GITHUB_WORKFLOW !== 'protected-release' ||
      process.env.GITHUB_REPOSITORY !== 'ruvnet/ruv-FANN' || process.env.GITHUB_REF !== 'refs/heads/main' ||
      process.env.GITHUB_SHA !== expectedSha || currentSha() !== expectedSha)
    throw new Error('Publication requires protected-npm-modules on exact reviewed main SHA');
  const mainSha = run('gh', ['api', 'repos/ruvnet/ruv-FANN/commits/main', '--jq', '.sha']);
  if (mainSha !== expectedSha) throw new Error('origin/main advanced beyond the sealed source SHA');
  const blockers = run('gh', ['api', 'repos/ruvnet/ruv-FANN/issues?labels=release-blocker&state=open&per_page=1', '--jq', 'length']);
  if (blockers !== '0') throw new Error('Open release-blocker issues prevent npm publication');
}
function tagCommit(tag) {
  let query = `repos/ruvnet/ruv-FANN/git/ref/tags/${tag}`;
  let result = run('gh', ['api', query, '--jq', '.object | [.type, .sha] | @tsv']);
  for (let depth = 0; depth < 5; depth++) {
    const [type, sha] = result.split('\t');
    if (type === 'commit') return sha;
    if (type !== 'tag' || !/^[0-9a-f]{40}$/.test(sha)) throw new Error('GitHub release tag does not resolve to a commit');
    result = run('gh', ['api', `repos/ruvnet/ruv-FANN/git/tags/${sha}`, '--jq', '.object | [.type, .sha] | @tsv']);
  }
  throw new Error('GitHub release tag nesting exceeds the safety limit');
}
const registryUrl = row => `https://registry.npmjs.org/${encodeURIComponent(row.name)}/-/${encodeURIComponent(row.name)}-${row.version}.tgz`;
async function publicBytes(row) {
  const response = await fetch(registryUrl(row), { redirect: 'error' });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Registry returned ${response.status} for ${row.name}`);
  return Buffer.from(await response.arrayBuffer());
}
async function waitForPublic(row, expectedHash, timeoutMs = 16 * 60_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const bytes = await publicBytes(row);
    if (bytes) {
      const actual = hash(bytes);
      if (actual !== expectedHash) throw new Error(`Published tarball hash mismatch for ${row.name}: expected ${expectedHash}; got ${actual}`);
      return bytes;
    }
    await new Promise(resolveSleep => setTimeout(resolveSleep, 10_000));
  }
  throw new Error(`Registry propagation timeout (still not visible after 16 minutes) for ${row.name}`);
}

async function publish() {
  const sealPath = resolve(need('seal'));
  const seal = await readJson(sealPath);
  protectedContext(seal.sourceSha);
  if (!process.env.NODE_AUTH_TOKEN) throw new Error('NPM_TOKEN is required in protected publishing job');
  const published = [];
  for (const row of seal.packages) {
    const tarball = resolve(dirname(sealPath), row.tarball);
    const bytes = await readFile(tarball);
    if (hash(bytes) !== row.sha256) throw new Error(`Sealed npm artifact changed: ${row.name}`);
    const existing = await publicBytes(row);
    if (existing) {
      if (hash(existing) !== row.sha256) throw new Error(`Version already exists with different bytes: ${row.name}@${row.version}`);
    } else {
      try {
        run('npm', ['publish', tarball, '--ignore-scripts', '--access', 'public', '--provenance', '--registry=https://registry.npmjs.org'], root);
      } catch (error) {
        // An earlier attempt may have uploaded before its runner failed. Retry only by proving bytes.
        try { await waitForPublic(row, row.sha256); }
        catch { throw error; }
      }
    }
    await waitForPublic(row, row.sha256);
    published.push({ name: row.name, version: row.version, filename: row.filename, sha256: row.sha256, status: existing ? 'existing-version-byte-equal' : 'published-byte-verified' });
  }
  const receipt = { schema: 'ruv-fann.npm-publication/1', sourceSha: seal.sourceSha, releaseTag: seal.releaseTag, packages: published };
  await saveNew(resolve(need('output')), receipt);
}

async function publicVerify() {
  const seal = await readJson(resolve(need('seal')));
  const publication = await readJson(resolve(need('publication')));
  if (publication.sourceSha !== seal.sourceSha || publication.releaseTag !== seal.releaseTag) throw new Error('Publication receipt does not match candidate seal');
  for (const row of seal.packages) await waitForPublic(row, row.sha256);
  const temp = await mkdtemp(join(tmpdir(), 'ruv-fann-public-consumer-'));
  const env = stripCredentials();
  try {
    const packageJson = { name: 'ruv-fann-public-consumer', version: '1.0.0', private: true, type: 'module', dependencies: Object.fromEntries(seal.packages.map(row => [row.name, row.version])) };
    await writeFile(join(temp, 'package.json'), JSON.stringify(packageJson));
    await writeFile(join(temp, '.npmrc'), 'registry=https://registry.npmjs.org\n');
    env.NPM_CONFIG_USERCONFIG = join(temp, '.npmrc');
    const lock = { name: 'ruv-fann-public-consumer', version: '1.0.0', lockfileVersion: 3, requires: true,
      packages: { '': packageJson } };
    for (const row of seal.packages) {
      const bytes = await publicBytes(row);
      if (!bytes || hash(bytes) !== row.sha256) throw new Error(`Public tarball changed before install: ${row.name}`);
      lock.packages[`node_modules/${row.name}`] = { version: row.version, resolved: registryUrl(row),
        integrity: `sha512-${createHash('sha512').update(bytes).digest('base64')}` };
      if (row.name === 'ruv-fann') lock.packages[`node_modules/${row.name}`].dependencies = { 'ruv-fann-wasm': '0.1.0' };
    }
    await writeFile(join(temp, 'package-lock.json'), JSON.stringify(lock));
    run('npm', ['ci', '--ignore-scripts', '--no-audit', '--no-fund', '--registry=https://registry.npmjs.org'], temp, env);
    const script = opts['consumer-script'] ? await readFile(resolve(opts['consumer-script']), 'utf8') : `import assert from 'node:assert/strict';\nimport {loadRuntime} from 'ruv-fann';\nimport {supportsSIMD} from 'ruv-fann-wasm';\nif(!supportsSIMD()) throw new Error('SIMD unavailable on required x64 runner');\nfor(const simd of [false,true]){const runtime=await loadRuntime({simd});const model={schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[0],bias:[0],activation:'Linear',steepness:1}]};const p=runtime.prepare(model);assert.deepEqual([...p.predict(Float64Array.of(1,2))],[0,0]);p.dispose();const result=runtime.train(model,{inputs:Float64Array.of(0,1,2),targets:Float64Array.of(1,3,5),rows:3,epochs:300,learningRate:0.05,optimizer:'adamw'});assert.ok(result.finalLoss<result.initialLoss/1000);}\nconsole.log('public npm install and real WASM inference/training passed');\n`;
    await writeFile(join(temp, 'verify.mjs'), script);
    run(process.execPath, ['verify.mjs'], temp, env);
  } finally { await rm(temp, { recursive: true, force: true }); }
  await saveNew(resolve(need('output')), { schema: 'ruv-fann.npm-public-consumer/1', sourceSha: seal.sourceSha, releaseTag: seal.releaseTag, status: 'passed', packageDigests: seal.packages.map(({ name, version, sha256 }) => ({ name, version, sha256 })) });
}

async function githubRelease() {
  const sealPath = resolve(need('seal'));
  const seal = await readJson(sealPath);
  protectedContext(seal.sourceSha);
  const publication = await readJson(resolve(need('publication')));
  const consumer = await readJson(resolve(need('public-consumer')));
  if (publication.sourceSha !== seal.sourceSha || consumer.sourceSha !== seal.sourceSha || consumer.status !== 'passed') throw new Error('Publication evidence mismatch');
  if (!process.env.GH_TOKEN) throw new Error('GH_TOKEN required for release creation');
  const receiptPath = resolve(need('release-receipt'));
  const receipt = { schema: 'ruv-fann.npm-release/1', sourceSha: seal.sourceSha, releaseTag: seal.releaseTag,
    packages: seal.packages.map(({ name, version, filename, sha256 }) => ({ name, version, filename, sha256 })),
    publicConsumerReceiptSha256: hash(await readFile(resolve(need('public-consumer')))) };
  await saveNew(receiptPath, receipt);
  const assetPaths = [...seal.packages.map(row => resolve(dirname(sealPath), row.tarball)), resolve(need('publication')), resolve(need('public-consumer')), receiptPath];
  let assets;
  try {
    const existing = execFileSync('gh', ['release', 'view', seal.releaseTag, '--json', 'assets'], { cwd: root, env: process.env, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
    assets = JSON.parse(existing).assets;
  }
  catch {
    run('gh', ['release', 'create', seal.releaseTag, ...assetPaths, '--target', seal.sourceSha, '--title', seal.releaseTag,
      '--notes', `Standalone ruv-FANN npm modules from source ${seal.sourceSha}. Exact npm archives and receipts are attached.`], root);
    assets = [];
  }
  if (assets.length) for (const path of assetPaths) {
    const matching = assets.find(asset => asset.name === basename(path));
    if (!matching) throw new Error(`Existing GitHub release lacks expected asset ${basename(path)}`);
  }
  if (tagCommit(seal.releaseTag) !== seal.sourceSha) throw new Error('GitHub release tag resolves to a different source SHA');
  // Existing releases are accepted only after download and byte comparison below.
  const temp = await mkdtemp(join(tmpdir(), 'ruv-fann-gh-assets-'));
  try {
    run('gh', ['release', 'download', seal.releaseTag, '--dir', temp], root);
    for (const row of seal.packages) if (hash(await readFile(join(temp, row.filename))) !== row.sha256) throw new Error(`GitHub asset hash mismatch for ${row.filename}`);
    for (const path of [resolve(need('publication')), resolve(need('public-consumer')), receiptPath]) {
      if (hash(await readFile(join(temp, basename(path)))) !== hash(await readFile(path))) throw new Error(`GitHub receipt asset mismatch: ${basename(path)}`);
    }
  } finally { await rm(temp, { recursive: true, force: true }); }
}

const modes = { pack, consumer, seal, publish, 'verify-public': publicVerify, 'github-release': githubRelease };
if (!modes[mode]) throw new Error(`Unsupported mode: ${mode}`);
await modes[mode]();
