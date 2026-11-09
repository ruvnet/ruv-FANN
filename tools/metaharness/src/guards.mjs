/** Input and process boundaries for the private MetaHarness tool. No shell or installers. */
import { constants } from 'node:fs';
import { lstat, mkdir, open, readdir, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

export const TOOL_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const PINS = Object.freeze({ referenceCommit: 'b27878cf39d3ada6b2c4d1bcc3e9258a70c6cfd6',
  repository: 'https://github.com/ruvnet/metaharness', metaharness: '0.4.21', darwin: '0.13.0' });
export const DEFAULTS = Object.freeze({ variant: 'core', model: 'tiny', generations: 2,
  children: 2, seed: 7, samples: 20, totalRows: 512, timeoutMs: 60000, baselineBatchRows: 1 });
const INTEGER_OPTIONS = { generations: [1, 3], children: [1, 2], seed: [0, 4294967295],
  samples: [5, 20], totalRows: [128, 1024], timeoutMs: [100, 120000], baselineBatchRows: [1, 128] };
const FLAGS = { '--repo': 'repo', '--output': 'output', '--variant': 'variant', '--model': 'model',
  '--generations': 'generations', '--children': 'children', '--seed': 'seed', '--samples': 'samples',
  '--total-rows': 'totalRows', '--timeout-ms': 'timeoutMs', '--baseline-batch-rows': 'baselineBatchRows' };
export class InputError extends Error {}
export class ExperimentError extends Error {}

/** Strict integer parsing rejects NaN, Infinity, overflow, whitespace and partial values. */
export function integer(value, min, max, name) {
  if ((typeof value !== 'number' && typeof value !== 'string') ||
      (typeof value === 'string' && !/^(0|[1-9][0-9]*)$/.test(value)))
    throw new InputError(`${name} must be an integer from ${min} to ${max}`);
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < min || number > max)
    throw new InputError(`${name} must be an integer from ${min} to ${max}`);
  return number;
}

export function validateOptions(options = {}) {
  const allowed = new Set([...Object.keys(DEFAULTS), 'repo', 'output', 'operation']);
  if (options === null || typeof options !== 'object' || Array.isArray(options) ||
      Object.keys(options).some(key => !allowed.has(key))) throw new InputError('unknown options');
  const result = { ...DEFAULTS, ...options };
  for (const [name, bounds] of Object.entries(INTEGER_OPTIONS))
    result[name] = integer(result[name], ...bounds, name);
  if (!['core', 'simd'].includes(result.variant) || !['tiny', 'medium', 'large'].includes(result.model))
    throw new InputError('unsupported variant or model');
  if (1 + result.children + 2 * result.children * (result.generations - 1) > 12)
    throw new InputError('numeric search exceeds the independent 12-dispatch limit');
  for (const name of ['repo', 'output']) if (result[name] !== undefined &&
    (typeof result[name] !== 'string' || !result[name] || result[name].includes('\0')))
    throw new InputError(`${name} must be a nonempty path`);
  return result;
}

export function parseArgs(argv) {
  const operation = argv[0];
  if (!['plan', 'score', 'optimize'].includes(operation)) throw new InputError('expected plan, score or optimize');
  const values = { operation, repo: '.' };
  const seen = new Set();
  for (let index = 1; index < argv.length; index += 2) {
    const key = FLAGS[argv[index]];
    if (!key || seen.has(key) || index + 1 >= argv.length) throw new InputError('unknown, duplicate or missing flag value');
    seen.add(key); values[key] = argv[index + 1];
  }
  if (operation !== 'plan' && !values.output) throw new InputError('--output must name a new external directory');
  if (operation === 'plan' && values.output) throw new InputError('plan is read-only; omit --output');
  return validateOptions(values);
}

export function inside(root, value) {
  const part = relative(root, value);
  return part === '' || (!part.startsWith('..' + sep) && part !== '..' && !isAbsolute(part));
}

export async function ordinaryAncestry(value) {
  let current = resolve(value);
  for (;;) {
    try { if ((await lstat(current)).isSymbolicLink()) throw new InputError(`symlink ancestry is unsupported: ${current}`); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
}

/** Read one contained ordinary file with a byte cap and no final-component symlink. */
export async function fileBytes(root, name, limit = 8 * 1024 * 1024) {
  if (typeof name !== 'string' || name.includes('\0') || name.includes('\\') || isAbsolute(name))
    throw new InputError('invalid relative input path');
  const path = resolve(root, name);
  if (!inside(root, path)) throw new InputError('input path escapes its root');
  let current = path;
  while (current !== root) {
    const info = await lstat(current);
    if (info.isSymbolicLink()) throw new InputError(`symlink input rejected: ${name}`);
    if ((current === path && !info.isFile()) || (current !== path && !info.isDirectory()))
      throw new InputError(`special or non-file input rejected: ${name}`);
    current = dirname(current);
  }
  const handle = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0) | (constants.O_NONBLOCK ?? 0));
  try {
    const info = await handle.stat();
    if (!info.isFile() || info.size > limit) throw new InputError(`input must be an ordinary file of at most ${limit} bytes: ${name}`);
    const buffer = Buffer.alloc(Math.min(limit + 1, info.size + 1));
    let used = 0;
    while (used < buffer.length) {
      const { bytesRead } = await handle.read(buffer, used, buffer.length - used, null);
      if (bytesRead === 0) break;
      used += bytesRead;
    }
    if (used > limit || used !== info.size || (await handle.stat()).size !== info.size)
      throw new InputError(`input changed while reading its bounded bytes: ${name}`);
    return buffer.subarray(0, used);
  } finally { await handle.close(); }
}

export function digest(bytes) { return createHash('sha256').update(bytes).digest('hex'); }

export async function repository(value) {
  await ordinaryAncestry(value);
  const root = await realpath(resolve(value));
  const text = (await fileBytes(root, 'Cargo.toml')).toString('utf8');
  let packageSection = false, found = false;
  for (const line of text.split(/\r?\n/)) {
    if (/^\s*\[/.test(line)) packageSection = /^\s*\[package\]\s*$/.test(line);
    if (packageSection && /^\s*name\s*=\s*"ruv-fann"\s*(?:#.*)?$/.test(line)) found = true;
  }
  if (!found) throw new InputError('--repo must be the actual root ruv-fann checkout');
  return root;
}

export async function externalOutput(value, repo) {
  const output = resolve(value);
  await ordinaryAncestry(output);
  if (inside(repo, output) || inside(output, repo) || inside(TOOL_ROOT, output))
    throw new InputError('output must be outside the checkout and tool directory');
  try { await lstat(output); throw new InputError('output already exists; receipts are never overwritten'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output, { mode: 0o700 });
  return output;
}

const REPO_INPUTS = [ 'README.md', 'Cargo.toml', 'Cargo.lock', 'package.json', 'package-lock.json',
  'ruv-swarm/Cargo.lock', 'neuro-divergent/Cargo.lock', 'cuda-wasm/Cargo.lock', 'opencv-rust/Cargo.lock',
  'config/rune-probe/Cargo.lock', 'ruv-swarm/npm/package-lock.json', 'cuda-wasm/package-lock.json',
  'ruv-swarm/npm/test-npm-install-v106/package-lock.json', 'ruv-swarm/npm/wasm/build-info.json',
  'ruv-swarm/npm/wasm/ruv_swarm_wasm.js', 'ruv-swarm/npm/wasm/ruv_swarm_wasm_bg.wasm',
  'ruv-swarm/npm/wasm/ruv_swarm_simd.wasm', 'ruv-swarm/npm/scripts/wasm-provenance.js' ];
const TOOL_INPUTS = ['package.json', 'package-lock.json'];
const DEPENDENCY_INPUTS = [ 'node_modules/metaharness/package.json', 'node_modules/metaharness/dist/bin.js',
  ...['index', 'repo-scorecard', 'analyze-repo', 'genome-scorers', 'constraints']
    .map(name => `node_modules/metaharness/dist/${name}.js`), 'node_modules/@metaharness/darwin/package.json',
  ...['index', 'numeric-evolve', 'numeric-mutator', 'numeric-archive', 'evolve', 'pareto']
    .map(name => `node_modules/@metaharness/darwin/dist/${name}.js`) ];

/** Hash exact benchmark/tool inputs and all ten existing lockfiles, never agent state. */
export async function inputSnapshot(repo, { dependencies = true } = {}) {
  const result = {};
  for (const name of REPO_INPUTS) result['repo/' + name] = digest(await fileBytes(repo, name));
  for (const name of [...TOOL_INPUTS, ...(dependencies ? DEPENDENCY_INPUTS : [])])
    result['tool/' + name] = digest(await fileBytes(TOOL_ROOT, name));
  const entries = await readdir(join(TOOL_ROOT, 'src'), { withFileTypes: true });
  if (entries.length > 64) throw new InputError('tool source directory exceeds its entry budget');
  for (const item of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    if (item.isSymbolicLink() || (!item.isFile() && !item.isDirectory())) throw new InputError('special/symlink tool source');
    if (item.isFile() && item.name.endsWith('.mjs')) {
      const name = 'src/' + item.name;
      result['tool/' + name] = digest(await fileBytes(TOOL_ROOT, name));
    }
  }
  return result;
}

export async function packagePins() {
  const lock = JSON.parse((await fileBytes(TOOL_ROOT, 'package-lock.json')).toString('utf8'));
  const result = { repository: PINS.repository, referenceCommit: PINS.referenceCommit };
  for (const [key, name] of [['metaharness', 'metaharness'], ['darwin', '@metaharness/darwin']]) {
    const item = lock.packages?.['node_modules/' + name];
    const installed = JSON.parse((await fileBytes(TOOL_ROOT, `node_modules/${name}/package.json`)).toString('utf8'));
    if (item?.version !== PINS[key] || installed.version !== PINS[key] || !/^sha512-[A-Za-z0-9+/]+=*$/.test(item.integrity))
      throw new InputError('installed package or lock differs from the exact version/integrity pin');
    result[key] = { version: PINS[key], npmIntegrity: item.integrity };
  }
  return result;
}

/** Upstream score lazily embeds when this optional package can resolve. Never load it. */
export function assertNoScoreModel(analyzer = join(TOOL_ROOT, 'node_modules/metaharness/dist/analyze-repo.js')) {
  try {
    const resolved = createRequire(analyzer).resolve('@ruvector/ruvllm');
    throw new InputError(`score refuses a resolvable optional embedding model: ${resolved}`);
  } catch (error) {
    if (error.code !== 'MODULE_NOT_FOUND') throw error;
  }
}

/** Env is an allowlist; provider keys, NODE_OPTIONS/loaders and host identities are omitted. */
export function childEnvironment() { return { LANG: 'C.UTF-8', TZ: 'UTC' }; }

/** One process-group deadline and combined-output cap, including synchronous WASM. */
export function boundedProcess(argv, { cwd, timeoutMs, maxBytes = 262144, env = childEnvironment() }) {
  if (process.platform === 'win32') return Promise.reject(new InputError('bounded process groups require POSIX'));
  return new Promise((resolvePromise, reject) => {
    const child = spawn(process.execPath, argv, { cwd, env, shell: false, detached: true,
      stdio: ['ignore', 'pipe', 'pipe'] });
    const chunks = [], errors = []; let count = 0, reason = null;
    const kill = () => {
      if (!Number.isInteger(child.pid)) return;
      try { process.kill(-child.pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
    };
    const timer = setTimeout(() => { reason = 'timeout'; kill(); }, timeoutMs);
    const capture = list => bytes => {
      count += bytes.length;
      if (count > maxBytes) { reason = 'output_limit'; kill(); return; }
      list.push(bytes);
    };
    child.stdout.on('data', capture(chunks)); child.stderr.on('data', capture(errors));
    child.on('error', error => { clearTimeout(timer); reject(new ExperimentError(error.message)); });
    child.on('close', (code, signal) => {
      clearTimeout(timer); kill();
      const result = { exitCode: code, signal, stdout: Buffer.concat(chunks).toString('utf8'),
        stderr: Buffer.concat(errors).toString('utf8'), reason };
      if (reason || code !== 0) reject(Object.assign(new ExperimentError(`child failed: ${reason ?? code}`), { result }));
      else resolvePromise(result);
    });
  });
}

/** Reject non-JSON objects and non-finite numbers before evidence serialization. */
export function finiteJson(value, depth = 0, counter = { nodes: 0 }) {
  if (++counter.nodes > 100000 || depth > 20) throw new ExperimentError('evidence exceeds its structural budget');
  if (typeof value === 'number' && !Number.isFinite(value)) throw new ExperimentError('non-finite evidence');
  if (value === null || ['string', 'number', 'boolean'].includes(typeof value)) return;
  if (typeof value !== 'object' || (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype))
    throw new ExperimentError('evidence must contain only JSON values');
  for (const child of Object.values(value)) finiteJson(child, depth + 1, counter);
}
