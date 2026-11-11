import { open, mkdir, lstat } from 'node:fs/promises';
import { constants } from 'node:fs';
import { resolve, dirname, basename, parse, relative, join, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CONTRACTS, CONTRACT_LIMITS, validateCatalog, validateModuleManifest } from 'ruvnet-constellation-contracts';

/** Read only an explicitly selected local catalog; no registry or network access. */
export async function readCatalog(path = new URL('../catalog.json', import.meta.url)) {
  if (path instanceof URL) path = fileURLToPath(path);
  if (typeof path !== 'string' || !path.length || path.length > 4096 || /[\x00-\x1f\x7f]/.test(path)) throw new TypeError('catalog path must be bounded local text');
  const handle = await open(path, constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    if (!(await handle.stat()).isFile()) throw new TypeError('catalog must be a regular file');
    const bytes = Buffer.alloc(CONTRACT_LIMITS.maxBytes + 1);
    let length = 0;
    while (length < bytes.length) {
      const { bytesRead } = await handle.read(bytes, length, bytes.length - length, null);
      if (!bytesRead) break;
      length += bytesRead;
    }
    if (length > CONTRACT_LIMITS.maxBytes) throw new RangeError('catalog exceeds byte limit');
    return validateCatalog(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(0, length))));
  } finally { await handle.close(); }
}

/** Planning is data only. Candidate sources never become purported public install specs. */
export function planProfile(input, id) {
  const catalog = validateCatalog(input);
  if (typeof id !== 'string' || !/^[a-z][a-z0-9-]{0,47}$/.test(id)) throw new TypeError('profile must be a bounded identifier');
  const profile = catalog.profiles.find(profile => profile.id === id);
  if (!profile) throw new Error(`unknown constellation profile: ${id}`);
  const entries = new Map(catalog.packages.map(entry => [entry.name, entry]));
  const packages = profile.packages.map(name => entries.get(name));
  return { schema: 'ruvnet.constellation-plan/1', profile: id, description: profile.description,
    packages, registrySpecs: packages.filter(entry => entry.status === 'published').map(entry => `${entry.name}@${entry.version}`),
    localCandidates: packages.filter(entry => entry.status === 'candidate').map(entry => ({name:entry.name, directory:entry.directory, repository:entry.repository})),
    publicInstallAvailable: packages.every(entry => entry.status === 'published'), automaticInstall: false, telemetry: 'disabled' };
}
function argumentsRecord(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError('scaffold options must be a plain record');
  const result = {};
  for (const key of Reflect.ownKeys(value)) {
    const property = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !['directory','name','description','integrations'].includes(key) || !property || !Object.hasOwn(property,'value'))
      throw new TypeError('unknown scaffold option or accessor');
    result[key] = property.value;
  }
  if (!Object.hasOwn(result, 'directory') || !Object.hasOwn(result, 'name')) throw new TypeError('scaffold requires directory and name');
  return result;
}
function markdown(value) { return value.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replace(/[\\`*_{}\[\]()#+.!|]/g, '\\$&'); }
function directoryAlias(handle) { return `/proc/self/fd/${handle.fd}`; }
async function closeDirectories(resources) {
  const results = await Promise.allSettled([...resources].reverse().map(resource => resource.handle.close()));
  const failed = results.find(result => result.status === 'rejected');
  if (failed) throw failed.reason;
}
async function verifyDirectories(resources) {
  for (const resource of resources) {
    const held = await resource.handle.stat();
    const entry = await lstat(resource.parent ? `${directoryAlias(resource.parent)}/${resource.name}` : resource.root);
    if (!held.isDirectory() || entry.isSymbolicLink() || !entry.isDirectory() || entry.dev !== held.dev || entry.ino !== held.ino)
      throw new Error('scaffold directory relationship changed during creation');
  }
}
async function openParentDirectories(path) {
  if (process.platform !== 'linux' || typeof constants.O_DIRECTORY !== 'number' || typeof constants.O_NOFOLLOW !== 'number')
    throw new Error('scaffolding requires Linux directory descriptors and /proc/self/fd; catalog and planning remain portable');
  const root = parse(path).root, parts = relative(root, path).split(sep).filter(Boolean), resources = [];
  if (parts.length > 64) throw new RangeError('scaffold parent path exceeds directory depth limit');
  const flags = constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW;
  try {
    // Each next component resolves through a held descriptor. No unverified ancestor
    // path is reused, and O_NOFOLLOW applies to the newly opened component.
    let current = await open(root, flags);
    resources.push({ handle: current, root });
    // Verify the kernel-provided descriptor namespace before creating any output.
    const proc = await lstat('/proc/self/fd');
    if (!proc.isDirectory()) throw new Error('Linux /proc/self/fd directory is unavailable');
    for (const name of parts) {
      const parent = current;
      try { current = await open(`${directoryAlias(parent)}/${name}`, flags); }
      catch (error) { throw new TypeError('scaffold parent path must consist of existing real directories', {cause:error}); }
      resources.push({ handle: current, parent, name });
    }
    await verifyDirectories(resources);
    return resources;
  } catch (error) { await closeDirectories(resources); throw error; }
}
async function createDirectory(resources, parent, name) {
  await verifyDirectories(resources);
  const path = `${directoryAlias(parent)}/${name}`;
  // mkdir refuses an existing target, including dangling symlinks. Opening the
  // result refuses substitution by a symlink before its descriptor is acquired.
  await mkdir(path, {mode:0o700});
  const handle = await open(path, constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW);
  resources.push({handle,parent,name});
  await verifyDirectories(resources);
  return handle;
}
async function writeNew(directoryHandle, name, content) {
  // Only fixed leaf names reach this function. O_EXCL prevents existing leaf
  // replacement; the held parent descriptor prevents intermediate symlink races.
  const path = `${directoryAlias(directoryHandle)}/${name}`;
  const handle = await open(path, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
  try { await handle.writeFile(content); } finally { await handle.close(); }
}
const adapterSource = `// Bounded JSON adapter template. Define and independently test domain mapping before release.
export function adaptRecord(input) {
  let nodes = 0, text = 0;
  const ancestors = new Set();
  function copy(value, depth) {
    if (++nodes > 8192 || depth > 8) throw new RangeError('adapter structure exceeds limits');
    if (value === null || typeof value === 'boolean') return value;
    if (typeof value === 'number' && Number.isFinite(value)) return value === 0 ? 0 : value;
    if (typeof value === 'string') { text += value.length; if (text > 65536) throw new RangeError('adapter text exceeds limit'); return value; }
    if (!value || typeof value !== 'object' || ancestors.has(value)) throw new TypeError('adapter accepts finite acyclic JSON data only');
    const array = Array.isArray(value);
    if (!(array ? Object.getPrototypeOf(value) === Array.prototype : [Object.prototype, null].includes(Object.getPrototypeOf(value)))) throw new TypeError('adapter requires plain data');
    if (array && value.length > 8192) throw new RangeError('adapter array exceeds limit');
    const result = array ? [] : Object.create(null); ancestors.add(value);
    for (const key of Reflect.ownKeys(value)) {
      if (array && key === 'length') continue;
      if (typeof key !== 'string' || ['__proto__','constructor','prototype'].includes(key) || (array && (!/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= value.length))) throw new TypeError('adapter contains unsafe properties');
      text += key.length; if (text > 65536) throw new RangeError('adapter text exceeds limit');
      const property = Object.getOwnPropertyDescriptor(value, key);
      if (!property || !Object.hasOwn(property, 'value')) throw new TypeError('adapter accessors are forbidden');
      result[key] = copy(property.value, depth + 1);
    }
    ancestors.delete(value);
    if (array && (result.length !== value.length || Object.keys(result).length !== value.length)) throw new TypeError('adapter arrays must be dense');
    return result;
  }
  const result = copy(input, 0);
  if (new TextEncoder().encode(JSON.stringify(result)).length > 65536) throw new RangeError('adapter bytes exceed limit');
  return result;
}
`;
const templateTests = `import test from 'node:test';
import assert from 'node:assert/strict';
import { adaptRecord } from '../src/index.js';
test('copies JSON without retaining caller objects', () => {
  const input = {format:'example', values:[1,2,null]}; const output = adaptRecord(input);
  input.values[0] = 10; assert.equal(output.values[0],1); assert.equal(output.format,'example');
});
test('rejects executable, nonfinite and unsafe inputs without invoking getters', () => {
  let calls=0; const input={get value(){calls++; return 1;}};
  assert.throws(()=>adaptRecord(input));assert.equal(calls,0);
  for(const value of [Infinity,()=>{},new Array(2),JSON.parse('{"__proto__":{"polluted":true}}')]) assert.throws(()=>adaptRecord(value));
  assert.equal({}.polluted,undefined);
});
test('bounds copied data and refuses cycles', () => {
  const cycle={};cycle.self=cycle;
  assert.throws(()=>adaptRecord(cycle));assert.throws(()=>adaptRecord('x'.repeat(65537)));
});
`;
const benchmarkSource = `import {performance} from 'node:perf_hooks';
import assert from 'node:assert/strict';
import {adaptRecord} from './src/index.js';
const input={format:'example',values:[1,2,3]},iterations=1000,start=performance.now();
for(let index=0;index<iterations;index++)assert.equal(adaptRecord(input).values[2],3);
console.log(JSON.stringify({schema:'ruvnet.satellite-benchmark/1',iterations,elapsedMs:performance.now()-start,note:'Local JSON-copy observation, not a domain accuracy or WASM benchmark.'}));
`;

/** Create a reviewable local candidate. Never overwrites a target or installs dependencies. */
export async function scaffoldSatellite(options) {
  // Capture all configuration before asynchronous filesystem operations.
  const config = argumentsRecord(options), directory = config.directory;
  if (typeof directory !== 'string' || !directory.length || directory.length > 4096 || /[\x00-\x1f\x7f]/.test(directory) ||
      directory.split(/[\\/]/).some(part => part === '..') || (sep === '/' && directory.includes('\\')))
    throw new TypeError('directory must be bounded text without traversal, control characters or ambiguous separators');
  const module = validateModuleManifest({schema:CONTRACTS.module,name:config.name,version:'0.1.0',role:'satellite',contracts:[CONTRACTS.module],integrations:config.integrations === undefined ? [] : config.integrations});
  const description = config.description === undefined ? 'Bounded JSON adapter candidate; replace mapping with independently tested domain logic.' : config.description;
  if (typeof description !== 'string' || !description.length || description.length > 2000 || /[\x00-\x1f\x7f]/.test(description)) throw new TypeError('description must be bounded plain text');
  const target = resolve(directory), parent = dirname(target);
  if (target === parent) throw new TypeError('scaffold target cannot be a filesystem root');
  const resources = await openParentDirectories(parent);
  try {
    const parentHandle = resources.at(-1).handle;
    const rootHandle = await createDirectory(resources, parentHandle, basename(target));
    const sourceHandle = await createDirectory(resources, rootHandle, 'src');
    const testHandle = await createDirectory(resources, rootHandle, 'test');
  const manifest = {name:module.name,version:module.version,description,type:'module',main:'./src/index.js',types:'./src/index.d.ts',
    exports:{'.':{types:'./src/index.d.ts',import:'./src/index.js'}},files:['src','constellation.json','README.md','LICENSE'],
    engines:{node:'>=20.9.0'},scripts:{test:'node --test test/*.test.mjs',check:'npm test',benchmark:'node benchmark.mjs','pack:check':'npm pack --dry-run --ignore-scripts'},license:'MIT',publishConfig:{access:'public'}};
  const integrationLinks = module.integrations.length ? module.integrations.map(name => `- [${name}](https://www.npmjs.com/package/${name}) — optional integration metadata; no dependency is installed.`).join('\n') : 'No integration is selected. Add only integrations the module actually uses.';
  const readme = `# ${markdown(module.name)}\n\n${markdown(description)}\n\nThis is a local candidate with a bounded identity JSON adapter template. It has not been published or proven useful for a domain. Replace the mapping and add independent domain cases before release.\n\n\`adaptRecord(value)\` owns finite acyclic JSON, rejects accessors/prototype keys, and bounds depth, elements and bytes. It executes no callbacks and performs no network requests. These checks are not a WSP or domain-accuracy benchmark.\n\nRun \`npm test\`, \`npm run benchmark\` and \`npm run pack:check\`. No install or provider is required for the template.\n\n## Optional constellation integrations\n\n${integrationLinks}\n\nDiscover [MetaHarness](https://www.npmjs.com/package/metaharness), [Ruflo](https://www.npmjs.com/package/@claude-flow/cli), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector) and [WSP](https://www.npmjs.com/package/ruv-fann-wsp) independently. The template does not force any nucleus dependency.\n\n## Publication preparation\n\nReview the generated files, package name/namespace ownership, license and quality evidence. Configure this package's npm trusted publisher for the exact owner, repository, workflow and optional environment; use a protected exact-source/archive release with supported Node/npm and OIDC provenance. The scaffold neither configures a publisher nor publishes. See [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/).\n`;
  const files = new Map([
    ['package.json',JSON.stringify(manifest,null,2)+'\n'],['constellation.json',JSON.stringify(module,null,2)+'\n'],
    ['README.md',readme],['src/index.js',adapterSource],['src/index.d.ts','/** Owns bounded finite JSON data. Replace this template with domain-specific types. */\nexport function adaptRecord(input: unknown): unknown;\n'],
    ['test/adapter.test.mjs',templateTests],['benchmark.mjs',benchmarkSource],
    ['LICENSE',`MIT License\n\nCopyright (c) ${new Date().getUTCFullYear()} contributors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n`]
  ]);
  for (const [path,content] of files) {
    await verifyDirectories(resources);
    const parts = path.split('/');
    const directoryHandle = parts.length === 1 ? rootHandle : parts[0] === 'src' ? sourceHandle : testHandle;
    await writeNew(directoryHandle, parts.at(-1), content);
  }
  await verifyDirectories(resources);
  return {schema:'ruvnet.constellation-scaffold/1',directory:target,name:module.name,status:'candidate',files:[...files.keys()],automaticInstall:false,telemetry:'disabled',published:false};
  } finally { await closeDirectories(resources); }
}
