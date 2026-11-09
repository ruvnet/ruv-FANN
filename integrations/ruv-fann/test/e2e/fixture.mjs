/** Repository-only acceptance helpers. No runtime dependency on these paths. */
import { readFile, lstat } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { resolve, join, relative, sep, isAbsolute, dirname } from 'node:path';
import { verifyWasmBuild } from '../../../../ruv-swarm/npm/scripts/wasm-provenance.js';

export const repo = fileURLToPath(new URL('../../../../', import.meta.url));
export const packageRoot = resolve(repo, 'integrations/ruv-fann');
export const sha = bytes => createHash('sha256').update(bytes).digest('hex');
export const vectorBytes = () => readFile(join(packageRoot, 'node_modules/@ruvector/wasm/pkg/ruvector_wasm_bg.wasm'));

export async function externalEvidencePath(value) {
  const output = resolve(value), root = resolve(repo);
  const inside = (parent, child) => {
    const part = relative(parent, child);
    return part === '' || (!isAbsolute(part) && part !== '..' && !part.startsWith('..' + sep));
  };
  if (inside(root, output) || inside(output, root)) throw new Error('Evidence output must be external to the checkout');
  for (let current = output;; current = dirname(current)) {
    try { if ((await lstat(current)).isSymbolicLink()) throw new Error('Evidence output rejects symlink ancestry'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (dirname(current) === current) break;
  }
  return output;
}

export async function sourceSnapshot() {
  await verifyWasmBuild(join(repo, 'ruv-swarm/npm'), repo, ['ruv_swarm_wasm_bg.wasm', 'ruv_swarm_simd.wasm']);
  const files = ['ruv-swarm/npm/wasm/build-info.json', 'ruv-swarm/npm/wasm/ruv_swarm_wasm.js',
    'ruv-swarm/npm/wasm/ruv_swarm_wasm_bg.wasm', 'ruv-swarm/npm/wasm/ruv_swarm_simd.wasm',
    'ruv-swarm/crates/ruv-swarm-wasm/tests/fixtures/prepared-dense.json',
    'ruv-swarm/crates/ruv-swarm-wasm/tests/fixtures/prepared-sparse.json'];
  return Object.fromEntries(await Promise.all(files.map(async name => [name, sha(await readFile(join(repo, name)))])));
}

export async function loadPrepared(variant, name = 'dense') {
  const binaryName = variant === 'simd' ? 'ruv_swarm_simd.wasm' : 'ruv_swarm_wasm_bg.wasm';
  const binary = await readFile(join(repo, 'ruv-swarm/npm/wasm', binaryName));
  const url = pathToFileURL(join(repo, 'ruv-swarm/npm/wasm/ruv_swarm_wasm.js'));
  url.searchParams.set('integration', randomUUID());
  const mod = await import(url.href);
  await mod.default({ module_or_path: binary });
  const capabilities = JSON.parse(mod.get_features());
  if (capabilities.simd_support !== (variant === 'simd')) throw new Error('Incorrect FANN variant loaded');
  const bytes = await readFile(join(repo, `ruv-swarm/crates/ruv-swarm-wasm/tests/fixtures/prepared-${name}.json`));
  const fixture = JSON.parse(bytes);
  const json = JSON.stringify(fixture.network);
  return { prepared: mod.WasmPreparedNetwork.from_network_json(json), fixture,
    modelId: `native-${name}:${sha(Buffer.from(json))}`, modelJsonSha256: sha(Buffer.from(json)), binarySha256: sha(binary) };
}

export function workload(fixture, rows = 131) {
  return Float64Array.from({ length: rows * fixture.input.length }, (_, i) =>
    fixture.input[i % fixture.input.length] + ((Math.floor(i / fixture.input.length) * 17 + i % fixture.input.length * 11) % 67 - 33) / 16);
}
