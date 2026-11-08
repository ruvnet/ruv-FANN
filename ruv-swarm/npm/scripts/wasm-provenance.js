import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';

const digest = data => createHash('sha256').update(data).digest('hex');
const artifactName = file => typeof file === 'string' && /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(file);

export function canonicalSourcePaths(files, separator = path.sep) {
  // Normalize before both deduplication and ordering: fixed recipe paths and
  // filesystem traversal can use different separators on Windows. On POSIX,
  // backslashes are literal filename characters and must remain untouched.
  return [...new Set(files.map(file => file.split(separator).join('/')))].sort();
}

// Read the filesystem rather than git's index: newly added source modules are
// build inputs too. Generated outputs and runtime databases are excluded.
export async function rustSourceDigest(repository) {
  const files = [];
  const visit = async(directory, include) => {
    for (const item of await fs.readdir(path.join(repository, directory), { withFileTypes: true })) {
      if (['target', 'node_modules', '.git'].includes(item.name)) {
        continue;
      }
      const relative = path.join(directory, item.name);
      if (item.isSymbolicLink()) {
        throw new Error(`Build source contains an unsupported symbolic link: ${relative}`);
      }
      if (item.isDirectory()) {
        await visit(relative, include);
      } else if (item.isFile() && include(item.name)) {
        files.push(relative);
      }
    }
  };
  for (const file of ['Cargo.toml', 'Cargo.lock', 'ruv-swarm/Cargo.toml', 'ruv-swarm/Cargo.lock',
    'package.json', 'package-lock.json', 'ruv-swarm/npm/package.json', 'ruv-swarm/npm/package-lock.json',
    'ruv-swarm/npm/scripts/build.js', 'ruv-swarm/npm/scripts/wasm-provenance.js',
    'ruv-swarm/npm/scripts/validate-package.js', 'ruv-swarm/npm/wasm/wasm-bindings-loader.mjs']) {
    await fs.access(path.join(repository, file));
    files.push(file);
  }
  for (const directory of ['src', 'ruv-swarm/crates', 'ruv-swarm/ml-training', 'ruv-swarm/benchmarking']) {
    await visit(directory, name => /\.(rs|toml|wgsl)$/.test(name) || name === 'Cargo.lock' || /^LICENSE(?:[-.]|$)/i.test(name));
  }
  for (const directory of ['ruv-swarm/npm/src', 'ruv-swarm/npm/bin']) {
    await visit(directory, name => /\.(m?js|cjs)$/.test(name) || (name.endsWith('.d.ts') && name !== 'wasm-types.d.ts'));
  }
  for (const file of ['build.rs', 'rust-toolchain', 'rust-toolchain.toml', '.cargo/config', '.cargo/config.toml',
    'ruv-swarm/build.rs', 'ruv-swarm/rust-toolchain', 'ruv-swarm/rust-toolchain.toml',
    'ruv-swarm/.cargo/config', 'ruv-swarm/.cargo/config.toml',
    'LICENSE', 'LICENSE-MIT', 'LICENSE-APACHE', 'ruv-swarm/LICENSE-MIT', 'ruv-swarm/LICENSE-APACHE',
    'ruv-swarm/npm/LICENSE-MIT', 'ruv-swarm/npm/LICENSE-APACHE']) {
    try {
      await fs.access(path.join(repository, file));
      files.push(file);
    } catch (error) {
      if (error.code !== 'ENOENT') {
        throw error;
      }
    }
  }
  const hash = createHash('sha256');
  for (const relative of canonicalSourcePaths(files)) {
    const data = await fs.readFile(path.join(repository, relative));
    hash.update(`${relative}\0${ data.length }\0`);
    hash.update(data);
  }
  return hash.digest('hex');
}

function validateReceipt(receipt) {
  if (receipt.schemaVersion !== 2 || !/^[a-f0-9]{64}$/.test(receipt.sourceSha256) ||
      !receipt.toolchain?.rustc || !receipt.toolchain?.wasmPack ||
      !Array.isArray(receipt.configurations) || receipt.configurations.length === 0 ||
      !receipt.artifacts || typeof receipt.artifacts !== 'object') {
    throw new Error('Invalid WASM build receipt');
  }
  const compiled = new Set();
  for (const config of receipt.configurations) {
    if (config.target !== 'wasm32-unknown-unknown' || config.bindingTarget !== 'web' ||
        config.profile !== 'release' || typeof config.defaultFeatures !== 'boolean' ||
        !Array.isArray(config.features) || !config.features.every(feature => typeof feature === 'string') ||
        typeof config.rustFlags !== 'string' || !Array.isArray(config.artifacts) || config.artifacts.length === 0) {
      throw new Error('Invalid WASM build configuration');
    }
    for (const file of config.artifacts) {
      if (!artifactName(file) || !file.endsWith('.wasm') || compiled.has(file) ||
          (file.includes('simd') && !config.rustFlags.includes('+simd128'))) {
        throw new Error('Invalid WASM artifact configuration');
      }
      compiled.add(file);
    }
  }
  for (const [file, hash] of Object.entries(receipt.artifacts)) {
    if (!artifactName(file) || !/^[a-f0-9]{64}$/.test(hash) ||
        (file.endsWith('.wasm') && !compiled.has(file))) {
      throw new Error('Invalid WASM artifact evidence');
    }
  }
  for (const file of compiled) {
    if (!receipt.artifacts[file]) {
      throw new Error('Missing compiled WASM artifact evidence');
    }
  }
}

export async function recordWasmBuild(npmRoot, { sourceSha256, artifacts, toolchain, configurations }) {
  const current = await rustSourceDigest(path.resolve(npmRoot, '../..'));
  if (current !== sourceSha256) {
    throw new Error('Build sources changed while WASM was building; rebuild');
  }
  const hashes = {};
  // Callers list only artifacts compiled during this build. Existing binaries
  // never gain a receipt merely because they happen to be in the output folder.
  for (const file of artifacts) {
    if (!artifactName(file) || file === 'build-info.json') {
      throw new Error('Invalid WASM artifact name');
    }
    hashes[file] = digest(await fs.readFile(path.join(npmRoot, 'wasm', file)));
  }
  const receipt = { schemaVersion: 2, sourceSha256, toolchain, configurations, artifacts: hashes };
  validateReceipt(receipt);
  await fs.writeFile(path.join(npmRoot, 'wasm', 'build-info.json'), `${JSON.stringify(receipt, null, 2) }\n`);
}

export async function verifyWasmBuild(npmRoot, repository = path.resolve(npmRoot, '../..'), artifacts) {
  let receipt;
  try {
    receipt = JSON.parse(await fs.readFile(path.join(npmRoot, 'wasm', 'build-info.json'), 'utf8'));
  } catch {
    throw new Error('Missing WASM build receipt; rebuild artifacts from the reviewed Rust sources before publishing');
  }
  validateReceipt(receipt);
  // Consumers verify the packaged bytes without needing the source checkout.
  // prepack passes a repository and additionally rejects stale build inputs.
  if (repository !== null && receipt.sourceSha256 !== await rustSourceDigest(repository)) {
    throw new Error('WASM build receipt does not match the current build sources and lockfiles');
  }
  const requested = artifacts ?? (await fs.readdir(path.join(npmRoot, 'wasm'))).filter(file => file.endsWith('.wasm'));
  for (const file of new Set([...requested, ...Object.keys(receipt.artifacts)])) {
    if (!artifactName(file)) {
      throw new Error('Invalid WASM artifact name');
    }
    const actual = digest(await fs.readFile(path.join(npmRoot, 'wasm', file)));
    if (receipt.artifacts?.[file] !== actual) {
      throw new Error(`WASM artifact has no matching build evidence: ${file}`);
    }
  }
}
