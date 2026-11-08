#!/usr/bin/env node
import { verifyWasmBuild } from './wasm-provenance.js';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pkg = JSON.parse(await fs.readFile(path.join(root, 'package.json'), 'utf8'));
for (const item of pkg.files) {
  const resolved = path.resolve(root, item);
  const relative = path.relative(root, resolved);
  if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error(`Published file escapes the package: ${item}`);
  }
  await fs.stat(resolved);
}
for (const item of [pkg.main, pkg.types, ...Object.values(pkg.bin)]) {
  if (!item) {
    throw new Error('Missing package entry point');
  }
  const stat = await fs.stat(path.join(root, item));
  if (!stat.isFile()) {
    throw new Error(`Entry point is not a file: ${item}`);
  }
}
await import(pathToFileURL(path.join(root, pkg.main)).href);
const artifacts = pkg.files.filter(item => item.startsWith('wasm/') && item !== 'wasm/build-info.json').map(item => path.basename(item));
if (!artifacts.some(item => item.endsWith('.wasm'))) {
  throw new Error('No explicit WASM artifacts declared for publication');
}
for (const item of artifacts) {
  if (item.endsWith('.wasm')) {
    const bytes = await fs.readFile(path.join(root, 'wasm', item));
    if (!WebAssembly.validate(bytes)) {
      throw new Error(`Invalid WASM: ${item}`);
    }
  }
}
const artifactOnly = process.argv.includes('--artifact-only');
await verifyWasmBuild(root, artifactOnly ? null : undefined, artifacts);
console.error(artifactOnly ? 'Entry points and packaged WASM build artifacts validate.' :
  'Entry points, WASM artifacts, and current source build receipts validate.');
