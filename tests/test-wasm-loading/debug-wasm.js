import { fileURLToPath } from 'url';
import path, { dirname, join } from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Simulate the wasm-loader.js environment
const baseDir = '/home/codespace/nvm/current/lib/node_modules/ruv-swarm/src';

console.log('Base directory:', baseDir);
console.log('\nChecking path candidates:\n');

const candidates = [
  {
    description: 'Local development (relative to src/)',
    wasmDir: path.join(baseDir, '..', 'wasm'),
  },
  {
    description: 'NPM package installation (adjacent to src/)',
    wasmDir: path.join(baseDir, '..', '..', 'wasm'),
  },
];

for (const candidate of candidates) {
  console.log(`${candidate.description}:`);
  console.log(`  Path: ${candidate.wasmDir}`);
  try {
    fs.accessSync(candidate.wasmDir);
    const files = fs.readdirSync(candidate.wasmDir);
    console.log(`  ✅ Exists! Files: ${files.filter(f => f.endsWith('.wasm')).join(', ')}`);
  } catch (error) {
    console.log(`  ❌ Not found: ${error.message}`);
  }
  console.log();
}
