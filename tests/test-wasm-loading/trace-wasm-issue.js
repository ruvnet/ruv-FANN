import { WasmModuleLoader } from '/home/codespace/nvm/current/lib/node_modules/ruv-swarm/src/wasm-loader.js';

async function traceWasmIssue() {
  console.log('=== Tracing WASM Loading Issue ===\n');

  const loader = new WasmModuleLoader();
  console.log('1. WasmModuleLoader created');
  console.log('   Base directory:', loader.baseDir);

  try {
    console.log('\n2. Initializing loader with progressive strategy...');
    await loader.initialize('progressive');

    console.log('\n3. Module status:');
    const status = loader.getModuleStatus();
    console.log(JSON.stringify(status, null, 2));

    console.log('\n4. Checking loaded modules:');
    for (const [name, module] of loader.modules.entries()) {
      console.log(`   - ${name}:`, {
        isPlaceholder: module.isPlaceholder || false,
        hasMemory: !!module.memory,
        exports: module.exports ? Object.keys(module.exports).slice(0, 5) : [],
      });
    }

  } catch (error) {
    console.error('\n❌ Error during initialization:', error.message);
    console.error('Stack:', error.stack);
  }
}

// Also check the actual file system
import fs from 'fs';
import path from 'path';

console.log('\n=== File System Check ===\n');
const baseDir = '/home/codespace/nvm/current/lib/node_modules/ruv-swarm/src';
const wasmDir = path.join(baseDir, '..', 'wasm');

console.log('Checking:', wasmDir);
try {
  const files = fs.readdirSync(wasmDir);
  console.log('Files found:', files.filter(f => f.endsWith('.wasm') || f.endsWith('.mjs')).join(', '));
} catch (error) {
  console.log('Error:', error.message);
}

console.log('\n=== Running Trace ===\n');
traceWasmIssue().catch(console.error);
