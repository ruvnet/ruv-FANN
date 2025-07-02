/**
 * Debug test for WASM bindings
 * Checks for WebAssembly.instantiate errors
 */

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function testWasmDirectly() {
  console.log('🔍 Testing WASM bindings directly...\n');

  try {
    // Path to WASM file
    const wasmPath = path.join(__dirname, '..', 'wasm', 'ruv_swarm_wasm_bg.wasm');

    // Check if file exists
    try {
      await fs.access(wasmPath);
      console.log('✅ WASM file exists at:', wasmPath);
    } catch (e) {
      console.error('❌ WASM file not found:', wasmPath);
      return;
    }

    // Read WASM file
    const wasmBuffer = await fs.readFile(wasmPath);
    console.log('✅ WASM file loaded, size:', wasmBuffer.byteLength, 'bytes');

    // Create minimal imports to see what's missing
    const imports = {
      wbg: {},
      env: {
        memory: new WebAssembly.Memory({ initial: 256, maximum: 4096 }),
      },
    };

    // Try to instantiate
    console.log('\n🔧 Attempting WebAssembly.instantiate...\n');

    try {
      const { instance, module } = await WebAssembly.instantiate(wasmBuffer, imports);
      console.log('✅ WASM instantiation successful!');
      console.log('Exports:', Object.keys(instance.exports));
    } catch (instantiateError) {
      console.error('❌ WebAssembly.instantiate failed:', instantiateError.message);

      // Parse the error to find missing imports
      if (instantiateError.message.includes('Import #')) {
        console.log('\n🔍 Missing imports detected:');
        const importMatch = instantiateError.message.match(/Import #(\d+) module="([^"]+)" function="([^"]+)"/);
        if (importMatch) {
          console.log(`  - Module: ${importMatch[2]}`);
          console.log(`  - Function: ${importMatch[3]}`);
          console.log(`  - Import index: ${importMatch[1]}`);
        }
      }

      // List all imports the WASM file expects
      console.log('\n📋 Analyzing WASM imports...');
      const module = await WebAssembly.compile(wasmBuffer);
      const importsList = WebAssembly.Module.imports(module);

      console.log(`\nTotal imports needed: ${importsList.length}`);
      console.log('\nImports by module:');

      const importsByModule = {};
      importsList.forEach(imp => {
        if (!importsByModule[imp.module]) {
          importsByModule[imp.module] = [];
        }
        importsByModule[imp.module].push({
          name: imp.name,
          kind: imp.kind,
        });
      });

      for (const [moduleName, moduleImports] of Object.entries(importsByModule)) {
        console.log(`\n  ${moduleName}: (${moduleImports.length} imports)`);
        moduleImports.slice(0, 10).forEach(imp => {
          console.log(`    - ${imp.name} (${imp.kind})`);
        });
        if (moduleImports.length > 10) {
          console.log(`    ... and ${moduleImports.length - 10} more`);
        }
      }
    }

  } catch (error) {
    console.error('❌ Test failed:', error);
  }
}

// Run the test
testWasmDirectly();