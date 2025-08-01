import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs/promises';

async function testProperWasmLoading() {
  console.log('Testing proper WASM loading using wasm-bindgen...\n');

  try {
    // Import the wasm module correctly
    const wasmModulePath = '/home/codespace/nvm/current/lib/node_modules/ruv-swarm/wasm/ruv_swarm_wasm.js';
    console.log('1. Importing WASM module from:', wasmModulePath);

    const wasmModule = await import(wasmModulePath);
    console.log('✅ WASM module imported successfully');
    console.log('   Available exports:', Object.keys(wasmModule).slice(0, 10).join(', '), '...');

    // Initialize the WASM module
    console.log('\n2. Initializing WASM...');
    const wasmPath = path.join(path.dirname(wasmModulePath), 'ruv_swarm_wasm_bg.wasm');

    // Read the WASM file
    const wasmBuffer = await fs.readFile(wasmPath);
    console.log(`   WASM file size: ${wasmBuffer.length} bytes`);

    // Call the default export (which is __wbg_init)
    await wasmModule.default(wasmBuffer);
    console.log('✅ WASM initialized successfully!');

    // Test some functions
    console.log('\n3. Testing WASM functions...');

    if (wasmModule.get_version) {
      const version = wasmModule.get_version();
      console.log('   Version:', version);
    }

    if (wasmModule.get_features) {
      const features = wasmModule.get_features();
      console.log('   Features:', features);
    }

    if (wasmModule.detect_simd_capabilities) {
      const simd = wasmModule.detect_simd_capabilities();
      console.log('   SIMD capabilities:', simd);
    }

    if (wasmModule.create_neural_network) {
      console.log('\n4. Testing neural network creation...');
      try {
        const nn = wasmModule.create_neural_network(3, 'relu');
        console.log('   ✅ Neural network created:', nn);
      } catch (e) {
        console.log('   ❌ Neural network creation failed:', e.message);
      }
    }

  } catch (error) {
    console.error('\n❌ Error:', error.message);
    if (error.stack) {
      console.error('Stack:', error.stack.split('\n').slice(0, 5).join('\n'));
    }
  }
}

testProperWasmLoading().catch(console.error);
