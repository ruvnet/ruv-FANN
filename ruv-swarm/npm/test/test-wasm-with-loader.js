/**
 * Test WASM with actual loader
 */

import loader from '../wasm/wasm-bindings-loader.mjs';

async function testWithLoader() {
  console.log('🔍 Testing WASM with bindings loader...\n');

  try {
    await loader.initialize();
    console.log('✅ WASM loader initialized successfully!');

    // Test exports
    console.log('\nAvailable exports:');
    const exports = Object.keys(loader).filter(key => typeof loader[key] === 'function');
    exports.forEach(exp => console.log(`  - ${exp}`));

    // Test memory
    console.log('\nMemory usage:', loader.getTotalMemoryUsage(), 'bytes');

  } catch (error) {
    console.error('❌ Loader initialization failed:', error.message);
    console.error('Stack:', error.stack);
  }
}

testWithLoader();