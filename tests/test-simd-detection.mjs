#!/usr/bin/env node

/**
 * Test SIMD detection in ruv-swarm
 */

import { RuvSwarm } from './ruv-swarm/npm/src/index-enhanced.js';

console.log('🔍 Testing SIMD Detection...\n');

// Test the static method directly
console.log('1. Testing RuvSwarm.detectSIMDSupport() directly:');
try {
  const simdSupported = RuvSwarm.detectSIMDSupport();
  console.log('   SIMD Support (static method):', simdSupported);
} catch (error) {
  console.log('   Error in static method:', error.message);
}

// Test WebAssembly.validate directly
console.log('\n2. Testing WebAssembly.validate directly:');
try {
  const simdTestBytes = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 7, 8, 1, 4, 116, 101, 115, 116, 0, 0, 10, 15, 1, 13, 0, 65, 0, 253, 15, 253, 98, 11,
  ]);
  const isValid = WebAssembly.validate(simdTestBytes);
  console.log('   WebAssembly.validate result:', isValid);
} catch (error) {
  console.log('   Error in WebAssembly.validate:', error.message);
}

// Test platform capabilities
console.log('\n3. Testing platform capabilities:');
console.log('   Node.js version:', process.version);
console.log('   Platform:', process.platform);
console.log('   Architecture:', process.arch);

// Test if WebAssembly is available
console.log('\n4. Testing WebAssembly availability:');
console.log('   WebAssembly available:', typeof WebAssembly !== 'undefined');
console.log('   WebAssembly.instantiate available:', typeof WebAssembly.instantiate === 'function');
console.log('   WebAssembly.validate available:', typeof WebAssembly.validate === 'function');

// Test SIMD with different bytes
console.log('\n5. Testing simplified SIMD detection:');
try {
  // This is a minimal SIMD test module that should pass
  const minimalSIMD = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, // magic
    0x01, 0x00, 0x00, 0x00, // version
    // Type section with SIMD type
    0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
  ]);
  const minimal = WebAssembly.validate(minimalSIMD);
  console.log('   Minimal WASM valid:', minimal);
} catch (error) {
  console.log('   Error in minimal test:', error.message);
}

// Initialize full ruv-swarm and check features
console.log('\n6. Testing full ruv-swarm initialization:');
try {
  const instance = await RuvSwarm.initialize({
    useSIMD: true,
    debug: true,
  });
  console.log('   Initialized features:', instance.features);
} catch (error) {
  console.log('   Error during initialization:', error.message);
}
