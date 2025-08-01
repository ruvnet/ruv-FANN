#!/usr/bin/env node

/**
 * Test SIMD fix by forcing new instance
 */

import { RuvSwarm } from './ruv-swarm/npm/src/index-enhanced.js';

console.log('🔧 Testing SIMD Fix with Fresh Instance...\n');

// Clear global cache to force new initialization
if (global._ruvSwarmInstance) {
  console.log('Clearing cached instance...');
  delete global._ruvSwarmInstance;
  global._ruvSwarmInitialized = 0;
}

// Test static method first
console.log('1. Static SIMD Detection:');
const simdSupported = RuvSwarm.detectSIMDSupport();
console.log('   SIMD Support (static):', simdSupported);

// Initialize new instance
console.log('\n2. Fresh Instance Initialization:');
try {
  const instance = await RuvSwarm.initialize({
    useSIMD: true,
    debug: false,
  });

  console.log('   SIMD Support (instance):', instance.features.simd_support);
  console.log('   All Features:', instance.features);
} catch (error) {
  console.log('   Error:', error.message);
}

console.log('\n✅ Test complete!');
