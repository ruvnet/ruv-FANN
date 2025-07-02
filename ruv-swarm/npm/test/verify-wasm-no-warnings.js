/**
 * Verify WASM loads without warnings
 */

import { RuvSwarm } from '../src/index-enhanced.js';

async function verifyNoWarnings() {
  console.log('🔍 Verifying WASM loads without warnings...\n');

  // Capture console warnings
  const originalWarn = console.warn;
  const warnings = [];
  console.warn = (...args) => {
    warnings.push(args.join(' '));
    originalWarn.apply(console, args);
  };

  try {
    // Initialize RuvSwarm
    const swarm = await RuvSwarm.initialize({
      loadingStrategy: 'progressive',
      enableNeuralNetworks: true,
      enableForecasting: true,
      useSIMD: true,
      debug: false, // Disable debug to see only warnings
    });

    // Create a swarm and spawn agents
    const testSwarm = await swarm.createSwarm({
      name: 'test-swarm',
      topology: 'mesh',
      maxAgents: 5,
    });

    // Spawn different agent types
    await testSwarm.spawn({ type: 'researcher' });
    await testSwarm.spawn({ type: 'coder' });
    await testSwarm.spawn({ type: 'analyst' });

    // Check features
    const features = swarm.features;

    // Restore console.warn
    console.warn = originalWarn;

    // Results
    console.log('\n📊 Results:');
    console.log('✅ WASM initialized successfully');
    console.log('✅ Features detected:', features);
    console.log(`✅ Swarm created with ${testSwarm.agents.size} agents`);

    if (warnings.length === 0) {
      console.log('\n🎉 SUCCESS: No WASM warnings detected!');
    } else {
      console.log(`\n⚠️  Found ${warnings.length} warnings:`);
      warnings.forEach((w, i) => console.log(`  ${i + 1}. ${w}`));
      process.exit(1);
    }

  } catch (error) {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
  }
}

verifyNoWarnings();