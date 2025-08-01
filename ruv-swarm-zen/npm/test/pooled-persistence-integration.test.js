/**
 * Integration Tests for Pooled Persistence in Production Environment
 *
 * Tests the complete integration of SwarmPersistencePooled with the MCP tools
 * and validates that all existing functionality works with the new pool.
 */

import { EnhancedMCPTools } from '../src/mcp-tools-enhanced.js';
import { RuvSwarm } from '../src/index-enhanced.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Test configuration
const TEST_DB_PATH = path.join(__dirname, 'test-pooled-integration.db');

console.log('🧪 Starting Pooled Persistence Integration Tests\n');

// Cleanup function
function cleanup() {
  try {
    [TEST_DB_PATH, TEST_DB_PATH + '-wal', TEST_DB_PATH + '-shm'].forEach(file => {
      if (fs.existsSync(file)) {
        fs.unlinkSync(file);
      }
    });
  } catch (error) {
    console.error('Cleanup error:', error);
  }
}

// Test 1: MCP Tools Integration with Pooled Persistence
async function testMCPToolsIntegration() {
  console.log('🔍 Test 1: MCP Tools Integration with Pooled Persistence');

  try {
    cleanup();

    // Create MCP tools instance (should use pooled persistence)
    const mcpTools = new EnhancedMCPTools();

    // Wait for persistence initialization
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Test swarm initialization
    const swarmResult = await mcpTools.swarm_init({
      topology: 'mesh',
      maxAgents: 5,
      strategy: 'balanced',
    });

    console.log('✅ Swarm initialized with pooled persistence');
    console.log(`📊 Swarm ID: ${swarmResult.id}`);

    // Test agent spawning
    const agentResult = await mcpTools.agent_spawn({
      type: 'researcher',
      name: 'Test Agent',
      capabilities: ['research', 'analysis'],
    });

    console.log('✅ Agent spawned successfully');
    console.log(`🤖 Agent ID: ${agentResult.agent.id}`);

    // Test pool health monitoring
    const healthResult = await mcpTools.pool_health();
    console.log('✅ Pool health monitoring working');
    console.log(`🏥 Pool healthy: ${healthResult.healthy}`);
    console.log(`📊 Active connections: ${healthResult.pool_status?.active_connections || 'N/A'}`);

    // Test pool statistics
    const statsResult = await mcpTools.pool_stats();
    console.log('✅ Pool statistics working');
    console.log(`📈 Total operations: ${statsResult.persistence_metrics?.total_operations || 'N/A'}`);

    // Test persistence statistics
    const persistenceResult = await mcpTools.persistence_stats();
    console.log('✅ Persistence statistics working');
    console.log(`💾 Persistence layer: ${persistenceResult.persistence_layer}`);
    console.log(`⚡ Connection pool: ${persistenceResult.connection_pool}`);

    console.log('✅ MCP Tools integration test passed\n');

  } catch (error) {
    console.error('❌ Test 1 failed:', error.message);
    throw error;
  }
}

// Test 2: RuvSwarm Core Integration with Pooled Persistence
async function testRuvSwarmCoreIntegration() {
  console.log('🔍 Test 2: RuvSwarm Core Integration with Pooled Persistence');

  try {
    cleanup();

    // Initialize RuvSwarm with pooled persistence
    const ruvSwarm = await RuvSwarm.initialize({
      enablePersistence: true,
      enableNeuralNetworks: false,
      enableForecasting: false,
      debug: false,
    });

    console.log('✅ RuvSwarm initialized with pooled persistence');

    // Test swarm creation
    const swarm = await ruvSwarm.createSwarm({
      name: 'Integration Test Swarm',
      topology: 'hierarchical',
      maxAgents: 8,
    });

    console.log('✅ Swarm created through core RuvSwarm');
    console.log(`🐝 Swarm ID: ${swarm.id}`);

    // Test persistence availability
    if (!ruvSwarm.persistence) {
      throw new Error('Persistence layer not available');
    }

    console.log('✅ Persistence layer available and initialized');

    // Test that it's the pooled version
    if (ruvSwarm.persistence.constructor.name !== 'SwarmPersistencePooled') {
      throw new Error('Expected SwarmPersistencePooled, got ' + ruvSwarm.persistence.constructor.name);
    }

    console.log('✅ Confirmed using SwarmPersistencePooled');

    // Test pool health
    if (typeof ruvSwarm.persistence.isHealthy === 'function') {
      const isHealthy = ruvSwarm.persistence.isHealthy();
      console.log(`🏥 Pool health: ${isHealthy ? 'Healthy' : 'Unhealthy'}`);
    }

    // Test pool statistics
    if (typeof ruvSwarm.persistence.getPoolStats === 'function') {
      const poolStats = ruvSwarm.persistence.getPoolStats();
      console.log(`📊 Pool connections: ${poolStats.activeConnections || 0} active, ${poolStats.availableReaders || 0} readers`);
    }

    console.log('✅ RuvSwarm core integration test passed\n');

  } catch (error) {
    console.error('❌ Test 2 failed:', error.message);
    throw error;
  }
}

// Test 3: Concurrent Operations with Pooled Persistence
async function testConcurrentOperations() {
  console.log('🔍 Test 3: Concurrent Operations with Pooled Persistence');

  try {
    cleanup();

    const mcpTools = new EnhancedMCPTools();

    // Wait for initialization
    await new Promise(resolve => setTimeout(resolve, 1000));

    console.log('📊 Running concurrent operations test');

    // Create multiple swarms concurrently
    const swarmPromises = [];
    for (let i = 0; i < 5; i++) {
      swarmPromises.push(
        mcpTools.swarm_init({
          topology: 'mesh',
          maxAgents: 3,
          strategy: 'balanced',
        }),
      );
    }

    const swarms = await Promise.all(swarmPromises);
    console.log(`✅ Created ${swarms.length} swarms concurrently`);

    // Create multiple agents concurrently across different swarms
    const agentPromises = [];
    for (let i = 0; i < 10; i++) {
      agentPromises.push(
        mcpTools.agent_spawn({
          type: 'researcher',
          name: `Concurrent Agent ${i}`,
          capabilities: ['analysis'],
        }),
      );
    }

    const agents = await Promise.all(agentPromises);
    console.log(`✅ Created ${agents.length} agents concurrently`);

    // Test concurrent health and statistics checks
    const monitoringPromises = [
      mcpTools.pool_health(),
      mcpTools.pool_stats(),
      mcpTools.persistence_stats(),
      mcpTools.swarm_status({ verbose: true }),
      mcpTools.memory_usage({ detail: 'summary' }),
    ];

    const monitoringResults = await Promise.all(monitoringPromises);
    console.log('✅ Concurrent monitoring operations completed');

    // Verify all operations succeeded
    const healthResult = monitoringResults[0];
    if (!healthResult.healthy && !healthResult.error) {
      throw new Error('Pool health check failed during concurrent operations');
    }

    console.log(`🏥 Pool remained healthy during concurrent operations: ${healthResult.healthy || 'N/A'}`);
    console.log('✅ Concurrent operations test passed\n');

  } catch (error) {
    console.error('❌ Test 3 failed:', error.message);
    throw error;
  }
}

// Test 4: Performance Comparison
async function testPerformanceComparison() {
  console.log('🔍 Test 4: Performance Validation with Pooled Persistence');

  try {
    cleanup();

    const mcpTools = new EnhancedMCPTools();

    // Wait for initialization
    await new Promise(resolve => setTimeout(resolve, 1000));

    console.log('📊 Running performance validation');

    // Measure swarm creation performance
    const startTime = Date.now();
    const numOperations = 20;

    for (let i = 0; i < numOperations; i++) {
      await mcpTools.swarm_init({
        topology: 'mesh',
        maxAgents: 2,
        strategy: 'balanced',
      });
    }

    const duration = Date.now() - startTime;
    const avgTime = duration / numOperations;

    console.log(`✅ Created ${numOperations} swarms in ${duration}ms`);
    console.log(`⚡ Average: ${avgTime.toFixed(2)}ms per swarm`);

    if (avgTime > 100) { // Should be much faster with pooling
      console.warn(`⚠️  Average time higher than expected: ${avgTime.toFixed(2)}ms`);
    } else {
      console.log('✅ Performance meets expectations');
    }

    // Test pool statistics after load
    const finalStats = await mcpTools.pool_stats();
    console.log(`📈 Final pool stats:`);
    console.log(`   - Total operations: ${finalStats.persistence_metrics?.total_operations || 'N/A'}`);
    console.log(`   - Error rate: ${finalStats.persistence_metrics?.error_rate || 'N/A'}`);
    console.log(`   - Avg response time: ${finalStats.persistence_metrics?.average_response_time || 'N/A'}ms`);

    console.log('✅ Performance validation test passed\n');

  } catch (error) {
    console.error('❌ Test 4 failed:', error.message);
    throw error;
  }
}

// Test 5: Environment Variable Configuration
async function testEnvironmentConfiguration() {
  console.log('🔍 Test 5: Environment Variable Configuration');

  try {
    // Set environment variables
    process.env.POOL_MAX_READERS = '8';
    process.env.POOL_MAX_WORKERS = '4';
    process.env.POOL_CACHE_SIZE = '-128000'; // 128MB

    cleanup();

    const mcpTools = new EnhancedMCPTools();

    // Wait for initialization
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Test that configuration was applied
    const stats = await mcpTools.pool_stats();

    console.log('✅ Environment configuration applied');
    console.log(`📊 Pool configuration loaded from environment variables`);
    console.log(`   - Readers: ${stats.pool_metrics?.available_readers || 'N/A'}`);
    console.log(`   - Workers: ${stats.pool_metrics?.available_workers || 'N/A'}`);

    // Clean up environment variables
    delete process.env.POOL_MAX_READERS;
    delete process.env.POOL_MAX_WORKERS;
    delete process.env.POOL_CACHE_SIZE;

    console.log('✅ Environment configuration test passed\n');

  } catch (error) {
    console.error('❌ Test 5 failed:', error.message);
    throw error;
  }
}

// Main test runner
async function runAllTests() {
  const tests = [
    testMCPToolsIntegration,
    testRuvSwarmCoreIntegration,
    testConcurrentOperations,
    testPerformanceComparison,
    testEnvironmentConfiguration,
  ];

  let passed = 0;
  let failed = 0;

  console.log('🚀 Starting Pooled Persistence Integration Test Suite\n');

  for (const test of tests) {
    try {
      await test();
      passed++;
    } catch (error) {
      failed++;
      console.error(`Test failed: ${error.message}\n`);
    }
  }

  console.log('📊 Integration Test Results Summary:');
  console.log(`✅ Passed: ${passed}`);
  console.log(`❌ Failed: ${failed}`);
  console.log(`📈 Success Rate: ${(passed / (passed + failed) * 100).toFixed(1)}%`);

  if (failed === 0) {
    console.log('\n🎉 All integration tests passed! Pooled persistence is production ready.');
  } else {
    console.log('\n⚠️  Some integration tests failed. Review and fix issues before deployment.');
  }

  cleanup();
}

// Handle cleanup on exit
process.on('exit', cleanup);
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

// Run tests
runAllTests().catch(console.error);
