#!/usr/bin/env node
/**
 * MCP Functionality Validation
 * Tests actual tool execution and validates responses
 */

const { EnhancedMCPTools } = require('../npm/src/mcp-tools-enhanced');

async function validateMcpFunctionality() {
  console.log('🔧 MCP Tool Functionality Validation\n');

  const results = {
    tools_tested: 0,
    tools_passed: 0,
    tools_failed: 0,
    execution_times: {},
    errors: [],
  };

  const mcpTools = new EnhancedMCPTools();

  try {
    // Initialize the system
    console.log('Initializing MCP tools...');
    await mcpTools.initialize();
    console.log('✅ Initialization successful\n');

    // Test 1: features_detect
    console.log('1. Testing features_detect...');
    const start1 = performance.now();
    try {
      const features = await mcpTools.features_detect({ category: 'all' });
      const time1 = performance.now() - start1;

      console.log(`✅ features_detect passed (${time1.toFixed(1)}ms)`);
      console.log(`   WASM modules: ${Object.keys(features.wasm.modules_loaded || {}).length}`);
      console.log(`   SIMD support: ${features.wasm.simd_support ? 'Yes' : 'No'}`);

      results.tools_tested++;
      results.tools_passed++;
      results.execution_times.features_detect = time1;
    } catch (error) {
      console.log(`❌ features_detect failed: ${error.message}`);
      results.tools_tested++;
      results.tools_failed++;
      results.errors.push({ tool: 'features_detect', error: error.message });
    }

    // Test 2: memory_usage
    console.log('\n2. Testing memory_usage...');
    const start2 = performance.now();
    try {
      const memory = await mcpTools.memory_usage({ detail: 'summary' });
      const time2 = performance.now() - start2;

      console.log(`✅ memory_usage passed (${time2.toFixed(1)}ms)`);
      console.log(`   Total: ${memory.total_mb.toFixed(2)}MB`);
      console.log(`   WASM: ${memory.wasm_mb.toFixed(2)}MB`);

      results.tools_tested++;
      results.tools_passed++;
      results.execution_times.memory_usage = time2;
    } catch (error) {
      console.log(`❌ memory_usage failed: ${error.message}`);
      results.tools_tested++;
      results.tools_failed++;
      results.errors.push({ tool: 'memory_usage', error: error.message });
    }

    // Test 3: swarm_init
    console.log('\n3. Testing swarm_init...');
    const start3 = performance.now();
    let swarmId;
    try {
      const swarm = await mcpTools.swarm_init({
        topology: 'mesh',
        maxAgents: 5,
        strategy: 'balanced',
      });
      const time3 = performance.now() - start3;
      swarmId = swarm.id;

      console.log(`✅ swarm_init passed (${time3.toFixed(1)}ms)`);
      console.log(`   Swarm ID: ${swarm.id}`);
      console.log(`   Topology: ${swarm.topology}`);
      console.log(`   Features: ${Object.entries(swarm.features).filter(([k,v]) => v).map(([k]) => k).join(', ')}`);

      results.tools_tested++;
      results.tools_passed++;
      results.execution_times.swarm_init = time3;
    } catch (error) {
      console.log(`❌ swarm_init failed: ${error.message}`);
      results.tools_tested++;
      results.tools_failed++;
      results.errors.push({ tool: 'swarm_init', error: error.message });
    }

    // Test 4: agent_spawn
    console.log('\n4. Testing agent_spawn...');
    const start4 = performance.now();
    let agentId;
    try {
      const agent = await mcpTools.agent_spawn({
        type: 'researcher',
        name: 'test-researcher',
        capabilities: ['analysis', 'research'],
      });
      const time4 = performance.now() - start4;
      agentId = agent.agent.id;

      console.log(`✅ agent_spawn passed (${time4.toFixed(1)}ms)`);
      console.log(`   Agent ID: ${agent.agent.id}`);
      console.log(`   Type: ${agent.agent.type}`);
      console.log(`   Cognitive Pattern: ${agent.agent.cognitive_pattern}`);

      results.tools_tested++;
      results.tools_passed++;
      results.execution_times.agent_spawn = time4;
    } catch (error) {
      console.log(`❌ agent_spawn failed: ${error.message}`);
      results.tools_tested++;
      results.tools_failed++;
      results.errors.push({ tool: 'agent_spawn', error: error.message });
    }

    // Test 5: task_orchestrate
    console.log('\n5. Testing task_orchestrate...');
    const start5 = performance.now();
    try {
      const task = await mcpTools.task_orchestrate({
        task: 'Analyze performance metrics',
        priority: 'medium',
        strategy: 'adaptive',
      });
      const time5 = performance.now() - start5;

      console.log(`✅ task_orchestrate passed (${time5.toFixed(1)}ms)`);
      console.log(`   Task ID: ${task.taskId}`);
      console.log(`   Status: ${task.status}`);
      console.log(`   Assigned Agents: ${task.assigned_agents.length}`);

      results.tools_tested++;
      results.tools_passed++;
      results.execution_times.task_orchestrate = time5;
    } catch (error) {
      console.log(`❌ task_orchestrate failed: ${error.message}`);
      results.tools_tested++;
      results.tools_failed++;
      results.errors.push({ tool: 'task_orchestrate', error: error.message });
    }

    // Test 6: swarm_status
    console.log('\n6. Testing swarm_status...');
    const start6 = performance.now();
    try {
      const status = await mcpTools.swarm_status({ verbose: false });
      const time6 = performance.now() - start6;

      console.log(`✅ swarm_status passed (${time6.toFixed(1)}ms)`);
      console.log(`   Active Swarms: ${status.active_swarms}`);
      console.log(`   Memory Usage: ${(status.global_metrics.memoryUsage / (1024 * 1024)).toFixed(2)}MB`);

      results.tools_tested++;
      results.tools_passed++;
      results.execution_times.swarm_status = time6;
    } catch (error) {
      console.log(`❌ swarm_status failed: ${error.message}`);
      results.tools_tested++;
      results.tools_failed++;
      results.errors.push({ tool: 'swarm_status', error: error.message });
    }

    // Test 7: benchmark_run (quick test)
    console.log('\n7. Testing benchmark_run...');
    const start7 = performance.now();
    try {
      const benchmarks = await mcpTools.benchmark_run({
        type: 'wasm',
        iterations: 3,
      });
      const time7 = performance.now() - start7;

      console.log(`✅ benchmark_run passed (${time7.toFixed(1)}ms)`);
      console.log(`   Benchmark Type: ${benchmarks.benchmark_type}`);
      console.log(`   Total Time: ${benchmarks.performance.total_benchmark_time_ms.toFixed(1)}ms`);

      results.tools_tested++;
      results.tools_passed++;
      results.execution_times.benchmark_run = time7;
    } catch (error) {
      console.log(`❌ benchmark_run failed: ${error.message}`);
      results.tools_tested++;
      results.tools_failed++;
      results.errors.push({ tool: 'benchmark_run', error: error.message });
    }

    // Test 8: neural_status
    console.log('\n8. Testing neural_status...');
    const start8 = performance.now();
    try {
      const neural = await mcpTools.neural_status({ agentId });
      const time8 = performance.now() - start8;

      console.log(`✅ neural_status passed (${time8.toFixed(1)}ms)`);
      console.log(`   Available: ${neural.available}`);
      if (neural.available) {
        console.log(`   Activation Functions: ${neural.activation_functions}`);
        console.log(`   SIMD Acceleration: ${neural.simd_acceleration ? 'Yes' : 'No'}`);
      }

      results.tools_tested++;
      results.tools_passed++;
      results.execution_times.neural_status = time8;
    } catch (error) {
      console.log(`❌ neural_status failed: ${error.message}`);
      results.tools_tested++;
      results.tools_failed++;
      results.errors.push({ tool: 'neural_status', error: error.message });
    }

    // Test 9: neural_patterns
    console.log('\n9. Testing neural_patterns...');
    const start9 = performance.now();
    try {
      const patterns = await mcpTools.neural_patterns({ pattern: 'all' });
      const time9 = performance.now() - start9;

      console.log(`✅ neural_patterns passed (${time9.toFixed(1)}ms)`);
      console.log(`   Patterns Available: ${Object.keys(patterns).length}`);
      console.log(`   Types: ${Object.keys(patterns).join(', ')}`);

      results.tools_tested++;
      results.tools_passed++;
      results.execution_times.neural_patterns = time9;
    } catch (error) {
      console.log(`❌ neural_patterns failed: ${error.message}`);
      results.tools_tested++;
      results.tools_failed++;
      results.errors.push({ tool: 'neural_patterns', error: error.message });
    }

    // Calculate average execution time
    const times = Object.values(results.execution_times);
    const avgTime = times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0;

    console.log('\n' + '='.repeat(50));
    console.log('📊 VALIDATION SUMMARY');
    console.log('='.repeat(50));
    console.log(`Total Tools Tested: ${results.tools_tested}`);
    console.log(`✅ Passed: ${results.tools_passed}`);
    console.log(`❌ Failed: ${results.tools_failed}`);
    console.log(`Average Execution Time: ${avgTime.toFixed(1)}ms`);

    if (results.errors.length > 0) {
      console.log('\n❌ Errors:');
      results.errors.forEach(e => {
        console.log(`  - ${e.tool}: ${e.error}`);
      });
    }

    // Performance analysis
    console.log('\n⚡ Performance Analysis:');
    Object.entries(results.execution_times).forEach(([tool, time]) => {
      const status = time < 100 ? '🟢' : time < 500 ? '🟡' : '🔴';
      console.log(`  ${status} ${tool}: ${time.toFixed(1)}ms`);
    });

    return results;

  } catch (error) {
    console.error('Fatal error:', error);
    results.errors.push({ tool: 'system', error: error.message });
    return results;
  }
}

// Run validation
if (require.main === module) {
  validateMcpFunctionality().then(results => {
    console.log('\nValidation complete.');
    process.exit(results.tools_failed > 0 ? 1 : 0);
  }).catch(error => {
    console.error('Validation failed:', error);
    process.exit(1);
  });
}

module.exports = { validateMcpFunctionality };
