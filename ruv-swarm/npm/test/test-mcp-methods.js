#!/usr/bin/env node

/**
 * Test script for the new MCP tool methods
 * Tests: agent_metrics, swarm_monitor, neural_train, task_results
 */

import { EnhancedMCPTools } from '../src/mcp-tools-enhanced';
import { RuvSwarm } from '../src/index-enhanced';

async function runTests() {
  console.log('🧪 Testing New MCP Tool Methods\n');

  const mcpTools = new EnhancedMCPTools();
  let testsPassed = 0;
  let testsTotal = 0;

  try {
    // Initialize the MCP tools
    console.log('📋 Initializing MCP Tools...');
    await mcpTools.initialize();
    console.log('✅ MCP Tools initialized\n');

    // Test 1: Initialize a swarm
    testsTotal++;
    console.log('🟢 Test 1: Initialize Swarm');
    const swarmResult = await mcpTools.swarm_init({
      topology: 'mesh',
      maxAgents: 5,
      strategy: 'balanced',
    });
    console.log(`   Swarm ID: ${swarmResult.id}`);
    console.log(`   Features: ${Object.keys(swarmResult.features).filter(f => swarmResult.features[f]).join(', ')}`);
    testsPassed++;
    console.log('   ✅ PASSED\n');

    // Test 2: Spawn agents
    testsTotal++;
    console.log('🟢 Test 2: Spawn Agents');
    const agents = [];
    for (let i = 0; i < 3; i++) {
      const agent = await mcpTools.agent_spawn({
        type: ['researcher', 'coder', 'analyst'][i],
        name: `TestAgent${i + 1}`,
        swarmId: swarmResult.id,
      });
      agents.push(agent.agent);
      console.log(`   Agent ${i + 1}: ${agent.agent.name} (${agent.agent.type})`);
    }
    testsPassed++;
    console.log('   ✅ PASSED\n');

    // Test 3: Agent Metrics
    testsTotal++;
    console.log('🟢 Test 3: Agent Metrics');
    try {
      const metrics = await mcpTools.agent_metrics({
        swarmId: swarmResult.id,
        metricType: 'all',
      });
      console.log(`   Total agents: ${metrics.total_agents}`);
      console.log(`   Average performance: ${(metrics.summary.avg_performance * 100).toFixed(1)}%`);
      console.log(`   Active agents: ${metrics.summary.active_agents}`);

      // Test specific agent metrics
      const agentMetrics = await mcpTools.agent_metrics({
        agentId: agents[0].id,
        metricType: 'performance',
      });
      console.log(`   Agent ${agents[0].name} performance: ${(agentMetrics.agents[0].performance.accuracy_score * 100).toFixed(1)}%`);

      testsPassed++;
      console.log('   ✅ PASSED\n');
    } catch (error) {
      console.log(`   ❌ FAILED: ${error.message}\n`);
    }

    // Test 4: Swarm Monitor
    testsTotal++;
    console.log('🟢 Test 4: Swarm Monitor');
    try {
      const monitoring = await mcpTools.swarm_monitor({
        swarmId: swarmResult.id,
        includeAgents: true,
        includeTasks: true,
        includeMetrics: true,
      });
      console.log(`   Monitoring session: ${monitoring.monitoring_session_id}`);
      console.log(`   Health score: ${(monitoring.swarms[0].health_score * 100).toFixed(1)}%`);
      console.log(`   CPU usage: ${monitoring.swarms[0].resource_utilization.cpu_usage_percent.toFixed(1)}%`);
      console.log(`   Memory usage: ${monitoring.swarms[0].resource_utilization.memory_usage_mb.toFixed(1)}MB`);

      testsPassed++;
      console.log('   ✅ PASSED\n');
    } catch (error) {
      console.log(`   ❌ FAILED: ${error.message}\n`);
    }

    // Test 5: Neural Training
    testsTotal++;
    console.log('🟢 Test 5: Neural Training');
    try {
      const neuralResult = await mcpTools.neural_train({
        agentId: agents[0].id,
        iterations: 5,
        learningRate: 0.001,
        modelType: 'feedforward',
      });
      console.log(`   Training complete: ${neuralResult.training_complete}`);
      console.log(`   Final accuracy: ${(neuralResult.final_accuracy * 100).toFixed(1)}%`);
      console.log(`   Final loss: ${neuralResult.final_loss.toFixed(4)}`);
      console.log(`   Training time: ${neuralResult.training_time_ms}ms`);
      console.log(`   Neural network ID: ${neuralResult.neural_network_id}`);

      testsPassed++;
      console.log('   ✅ PASSED\n');
    } catch (error) {
      console.log(`   ❌ FAILED: ${error.message}\n`);
    }

    // Test 6: Create and test task results
    testsTotal++;
    console.log('🟢 Test 6: Task Orchestration and Results');
    try {
      // Create a task
      const taskResult = await mcpTools.task_orchestrate({
        task: 'Test data processing task',
        priority: 'high',
        strategy: 'parallel',
        swarmId: swarmResult.id,
      });
      console.log(`   Task created: ${taskResult.taskId}`);
      console.log(`   Assigned agents: ${taskResult.assigned_agents.length}`);

      // Simulate task completion by updating database
      mcpTools.persistence.updateTask(taskResult.taskId, {
        status: 'completed',
        result: JSON.stringify({ output: 'Task completed successfully', data: [1, 2, 3, 4, 5] }),
        execution_time_ms: 1500,
        completed_at: new Date().toISOString(),
      });

      // Test task results - summary format
      const results = await mcpTools.task_results({
        taskId: taskResult.taskId,
        format: 'summary',
      });
      console.log(`   Task status: ${results.status}`);
      console.log(`   Success: ${results.success}`);
      console.log(`   Completion time: ${results.completion_time}ms`);

      // Test task results - detailed format
      const detailedResults = await mcpTools.task_results({
        taskId: taskResult.taskId,
        format: 'detailed',
        includeAgentResults: true,
      });
      console.log(`   Detailed results available: ${Boolean(detailedResults.final_result)}`);
      console.log(`   Agents involved: ${detailedResults.execution_summary.agents_involved}`);

      testsPassed++;
      console.log('   ✅ PASSED\n');
    } catch (error) {
      console.log(`   ❌ FAILED: ${error.message}\n`);
    }

    // Test 7: Error handling - Invalid task ID
    testsTotal++;
    console.log('🟢 Test 7: Error Handling (Invalid Task ID)');
    try {
      await mcpTools.task_results({
        taskId: 'invalid-task-id-12345',
        format: 'summary',
      });
      console.log('   ❌ FAILED: Should have thrown error for invalid task ID\n');
    } catch (error) {
      if (error.message.includes('Task not found')) {
        console.log(`   Expected error caught: ${error.message}`);
        testsPassed++;
        console.log('   ✅ PASSED\n');
      } else {
        console.log(`   ❌ FAILED: Unexpected error: ${error.message}\n`);
      }
    }

    // Test 8: Error handling - Missing required parameters
    testsTotal++;
    console.log('🟢 Test 8: Error Handling (Missing Parameters)');
    try {
      await mcpTools.neural_train({
        // Missing agentId
        iterations: 10,
      });
      console.log('   ❌ FAILED: Should have thrown error for missing agentId\n');
    } catch (error) {
      if (error.message.includes('agentId is required')) {
        console.log(`   Expected error caught: ${error.message}`);
        testsPassed++;
        console.log('   ✅ PASSED\n');
      } else {
        console.log(`   ❌ FAILED: Unexpected error: ${error.message}\n`);
      }
    }

    // Summary
    console.log('📊 Test Summary');
    console.log(`   Total tests: ${testsTotal}`);
    console.log(`   Passed: ${testsPassed}`);
    console.log(`   Failed: ${testsTotal - testsPassed}`);
    console.log(`   Success rate: ${((testsPassed / testsTotal) * 100).toFixed(1)}%`);

    if (testsPassed === testsTotal) {
      console.log('\n🎉 All tests passed! MCP methods are working correctly.');
    } else {
      console.log(`\n⚠️  ${testsTotal - testsPassed} test(s) failed. Please review the implementation.`);
    }

  } catch (error) {
    console.error('💥 Fatal error during testing:', error.message);
    console.error('Stack trace:', error.stack);
    process.exit(1);
  }
}

// Run tests if this script is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTests().catch(error => {
    console.error('💥 Test execution failed:', error.message);
    process.exit(1);
  });
}

export { runTests };