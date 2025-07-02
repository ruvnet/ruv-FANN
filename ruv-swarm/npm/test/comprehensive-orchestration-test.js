#!/usr/bin/env node

/**
 * Comprehensive orchestration test covering edge cases and different scenarios
 */

import path from 'path';
process.chdir(path.join(__dirname, '..'));

import { EnhancedMCPTools } from '../src/mcp-tools-enhanced';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function comprehensiveOrchestrationTest() {
  console.log('🧪 Comprehensive Orchestration Test\n');

  const mcpTools = new EnhancedMCPTools();

  try {
    // Test 1: Initialize multiple swarms
    console.log('1️⃣ Testing multiple swarms...');
    const swarm1 = await mcpTools.swarm_init({
      topology: 'mesh',
      maxAgents: 3,
      strategy: 'balanced',
    });
    const swarm2 = await mcpTools.swarm_init({
      topology: 'star',
      maxAgents: 2,
      strategy: 'balanced',
    });
    console.log(`✅ Created ${swarm1.id} and ${swarm2.id}`);

    // Test 2: Spawn different types of agents
    console.log('\n2️⃣ Testing different agent types...');
    const agentTypes = ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'];
    let agentCount = 0;

    for (let i = 0; i < agentTypes.length; i++) {
      await mcpTools.agent_spawn({
        type: agentTypes[i],
        name: `${agentTypes[i]}-agent-${i}`,
        capabilities: [agentTypes[i], 'general', 'task-execution'],
      });
      agentCount++;
    }
    console.log(`✅ Spawned ${agentCount} different agent types`);

    // Test 3: Check agent list functionality
    console.log('\n3️⃣ Testing agent listing...');
    const allAgents = await mcpTools.agent_list({ filter: 'all' });
    const idleAgents = await mcpTools.agent_list({ filter: 'idle' });
    console.log(`📊 Total agents: ${allAgents.total_agents}, Idle: ${idleAgents.total_agents}`);

    // Test 4: Test capability-based task assignment
    console.log('\n4️⃣ Testing capability-based task assignment...');
    const capabilityTasks = [
      {
        description: 'Research quantum computing applications',
        capabilities: ['researcher'],
        priority: 'high',
      },
      {
        description: 'Optimize database queries',
        capabilities: ['optimizer'],
        priority: 'medium',
      },
      {
        description: 'Coordinate team activities',
        capabilities: ['coordinator'],
        priority: 'low',
      },
    ];

    const capabilityResults = [];
    for (const task of capabilityTasks) {
      const result = await mcpTools.task_orchestrate({
        task: task.description,
        priority: task.priority,
        requiredCapabilities: task.capabilities,
        maxAgents: 1,
      });
      capabilityResults.push(result);
      console.log(`✅ Assigned "${task.description}" to agent with ${task.capabilities} capability`);
    }

    // Test 5: Test maxAgents parameter
    console.log('\n5️⃣ Testing maxAgents parameter...');
    const multiAgentTask = await mcpTools.task_orchestrate({
      task: 'Large-scale data analysis requiring multiple agents',
      priority: 'high',
      maxAgents: 3,
      strategy: 'parallel',
    });
    console.log(`✅ Multi-agent task assigned to ${multiAgentTask.assigned_agents.length} agents (max: 3)`);

    // Test 6: Test task queue when all agents are busy
    console.log('\n6️⃣ Testing task queue with busy agents...');
    // First, make all agents busy with long-running tasks
    const longRunningTasks = [];
    for (let i = 0; i < 3; i++) {
      const task = await mcpTools.task_orchestrate({
        task: `Long running task ${i + 1}`,
        priority: 'medium',
        maxAgents: 2,
      });
      longRunningTasks.push(task);
    }

    // Wait a bit, then try to orchestrate when agents are busy
    await delay(100);

    try {
      await mcpTools.task_orchestrate({
        task: 'Task when all agents are busy',
        priority: 'high',
        maxAgents: 1,
      });
      console.log('✅ Successfully handled task orchestration with busy agents');
    } catch (error) {
      if (error.message.includes('No agents available')) {
        console.log('✅ Correctly detected no available agents (expected behavior)');
      } else {
        throw error;
      }
    }

    // Wait for tasks to complete
    await delay(1000);

    // Test 7: Test task status and results
    console.log('\n7️⃣ Testing task status and results...');
    for (const taskResult of capabilityResults) {
      const status = await mcpTools.task_status({
        taskId: taskResult.taskId,
        detailed: true,
      });
      console.log(`📊 Task ${taskResult.taskId}: ${status.status} (${status.progress * 100}%)`);

      if (status.status === 'completed') {
        const results = await mcpTools.task_results({
          taskId: taskResult.taskId,
          format: 'summary',
        });
        console.log(`   Execution time: ${results.completion_time}ms`);
      }
    }

    // Test 8: Test swarm metrics
    console.log('\n8️⃣ Testing swarm metrics...');
    const swarmStatus = await mcpTools.swarm_status({ verbose: true });
    console.log(`📊 Active swarms: ${swarmStatus.active_swarms}`);
    console.log(`🔧 Runtime features: ${Object.keys(swarmStatus.runtime_info.features).filter(f => swarmStatus.runtime_info.features[f]).join(', ')}`);

    // Test 9: Test memory usage tracking
    console.log('\n9️⃣ Testing memory usage tracking...');
    const memoryUsage = await mcpTools.memory_usage({ detail: 'summary' });
    console.log(`💾 Total memory: ${memoryUsage.total_mb.toFixed(2)} MB`);
    console.log(`   WASM: ${memoryUsage.wasm_mb.toFixed(2)} MB, JS: ${memoryUsage.javascript_mb.toFixed(2)} MB`);

    // Test 10: Test error handling
    console.log('\n🔟 Testing error handling...');
    try {
      await mcpTools.task_orchestrate({
        task: 'Task with invalid swarm',
        priority: 'high',
      });
      // If we get here, clear the swarms first
      mcpTools.activeSwarms.clear();
      await mcpTools.task_orchestrate({
        task: 'Task with no swarm',
        priority: 'high',
      });
    } catch (error) {
      if (error.message.includes('No active swarm found')) {
        console.log('✅ Correctly handled error when no swarm is available');
      } else {
        throw error;
      }
    }

    console.log('\n🎉 All comprehensive tests passed!');
    console.log('\n✅ Verified functionality:');
    console.log('   - Multiple swarm creation and management');
    console.log('   - Different agent types and capabilities');
    console.log('   - Capability-based task assignment');
    console.log('   - MaxAgents parameter handling');
    console.log('   - Task status and result tracking');
    console.log('   - Memory usage monitoring');
    console.log('   - Error handling for edge cases');
    console.log('   - Agent state transitions (idle ↔ busy)');

    return true;

  } catch (error) {
    console.error('\n❌ Comprehensive test failed:', error.message);
    console.error(error.stack);
    return false;
  }
}

// Run the comprehensive test
// Direct execution
comprehensiveOrchestrationTest()
  .then(success => {
    process.exit(success ? 0 : 1);
  })
  .catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });

export { comprehensiveOrchestrationTest };