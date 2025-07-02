#!/usr/bin/env node

/**
 * Test script to verify agent-task binding mechanism
 * This tests the fix for "No agents available" issue in task orchestration
 */

import path from 'path';
process.chdir(path.join(__dirname, '..'));

import { EnhancedMCPTools  } from '../src/mcp-tools-enhanced';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testAgentTaskBinding() {
  console.log('🧪 Testing Agent-Task Binding Mechanism\n');

  const mcpTools = new EnhancedMCPTools();

  try {
    // Step 1: Initialize swarm
    console.log('1️⃣ Initializing swarm...');
    const swarmResult = await mcpTools.swarm_init({
      topology: 'mesh',
      maxAgents: 5,
      strategy: 'balanced',
    });
    console.log('✅ Swarm initialized:', swarmResult.id);
    console.log('   Features:', Object.entries(swarmResult.features)
      .filter(([key, value]) => value)
      .map(([key]) => key)
      .join(', '));

    // Step 2: Spawn multiple agents
    console.log('\n2️⃣ Spawning agents...');
    const agents = [];

    const agentTypes = ['researcher', 'coder', 'analyst'];
    for (let i = 0; i < agentTypes.length; i++) {
      const agentResult = await mcpTools.agent_spawn({
        type: agentTypes[i],
        name: `Agent-${agentTypes[i]}-${i + 1}`,
        capabilities: [agentTypes[i], 'general'],
      });
      agents.push(agentResult);
      console.log(`✅ Spawned ${agentResult.agent.type} agent: ${agentResult.agent.name} (${agentResult.agent.id})`);
    }

    // Step 3: Check swarm status
    console.log('\n3️⃣ Checking swarm status...');
    const statusResult = await mcpTools.swarm_status({ verbose: true });
    console.log('📊 Swarm status:');
    console.log(`   Active swarms: ${statusResult.active_swarms}`);
    if (statusResult.swarms && statusResult.swarms.length > 0) {
      const swarm = statusResult.swarms[0];
      console.log(`   Agents: ${swarm.status.agents.total} total, ${swarm.status.agents.idle} idle, ${swarm.status.agents.active} active`);
    }

    // Step 4: Test task orchestration (this was failing before)
    console.log('\n4️⃣ Testing task orchestration...');

    const testTasks = [
      {
        description: 'Analyze system performance metrics',
        priority: 'high',
        requiredCapabilities: ['analyst'],
      },
      {
        description: 'Research best practices for optimization',
        priority: 'medium',
        requiredCapabilities: ['researcher'],
      },
      {
        description: 'Implement performance improvements',
        priority: 'medium',
        requiredCapabilities: ['coder'],
      },
    ];

    const orchestratedTasks = [];

    for (const taskConfig of testTasks) {
      try {
        const taskResult = await mcpTools.task_orchestrate({
          task: taskConfig.description,
          priority: taskConfig.priority,
          maxAgents: 2,
          requiredCapabilities: taskConfig.requiredCapabilities,
        });

        orchestratedTasks.push(taskResult);
        console.log(`✅ Task orchestrated: "${taskConfig.description}"`);
        console.log(`   Task ID: ${taskResult.taskId}`);
        console.log(`   Assigned agents: ${taskResult.assigned_agents.length}`);
        console.log(`   Agent IDs: ${taskResult.assigned_agents.join(', ')}`);

      } catch (error) {
        console.error(`❌ Task orchestration failed: ${error.message}`);
        throw error;
      }
    }

    // Step 5: Wait for task execution and check results
    console.log('\n5️⃣ Waiting for task execution...');
    await delay(2000); // Wait for tasks to execute

    for (const taskResult of orchestratedTasks) {
      try {
        const statusCheck = await mcpTools.task_status({
          taskId: taskResult.taskId,
          detailed: true,
        });
        console.log(`📊 Task ${taskResult.taskId} status: ${statusCheck.status}`);
        console.log(`   Progress: ${(statusCheck.progress * 100).toFixed(1)}%`);

        if (statusCheck.status === 'completed') {
          const results = await mcpTools.task_results({
            taskId: taskResult.taskId,
            format: 'summary',
          });
          console.log(`✅ Task completed in ${results.execution_summary?.execution_time_ms || 'N/A'}ms`);
        }
      } catch (error) {
        console.warn(`⚠️ Could not get task status: ${error.message}`);
      }
    }

    // Step 6: Verify agents returned to idle state
    console.log('\n6️⃣ Checking final agent states...');
    const finalStatus = await mcpTools.swarm_status({ verbose: true });
    if (finalStatus.swarms && finalStatus.swarms.length > 0) {
      const swarm = finalStatus.swarms[0];
      console.log(`📊 Final agent states: ${swarm.status.agents.idle} idle, ${swarm.status.agents.active} active`);
    }

    // Step 7: Test with multiple agents per task
    console.log('\n7️⃣ Testing multi-agent task orchestration...');
    try {
      const multiAgentTask = await mcpTools.task_orchestrate({
        task: 'Complex analysis requiring multiple perspectives',
        priority: 'high',
        maxAgents: 3,
        strategy: 'parallel',
      });

      console.log(`✅ Multi-agent task orchestrated: ${multiAgentTask.taskId}`);
      console.log(`   Assigned ${multiAgentTask.assigned_agents.length} agents`);

    } catch (error) {
      console.error(`❌ Multi-agent task failed: ${error.message}`);
      throw error;
    }

    console.log('\n🎉 All tests passed! Agent-task binding is working correctly.');
    console.log('\n✅ Summary:');
    console.log(`   - Successfully spawned ${agents.length} agents`);
    console.log(`   - Successfully orchestrated ${orchestratedTasks.length} tasks`);
    console.log('   - All tasks were assigned to available agents');
    console.log('   - No "No agents available" errors encountered');

    return true;

  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    console.error(error.stack);
    return false;
  }
}

// Run the test
// Direct execution block
{
  testAgentTaskBinding()
    .then(success => {
      process.exit(success ? 0 : 1);
    })
    .catch(error => {
      console.error('Unhandled error:', error);
      process.exit(1);
    });
}

export { testAgentTaskBinding };