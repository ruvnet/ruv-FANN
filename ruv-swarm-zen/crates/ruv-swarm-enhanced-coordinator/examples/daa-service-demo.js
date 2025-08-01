/**
 * DAA Service Demo
 * Demonstrates comprehensive agent lifecycle management, state persistence,
 * and multi-agent workflow coordination with < 1ms latency
 */

import { daaService } from '../npm/src/daa-service.js';

// Helper to format duration
const formatDuration = (ms) => {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
  return `${ms.toFixed(2)}ms`;
};

// Performance monitoring
const measurePerformance = async (name, fn) => {
  const start = performance.now();
  const result = await fn();
  const duration = performance.now() - start;
  console.log(`⏱️  ${name}: ${formatDuration(duration)}`);
  return result;
};

async function demonstrateDAAService() {
  console.log('🚀 DAA Service Demonstration\n');

  try {
    // Initialize the service
    await measurePerformance('Service initialization', async () => {
      await daaService.initialize();
    });

    // Set up event listeners
    daaService.on('decisionMade', ({ agentId, latency, withinThreshold }) => {
      const status = withinThreshold ? '✅' : '⚠️';
      console.log(`${status} Decision latency for ${agentId}: ${formatDuration(latency)}`);
    });

    daaService.on('workflowStepCompleted', ({ workflowId, stepId, duration }) => {
      console.log(`📋 Workflow ${workflowId} step ${stepId} completed in ${formatDuration(duration)}`);
    });

    console.log('\n1️⃣ Agent Lifecycle Management\n');

    // Create multiple agents with different capabilities
    const agents = await measurePerformance('Batch agent creation', async () => {
      return await daaService.batchCreateAgents([
        { id: 'analyzer-001', capabilities: ['decision_making', 'learning', 'prediction'] },
        { id: 'optimizer-001', capabilities: ['resource_optimization', 'self_monitoring'] },
        { id: 'coordinator-001', capabilities: ['coordination', 'goal_planning'] },
        { id: 'healer-001', capabilities: ['self_healing', 'memory_management'] },
      ]);
    });

    console.log(`✅ Created ${agents.filter(r => r.success).length} agents\n`);

    console.log('2️⃣ Cross-Boundary Communication Performance\n');

    // Test decision making latency
    const decisionContext = {
      environment_state: {
        environment_type: 'Dynamic',
        conditions: { temperature: 0.7, pressure: 0.5, volatility: 0.8 },
        stability: 0.4,
        resource_availability: 0.9,
      },
      available_actions: [
        {
          id: 'optimize',
          action_type: 'Compute',
          cost: 0.2,
          expected_reward: 0.8,
          risk: 0.1,
          prerequisites: [],
        },
        {
          id: 'adapt',
          action_type: 'Learn',
          cost: 0.3,
          expected_reward: 0.9,
          risk: 0.2,
          prerequisites: [],
        },
      ],
      goals: [
        {
          id: 'efficiency',
          description: 'Maximize resource efficiency',
          goal_type: 'Efficiency',
          priority: 8,
          progress: 0.6,
          success_criteria: ['resource_usage < 0.7', 'performance > 0.8'],
        },
      ],
      history: [],
      constraints: {
        max_memory_mb: 512,
        max_cpu_usage: 0.8,
        max_network_mbps: 100,
        max_execution_time: 60,
        energy_budget: 1000,
      },
      time_pressure: 0.3,
      uncertainty: 0.4,
    };

    // Make multiple decisions to test latency
    console.log('Testing cross-boundary call latency (target < 1ms):\n');

    for (let i = 0; i < 5; i++) {
      await daaService.makeDecision('analyzer-001', decisionContext);
      await new Promise(resolve => setTimeout(resolve, 100)); // Small delay between calls
    }

    console.log('\n3️⃣ Multi-Agent Workflow Coordination\n');

    // Create a complex workflow
    const workflow = await daaService.createWorkflow(
      'data-processing-pipeline',
      [
        {
          id: 'analyze',
          name: 'Data Analysis',
          task: {
            method: 'make_decision',
            args: [JSON.stringify(decisionContext)],
          },
        },
        {
          id: 'optimize',
          name: 'Resource Optimization',
          task: {
            method: 'optimize_resources',
            args: [],
          },
        },
        {
          id: 'coordinate',
          name: 'Agent Coordination',
          task: async (agent) => {
            // Custom coordination logic
            return `Coordinated by ${agent.id}`;
          },
        },
        {
          id: 'report',
          name: 'Generate Report',
          task: async (agent) => {
            const status = await agent.get_status();
            return { agent: agent.id, status: JSON.parse(status) };
          },
        },
      ],
      {
        'optimize': ['analyze'],
        'coordinate': ['analyze', 'optimize'],
        'report': ['coordinate'],
      },
    );

    console.log(`📋 Created workflow: ${workflow.id}\n`);

    // Execute workflow steps
    await measurePerformance('Execute analysis step', async () => {
      await daaService.executeWorkflowStep('data-processing-pipeline', 'analyze', ['analyzer-001']);
    });

    await measurePerformance('Execute optimization step', async () => {
      await daaService.executeWorkflowStep('data-processing-pipeline', 'optimize', ['optimizer-001']);
    });

    await measurePerformance('Execute coordination step', async () => {
      await daaService.executeWorkflowStep('data-processing-pipeline', 'coordinate', ['coordinator-001']);
    });

    await measurePerformance('Execute reporting step', async () => {
      await daaService.executeWorkflowStep('data-processing-pipeline', 'report', ['analyzer-001', 'optimizer-001']);
    });

    // Get workflow status
    const workflowStatus = daaService.workflows.getWorkflowStatus('data-processing-pipeline');
    console.log('\n📊 Workflow Status:', JSON.stringify(workflowStatus, null, 2));

    console.log('\n4️⃣ State Persistence & Synchronization\n');

    // Synchronize states across agents
    const synchronizedStates = await measurePerformance('State synchronization', async () => {
      return await daaService.synchronizeStates(['analyzer-001', 'optimizer-001', 'coordinator-001']);
    });

    console.log(`🔄 Synchronized ${synchronizedStates.size} agent states\n`);

    console.log('5️⃣ Performance Metrics\n');

    // Get comprehensive performance metrics
    const metrics = daaService.getPerformanceMetrics();

    console.log('System Performance:');
    console.log(`  • Total Agents: ${metrics.system.totalAgents}`);
    console.log(`  • Active Workflows: ${metrics.system.activeWorkflows}`);
    console.log('\nAverage Latencies:');
    console.log(`  • Cross-boundary calls: ${formatDuration(metrics.system.averageLatencies.crossBoundaryCall)}`);
    console.log(`  • Agent spawn: ${formatDuration(metrics.system.averageLatencies.agentSpawn)}`);
    console.log(`  • State sync: ${formatDuration(metrics.system.averageLatencies.stateSync)}`);
    console.log(`  • Workflow steps: ${formatDuration(metrics.system.averageLatencies.workflowStep)}`);

    console.log('\nAgent Metrics:');
    for (const [agentId, agentMetrics] of Object.entries(metrics.agents)) {
      console.log(`\n  ${agentId}:`);
      console.log(`    • Decisions: ${agentMetrics.decisionsMade}`);
      console.log(`    • Avg Response: ${formatDuration(agentMetrics.averageResponseTime)}`);
      console.log(`    • Errors: ${agentMetrics.errors}`);
      console.log(`    • Uptime: ${(agentMetrics.uptime / 1000).toFixed(1)}s`);
    }

    console.log('\n6️⃣ Resource Optimization\n');

    // Optimize resources
    const optimization = await measurePerformance('Resource optimization', async () => {
      return await daaService.optimizeResources();
    });

    console.log('Optimization result:', optimization);

    console.log('\n7️⃣ Batch Operations Demo\n');

    // Batch decision making
    const batchDecisions = await measurePerformance('Batch decisions (10 decisions)', async () => {
      const decisions = [];
      for (let i = 0; i < 10; i++) {
        decisions.push({
          agentId: agents[i % agents.length].agent.id,
          context: { ...decisionContext, uncertainty: Math.random() },
        });
      }
      return await daaService.batchMakeDecisions(decisions);
    });

    const successCount = batchDecisions.filter(r => r.success).length;
    console.log(`✅ ${successCount}/${batchDecisions.length} decisions succeeded\n`);

    console.log('8️⃣ Service Status\n');

    // Get comprehensive service status
    const status = daaService.getStatus();
    console.log(JSON.stringify(status, null, 2));

    console.log('\n🧹 Cleanup\n');

    // Clean up resources
    await measurePerformance('Service cleanup', async () => {
      await daaService.cleanup();
    });

    console.log('\n✅ DAA Service demonstration completed!');

  } catch (error) {
    console.error('❌ Error during demonstration:', error);
  }
}

// Advanced workflow example
async function demonstrateAdvancedWorkflow() {
  console.log('\n\n🎯 Advanced Multi-Agent Workflow Demo\n');

  await daaService.initialize();

  // Create specialized agents
  const specialists = await daaService.batchCreateAgents([
    { id: 'perception-agent', capabilities: ['self_monitoring', 'prediction'] },
    { id: 'planning-agent', capabilities: ['goal_planning', 'decision_making'] },
    { id: 'execution-agent', capabilities: ['coordination', 'resource_optimization'] },
    { id: 'learning-agent', capabilities: ['learning', 'memory_management'] },
  ]);

  // Create a perception-action-learning loop workflow
  const palWorkflow = await daaService.createWorkflow(
    'perception-action-learning',
    [
      {
        id: 'perceive',
        name: 'Environmental Perception',
        task: async (agent) => {
          // Simulate perception
          return {
            detected: ['obstacle', 'resource', 'threat'],
            confidence: 0.85,
          };
        },
      },
      {
        id: 'plan',
        name: 'Action Planning',
        task: async (agent) => {
          // Simulate planning based on perception
          return {
            actions: ['avoid_obstacle', 'collect_resource', 'evade_threat'],
            priority: [0.9, 0.7, 0.95],
          };
        },
      },
      {
        id: 'execute',
        name: 'Action Execution',
        task: async (agent) => {
          // Simulate execution
          return {
            executed: true,
            success_rate: 0.92,
          };
        },
      },
      {
        id: 'learn',
        name: 'Experience Learning',
        task: async (agent) => {
          // Simulate learning from experience
          const feedback = daaService.wasmModule.WasmUtils.create_feedback(0.9, 0.85);
          return await agent.adapt(feedback);
        },
      },
    ],
    {
      'plan': ['perceive'],
      'execute': ['plan'],
      'learn': ['execute'],
    },
  );

  // Execute the PAL loop
  console.log('Executing Perception-Action-Learning loop...\n');

  for (const step of ['perceive', 'plan', 'execute', 'learn']) {
    const agentMap = {
      'perceive': 'perception-agent',
      'plan': 'planning-agent',
      'execute': 'execution-agent',
      'learn': 'learning-agent',
    };

    const result = await daaService.executeWorkflowStep(
      'perception-action-learning',
      step,
      [agentMap[step]],
    );

    console.log(`✓ ${step}: ${JSON.stringify(result)}`);
  }

  console.log('\n✅ Advanced workflow completed!');
}

// Run demonstrations
(async () => {
  await demonstrateDAAService();
  await demonstrateAdvancedWorkflow();
})();
