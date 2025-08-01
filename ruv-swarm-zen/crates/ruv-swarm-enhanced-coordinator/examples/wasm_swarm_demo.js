// Example demonstrating ruv-swarm WASM capabilities with neural coordination

import init, {
  WasmSwarmOrchestrator,
  CognitiveDiversityEngine,
  NeuralSwarmCoordinator,
  CognitiveNeuralArchitectures,
} from '../pkg/ruv_swarm_wasm.js';

async function runSwarmDemo() {
  // Initialize WASM module
  await init();

  console.log('🚀 ruv-swarm WASM Demo - Neural Swarm Orchestration');
  console.log('=====================================================\n');

  // 1. Create Swarm Orchestrator
  const orchestrator = new WasmSwarmOrchestrator();
  console.log('✅ Created WasmSwarmOrchestrator');

  // 2. Create swarm with mesh topology
  const swarmConfig = {
    name: 'Neural Research Swarm',
    topology_type: 'mesh',
    max_agents: 10,
    enable_cognitive_diversity: true,
    cognitive_balance_threshold: 0.7,
  };

  const swarmResult = orchestrator.create_swarm(swarmConfig);
  console.log('✅ Created swarm:', swarmResult);
  const swarmId = swarmResult.swarm_id;

  // 3. Initialize Cognitive Diversity Engine
  const diversityEngine = new CognitiveDiversityEngine();
  console.log('✅ Created CognitiveDiversityEngine');

  // Get available cognitive patterns
  const patterns = diversityEngine.get_cognitive_patterns();
  console.log('📊 Available cognitive patterns:', Object.keys(patterns));

  // 4. Initialize Neural Swarm Coordinator
  const neuralCoordinator = new NeuralSwarmCoordinator('mesh');
  console.log('✅ Created NeuralSwarmCoordinator with mesh topology');

  // 5. Spawn diverse agents
  const agentTypes = [
    { type: 'researcher', count: 2 },
    { type: 'coder', count: 2 },
    { type: 'analyst', count: 1 },
    { type: 'optimizer', count: 1 },
    { type: 'coordinator', count: 1 },
  ];

  const spawnedAgents = [];

  for (const agentSpec of agentTypes) {
    for (let i = 0; i < agentSpec.count; i++) {
      const agentConfig = {
        agent_type: agentSpec.type,
        name: `${agentSpec.type}-${i+1}`,
        max_agents: 10,
      };

      const agent = orchestrator.spawn_agent(swarmId, agentConfig);
      spawnedAgents.push(agent);
      console.log(`✅ Spawned agent: ${agent.name} (${agent.cognitive_pattern})`);
    }
  }

  // 6. Check swarm diversity
  const agentComposition = spawnedAgents.map(agent => ({
    agent_id: agent.agent_id,
    agent_type: agent.type,
    cognitive_pattern: agent.cognitive_pattern,
    capabilities: agent.capabilities,
  }));

  const diversityAnalysis = diversityEngine.analyze_swarm_diversity(agentComposition);
  console.log('\n📊 Swarm Diversity Analysis:');
  console.log('- Overall diversity score:', diversityAnalysis.diversity_metrics.overall_diversity_score);
  console.log('- Pattern distribution:', diversityAnalysis.diversity_metrics.pattern_distribution);
  console.log('- Recommendations:', diversityAnalysis.recommendations);

  // 7. Orchestrate a complex task
  const taskConfig = {
    description: 'Research and implement a new neural network architecture for time series forecasting',
    priority: 'high',
    required_capabilities: ['data_analysis', 'code_generation', 'optimization'],
    max_agents: 5,
    estimated_duration_ms: 60000,
  };

  const taskResult = orchestrator.orchestrate_task(swarmId, taskConfig);
  console.log('\n📋 Task Orchestration Result:');
  console.log('- Task ID:', taskResult.task_id);
  console.log('- Assigned agents:', taskResult.assigned_agents);
  console.log('- Distribution plan:', taskResult.distribution_plan);

  // 8. Configure distributed neural training
  const trainingConfig = {
    training_mode: 'DataParallel',
    agent_ids: taskResult.assigned_agents,
    dataset_config: {
      dataset_size: 10000,
      feature_dim: 64,
      num_classes: 10,
    },
    optimization_config: {
      optimizer: 'adam',
      learning_rate: 0.001,
      batch_size: 32,
    },
    synchronization_interval: 100,
  };

  const trainingResult = neuralCoordinator.coordinate_neural_training(trainingConfig);
  console.log('\n🧠 Neural Training Coordination:');
  console.log('- Final loss:', trainingResult.final_loss);
  console.log('- Epochs completed:', trainingResult.epochs_completed);
  console.log('- Convergence achieved:', trainingResult.convergence_achieved);

  // 9. Synchronize agent knowledge
  const syncRequest = {
    sync_type: 'Knowledge',
    participating_agents: taskResult.assigned_agents,
    knowledge_domains: ['neural_architectures', 'time_series_patterns'],
    sync_depth: {
      layers: [2, 3, 4],
      percentage: 0.8,
    },
  };

  const syncResult = neuralCoordinator.synchronize_agent_knowledge(syncRequest);
  console.log('\n🔄 Knowledge Synchronization:');
  console.log('- Sync type:', syncResult.sync_type);
  console.log('- Operations:', syncResult.operations.length);
  console.log('- Sync time:', syncResult.sync_time_ms, 'ms');

  // 10. Get cognitive neural architectures
  const neuralArchitectures = new CognitiveNeuralArchitectures();
  console.log('\n🏗️ Cognitive Neural Architectures:');

  // Get architecture for each cognitive pattern
  const architecturePatterns = ['convergent', 'divergent', 'systems', 'critical', 'lateral'];
  for (const pattern of architecturePatterns) {
    const architecture = neuralArchitectures.get_architecture_for_pattern(pattern);
    console.log(`- ${pattern}:`, architecture);
  }

  // 11. Monitor swarm performance
  const monitoringData = orchestrator.monitor_swarm(swarmId, 5000);
  console.log('\n📈 Swarm Monitoring:');
  console.log('- Active connections:', monitoringData.real_time_metrics.active_connections);
  console.log('- Memory usage:', monitoringData.real_time_metrics.memory_usage_mb, 'MB');
  console.log('- Agent activity:', monitoringData.agent_activity);

  // 12. Get final swarm status
  const finalStatus = orchestrator.get_swarm_status(swarmId, true);
  console.log('\n📊 Final Swarm Status:');
  console.log('- Total agents:', finalStatus.agents.total);
  console.log('- Active agents:', finalStatus.agents.active);
  console.log('- Tasks completed:', finalStatus.tasks.completed);
  console.log('- Average task completion time:', finalStatus.performance.avg_task_completion_time, 'ms');
  console.log('- Success rate:', finalStatus.performance.success_rate);

  console.log('\n✨ Demo completed successfully!');
}

// Run the demo
runSwarmDemo().catch(console.error);
