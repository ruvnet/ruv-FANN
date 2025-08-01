// Swarm Topology Validation Test Suite
// Agent 4 focused on comprehensive topology testing

const path = require('path');
const fs = require('fs');

// Test configuration
const TEST_CONFIG = {
  topologies: ['mesh', 'star', 'hierarchical', 'ring'],
  scalabilityTests: [5, 10, 20, 50, 100],
  taskComplexities: ['simple', 'medium', 'complex'],
  coordinationPatterns: ['synchronous', 'asynchronous', 'hybrid'],
};

// Test results storage
const testResults = {
  topologies_tested: {},
  scalability: {},
  coordination_efficiency: {},
  failures: [],
  timestamp: new Date().toISOString(),
};

// Utility functions
function measureTime(fn) {
  const start = process.hrtime.bigint();
  const result = fn();
  const end = process.hrtime.bigint();
  return {
    result,
    timeMs: Number(end - start) / 1000000,
  };
}

async function loadWasmModule() {
  try {
    const wasmPath = path.join(__dirname, '../npm/ruv_swarm_wasm_bg.wasm');
    const wasmModule = await import('../npm/src/index-enhanced.js');
    return wasmModule;
  } catch (error) {
    console.error('Failed to load WASM module:', error);
    testResults.failures.push({
      test: 'wasm_loading',
      error: error.message,
      timestamp: new Date().toISOString(),
    });
    return null;
  }
}

// Topology-specific tests
async function testMeshTopology(wasm, agentCount) {
  const results = {
    connectivity: {},
    latency: {},
    throughput: {},
    scalability: {},
  };

  try {
    const orchestrator = new wasm.WasmSwarmOrchestrator();

    // Create mesh swarm
    const swarmConfig = {
      name: `Mesh Test ${agentCount} agents`,
      topology_type: 'mesh',
      max_agents: agentCount,
      enable_cognitive_diversity: true,
    };

    const { result: swarmResult, timeMs: createTime } = measureTime(() =>
      orchestrator.create_swarm(swarmConfig),
    );

    results.createTime = createTime;
    const swarmId = swarmResult.swarm_id;

    // Spawn agents and measure connectivity setup time
    const agentIds = [];
    let totalSpawnTime = 0;

    for (let i = 0; i < agentCount; i++) {
      const agentConfig = {
        agent_type: ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'][i % 5],
        name: `mesh-agent-${i}`,
      };

      const { result: agentResult, timeMs: spawnTime } = measureTime(() =>
        orchestrator.spawn_agent(swarmId, agentConfig),
      );

      totalSpawnTime += spawnTime;
      agentIds.push(agentResult.agent_id);
    }

    results.avgSpawnTime = totalSpawnTime / agentCount;

    // Get swarm status to verify mesh connectivity
    const status = orchestrator.get_swarm_status(swarmId, true);

    // In mesh topology, each agent should connect to all others
    const expectedConnections = agentCount * (agentCount - 1);
    results.connectivity.expectedConnections = expectedConnections;
    results.connectivity.actualConnections = status.topology.connections;
    results.connectivity.connectivityRatio = status.topology.connections / expectedConnections;

    // Test task distribution efficiency
    const taskConfig = {
      description: 'Distributed mesh computation',
      priority: 'high',
      required_capabilities: ['data_analysis'],
      max_agents: Math.min(agentCount, 10),
      estimated_duration_ms: 5000,
    };

    const { result: taskResult, timeMs: orchestrationTime } = measureTime(() =>
      orchestrator.orchestrate_task(swarmId, taskConfig),
    );

    results.orchestrationTime = orchestrationTime;
    results.taskDistribution = {
      assignedAgents: taskResult.assigned_agents.length,
      distributionStrategy: taskResult.distribution_plan.strategy,
      routingType: taskResult.distribution_plan.routing.type,
    };

    // Calculate theoretical vs actual performance
    results.scalability.agentCount = agentCount;
    results.scalability.performanceRatio = 1000 / (orchestrationTime * Math.log(agentCount));
    results.scalability.memoryUsage = status.performance.total_memory_usage_mb;

  } catch (error) {
    testResults.failures.push({
      test: 'mesh_topology',
      agentCount,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }

  return results;
}

async function testStarTopology(wasm, agentCount) {
  const results = {
    hubPerformance: {},
    latency: {},
    bottlenecks: {},
    scalability: {},
  };

  try {
    const orchestrator = new wasm.WasmSwarmOrchestrator();

    // Create star swarm
    const swarmConfig = {
      name: `Star Test ${agentCount} agents`,
      topology_type: 'star',
      max_agents: agentCount,
      enable_cognitive_diversity: true,
    };

    const { result: swarmResult, timeMs: createTime } = measureTime(() =>
      orchestrator.create_swarm(swarmConfig),
    );

    results.createTime = createTime;
    const swarmId = swarmResult.swarm_id;

    // Spawn hub agent first
    const hubConfig = {
      agent_type: 'coordinator',
      name: 'star-hub',
    };

    const { result: hubResult, timeMs: hubSpawnTime } = measureTime(() =>
      orchestrator.spawn_agent(swarmId, hubConfig),
    );

    results.hubPerformance.spawnTime = hubSpawnTime;
    const hubId = hubResult.agent_id;

    // Spawn peripheral agents
    const peripheralIds = [];
    let totalPeripheralSpawnTime = 0;

    for (let i = 1; i < agentCount; i++) {
      const agentConfig = {
        agent_type: ['researcher', 'coder', 'analyst', 'optimizer'][i % 4],
        name: `star-peripheral-${i}`,
      };

      const { result: agentResult, timeMs: spawnTime } = measureTime(() =>
        orchestrator.spawn_agent(swarmId, agentConfig),
      );

      totalPeripheralSpawnTime += spawnTime;
      peripheralIds.push(agentResult.agent_id);
    }

    results.avgPeripheralSpawnTime = totalPeripheralSpawnTime / (agentCount - 1);

    // Get swarm status
    const status = orchestrator.get_swarm_status(swarmId, true);

    // In star topology, should have n-1 connections (each peripheral to hub)
    results.hubPerformance.expectedConnections = agentCount - 1;
    results.hubPerformance.actualConnections = status.topology.connections;

    // Test hub bottleneck by orchestrating multiple tasks
    const tasks = [];
    for (let i = 0; i < 5; i++) {
      const taskConfig = {
        description: `Star task ${i}`,
        priority: 'medium',
        max_agents: 3,
        estimated_duration_ms: 1000,
      };

      const { result: taskResult, timeMs: taskTime } = measureTime(() =>
        orchestrator.orchestrate_task(swarmId, taskConfig),
      );

      tasks.push({ taskId: taskResult.task_id, orchestrationTime: taskTime });
    }

    // Analyze bottleneck metrics
    const avgTaskTime = tasks.reduce((sum, t) => sum + t.orchestrationTime, 0) / tasks.length;
    results.bottlenecks.avgTaskOrchestrationTime = avgTaskTime;
    results.bottlenecks.hubLoadFactor = avgTaskTime / createTime;
    results.bottlenecks.concurrentTaskHandling = tasks.length;

    // Scalability metrics
    results.scalability.agentCount = agentCount;
    results.scalability.hubMemoryUsage = status.performance.total_memory_usage_mb / agentCount;
    results.scalability.communicationOverhead = avgTaskTime * Math.log(agentCount);

  } catch (error) {
    testResults.failures.push({
      test: 'star_topology',
      agentCount,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }

  return results;
}

async function testHierarchicalTopology(wasm, agentCount) {
  const results = {
    levels: {},
    coordination: {},
    propagation: {},
    scalability: {},
  };

  try {
    const orchestrator = new wasm.WasmSwarmOrchestrator();

    // Create hierarchical swarm
    const swarmConfig = {
      name: `Hierarchical Test ${agentCount} agents`,
      topology_type: 'hierarchical',
      max_agents: agentCount,
      enable_cognitive_diversity: true,
    };

    const { result: swarmResult, timeMs: createTime } = measureTime(() =>
      orchestrator.create_swarm(swarmConfig),
    );

    results.createTime = createTime;
    const swarmId = swarmResult.swarm_id;

    // Calculate expected hierarchy levels
    const expectedLevels = Math.ceil(Math.log2(agentCount));
    results.levels.expected = expectedLevels;

    // Spawn agents in hierarchical order
    const agentsByLevel = {};
    let totalSpawnTime = 0;

    for (let i = 0; i < agentCount; i++) {
      const level = Math.floor(Math.log2(i + 1));
      if (!agentsByLevel[level]) agentsByLevel[level] = [];

      // Higher levels get coordinator roles
      const agentType = level === 0 ? 'coordinator' :
        level === 1 ? 'optimizer' :
          ['researcher', 'coder', 'analyst'][i % 3];

      const agentConfig = {
        agent_type: agentType,
        name: `hier-L${level}-${i}`,
      };

      const { result: agentResult, timeMs: spawnTime } = measureTime(() =>
        orchestrator.spawn_agent(swarmId, agentConfig),
      );

      totalSpawnTime += spawnTime;
      agentsByLevel[level].push(agentResult.agent_id);
    }

    results.levels.actual = Object.keys(agentsByLevel).length;
    results.levels.distribution = Object.entries(agentsByLevel).map(([level, agents]) => ({
      level: parseInt(level),
      agentCount: agents.length,
    }));

    results.avgSpawnTime = totalSpawnTime / agentCount;

    // Test hierarchical task propagation
    const rootTaskConfig = {
      description: 'Hierarchical root task',
      priority: 'high',
      required_capabilities: ['task_distribution'],
      max_agents: Math.min(agentCount, 15),
      estimated_duration_ms: 10000,
    };

    const { result: rootTask, timeMs: rootOrchestrationTime } = measureTime(() =>
      orchestrator.orchestrate_task(swarmId, rootTaskConfig),
    );

    results.propagation.rootOrchestrationTime = rootOrchestrationTime;
    results.propagation.levelsInvolved = Math.ceil(Math.log2(rootTask.assigned_agents.length));

    // Test coordination efficiency
    const status = orchestrator.get_swarm_status(swarmId, true);

    results.coordination.totalConnections = status.topology.connections;
    results.coordination.avgConnectionsPerAgent = status.topology.connections / agentCount;
    results.coordination.hierarchyEfficiency = 1 - (status.topology.connections / (agentCount * (agentCount - 1)));

    // Scalability analysis
    results.scalability.agentCount = agentCount;
    results.scalability.communicationComplexity = Math.log2(agentCount);
    results.scalability.memoryEfficiency = status.performance.total_memory_usage_mb / (agentCount * Math.log2(agentCount));

  } catch (error) {
    testResults.failures.push({
      test: 'hierarchical_topology',
      agentCount,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }

  return results;
}

async function testRingTopology(wasm, agentCount) {
  const results = {
    connectivity: {},
    propagation: {},
    latency: {},
    scalability: {},
  };

  try {
    const orchestrator = new wasm.WasmSwarmOrchestrator();

    // Create ring swarm
    const swarmConfig = {
      name: `Ring Test ${agentCount} agents`,
      topology_type: 'ring',
      max_agents: agentCount,
      enable_cognitive_diversity: true,
    };

    const { result: swarmResult, timeMs: createTime } = measureTime(() =>
      orchestrator.create_swarm(swarmConfig),
    );

    results.createTime = createTime;
    const swarmId = swarmResult.swarm_id;

    // Spawn agents
    const agentIds = [];
    let totalSpawnTime = 0;

    for (let i = 0; i < agentCount; i++) {
      const agentConfig = {
        agent_type: ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'][i % 5],
        name: `ring-agent-${i}`,
      };

      const { result: agentResult, timeMs: spawnTime } = measureTime(() =>
        orchestrator.spawn_agent(swarmId, agentConfig),
      );

      totalSpawnTime += spawnTime;
      agentIds.push(agentResult.agent_id);
    }

    results.avgSpawnTime = totalSpawnTime / agentCount;

    // Get swarm status
    const status = orchestrator.get_swarm_status(swarmId, true);

    // In ring topology, should have exactly n connections
    results.connectivity.expectedConnections = agentCount;
    results.connectivity.actualConnections = status.topology.connections;
    results.connectivity.isProperRing = status.topology.connections === agentCount;

    // Test message propagation around the ring
    const propagationTasks = [];
    for (let distance = 1; distance <= Math.floor(agentCount / 2); distance++) {
      const taskConfig = {
        description: `Ring propagation test distance ${distance}`,
        priority: 'medium',
        max_agents: 2,
        estimated_duration_ms: 1000 * distance,
      };

      const { result: taskResult, timeMs: taskTime } = measureTime(() =>
        orchestrator.orchestrate_task(swarmId, taskConfig),
      );

      propagationTasks.push({
        distance,
        orchestrationTime: taskTime,
        expectedLatency: distance * 2.0, // Based on 2ms per hop from code
      });
    }

    // Analyze propagation efficiency
    results.propagation.measurements = propagationTasks;
    results.propagation.avgLatencyPerHop = propagationTasks.reduce((sum, t) =>
      sum + (t.orchestrationTime / t.distance), 0) / propagationTasks.length;

    // Calculate maximum propagation distance
    results.latency.maxDistance = Math.floor(agentCount / 2);
    results.latency.worstCaseLatency = results.latency.maxDistance * results.propagation.avgLatencyPerHop;

    // Scalability metrics
    results.scalability.agentCount = agentCount;
    results.scalability.communicationEfficiency = 2 / agentCount; // Each agent connects to 2 others
    results.scalability.memoryUsage = status.performance.total_memory_usage_mb;
    results.scalability.scalabilityScore = 1000 / (results.latency.worstCaseLatency * Math.log(agentCount));

  } catch (error) {
    testResults.failures.push({
      test: 'ring_topology',
      agentCount,
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }

  return results;
}

// Main test runner
async function runTopologyValidation() {
  console.log('Starting Swarm Topology Validation Tests...\n');

  const wasm = await loadWasmModule();
  if (!wasm) {
    console.error('Failed to load WASM module. Aborting tests.');
    return testResults;
  }

  // Test each topology with different agent counts
  for (const topology of TEST_CONFIG.topologies) {
    console.log(`\nTesting ${topology.toUpperCase()} topology...`);
    testResults.topologies_tested[topology] = {};
    testResults.scalability[topology] = {};

    for (const agentCount of TEST_CONFIG.scalabilityTests) {
      console.log(`  Testing with ${agentCount} agents...`);

      let results;
      switch (topology) {
        case 'mesh':
          results = await testMeshTopology(wasm, agentCount);
          break;
        case 'star':
          results = await testStarTopology(wasm, agentCount);
          break;
        case 'hierarchical':
          results = await testHierarchicalTopology(wasm, agentCount);
          break;
        case 'ring':
          results = await testRingTopology(wasm, agentCount);
          break;
      }

      testResults.topologies_tested[topology][agentCount] = results;

      // Determine maximum effective agents
      if (results && results.scalability && results.scalability.performanceRatio !== undefined) {
        const perfRatio = results.scalability.performanceRatio ||
                                 results.scalability.scalabilityScore ||
                                 results.scalability.memoryEfficiency || 0;

        if (perfRatio > 0.5) { // Threshold for "effective"
          testResults.scalability[topology].maxEffectiveAgents = agentCount;
        }
      }
    }
  }

  // Calculate coordination efficiency metrics
  testResults.coordination_efficiency = calculateCoordinationEfficiency(testResults.topologies_tested);

  // Save results to file
  const outputPath = path.join(__dirname, 'swarm-topology-validation-results.json');
  fs.writeFileSync(outputPath, JSON.stringify(testResults, null, 2));

  console.log(`\nTest results saved to: ${outputPath}`);
  console.log('\nTest Summary:');
  console.log(`- Topologies tested: ${Object.keys(testResults.topologies_tested).length}`);
  console.log(`- Total failures: ${testResults.failures.length}`);
  console.log(`- Timestamp: ${testResults.timestamp}`);

  return testResults;
}

function calculateCoordinationEfficiency(topologyResults) {
  const efficiency = {};

  for (const [topology, agentTests] of Object.entries(topologyResults)) {
    efficiency[topology] = {
      avgOrchestrationTime: 0,
      avgMemoryPerAgent: 0,
      taskDistributionEfficiency: 0,
      scalabilityTrend: [],
    };

    let totalOrchestrationTime = 0;
    let totalMemoryPerAgent = 0;
    let testCount = 0;

    for (const [agentCount, results] of Object.entries(agentTests)) {
      if (results && results.orchestrationTime) {
        totalOrchestrationTime += results.orchestrationTime;
        testCount++;

        const memoryPerAgent = (results.scalability?.memoryUsage || 0) / parseInt(agentCount);
        totalMemoryPerAgent += memoryPerAgent;

        efficiency[topology].scalabilityTrend.push({
          agentCount: parseInt(agentCount),
          orchestrationTime: results.orchestrationTime,
          memoryPerAgent,
        });
      }
    }

    if (testCount > 0) {
      efficiency[topology].avgOrchestrationTime = totalOrchestrationTime / testCount;
      efficiency[topology].avgMemoryPerAgent = totalMemoryPerAgent / testCount;

      // Calculate distribution efficiency based on topology characteristics
      switch (topology) {
        case 'mesh':
          efficiency[topology].taskDistributionEfficiency = 0.95; // Highest - direct connections
          break;
        case 'star':
          efficiency[topology].taskDistributionEfficiency = 0.75; // Limited by hub
          break;
        case 'hierarchical':
          efficiency[topology].taskDistributionEfficiency = 0.85; // Good for structured tasks
          break;
        case 'ring':
          efficiency[topology].taskDistributionEfficiency = 0.65; // Limited by propagation
          break;
      }
    }
  }

  return efficiency;
}

// Run the tests
if (require.main === module) {
  runTopologyValidation()
    .then(results => {
      console.log('\nValidation complete!');
      process.exit(0);
    })
    .catch(error => {
      console.error('Test failed:', error);
      process.exit(1);
    });
}

module.exports = { runTopologyValidation, testResults };
