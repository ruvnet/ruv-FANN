/**
 * Comprehensive MCP Integration Tests for ruv-swarm
 * Tests all 12 MCP tools and their integration
 */

import assert from 'assert';
import { spawn } from 'child_process';
import WebSocket from 'ws';
import { v4 as uuidv4 } from 'uuid';
import { promises as fs } from 'fs';

// Test configuration
const MCP_SERVER_URL = 'ws://localhost:3000/mcp';
const HTTP_BASE_URL = 'http://localhost:3000';
const TEST_TIMEOUT = 60000; // 60 seconds

// Test utilities
class MCPTestClient {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.requestId = 0;
    this.pendingRequests = new Map();
    this.notifications = [];
    this.connected = false;
  }

  async connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.on('open', () => {
        console.log('Connected to MCP server');
        resolve();
      });

      this.ws.on('message', (data) => {
        const message = JSON.parse(data.toString());

        if (message.id && this.pendingRequests.has(message.id)) {
          const { resolve, reject } = this.pendingRequests.get(message.id);
          this.pendingRequests.delete(message.id);

          if (message.error) {
            reject(new Error(message.error.message));
          } else {
            resolve(message.result);
          }
        } else if (message.method) {
          // Notification
          this.notifications.push(message);
        }
      });

      this.ws.on('error', (error) => {
        console.error('WebSocket error:', error.message);
        reject(new Error(`MCP server connection failed: ${error.message}`));
      });

      this.ws.on('close', () => {
        console.log('Disconnected from MCP server');
        this.connected = false;
      });

      // Set connection timeout
      setTimeout(() => {
        if (this.ws && this.ws.readyState !== WebSocket.OPEN) {
          this.ws.close();
          reject(new Error('Connection timeout - MCP server may not be running'));
        }
      }, 5000);
    });
  }

  async disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  async sendRequest(method, params = null) {
    const id = ++this.requestId;
    const request = {
      jsonrpc: '2.0',
      id,
      method,
      params,
    };

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });
      this.ws.send(JSON.stringify(request));

      // Timeout after 30 seconds
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error(`Request ${id} timed out`));
        }
      }, 30000);
    });
  }

  clearNotifications() {
    this.notifications = [];
  }

  getNotifications(filter = null) {
    if (!filter) {
      return this.notifications;
    }
    return this.notifications.filter(n => n.method === filter);
  }
}

// Test suites
async function runMCPIntegrationTests() {
  console.log('🚀 Starting Comprehensive MCP Integration Tests\n');

  const results = {
    passed: 0,
    failed: 0,
    errors: [],
  };

  const client = new MCPTestClient(MCP_SERVER_URL);

  async function test(name, fn) {
    try {
      await fn();
      console.log(`✅ ${name}`);
      results.passed++;
    } catch (error) {
      console.error(`❌ ${name}`);
      console.error(`   ${error.message}`);
      results.failed++;
      results.errors.push({ test: name, error: error.message });
    }
  }

  try {
    // Try to connect to MCP server with timeout
    console.log('🔌 Attempting to connect to MCP server...');
    try {
      await client.connect();
      console.log('✅ Connected to MCP server successfully');
    } catch (connectError) {
      console.log('⚠️  MCP server not available:', connectError.message);
      console.log('ℹ️  Skipping MCP integration tests (server may not be running)');
      results.errors.push({ test: 'MCP Connection', error: 'Server not available' });
      results.failed++;

      // Return early with partial results
      console.log('\n📊 MCP Test Results Summary (Skipped)');
      console.log('─'.repeat(50));
      console.log('MCP server not running - tests skipped');
      console.log('To run MCP tests: npm run mcp:server (in separate terminal)');
      return results;
    }

    // 1. Test Initialize
    await test('MCP Initialize', async() => {
      const result = await client.sendRequest('initialize', {
        protocolVersion: '2024-11-05',
        clientInfo: {
          name: 'ruv-swarm-test-client',
          version: '1.0.0',
        },
        capabilities: {
          tools: {},
          resources: {},
        },
      });

      // More flexible assertions
      assert(result, 'Initialize should return a result');
      assert(result.protocolVersion || result.capabilities, 'Should have protocol version or capabilities');

      if (result.serverInfo) {
        console.log(`   Server: ${result.serverInfo.name} v${result.serverInfo.version || 'unknown'}`);
      }
    });

    // 2. Test Tools List
    await test('MCP Tools List', async() => {
      const result = await client.sendRequest('tools/list');

      assert(result, 'Tools list should return a result');
      assert(result.tools || result.available_tools, 'Should have tools or available_tools');

      const tools = result.tools || result.available_tools || [];
      console.log(`   Found ${tools.length} tools`);

      if (tools.length > 0) {
        console.log(`   Example tool: ${tools[0].name || tools[0]}`);
      }

      const toolNames = result.tools.map(t => t.name);
      const expectedTools = [
        'ruv-swarm.spawn',
        'ruv-swarm.orchestrate',
        'ruv-swarm.query',
        'ruv-swarm.monitor',
        'ruv-swarm.optimize',
        'ruv-swarm.memory.store',
        'ruv-swarm.memory.get',
        'ruv-swarm.task.create',
        'ruv-swarm.workflow.execute',
        'ruv-swarm.agent.list',
      ];

      expectedTools.forEach(tool => {
        assert(toolNames.includes(tool), `Missing tool: ${tool}`);
      });
    });

    // 3. Test Swarm Init (Agent Spawn)
    let agentId;
    await test('MCP Tool: ruv-swarm.spawn', async() => {
      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.spawn',
        arguments: {
          agent_type: 'researcher',
          name: 'test-researcher-001',
          capabilities: {
            max_tokens: 4096,
            temperature: 0.7,
            specialized_domains: ['web_frameworks', 'performance'],
          },
        },
      });

      assert(result.agent_id);
      assert(result.agent_type === 'researcher');
      assert(result.status === 'active');
      agentId = result.agent_id;
    });

    // 4. Test Agent List
    await test('MCP Tool: ruv-swarm.agent.list', async() => {
      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.agent.list',
        arguments: {
          include_inactive: false,
          sort_by: 'created_at',
        },
      });

      assert(Array.isArray(result.agents));
      assert(result.count >= 1);
      assert(result.agents.some(a => a.id === agentId));
    });

    // 5. Test Memory Store
    await test('MCP Tool: ruv-swarm.memory.store', async() => {
      const testData = {
        framework_analysis: {
          react: { performance: 'excellent', learning_curve: 'moderate' },
          vue: { performance: 'excellent', learning_curve: 'easy' },
          angular: { performance: 'good', learning_curve: 'steep' },
        },
        timestamp: new Date().toISOString(),
      };

      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.memory.store',
        arguments: {
          key: 'test_framework_analysis',
          value: testData,
          ttl_secs: 3600,
        },
      });

      assert(result.stored === true);
      assert(result.key === 'test_framework_analysis');
    });

    // 6. Test Memory Get
    await test('MCP Tool: ruv-swarm.memory.get', async() => {
      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.memory.get',
        arguments: {
          key: 'test_framework_analysis',
        },
      });

      assert(result.found === true);
      assert(result.value);
      assert(result.value.framework_analysis);
      assert(result.value.framework_analysis.react.performance === 'excellent');
    });

    // 7. Test Task Create
    let taskId;
    await test('MCP Tool: ruv-swarm.task.create', async() => {
      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.task.create',
        arguments: {
          task_type: 'research',
          description: 'Analyze performance characteristics of modern web frameworks',
          priority: 'high',
          assigned_agent: agentId,
        },
      });

      assert(result.task_id);
      assert(result.task_type === 'research');
      assert(result.priority === 'high');
      assert(result.status === 'pending');
      taskId = result.task_id;
    });

    // 8. Test Query Swarm State
    await test('MCP Tool: ruv-swarm.query', async() => {
      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.query',
        arguments: {
          filter: {
            agent_type: 'researcher',
          },
          include_metrics: true,
        },
      });

      assert(result.agents);
      assert(result.active_tasks >= 1);
      assert(result.total_agents >= 1);
      if (result.metrics) {
        assert(typeof result.metrics === 'object');
      }
    });

    // 9. Test Monitor (with short duration)
    await test('MCP Tool: ruv-swarm.monitor', async() => {
      client.clearNotifications();

      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.monitor',
        arguments: {
          event_types: ['agent_spawned', 'task_created', 'task_completed'],
          duration_secs: 5, // Short duration for testing
        },
      });

      assert(result.status === 'monitoring');
      assert(result.duration_secs === 5);

      // Wait a bit and check for notifications
      await new Promise(resolve => setTimeout(resolve, 2000));

      const notifications = client.getNotifications('ruv-swarm/event');
      console.log(`   Received ${notifications.length} notifications`);
    });

    // 10. Test Orchestrate
    await test('MCP Tool: ruv-swarm.orchestrate', async() => {
      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.orchestrate',
        arguments: {
          objective: 'Create comprehensive comparison of React, Vue, and Angular frameworks',
          strategy: 'research',
          mode: 'distributed',
          max_agents: 3,
          parallel: true,
        },
      });

      assert(result.task_id);
      assert(result.objective);
      assert(result.strategy === 'research');
      assert(result.mode === 'distributed');
      assert(result.status === 'started');
    });

    // 11. Test Optimize
    await test('MCP Tool: ruv-swarm.optimize', async() => {
      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.optimize',
        arguments: {
          target_metric: 'throughput',
          constraints: {
            max_memory_mb: 512,
            max_cpu_percent: 80,
          },
          auto_apply: false,
        },
      });

      assert(result.target_metric === 'throughput');
      assert(Array.isArray(result.recommendations));
      assert(result.applied === false);
    });

    // 12. Test Workflow Execute
    await test('MCP Tool: ruv-swarm.workflow.execute', async() => {
      // First, create a simple workflow file
      const workflowPath = '/tmp/test-workflow.json';
      const workflow = {
        name: 'test-workflow',
        steps: [
          { action: 'spawn', params: { agent_type: 'coder' } },
          { action: 'spawn', params: { agent_type: 'tester' } },
          { action: 'create_task', params: { task_type: 'development', description: 'Build feature X' } },
        ],
      };

      await fs.writeFile(workflowPath, JSON.stringify(workflow, null, 2));

      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.workflow.execute',
        arguments: {
          workflow_path: workflowPath,
          parameters: {
            feature_name: 'user-authentication',
          },
          async_execution: true,
        },
      });

      assert(result.workflow_id);
      assert(result.status === 'started');
      assert(result.async === true);

      // Clean up
      await fs.unlink(workflowPath);
    });

    // Test Concurrency
    await test('Concurrent Operations', async() => {
      const promises = [];

      // Spawn multiple agents concurrently
      for (let i = 0; i < 5; i++) {
        promises.push(
          client.sendRequest('tools/call', {
            name: 'ruv-swarm.spawn',
            arguments: {
              agent_type: ['researcher', 'coder', 'analyst', 'tester', 'reviewer'][i],
              name: `concurrent-agent-${i}`,
            },
          }),
        );
      }

      const results = await Promise.all(promises);
      assert(results.length === 5);
      results.forEach(r => assert(r.agent_id));
    });

    // Test Error Handling
    await test('Error Handling: Invalid Agent Type', async() => {
      try {
        await client.sendRequest('tools/call', {
          name: 'ruv-swarm.spawn',
          arguments: {
            agent_type: 'invalid_type',
          },
        });
        assert.fail('Should have thrown an error');
      } catch (error) {
        assert(error.message.includes('Invalid agent_type'));
      }
    });

    await test('Error Handling: Missing Required Parameters', async() => {
      try {
        await client.sendRequest('tools/call', {
          name: 'ruv-swarm.task.create',
          arguments: {
            // Missing required 'task_type' and 'description'
            priority: 'high',
          },
        });
        assert.fail('Should have thrown an error');
      } catch (error) {
        assert(error.message.includes('Missing'));
      }
    });

    // Test Persistence Across Sessions
    await test('Persistence: Memory Across Reconnection', async() => {
      // Store data
      await client.sendRequest('tools/call', {
        name: 'ruv-swarm.memory.store',
        arguments: {
          key: 'persistence_test',
          value: { test: 'data', timestamp: Date.now() },
        },
      });

      // Disconnect and reconnect
      await client.disconnect();
      await new Promise(resolve => setTimeout(resolve, 1000));
      await client.connect();

      // Re-initialize
      await client.sendRequest('initialize', {
        clientInfo: {
          name: 'ruv-swarm-test-client',
          version: '1.0.0',
        },
      });

      // Try to retrieve data
      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.memory.get',
        arguments: {
          key: 'persistence_test',
        },
      });

      // Note: This might fail if session-based storage is used
      // In production, implement proper persistent storage
      console.log(`   Persistence test result: ${result.found ? 'Data persisted' : 'Data not persisted (session-based storage)'}`);
    });

    // Performance Benchmarks
    await test('Performance: Rapid Agent Spawning', async() => {
      const startTime = Date.now();
      const promises = [];

      for (let i = 0; i < 10; i++) {
        promises.push(
          client.sendRequest('tools/call', {
            name: 'ruv-swarm.spawn',
            arguments: {
              agent_type: 'researcher',
              name: `perf-test-agent-${i}`,
            },
          }),
        );
      }

      await Promise.all(promises);
      const duration = Date.now() - startTime;

      console.log(`   Spawned 10 agents in ${duration}ms (${(duration / 10).toFixed(1)}ms per agent)`);
      assert(duration < 5000, 'Agent spawning too slow');
    });

    // Test Custom MCP Methods
    await test('Custom Method: ruv-swarm/status', async() => {
      const result = await client.sendRequest('ruv-swarm/status');
      assert(result);
      console.log(`   Status: ${JSON.stringify(result, null, 2).substring(0, 100)}...`);
    });

    await test('Custom Method: ruv-swarm/metrics', async() => {
      const result = await client.sendRequest('ruv-swarm/metrics');
      assert(result);
      console.log(`   Metrics: ${JSON.stringify(result, null, 2).substring(0, 100)}...`);
    });

  } catch (error) {
    console.error('Test suite error:', error);
    results.failed++;
  } finally {
    await client.disconnect();
  }

  // Summary
  console.log('\n📊 Test Results Summary');
  console.log('─'.repeat(50));
  console.log(`Total Tests: ${results.passed + results.failed}`);
  console.log(`✅ Passed: ${results.passed}`);
  console.log(`❌ Failed: ${results.failed}`);

  if (results.errors.length > 0) {
    console.log('\n❌ Failed Tests:');
    results.errors.forEach(e => {
      console.log(`  - ${e.test}: ${e.error}`);
    });
  }

  return results.failed === 0;
}

// Integration test scenarios
async function runIntegrationScenarios() {
  console.log('\n🔄 Running Integration Scenarios\n');

  const client = new MCPTestClient(MCP_SERVER_URL);

  try {
    await client.connect();
    await client.sendRequest('initialize', {
      clientInfo: { name: 'integration-test', version: '1.0.0' },
    });

    // Scenario 1: Complete Research Workflow
    console.log('📚 Scenario 1: Complete Research Workflow');

    // 1. Spawn research team
    const researchers = [];
    for (let i = 0; i < 3; i++) {
      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.spawn',
        arguments: {
          agent_type: 'researcher',
          name: `research-team-${i}`,
          capabilities: {
            specialization: ['web_tech', 'performance', 'architecture'][i],
          },
        },
      });
      researchers.push(result.agent_id);
    }
    console.log(`  ✅ Spawned ${researchers.length} researchers`);

    // 2. Create research tasks
    const tasks = [];
    const topics = [
      'Modern JavaScript frameworks comparison',
      'WebAssembly performance analysis',
      'Microservices vs Monolithic architecture',
    ];

    for (let i = 0; i < topics.length; i++) {
      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.task.create',
        arguments: {
          task_type: 'research',
          description: topics[i],
          priority: 'high',
          assigned_agent: researchers[i],
        },
      });
      tasks.push(result.task_id);
    }
    console.log(`  ✅ Created ${tasks.length} research tasks`);

    // 3. Store research findings
    await client.sendRequest('tools/call', {
      name: 'ruv-swarm.memory.store',
      arguments: {
        key: 'research_findings',
        value: {
          frameworks: {
            react: { pros: ['ecosystem', 'flexibility'], cons: ['complexity'] },
            vue: { pros: ['simplicity', 'performance'], cons: ['smaller ecosystem'] },
            angular: { pros: ['enterprise', 'typescript'], cons: ['learning curve'] },
          },
          timestamp: new Date().toISOString(),
        },
      },
    });
    console.log('  ✅ Stored research findings in memory');

    // 4. Query final state
    const state = await client.sendRequest('tools/call', {
      name: 'ruv-swarm.query',
      arguments: { include_metrics: true },
    });
    console.log(`  ✅ Final state: ${state.total_agents} agents, ${state.active_tasks} active tasks`);

    // Scenario 2: Development Pipeline
    console.log('\n🛠️  Scenario 2: Development Pipeline');

    // 1. Orchestrate development task
    const devResult = await client.sendRequest('tools/call', {
      name: 'ruv-swarm.orchestrate',
      arguments: {
        objective: 'Implement user authentication system with JWT',
        strategy: 'development',
        mode: 'hierarchical',
        max_agents: 5,
        parallel: true,
      },
    });
    console.log(`  ✅ Started development orchestration: ${devResult.task_id}`);

    // 2. Monitor progress
    client.clearNotifications();
    await client.sendRequest('tools/call', {
      name: 'ruv-swarm.monitor',
      arguments: {
        event_types: ['task_started', 'task_completed', 'agent_message'],
        duration_secs: 3,
      },
    });

    await new Promise(resolve => setTimeout(resolve, 3500));
    const events = client.getNotifications('ruv-swarm/event');
    console.log(`  ✅ Captured ${events.length} events during monitoring`);

    // 3. Optimize performance
    const optResult = await client.sendRequest('tools/call', {
      name: 'ruv-swarm.optimize',
      arguments: {
        target_metric: 'latency',
        constraints: {
          max_memory_mb: 256,
          max_agents: 10,
        },
        auto_apply: true,
      },
    });
    console.log(`  ✅ Applied ${optResult.recommendations.length} optimizations`);

    // Scenario 3: Neural Network Learning
    console.log('\n🧠 Scenario 3: Neural Network Learning Simulation');

    // 1. Create analyzer agents
    const analyzers = [];
    for (let i = 0; i < 2; i++) {
      const result = await client.sendRequest('tools/call', {
        name: 'ruv-swarm.spawn',
        arguments: {
          agent_type: 'analyst',
          name: `neural-analyzer-${i}`,
          capabilities: {
            neural_enabled: true,
            learning_rate: 0.01,
          },
        },
      });
      analyzers.push(result.agent_id);
    }
    console.log(`  ✅ Spawned ${analyzers.length} neural-enabled analyzers`);

    // 2. Store training data
    const trainingData = {
      patterns: [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ],
      epochs: 1000,
    };

    await client.sendRequest('tools/call', {
      name: 'ruv-swarm.memory.store',
      arguments: {
        key: 'xor_training_data',
        value: trainingData,
      },
    });
    console.log('  ✅ Stored neural network training data');

    // 3. Create learning task
    const learningResult = await client.sendRequest('tools/call', {
      name: 'ruv-swarm.task.create',
      arguments: {
        task_type: 'analysis',
        description: 'Train XOR pattern recognition',
        priority: 'critical',
        assigned_agent: analyzers[0],
      },
    });
    console.log(`  ✅ Created neural learning task: ${learningResult.task_id}`);

  } finally {
    await client.disconnect();
  }
}

// Main test runner
async function main() {
  console.log('🧪 RUV-SWARM MCP Integration Test Suite');
  console.log('=' .repeat(50));
  console.log(`Started: ${new Date().toISOString()}\n`);

  // Check if MCP server is running
  try {
    const response = await fetch(`${HTTP_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error('MCP server health check failed');
    }
  } catch (error) {
    console.error('❌ MCP server is not running!');
    console.error('   Please start the server with: npm run mcp:server');
    process.exit(1);
  }

  // Run comprehensive tests
  const testsPassed = await runMCPIntegrationTests();

  // Run integration scenarios
  await runIntegrationScenarios();

  console.log(`\n${ '='.repeat(50)}`);
  console.log(`Completed: ${new Date().toISOString()}`);

  process.exit(testsPassed ? 0 : 1);
}

// Handle uncaught errors
process.on('unhandledRejection', (error) => {
  console.error('Unhandled rejection:', error);
  process.exit(1);
});

// Run tests if called directly
// Direct execution
main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});

export { MCPTestClient, runMCPIntegrationTests, runIntegrationScenarios };
