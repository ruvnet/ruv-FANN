#!/usr/bin/env node
/**
 * Comprehensive MCP Tools Test Suite
 * Tests all 12 MCP tools with a 5-agent swarm
 */

import { spawn  } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Colors for output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

// Execute MCP tool
async function executeMcpTool(toolName, args = {}) {
  return new Promise((resolve, reject) => {
    const request = {
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: toolName,
        arguments: args,
      },
      id: Date.now(),
    };

    const mcpPath = path.join(__dirname, '..', 'mcp-server.sh');
    const mcp = spawn(mcpPath, [], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let output = '';
    let error = '';

    mcp.stdout.on('data', (data) => {
      output += data.toString();
    });

    mcp.stderr.on('data', (data) => {
      error += data.toString();
    });

    mcp.on('close', (code) => {
      try {
        const lines = output.trim().split('\n');
        for (const line of lines) {
          if (line.trim()) {
            const response = JSON.parse(line);
            if (response.id === request.id) {
              if (response.error) {
                reject(new Error(response.error.message));
              } else {
                resolve(response.result);
              }
              return;
            }
          }
        }
        reject(new Error('No valid response received'));
      } catch (err) {
        reject(err);
      }
    });

    // Send request
    mcp.stdin.write(`${JSON.stringify(request) }\n`);
    mcp.stdin.end();
  });
}

// Test all MCP tools
async function testAllMcpTools() {
  log('\n🚀 Starting Comprehensive MCP Tools Test Suite', 'cyan');
  log('=' .repeat(60), 'cyan');

  const results = {
    passed: 0,
    failed: 0,
    tools: {},
  };

  // Test 1: Initialize Swarm
  log('\n1. Testing swarm_init...', 'blue');
  try {
    const result = await executeMcpTool('swarm_init', {
      topology: 'mesh',
      maxAgents: 10,
      strategy: 'balanced',
    });
    log('✅ swarm_init: PASSED', 'green');
    log(`   Swarm ID: ${result.content[0].text}`, 'green');
    results.passed++;
    results.tools.swarm_init = { status: 'passed', result };
  } catch (error) {
    log('❌ swarm_init: FAILED', 'red');
    log(`   Error: ${error.message}`, 'red');
    results.failed++;
    results.tools.swarm_init = { status: 'failed', error: error.message };
  }

  // Test 2: Create 5 Agents in Parallel
  log('\n2. Creating 5-agent swarm in parallel...', 'blue');
  const agentTypes = ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'];
  const agentPromises = agentTypes.map((type, index) =>
    executeMcpTool('agent_spawn', {
      type,
      name: `agent-${type}-${index + 1}`,
      capabilities: [`capability-${index + 1}`, `capability-${index + 2}`],
    }),
  );

  try {
    const agents = await Promise.all(agentPromises);
    log('✅ Created 5 agents in parallel: PASSED', 'green');
    agents.forEach((agent, index) => {
      log(`   Agent ${index + 1}: ${agentTypes[index]}`, 'green');
    });
    results.passed++;
    results.tools.agent_spawn = { status: 'passed', agents };
  } catch (error) {
    log('❌ agent_spawn (parallel): FAILED', 'red');
    log(`   Error: ${error.message}`, 'red');
    results.failed++;
    results.tools.agent_spawn = { status: 'failed', error: error.message };
  }

  // Test 3: Swarm Status
  log('\n3. Testing swarm_status...', 'blue');
  try {
    const result = await executeMcpTool('swarm_status', { verbose: true });
    log('✅ swarm_status: PASSED', 'green');
    const text = result.content[0].text;
    log(`   ${text.split('\n')[0]}`, 'green');
    results.passed++;
    results.tools.swarm_status = { status: 'passed', result };
  } catch (error) {
    log('❌ swarm_status: FAILED', 'red');
    log(`   Error: ${error.message}`, 'red');
    results.failed++;
    results.tools.swarm_status = { status: 'failed', error: error.message };
  }

  // Test 4: Agent List
  log('\n4. Testing agent_list...', 'blue');
  try {
    const result = await executeMcpTool('agent_list', { filter: 'all' });
    log('✅ agent_list: PASSED', 'green');
    results.passed++;
    results.tools.agent_list = { status: 'passed', result };
  } catch (error) {
    log('❌ agent_list: FAILED', 'red');
    log(`   Error: ${error.message}`, 'red');
    results.failed++;
    results.tools.agent_list = { status: 'failed', error: error.message };
  }

  // Test 5: Task Orchestration
  log('\n5. Testing task_orchestrate...', 'blue');
  try {
    const result = await executeMcpTool('task_orchestrate', {
      task: 'Analyze system performance and generate optimization report',
      priority: 'high',
      strategy: 'adaptive',
      maxAgents: 3,
    });
    log('✅ task_orchestrate: PASSED', 'green');
    log(`   ${result.content[0].text.split('\n')[0]}`, 'green');
    results.passed++;
    results.tools.task_orchestrate = { status: 'passed', result };
  } catch (error) {
    log('❌ task_orchestrate: FAILED', 'red');
    log(`   Error: ${error.message}`, 'red');
    results.failed++;
    results.tools.task_orchestrate = { status: 'failed', error: error.message };
  }

  // Test 6: Task Status
  log('\n6. Testing task_status...', 'blue');
  try {
    const result = await executeMcpTool('task_status', { detailed: true });
    log('✅ task_status: PASSED', 'green');
    results.passed++;
    results.tools.task_status = { status: 'passed', result };
  } catch (error) {
    log('❌ task_status: FAILED', 'red');
    log(`   Error: ${error.message}`, 'red');
    results.failed++;
    results.tools.task_status = { status: 'failed', error: error.message };
  }

  // Test 7: Agent Metrics
  log('\n7. Testing agent_metrics...', 'blue');
  try {
    const result = await executeMcpTool('agent_metrics', { metric: 'all' });
    log('✅ agent_metrics: PASSED', 'green');
    results.passed++;
    results.tools.agent_metrics = { status: 'passed', result };
  } catch (error) {
    log('❌ agent_metrics: FAILED', 'red');
    log(`   Error: ${error.message}`, 'red');
    results.failed++;
    results.tools.agent_metrics = { status: 'failed', error: error.message };
  }

  // Test 8: Memory Usage
  log('\n8. Testing memory_usage...', 'blue');
  try {
    const result = await executeMcpTool('memory_usage', { detail: 'detailed' });
    log('✅ memory_usage: PASSED', 'green');
    log(`   ${result.content[0].text.split('\n')[0]}`, 'green');
    results.passed++;
    results.tools.memory_usage = { status: 'passed', result };
  } catch (error) {
    log('❌ memory_usage: FAILED', 'red');
    log(`   Error: ${error.message}`, 'red');
    results.failed++;
    results.tools.memory_usage = { status: 'failed', error: error.message };
  }

  // Test 9: Features Detect
  log('\n9. Testing features_detect...', 'blue');
  try {
    const result = await executeMcpTool('features_detect', { category: 'all' });
    log('✅ features_detect: PASSED', 'green');
    const features = result.content[0].text.split('\n').slice(0, 3).join(', ');
    log(`   ${features}`, 'green');
    results.passed++;
    results.tools.features_detect = { status: 'passed', result };
  } catch (error) {
    log('❌ features_detect: FAILED', 'red');
    log(`   Error: ${error.message}`, 'red');
    results.failed++;
    results.tools.features_detect = { status: 'failed', error: error.message };
  }

  // Test 10: Benchmark Run
  log('\n10. Testing benchmark_run...', 'blue');
  try {
    const result = await executeMcpTool('benchmark_run', {
      type: 'wasm',
      iterations: 5,
    });
    log('✅ benchmark_run: PASSED', 'green');
    results.passed++;
    results.tools.benchmark_run = { status: 'passed', result };
  } catch (error) {
    log('❌ benchmark_run: FAILED', 'red');
    log(`   Error: ${error.message}`, 'red');
    results.failed++;
    results.tools.benchmark_run = { status: 'failed', error: error.message };
  }

  // Test 11: Swarm Monitor (short duration)
  log('\n11. Testing swarm_monitor...', 'blue');
  try {
    const result = await executeMcpTool('swarm_monitor', {
      duration: 2,
      interval: 1,
    });
    log('✅ swarm_monitor: PASSED', 'green');
    results.passed++;
    results.tools.swarm_monitor = { status: 'passed', result };
  } catch (error) {
    log('❌ swarm_monitor: FAILED', 'red');
    log(`   Error: ${error.message}`, 'red');
    results.failed++;
    results.tools.swarm_monitor = { status: 'failed', error: error.message };
  }

  // Test 12: Task Results (if we have a task ID)
  log('\n12. Testing task_results...', 'blue');
  try {
    // First get a task ID from task_status
    const statusResult = await executeMcpTool('task_status', {});
    const taskIdMatch = statusResult.content[0].text.match(/task_[\w]+/);

    if (taskIdMatch) {
      const taskId = taskIdMatch[0];
      const result = await executeMcpTool('task_results', {
        taskId,
        format: 'detailed',
      });
      log('✅ task_results: PASSED', 'green');
      results.passed++;
      results.tools.task_results = { status: 'passed', result };
    } else {
      log('⚠️  task_results: SKIPPED (no tasks found)', 'yellow');
      results.tools.task_results = { status: 'skipped', reason: 'No tasks available' };
    }
  } catch (error) {
    log('❌ task_results: FAILED', 'red');
    log(`   Error: ${error.message}`, 'red');
    results.failed++;
    results.tools.task_results = { status: 'failed', error: error.message };
  }

  // Summary
  log(`\n${ '=' .repeat(60)}`, 'cyan');
  log('📊 Test Summary', 'cyan');
  log('=' .repeat(60), 'cyan');
  log(`Total Tests: ${results.passed + results.failed}`, 'blue');
  log(`Passed: ${results.passed}`, 'green');
  log(`Failed: ${results.failed}`, results.failed > 0 ? 'red' : 'green');

  // Tool-by-tool summary
  log('\nTool Results:', 'blue');
  Object.entries(results.tools).forEach(([tool, result]) => {
    const status = result.status === 'passed' ? '✅' : result.status === 'failed' ? '❌' : '⚠️';
    const color = result.status === 'passed' ? 'green' : result.status === 'failed' ? 'red' : 'yellow';
    log(`  ${status} ${tool}: ${result.status.toUpperCase()}`, color);
  });

  // Performance metrics
  if (results.tools.memory_usage?.status === 'passed') {
    log('\n📈 System Metrics:', 'blue');
    const memText = results.tools.memory_usage.result.content[0].text;
    log(`  ${memText.split('\n')[0]}`, 'cyan');
  }

  // Final result
  log(`\n${ '=' .repeat(60)}`, 'cyan');
  if (results.failed === 0) {
    log('🎉 All MCP tools are working correctly!', 'green');
    log('✨ The 5-agent swarm was successfully created and tested.', 'green');
  } else {
    log('⚠️  Some tests failed. Please check the errors above.', 'red');
  }
  log('=' .repeat(60), 'cyan');

  // Save detailed results
  import fs from 'fs';
  const resultsPath = path.join(__dirname, 'mcp-test-results.json');
  fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
  log(`\n📄 Detailed results saved to: ${resultsPath}`, 'cyan');
}

// Run tests
testAllMcpTools().catch(error => {
  log('\n💥 Fatal error during testing:', 'red');
  log(error.message, 'red');
  process.exit(1);
});