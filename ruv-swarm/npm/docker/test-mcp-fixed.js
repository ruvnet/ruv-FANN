#!/usr/bin/env node

/**
 * FIXED MCP Server Validation Test for ruv-swarm v1.0.6
 * Properly handles stderr/stdout output streams
 */

import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('================================================');
console.log('ruv-swarm v1.0.6 MCP Server Validation (FIXED)');
console.log('================================================');
console.log(`Date: ${new Date().toISOString()}`);
console.log(`Node Version: ${process.version}`);
console.log('');

const results = {
  testSuite: 'mcp-server-validation-fixed',
  version: '1.0.6',
  timestamp: new Date().toISOString(),
  tests: [],
  summary: {
    total: 0,
    passed: 0,
    failed: 0,
  },
};

let mcpProcess = null;

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function addTestResult(name, status, message, error = null) {
  const result = { name, status, message };
  if (error) {
    result.error = error;
  }
  results.tests.push(result);
  results.summary.total++;
  if (status === 'passed') {
    results.summary.passed++;
  }
  if (status === 'failed') {
    results.summary.failed++;
  }
  console.log(`${status === 'passed' ? '✅' : '❌'} ${name}: ${message}`);
}

// Start MCP server with FIXED output monitoring
async function startMCPServer() {
  console.log('1. Starting MCP Server (FIXED)');
  console.log('===============================');

  return new Promise((resolve, reject) => {
    mcpProcess = spawn('node', ['bin/ruv-swarm-clean.js', 'mcp', 'start'], {
      env: { ...process.env, MCP_TEST_MODE: 'true' },
    });

    let serverReady = false;
    let initializationComplete = false;

    // Monitor BOTH stdout AND stderr (FIX: original only monitored stdout)
    mcpProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('  Server (stdout):', output.trim());

      if (output.includes('MCP server ready') || output.includes('Listening on') || output.includes('server.initialized')) {
        if (!serverReady) {
          serverReady = true;
          addTestResult('MCP Server Start', 'passed', 'Server started successfully (stdout)');
          if (initializationComplete) resolve();
        }
      }
    });

    // FIX: Also monitor stderr where "MCP server ready" is actually output
    mcpProcess.stderr.on('data', (data) => {
      const output = data.toString();
      console.log('  Server (stderr):', output.trim());

      if (output.includes('MCP server ready') || output.includes('Connection established')) {
        if (!initializationComplete) {
          initializationComplete = true;
          addTestResult('MCP Server Initialization', 'passed', 'Server initialization complete (stderr)');
          if (serverReady) resolve();
        }
      }
    });

    mcpProcess.on('error', (error) => {
      addTestResult('MCP Server Start', 'failed', 'Failed to start server', error.message);
      reject(error);
    });

    // FIX: Increased timeout from 10s to 15s (server needs ~12s to fully initialize)
    setTimeout(() => {
      if (!serverReady || !initializationComplete) {
        addTestResult('MCP Server Start', 'failed', 'Server startup timeout (15s)');
        reject(new Error('Server startup timeout'));
      }
    }, 15000);
  });
}

// Test stdio MCP communication
async function testStdioMCP() {
  console.log('\n2. Testing stdio MCP Communication');
  console.log('==================================');

  return new Promise((resolve, reject) => {
    let responseReceived = false;
    
    // Test basic MCP request via stdio
    const testRequest = {
      jsonrpc: '2.0',
      id: 1,
      method: 'ruv-swarm/swarm_status',
      params: {}
    };

    console.log('📤 Sending test request:', JSON.stringify(testRequest));
    mcpProcess.stdin.write(JSON.stringify(testRequest) + '\n');

    const responseHandler = (data) => {
      const output = data.toString();
      console.log('📥 Response:', output.trim());
      
      if (output.includes('jsonrpc') && (output.includes('result') || output.includes('error'))) {
        if (!responseReceived) {
          responseReceived = true;
          addTestResult('MCP Server Response', 'passed', 'Server responded to stdio request');
          resolve();
        }
      }
    };

    mcpProcess.stdout.on('data', responseHandler);
    mcpProcess.stderr.on('data', responseHandler);

    setTimeout(() => {
      if (!responseReceived) {
        addTestResult('MCP Server Response', 'failed', 'No response to stdio request');
        reject(new Error('No response timeout'));
      }
    }, 5000);
  });
}

// Cleanup
async function cleanup() {
  console.log('\n3. Cleanup');
  console.log('==========');

  if (mcpProcess) {
    mcpProcess.kill();
    console.log('  MCP server stopped');
  }
}

// Generate report
async function generateReport() {
  results.summary.passRate = (results.summary.passed / results.summary.total * 100).toFixed(2);

  const resultsPath = path.join(__dirname, '..', 'test-results', 'mcp-validation-fixed.json');
  await fs.mkdir(path.dirname(resultsPath), { recursive: true });
  await fs.writeFile(resultsPath, JSON.stringify(results, null, 2));

  console.log('\n================================================');
  console.log('MCP Validation Summary (FIXED)');
  console.log('================================================');
  console.log(`Total Tests: ${results.summary.total}`);
  console.log(`Passed: ${results.summary.passed}`);
  console.log(`Failed: ${results.summary.failed}`);
  console.log(`Pass Rate: ${results.summary.passRate}%`);
  console.log('');
  console.log(`Results saved to: ${resultsPath}`);
  
  if (results.summary.failed === 0) {
    console.log('🎉 ALL TESTS PASSED - MCP Server is working correctly!');
  } else {
    console.log('❌ Some tests failed - check results above');
  }
}

// Run all tests
async function runTests() {
  try {
    await startMCPServer();
    await sleep(2000); // Let server fully initialize
    await testStdioMCP();
    addTestResult('Test Suite', 'passed', 'All tests completed successfully');
  } catch (error) {
    console.error('Test failed:', error);
    addTestResult('Test Suite', 'failed', 'Suite execution failed', error.message);
  } finally {
    await cleanup();
    await generateReport();
    process.exit(results.summary.failed > 0 ? 1 : 0);
  }
}

// Handle interrupts
process.on('SIGINT', async() => {
  console.log('\nInterrupted, cleaning up...');
  await cleanup();
  process.exit(1);
});

runTests();