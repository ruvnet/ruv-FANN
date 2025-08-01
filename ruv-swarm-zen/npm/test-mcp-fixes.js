#!/usr/bin/env node

/**
 * Test script to verify MCP server fixes for issue #65
 */

import { spawn } from 'child_process';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

console.log('🧪 Testing MCP Server Fixes for Issue #65');
console.log('========================================\n');

// Start the MCP server
console.log('1. Starting MCP server in stdio mode...');
const mcpServer = spawn('node', [join(__dirname, 'bin/ruv-swarm-clean.js'), 'mcp', 'start'], {
  stdio: ['pipe', 'pipe', 'inherit'],
});

let serverReady = false;
let testsPassed = 0;
let testsFailed = 0;

// Handle server output
mcpServer.stdout.on('data', (data) => {
  const lines = data.toString().split('\n').filter(line => line.trim());

  for (const line of lines) {
    try {
      const message = JSON.parse(line);
      console.log('✅ Received:', message.method || message.result?.success || 'response');

      if (message.method === 'server.initialized') {
        serverReady = true;
        console.log('✅ Server initialized successfully\n');
        runTests();
      }
    } catch (e) {
      console.log('📝 Server output:', line);
    }
  }
});

// Test functions
function sendRequest(method, params = {}) {
  const request = {
    jsonrpc: '2.0',
    id: Date.now(),
    method,
    params,
  };

  console.log(`📤 Sending: ${method}`);
  mcpServer.stdin.write(JSON.stringify(request) + '\n');

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      testsFailed++;
      console.log(`❌ Timeout waiting for response to ${method}\n`);
      reject(new Error('Timeout'));
    }, 5000);

    const handler = (data) => {
      const lines = data.toString().split('\n').filter(line => line.trim());
      for (const line of lines) {
        try {
          const response = JSON.parse(line);
          if (response.id === request.id) {
            clearTimeout(timeout);
            mcpServer.stdout.removeListener('data', handler);

            if (response.error) {
              testsFailed++;
              console.log(`❌ Error response: ${response.error.message}\n`);
              reject(response.error);
            } else {
              testsPassed++;
              console.log(`✅ Success response received\n`);
              resolve(response.result);
            }
          }
        } catch (e) {
          // Not JSON, ignore
        }
      }
    };

    mcpServer.stdout.on('data', handler);
  });
}

async function runTests() {
  console.log('2. Running test sequence...\n');

  try {
    // Test 1: List tools
    console.log('Test 1: List available tools');
    await sendRequest('tools/list');

    // Test 2: Initialize swarm
    console.log('Test 2: Initialize swarm');
    await sendRequest('tools/call', {
      name: 'swarm_init',
      arguments: {
        topology: 'mesh',
        maxAgents: 3,
      },
    });

    // Test 3: Check swarm status
    console.log('Test 3: Check swarm status');
    await sendRequest('tools/call', {
      name: 'swarm_status',
      arguments: {},
    });

    // Test 4: Test connection stability
    console.log('Test 4: Testing connection stability (30s idle)...');
    await new Promise(resolve => setTimeout(resolve, 30000));

    // Test 5: Verify connection still active
    console.log('Test 5: Verify connection still active');
    await sendRequest('tools/call', {
      name: 'swarm_status',
      arguments: {},
    });

    console.log('\n✅ All tests completed!');
    console.log(`Tests passed: ${testsPassed}`);
    console.log(`Tests failed: ${testsFailed}`);

  } catch (error) {
    console.error('❌ Test failed:', error);
  } finally {
    // Test disconnection handling
    console.log('\n3. Testing graceful shutdown...');
    mcpServer.stdin.end();

    setTimeout(() => {
      if (mcpServer.exitCode === null) {
        console.log('❌ Server did not exit after stdin close');
        mcpServer.kill();
      } else {
        console.log('✅ Server exited gracefully with code:', mcpServer.exitCode);
      }
      process.exit(testsFailed > 0 ? 1 : 0);
    }, 2000);
  }
}

// Handle errors
mcpServer.on('error', (error) => {
  console.error('❌ Server error:', error);
  process.exit(1);
});

mcpServer.on('exit', (code, signal) => {
  console.log(`\n📊 Server exited with code ${code} and signal ${signal}`);
});

// Handle test script termination
process.on('SIGINT', () => {
  console.log('\n🛑 Test interrupted, cleaning up...');
  mcpServer.kill();
  process.exit(1);
});
