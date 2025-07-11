#!/usr/bin/env node

/**
 * Test MCP server in stdio mode (like Claude Code uses)
 */

import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';

console.log('🧪 Testing MCP Server in stdio mode (Claude Code compatible)');
console.log('============================================================');

const testResults = {
  testSuite: 'mcp-stdio-validation',
  version: '1.0.6',
  timestamp: new Date().toISOString(),
  tests: [],
  summary: { total: 0, passed: 0, failed: 0 }
};

function addResult(name, status, message) {
  testResults.tests.push({ name, status, message });
  testResults.summary.total++;
  if (status === 'passed') testResults.summary.passed++;
  if (status === 'failed') testResults.summary.failed++;
  console.log(`${status === 'passed' ? '✅' : '❌'} ${name}: ${message}`);
}

async function testStdioMCP() {
  console.log('\n📡 Testing stdio MCP server...');
  
  return new Promise((resolve, reject) => {
    const mcpProcess = spawn('node', ['bin/ruv-swarm-clean.js', 'mcp', 'start'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, MCP_TEST_MODE: 'true' }
    });

    let initialized = false;
    let responseReceived = false;
    
    mcpProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('📤 Server output:', output.trim());
      
      if (output.includes('MCP server ready') || output.includes('Connection established')) {
        if (!initialized) {
          initialized = true;
          addResult('MCP Server Stdio Start', 'passed', 'Server initialized in stdio mode');
          
          // Test basic MCP request
          const testRequest = {
            jsonrpc: '2.0',
            id: 1,
            method: 'ruv-swarm/swarm_status',
            params: {}
          };
          
          console.log('📤 Sending test request:', JSON.stringify(testRequest));
          mcpProcess.stdin.write(JSON.stringify(testRequest) + '\n');
        }
      }
      
      // Check for JSON-RPC response
      if (output.includes('jsonrpc') && output.includes('result')) {
        if (!responseReceived) {
          responseReceived = true;
          addResult('MCP Server Response', 'passed', 'Server responded to test request');
          mcpProcess.kill();
          resolve();
        }
      }
    });

    mcpProcess.stderr.on('data', (data) => {
      console.log('📥 Server stderr:', data.toString());
    });

    mcpProcess.on('error', (error) => {
      addResult('MCP Server Stdio Start', 'failed', `Process error: ${error.message}`);
      reject(error);
    });

    mcpProcess.on('exit', (code) => {
      if (code !== 0 && code !== null) {
        addResult('MCP Server Exit', 'failed', `Process exited with code ${code}`);
      } else if (responseReceived) {
        addResult('MCP Server Exit', 'passed', 'Process exited cleanly');
      }
      resolve();
    });

    // Timeout after 15 seconds
    setTimeout(() => {
      if (!initialized) {
        addResult('MCP Server Stdio Start', 'failed', 'Server startup timeout (15s)');
        mcpProcess.kill();
        reject(new Error('Startup timeout'));
      } else if (!responseReceived) {
        addResult('MCP Server Response', 'failed', 'No response to test request');
        mcpProcess.kill();
        resolve();
      }
    }, 15000);
  });
}

async function saveResults() {
  testResults.summary.passRate = (testResults.summary.passed / testResults.summary.total * 100).toFixed(2);
  
  const resultsPath = '/app/test-results/mcp-stdio-validation.json';
  await fs.mkdir(path.dirname(resultsPath), { recursive: true });
  await fs.writeFile(resultsPath, JSON.stringify(testResults, null, 2));
  
  console.log('\n📊 Test Results Summary:');
  console.log('========================');
  console.log(`Total: ${testResults.summary.total}`);
  console.log(`Passed: ${testResults.summary.passed}`);
  console.log(`Failed: ${testResults.summary.failed}`);
  console.log(`Pass Rate: ${testResults.summary.passRate}%`);
  console.log(`Results saved to: ${resultsPath}`);
}

async function runTests() {
  try {
    await testStdioMCP();
  } catch (error) {
    console.error('❌ Test failed:', error);
    addResult('Test Suite', 'failed', 'Suite execution failed');
  } finally {
    await saveResults();
    process.exit(testResults.summary.failed > 0 ? 1 : 0);
  }
}

runTests();