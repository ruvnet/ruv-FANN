#!/usr/bin/env node
/**
 * Comprehensive Test Suite for Issue #91 MCP Server Fixes
 * Tests: timeout resolution, ANSI escape handling, JSON parsing, notifications/initialized
 *
 * Author: Claude Code Assistant
 * Version: 1.0.0
 */

import { spawn } from 'child_process';
import { readFileSync, writeFileSync } from 'fs';
import { strict as assert } from 'assert';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class MCPServerTester {
  constructor() {
    this.results = {
      totalTests: 0,
      passed: 0,
      failed: 0,
      errors: [],
    };
  }

  async runTest(name, testFn) {
    this.results.totalTests++;
    console.log(`\n🧪 Running: ${name}`);

    try {
      await testFn();
      this.results.passed++;
      console.log(`✅ PASSED: ${name}`);
      return true;
    } catch (error) {
      this.results.failed++;
      this.results.errors.push({ name, error: error.message });
      console.log(`❌ FAILED: ${name}: ${error.message}`);
      return false;
    }
  }

  // Test 1: Server starts without ANSI escape codes
  async testServerStartup() {
    return new Promise((resolve, reject) => {
      const serverPath = join(__dirname, '..', 'bin', 'ruv-swarm-mcp-enhanced.js');
      const server = spawn('node', [serverPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, MCP_TEST_MODE: 'true', DEBUG: 'false' },
      });

      let stdoutData = '';
      let stderrData = '';
      let hasAnsiEscape = false;

      server.stdout.on('data', (data) => {
        stdoutData += data.toString();

        // Check for ANSI escape codes in stdout (this should NOT happen)
        if (data.toString().includes('\x1b[') || data.toString().includes('\u001b[')) {
          hasAnsiEscape = true;
        }
      });

      server.stderr.on('data', (data) => {
        stderrData += data.toString();
      });

      // Server should signal readiness
      const timeout = setTimeout(() => {
        server.kill();
        reject(new Error('Server failed to start within 5 seconds'));
      }, 5000);

      server.stderr.on('data', (data) => {
        if (data.toString().includes('Enhanced MCP server ready')) {
          clearTimeout(timeout);
          server.kill();

          // Validate no ANSI codes in stdout
          if (hasAnsiEscape) {
            reject(new Error('ANSI escape codes found in stdout - this breaks JSON-RPC'));
          } else {
            resolve();
          }
        }
      });

      server.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Server spawn error: ${error.message}`));
      });
    });
  }

  // Test 2: Test notifications/initialized handling
  async testNotificationsInitialized() {
    return new Promise((resolve, reject) => {
      const serverPath = join(__dirname, '..', 'bin', 'ruv-swarm-mcp-enhanced.js');
      const server = spawn('node', [serverPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, MCP_TEST_MODE: 'true', DEBUG: 'false' },
      });

      const responses = [];
      let buffer = '';

      server.stdout.on('data', (data) => {
        buffer += data.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              const response = JSON.parse(line.trim());
              responses.push(response);
            } catch (e) {
              // Ignore JSON parse errors for this test
            }
          }
        }
      });

      // Wait for server to be ready
      server.stderr.on('data', (data) => {
        if (data.toString().includes('Enhanced MCP server ready')) {
          // Test sequence: initialize -> notifications/initialized
          const initRequest = {
            jsonrpc: '2.0',
            id: 1,
            method: 'initialize',
            params: {
              protocolVersion: '2024-11-05',
              clientInfo: { name: 'test-client', version: '1.0.0' },
            },
          };

          const notificationRequest = {
            jsonrpc: '2.0',
            id: 2,
            method: 'notifications/initialized',
            params: {},
          };

          // Send requests
          server.stdin.write(JSON.stringify(initRequest) + '\n');
          server.stdin.write(JSON.stringify(notificationRequest) + '\n');

          // Wait for responses
          setTimeout(() => {
            server.kill();

            // Validate responses
            const initResponse = responses.find(r => r.id === 1);
            const notificationResponse = responses.find(r => r.id === 2);

            if (!initResponse) {
              reject(new Error('No response to initialize request'));
              return;
            }

            if (initResponse.error) {
              reject(new Error(`Initialize failed: ${initResponse.error.message}`));
              return;
            }

            if (!notificationResponse) {
              reject(new Error('No response to notifications/initialized request'));
              return;
            }

            // This should NOT return an error (previously it returned "Method not found")
            if (notificationResponse.error && notificationResponse.error.code === -32601) {
              reject(new Error('notifications/initialized still returns "Method not found" error'));
              return;
            }

            resolve();
          }, 2000);
        }
      });

      const timeout = setTimeout(() => {
        server.kill();
        reject(new Error('Test timeout'));
      }, 10000);

      server.on('exit', () => {
        clearTimeout(timeout);
      });
    });
  }

  // Test 3: JSON parsing with stderr output
  async testJsonParsingWithStderr() {
    return new Promise((resolve, reject) => {
      const serverPath = join(__dirname, '..', 'bin', 'ruv-swarm-mcp-enhanced.js');
      const server = spawn('node', [serverPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, MCP_TEST_MODE: 'true', DEBUG: 'true' }, // Enable debug logs
      });

      const jsonResponses = [];
      let stderrOutput = '';
      let buffer = '';

      server.stdout.on('data', (data) => {
        buffer += data.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              const response = JSON.parse(line.trim());
              jsonResponses.push(response);
            } catch (parseError) {
              reject(new Error(`JSON parse error: ${parseError.message}. Line: ${line.substring(0, 100)}`));
              return;
            }
          }
        }
      });

      server.stderr.on('data', (data) => {
        stderrOutput += data.toString();

        if (data.toString().includes('Enhanced MCP server ready')) {
          // Send a valid request that should generate logs
          const request = {
            jsonrpc: '2.0',
            id: 1,
            method: 'tools/list',
            params: {},
          };

          server.stdin.write(JSON.stringify(request) + '\n');

          setTimeout(() => {
            server.kill();

            // Validate that we got a valid JSON response despite stderr output
            if (jsonResponses.length === 0) {
              reject(new Error('No JSON responses received'));
              return;
            }

            const response = jsonResponses.find(r => r.id === 1);
            if (!response) {
              reject(new Error('No response to tools/list request'));
              return;
            }

            if (response.error) {
              reject(new Error(`tools/list failed: ${response.error.message}`));
              return;
            }

            // Validate stderr contains logs but stdout is clean JSON
            if (stderrOutput.length === 0) {
              reject(new Error('Expected stderr logging output'));
              return;
            }

            resolve();
          }, 1000);
        }
      });

      const timeout = setTimeout(() => {
        server.kill();
        reject(new Error('Test timeout'));
      }, 10000);

      server.on('exit', () => {
        clearTimeout(timeout);
      });
    });
  }

  // Test 4: Connection stability (no premature timeouts)
  async testConnectionStability() {
    return new Promise((resolve, reject) => {
      const serverPath = join(__dirname, '..', 'bin', 'ruv-swarm-mcp-enhanced.js');
      const server = spawn('node', [serverPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, MCP_TEST_MODE: 'true' },
      });

      let responseCount = 0;
      let buffer = '';

      server.stdout.on('data', (data) => {
        buffer += data.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              JSON.parse(line.trim());
              responseCount++;
            } catch (e) {
              // Ignore parse errors for this test
            }
          }
        }
      });

      server.stderr.on('data', (data) => {
        if (data.toString().includes('Enhanced MCP server ready')) {
          // Send multiple requests with delays to test stability
          let requestId = 1;

          const sendRequest = () => {
            if (requestId <= 5) {
              const request = {
                jsonrpc: '2.0',
                id: requestId,
                method: 'swarm_status',
                params: { verbose: false },
              };

              server.stdin.write(JSON.stringify(request) + '\n');
              requestId++;

              // Send next request after delay
              setTimeout(sendRequest, 1000);
            } else {
              // Wait a bit more then check results
              setTimeout(() => {
                server.kill();

                if (responseCount < 5) {
                  reject(new Error(`Expected 5 responses, got ${responseCount} - server may have timed out`));
                } else {
                  resolve();
                }
              }, 1000);
            }
          };

          sendRequest();
        }
      });

      const timeout = setTimeout(() => {
        server.kill();
        reject(new Error('Connection stability test timeout'));
      }, 15000);

      server.on('exit', () => {
        clearTimeout(timeout);
      });
    });
  }

  // Test 5: Error handling improvement
  async testErrorHandling() {
    return new Promise((resolve, reject) => {
      const serverPath = join(__dirname, '..', 'bin', 'ruv-swarm-mcp-enhanced.js');
      const server = spawn('node', [serverPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, MCP_TEST_MODE: 'true' },
      });

      let errorResponse = null;
      let buffer = '';

      server.stdout.on('data', (data) => {
        buffer += data.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              const response = JSON.parse(line.trim());
              if (response.error) {
                errorResponse = response;
              }
            } catch (e) {
              // Ignore parse errors
            }
          }
        }
      });

      server.stderr.on('data', (data) => {
        if (data.toString().includes('Enhanced MCP server ready')) {
          // Send an unknown method to test error handling
          const request = {
            jsonrpc: '2.0',
            id: 1,
            method: 'unknown_method_test',
            params: {},
          };

          server.stdin.write(JSON.stringify(request) + '\n');

          setTimeout(() => {
            server.kill();

            if (!errorResponse) {
              reject(new Error('Expected error response for unknown method'));
              return;
            }

            // Validate error response structure
            if (errorResponse.error.code !== -32601) {
              reject(new Error(`Expected error code -32601, got ${errorResponse.error.code}`));
              return;
            }

            // Check for improved error message
            if (!errorResponse.error.data || !errorResponse.error.data.includes('Supported methods:')) {
              reject(new Error('Error response should include helpful information about supported methods'));
              return;
            }

            resolve();
          }, 1000);
        }
      });

      const timeout = setTimeout(() => {
        server.kill();
        reject(new Error('Error handling test timeout'));
      }, 10000);

      server.on('exit', () => {
        clearTimeout(timeout);
      });
    });
  }

  // Test 6: Memory and resource management
  async testResourceManagement() {
    return new Promise((resolve, reject) => {
      const serverPath = join(__dirname, '..', 'bin', 'ruv-swarm-mcp-enhanced.js');
      const server = spawn('node', [serverPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, MCP_TEST_MODE: 'true' },
      });

      let resourceResponse = null;
      let buffer = '';

      server.stdout.on('data', (data) => {
        buffer += data.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              const response = JSON.parse(line.trim());
              if (response.result && response.result.resources) {
                resourceResponse = response;
              }
            } catch (e) {
              // Ignore parse errors
            }
          }
        }
      });

      server.stderr.on('data', (data) => {
        if (data.toString().includes('Enhanced MCP server ready')) {
          // Test resources/list
          const request = {
            jsonrpc: '2.0',
            id: 1,
            method: 'resources/list',
            params: {},
          };

          server.stdin.write(JSON.stringify(request) + '\n');

          setTimeout(() => {
            server.kill();

            if (!resourceResponse) {
              reject(new Error('No response to resources/list'));
              return;
            }

            // Validate resource structure
            const resources = resourceResponse.result.resources;
            if (!Array.isArray(resources) || resources.length === 0) {
              reject(new Error('Expected array of resources'));
              return;
            }

            // Check resource properties
            const resource = resources[0];
            if (!resource.uri || !resource.name || !resource.description) {
              reject(new Error('Resource missing required properties'));
              return;
            }

            resolve();
          }, 1000);
        }
      });

      const timeout = setTimeout(() => {
        server.kill();
        reject(new Error('Resource management test timeout'));
      }, 10000);

      server.on('exit', () => {
        clearTimeout(timeout);
      });
    });
  }

  async runAllTests() {
    console.log('🚀 Starting MCP Server Timeout & JSON Parsing Fixes Test Suite\n');
    console.log('Testing fixes for Issue #91');

    await this.runTest('Server Startup Without ANSI Codes', () => this.testServerStartup());
    await this.runTest('notifications/initialized Handling', () => this.testNotificationsInitialized());
    await this.runTest('JSON Parsing with stderr Output', () => this.testJsonParsingWithStderr());
    await this.runTest('Connection Stability', () => this.testConnectionStability());
    await this.runTest('Enhanced Error Handling', () => this.testErrorHandling());
    await this.runTest('Resource Management', () => this.testResourceManagement());

    // Report results
    console.log('\n📊 Test Results Summary:');
    console.log(`Total Tests: ${this.results.totalTests}`);
    console.log(`Passed: ${this.results.passed} ✅`);
    console.log(`Failed: ${this.results.failed} ❌`);

    if (this.results.failed > 0) {
      console.log('\n❌ Failed Tests:');
      this.results.errors.forEach(error => {
        console.log(`  - ${error.name}: ${error.error}`);
      });
    }

    const successRate = (this.results.passed / this.results.totalTests) * 100;
    console.log(`\nSuccess Rate: ${successRate.toFixed(1)}%`);

    if (successRate >= 90) {
      console.log('\n🎉 All critical fixes verified! Issue #91 resolved.');
      return true;
    } else {
      console.log('\n⚠️  Some tests failed. Additional fixes may be needed.');
      return false;
    }
  }
}

// Run tests if called directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const tester = new MCPServerTester();
  tester.runAllTests().then(success => {
    process.exit(success ? 0 : 1);
  }).catch(error => {
    console.error('Test suite error:', error.message);
    process.exit(1);
  });
}

export { MCPServerTester };
