#!/usr/bin/env node

/**
 * Comprehensive Test Runner for ruv-swarm
 * Executes all test suites and generates a detailed report
 */

import fs from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Test configuration
const TEST_SUITES = [
  {
    name: 'MCP Integration Tests',
    file: './mcp-integration.test.js',
    timeout: 120000, // 2 minutes
    requiresServer: true,
  },
  {
    name: 'Persistence Layer Tests',
    file: './persistence.test.js',
    timeout: 60000, // 1 minute
    requiresServer: false,
  },
  {
    name: 'Neural Network Integration Tests',
    file: './neural-integration.test.js',
    timeout: 90000, // 1.5 minutes
    requiresServer: false,
  },
  {
    name: 'Basic WASM Tests',
    file: './test.js',
    timeout: 30000, // 30 seconds
    requiresServer: false,
  },
];

// Test report structure
class TestReport {
  constructor() {
    this.startTime = new Date();
    this.suites = [];
    this.summary = {
      totalSuites: 0,
      passedSuites: 0,
      failedSuites: 0,
      totalTests: 0,
      passedTests: 0,
      failedTests: 0,
      totalDuration: 0,
    };
  }

  addSuite(suite) {
    this.suites.push(suite);
    this.summary.totalSuites++;

    if (suite.passed) {
      this.summary.passedSuites++;
    } else {
      this.summary.failedSuites++;
    }

    this.summary.totalTests += suite.totalTests || 0;
    this.summary.passedTests += suite.passedTests || 0;
    this.summary.failedTests += suite.failedTests || 0;
  }

  finalize() {
    this.endTime = new Date();
    this.summary.totalDuration = this.endTime - this.startTime;
  }

  async saveReport(filename) {
    const report = {
      metadata: {
        generatedAt: this.startTime.toISOString(),
        completedAt: this.endTime.toISOString(),
        duration: `${(this.summary.totalDuration / 1000).toFixed(2)}s`,
        environment: {
          node: process.version,
          platform: process.platform,
          arch: process.arch,
        },
      },
      summary: this.summary,
      suites: this.suites,
    };

    await fs.writeFile(filename, JSON.stringify(report, null, 2));
  }

  printSummary() {
    console.log('\n📊 Test Summary');
    console.log('═'.repeat(60));

    console.log('\n📦 Test Suites:');
    console.log(`  Total: ${this.summary.totalSuites}`);
    console.log(`  ✅ Passed: ${this.summary.passedSuites}`);
    console.log(`  ❌ Failed: ${this.summary.failedSuites}`);

    console.log('\n🧪 Individual Tests:');
    console.log(`  Total: ${this.summary.totalTests}`);
    console.log(`  ✅ Passed: ${this.summary.passedTests}`);
    console.log(`  ❌ Failed: ${this.summary.failedTests}`);

    console.log(`\n⏱️  Total Duration: ${(this.summary.totalDuration / 1000).toFixed(2)}s`);

    if (this.summary.failedSuites > 0) {
      console.log('\n❌ Failed Suites:');
      this.suites.filter(s => !s.passed).forEach(suite => {
        console.log(`  - ${suite.name}`);
        if (suite.errors && suite.errors.length > 0) {
          suite.errors.slice(0, 3).forEach(error => {
            console.log(`    • ${error}`);
          });
          if (suite.errors.length > 3) {
            console.log(`    ... and ${suite.errors.length - 3} more errors`);
          }
        }
      });
    }

    console.log(`\n${ '═'.repeat(60)}`);
    console.log(this.summary.failedSuites === 0 ? '✅ All tests passed!' : '❌ Some tests failed');
  }
}

// Test execution utilities
async function runTestSuite(suite, report) {
  console.log(`\n🚀 Running ${suite.name}`);
  console.log('─'.repeat(50));

  const suiteStartTime = Date.now();
  const suiteResult = {
    name: suite.name,
    file: suite.file,
    startTime: new Date().toISOString(),
    passed: false,
    totalTests: 0,
    passedTests: 0,
    failedTests: 0,
    duration: 0,
    output: [],
    errors: [],
  };

  return new Promise((resolve) => {
    const testProcess = spawn('node', [suite.file], {
      cwd: __dirname,
      env: { ...process.env, NODE_ENV: 'test' },
    });

    let output = '';
    let errorOutput = '';

    testProcess.stdout.on('data', (data) => {
      const text = data.toString();
      output += text;
      process.stdout.write(data);

      // Parse test results from output
      const passedMatches = text.match(/✅|✓/g);
      const failedMatches = text.match(/❌|✗/g);

      if (passedMatches) {
        suiteResult.passedTests += passedMatches.length;
      }
      if (failedMatches) {
        suiteResult.failedTests += failedMatches.length;
      }
    });

    testProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
      process.stderr.write(data);
    });

    const timeout = setTimeout(() => {
      testProcess.kill('SIGTERM');
      suiteResult.errors.push(`Test suite timed out after ${suite.timeout / 1000}s`);
    }, suite.timeout);

    testProcess.on('close', (code) => {
      clearTimeout(timeout);

      suiteResult.duration = Date.now() - suiteStartTime;
      suiteResult.totalTests = suiteResult.passedTests + suiteResult.failedTests;
      suiteResult.passed = code === 0 && suiteResult.failedTests === 0;
      suiteResult.exitCode = code;
      suiteResult.output = output.split('\n').filter(line => line.trim());

      if (errorOutput) {
        suiteResult.errors.push(...errorOutput.split('\n').filter(line => line.trim()));
      }

      console.log(`\n${suiteResult.passed ? '✅' : '❌'} ${suite.name} completed in ${(suiteResult.duration / 1000).toFixed(2)}s`);

      report.addSuite(suiteResult);
      resolve(suiteResult);
    });
  });
}

// MCP Server management
let mcpServer = null;

async function startMCPServer() {
  console.log('🚀 Starting MCP Server...');

  return new Promise((resolve, reject) => {
    mcpServer = spawn('npm', ['run', 'mcp:server'], {
      cwd: path.join(__dirname, '..'),
      detached: true,
    });

    let serverStarted = false;

    mcpServer.stdout.on('data', (data) => {
      const output = data.toString();
      console.log(`MCP Server: ${output.trim()}`);

      if (output.includes('Starting RUV-Swarm MCP server') || output.includes('listening')) {
        serverStarted = true;
        setTimeout(resolve, 2000); // Give server time to fully initialize
      }
    });

    mcpServer.stderr.on('data', (data) => {
      console.error(`MCP Server Error: ${data.toString()}`);
    });

    mcpServer.on('error', reject);

    // Timeout if server doesn't start
    setTimeout(() => {
      if (!serverStarted) {
        reject(new Error('MCP Server failed to start within timeout'));
      }
    }, 10000);
  });
}

async function stopMCPServer() {
  if (mcpServer) {
    console.log('\n🛑 Stopping MCP Server...');

    // Try graceful shutdown first
    mcpServer.kill('SIGTERM');

    // Force kill after timeout
    setTimeout(() => {
      if (mcpServer && !mcpServer.killed) {
        mcpServer.kill('SIGKILL');
      }
    }, 5000);

    mcpServer = null;
  }
}

// Performance monitoring
class PerformanceMonitor {
  constructor() {
    this.metrics = {
      cpu: [],
      memory: [],
    };
    this.interval = null;
  }

  start() {
    this.interval = setInterval(() => {
      const usage = process.cpuUsage();
      const memory = process.memoryUsage();

      this.metrics.cpu.push({
        timestamp: Date.now(),
        user: usage.user,
        system: usage.system,
      });

      this.metrics.memory.push({
        timestamp: Date.now(),
        heapUsed: memory.heapUsed,
        heapTotal: memory.heapTotal,
        rss: memory.rss,
      });
    }, 1000);
  }

  stop() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  getReport() {
    const avgCpu = this.metrics.cpu.reduce((sum, m) => sum + m.user + m.system, 0) /
                      (this.metrics.cpu.length || 1) / 1000000; // Convert to seconds

    const maxMemory = Math.max(...this.metrics.memory.map(m => m.heapUsed)) / 1024 / 1024; // MB

    return {
      averageCpuSeconds: avgCpu.toFixed(2),
      maxMemoryMB: maxMemory.toFixed(2),
      samples: this.metrics.cpu.length,
    };
  }
}

// Main test orchestration
async function main() {
  console.log('🧪 RUV-SWARM Comprehensive Test Suite');
  console.log('═'.repeat(60));
  console.log(`Started at: ${new Date().toISOString()}`);
  console.log(`Node version: ${process.version}`);
  console.log(`Platform: ${process.platform} ${process.arch}`);

  const report = new TestReport();
  const perfMonitor = new PerformanceMonitor();

  try {
    // Start performance monitoring
    perfMonitor.start();

    // Check if MCP server is needed
    const requiresServer = TEST_SUITES.some(suite => suite.requiresServer);

    if (requiresServer) {
      try {
        await startMCPServer();
        console.log('✅ MCP Server started successfully\n');
      } catch (error) {
        console.error('❌ Failed to start MCP Server:', error.message);
        console.log('⚠️  Skipping tests that require MCP server\n');
      }
    }

    // Run each test suite
    for (const suite of TEST_SUITES) {
      if (suite.requiresServer && !mcpServer) {
        console.log(`\n⚠️  Skipping ${suite.name} (requires MCP server)`);
        report.addSuite({
          name: suite.name,
          file: suite.file,
          passed: false,
          skipped: true,
          reason: 'MCP server not available',
          totalTests: 0,
          passedTests: 0,
          failedTests: 0,
        });
        continue;
      }

      await runTestSuite(suite, report);

      // Small delay between suites
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

  } catch (error) {
    console.error('\n❌ Test runner error:', error);
    process.exitCode = 1;
  } finally {
    // Stop performance monitoring
    perfMonitor.stop();

    // Stop MCP server if running
    await stopMCPServer();

    // Finalize report
    report.finalize();

    // Add performance metrics to report
    const perfReport = perfMonitor.getReport();
    console.log('\n📈 Performance Metrics:');
    console.log(`  Average CPU: ${perfReport.averageCpuSeconds}s`);
    console.log(`  Max Memory: ${perfReport.maxMemoryMB}MB`);

    // Save detailed report
    const reportPath = path.join(__dirname, `test-report-${Date.now()}.json`);
    await report.saveReport(reportPath);
    console.log(`\n📄 Detailed report saved to: ${reportPath}`);

    // Print summary
    report.printSummary();

    // Set exit code based on results
    process.exit(report.summary.failedSuites > 0 ? 1 : 0);
  }
}

// Handle interrupts
process.on('SIGINT', async() => {
  console.log('\n⚠️  Test run interrupted');
  await stopMCPServer();
  process.exit(1);
});

process.on('unhandledRejection', async(error) => {
  console.error('\n❌ Unhandled rejection:', error);
  await stopMCPServer();
  process.exit(1);
});

// Run tests
main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});

export { TestReport, runTestSuite };