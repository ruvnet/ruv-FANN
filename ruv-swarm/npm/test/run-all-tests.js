#!/usr/bin/env node

/**
 * Comprehensive Test Runner for ruv-swarm
 * Executes all test suites and generates a detailed report
 */

import fs from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath, pathToFileURL } from 'url';
import { dirname } from 'path';
import { setTimeout as delay } from 'node:timers/promises';
import { createServer } from 'node:net';
import os from 'node:os';

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
    name: 'Mock API Examples',
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
export function parseSuiteOutput(output) {
  const summary = output.match(/(?:✅\s*)?Passed:\s*(\d+)[\s\S]*?(?:❌\s*)?Failed:\s*(\d+)/);
  const examples = output.match(/Tests completed:\s*(\d+) passed,\s*(\d+) failed/);
  const counts = summary || examples;
  if (!counts) {
    return { totalTests: 0, passedTests: 0, failedTests: 0 };
  }
  const passedTests = Number(counts[1]);
  const failedTests = Number(counts[2]);
  return { totalTests: passedTests + failedTests, passedTests, failedTests };
}

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
    const testProcess = spawn(process.execPath, [suite.file], {
      cwd: __dirname,
      env: { ...process.env, NODE_ENV: 'test',
        ...(mcpContext ? { RUV_SWARM_TEST_HTTP_URL: mcpContext.url } : {}) },
    });

    let output = '';
    let errorOutput = '';

    testProcess.stdout.on('data', (data) => {
      const text = data.toString();
      output += text;
      process.stdout.write(data);

    });

    testProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
      process.stderr.write(data);
    });

    const timeout = setTimeout(() => {
      testProcess.kill('SIGTERM');
      suiteResult.errors.push(`Test suite timed out after ${suite.timeout / 1000}s`);
    }, suite.timeout);
    testProcess.on('error', error => suiteResult.errors.push(error.message));

    testProcess.on('close', (code) => {
      clearTimeout(timeout);

      suiteResult.duration = Date.now() - suiteStartTime;
      Object.assign(suiteResult, parseSuiteOutput(output));
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
let mcpContext = null;

async function reserveLoopbackPort() {
  const reservation = createServer();
  await new Promise((resolve, reject) => {
    reservation.once('error', reject);
    reservation.listen(0, '127.0.0.1', resolve);
  });
  const port = reservation.address().port;
  await new Promise((resolve, reject) => reservation.close(error => error ? reject(error) : resolve()));
  return port;
}

async function startMCPServer() {
  console.log('🚀 Starting MCP Server...');
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'ruv-mcp-integration-'));
  const address = `127.0.0.1:${await reserveLoopbackPort()}`;
  mcpContext = { directory, address, url: `http://${address}` };
  mcpServer = spawn('cargo', ['run', '--locked', '--bin', 'ruv-swarm-mcp'], {
    cwd: path.join(__dirname, '../../crates/ruv-swarm-mcp'),
    detached: process.platform !== 'win32',
    env: { ...process.env, RUST_LOG: 'ruv_swarm_mcp=info', RUV_SWARM_MCP_BIND_ADDR: address,
      RUV_SWARM_DB_PATH: path.join(directory, 'mcp.sqlite') },
  });
  const server = mcpServer;
  let startupError;
  let ownedListener = false;
  let startupLog = '';
  const checkListener = data => {
    startupLog = (startupLog + data.toString()).slice(-8192);
    // This message is emitted by the owned child only after bind succeeds.
    // A healthy service on a colliding port cannot satisfy this condition.
    if (startupLog.includes(`MCP server listening on ${address}`)) {
      ownedListener = true;
    }
  };
  server.on('error', error => {
    startupError = error;
  });
  server.stdout.on('data', data => {
    checkListener(data); process.stdout.write(data);
  });
  server.stderr.on('data', data => {
    checkListener(data); process.stderr.write(data);
  });
  const deadline = Date.now() + Number(process.env.MCP_START_TIMEOUT_MS || 600000);
  while (Date.now() < deadline) {
    if (startupError || server.exitCode !== null || server.signalCode !== null) {
      throw startupError || new Error(`MCP server exited during startup (${server.exitCode})`);
    }
    try {
      const response = ownedListener && await fetch(`${mcpContext.url}/health`, { signal: AbortSignal.timeout(750) });
      if (response?.ok && server.exitCode === null && server.signalCode === null) {
        return;
      }
    } catch {
      // Compilation or socket binding is still in progress.
    }
    await delay(250);
  }
  throw new Error('MCP server did not become healthy before the startup deadline');
}

async function stopMCPServer() {
  const context = mcpContext;
  mcpContext = null;
  try {
    if (mcpServer) {
      console.log('\n🛑 Stopping MCP Server...');
      const server = mcpServer;
      mcpServer = null;
      if (server.exitCode !== null || !server.pid) {
        return;
      }
      const closed = new Promise(resolve => server.once('close', resolve));
      const kill = signal => {
        try {
          if (process.platform === 'win32') {
            server.kill(signal);
          } else {
            process.kill(-server.pid, signal);
          }
        } catch (error) {
          if (error.code !== 'ESRCH') {
            throw error;
          }
        }
      };
      kill('SIGTERM');
      const force = setTimeout(() => kill('SIGKILL'), 3000);
      force.unref();
      await closed;
      clearTimeout(force);
    }
  } finally {
    if (context) {
      await fs.rm(context.directory, { recursive: true, force: true });
    }
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
        await stopMCPServer();
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
    const reportPath = path.join(__dirname, 'test-reports', `${process.pid}-${Date.now()}.json`);
    await fs.mkdir(path.dirname(reportPath), { recursive: true });
    await report.saveReport(reportPath);
    console.log(`\n📄 Detailed report saved to: ${reportPath}`);

    // Print summary
    report.printSummary();

    // Set exit code based on results
    process.exit(report.summary.failedSuites > 0 || process.exitCode ? 1 : 0);
  }
}

// Importing report utilities must not start servers or execute the suites.
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
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
  main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { TestReport, runTestSuite };
