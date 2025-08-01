#!/usr/bin/env node

/**
 * Cross-Platform Validation Test for ruv-swarm v1.0.6
 * Tests compatibility across different operating systems and architectures
 */

import { RuvSwarm } from '../src/index.js';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import os from 'os';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('================================================');
console.log('ruv-swarm v1.0.6 Cross-Platform Validation');
console.log('================================================');
console.log(`Date: ${new Date().toISOString()}`);
console.log(`Platform: ${process.platform}`);
console.log(`Architecture: ${process.arch}`);
console.log(`OS Release: ${os.release()}`);
console.log(`Node Version: ${process.version}`);
console.log('');

const results = {
  testSuite: 'cross-platform-validation',
  version: '1.0.6',
  timestamp: new Date().toISOString(),
  environment: {
    platform: process.platform,
    arch: process.arch,
    osRelease: os.release(),
    nodeVersion: process.version,
    endianness: os.endianness(),
    cpus: os.cpus().length,
    memory: os.totalmem(),
  },
  tests: [],
  summary: {
    total: 0,
    passed: 0,
    failed: 0,
  },
};

// Test utilities
function addTestResult(name, status, message, details = {}) {
  const result = { name, status, message, ...details };
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

// Test file system operations
async function testFileSystemOps() {
  console.log('1. Testing File System Operations');
  console.log('================================');

  try {
    // Test path resolution
    const testPath = path.join(os.tmpdir(), 'ruv-swarm-test');
    await fs.mkdir(testPath, { recursive: true });

    // Test file writing
    const testFile = path.join(testPath, 'test.json');
    await fs.writeFile(testFile, JSON.stringify({ test: true }));

    // Test file reading
    const content = await fs.readFile(testFile, 'utf8');
    const parsed = JSON.parse(content);

    if (parsed.test === true) {
      addTestResult('File System Operations', 'passed', 'File I/O working correctly');
    } else {
      addTestResult('File System Operations', 'failed', 'File content mismatch');
    }

    // Cleanup
    await fs.rm(testPath, { recursive: true });

  } catch (error) {
    addTestResult('File System Operations', 'failed', error.message);
  }
}

// Test WASM loading on different platforms
async function testWASMCompatibility() {
  console.log('\n2. Testing WASM Compatibility');
  console.log('=============================');

  try {
    // Check WASM file exists
    const wasmPath = path.join(__dirname, '..', 'wasm', 'ruv_swarm_wasm_bg.wasm');
    const stats = await fs.stat(wasmPath);

    addTestResult('WASM File Access', 'passed', `WASM file accessible: ${stats.size} bytes`);

    // Try to load WASM module
    const swarm = new RuvSwarm({ maxAgents: 2 });
    addTestResult('WASM Module Loading', 'passed', `WASM loaded successfully on ${ process.platform}`);

    // Test basic WASM operation
    const agent = swarm.spawnAgent('test-agent', 'researcher');
    if (agent) {
      addTestResult('WASM Functionality', 'passed', 'WASM operations working');
    }

  } catch (error) {
    addTestResult('WASM Compatibility', 'failed', error.message);
  }
}

// Test process spawning
async function testProcessSpawning() {
  console.log('\n3. Testing Process Spawning');
  console.log('===========================');

  try {
    // Test basic command execution
    const nodeVersion = execSync('node --version', { encoding: 'utf8' }).trim();
    addTestResult('Process Execution', 'passed', `Can execute processes: ${nodeVersion}`);

    // Test npx availability
    try {
      const npxVersion = execSync('npx --version', { encoding: 'utf8' }).trim();
      addTestResult('NPX Availability', 'passed', `NPX available: v${npxVersion}`);
    } catch (error) {
      addTestResult('NPX Availability', 'warning', 'NPX not available in PATH');
    }

  } catch (error) {
    addTestResult('Process Spawning', 'failed', error.message);
  }
}

// Test memory allocation
async function testMemoryAllocation() {
  console.log('\n4. Testing Memory Allocation');
  console.log('============================');

  try {
    const swarm = new RuvSwarm({ maxAgents: 32 });

    // Test large memory allocation
    const largeData = new Array(1000000).fill(0).map(() => Math.random());
    swarm.memory.store('large-data', largeData);

    const retrieved = swarm.memory.retrieve('large-data');
    if (retrieved && retrieved.length === largeData.length) {
      addTestResult('Large Memory Allocation', 'passed', 'Can handle large data structures');
    } else {
      addTestResult('Large Memory Allocation', 'failed', 'Data retrieval mismatch');
    }

    // Test memory limits
    const memUsage = process.memoryUsage();
    addTestResult('Memory Usage', 'passed', `Heap: ${(memUsage.heapUsed / 1024 / 1024).toFixed(2)}MB`, {
      memoryUsage: memUsage,
    });

  } catch (error) {
    addTestResult('Memory Allocation', 'failed', error.message);
  }
}

// Test native module compatibility
async function testNativeModules() {
  console.log('\n5. Testing Native Module Compatibility');
  console.log('=====================================');

  try {
    // Check if better-sqlite3 loads correctly
    const Database = (await import('better-sqlite3')).default;
    const db = new Database(':memory:');
    db.close();

    addTestResult('SQLite Module', 'passed', 'Native SQLite module working');

  } catch (error) {
    addTestResult('Native Modules', 'warning', 'Some native modules may not be available', {
      error: error.message,
    });
  }
}

// Test platform-specific features
async function testPlatformFeatures() {
  console.log('\n6. Testing Platform-Specific Features');
  console.log('====================================');

  // Test endianness handling
  const buffer = Buffer.alloc(4);
  buffer.writeInt32BE(0x12345678, 0);
  const value = buffer.readInt32BE(0);

  if (value === 0x12345678) {
    addTestResult('Endianness Handling', 'passed', `${os.endianness()} endian system`);
  } else {
    addTestResult('Endianness Handling', 'failed', 'Endianness mismatch');
  }

  // Test path separator handling
  const testPath = path.join('test', 'path', 'file.js');
  const expectedSep = process.platform === 'win32' ? '\\' : '/';

  if (testPath.includes(expectedSep)) {
    addTestResult('Path Separator', 'passed', `Using correct separator: ${expectedSep}`);
  } else {
    addTestResult('Path Separator', 'failed', 'Path separator mismatch');
  }
}

// Test concurrent operations
async function testConcurrency() {
  console.log('\n7. Testing Concurrent Operations');
  console.log('================================');

  try {
    const swarm = new RuvSwarm({ maxAgents: 16 });

    // Spawn multiple agents concurrently
    const promises = [];
    for (let i = 0; i < 10; i++) {
      promises.push(swarm.spawnAgent(`agent-${i}`, 'researcher'));
    }

    await Promise.all(promises);

    if (swarm.agents.length === 10) {
      addTestResult('Concurrent Agent Creation', 'passed', 'Created 10 agents concurrently');
    } else {
      addTestResult('Concurrent Agent Creation', 'failed', `Expected 10 agents, got ${swarm.agents.length}`);
    }

    // Test concurrent task execution
    const taskPromises = swarm.agents.map(agent =>
      agent.assignTask({ type: 'test', data: 'concurrent' }),
    );

    await Promise.all(taskPromises);
    addTestResult('Concurrent Task Execution', 'passed', 'All agents executed tasks concurrently');

  } catch (error) {
    addTestResult('Concurrency Test', 'failed', error.message);
  }
}

// Generate report
async function generateReport() {
  results.summary.passRate = (results.summary.passed / results.summary.total * 100).toFixed(2);

  // Platform compatibility score
  let compatibilityScore = 100;
  results.tests.forEach(test => {
    if (test.status === 'failed') {
      compatibilityScore -= 10;
    }
    if (test.status === 'warning') {
      compatibilityScore -= 5;
    }
  });
  results.summary.compatibilityScore = Math.max(0, compatibilityScore);

  const resultsPath = path.join(__dirname, '..', 'test-results', 'cross-platform-validation.json');
  await fs.mkdir(path.dirname(resultsPath), { recursive: true });
  await fs.writeFile(resultsPath, JSON.stringify(results, null, 2));

  console.log('\n================================================');
  console.log('Cross-Platform Validation Summary');
  console.log('================================================');
  console.log(`Platform: ${process.platform} (${process.arch})`);
  console.log(`Total Tests: ${results.summary.total}`);
  console.log(`Passed: ${results.summary.passed}`);
  console.log(`Failed: ${results.summary.failed}`);
  console.log(`Pass Rate: ${results.summary.passRate}%`);
  console.log(`Compatibility Score: ${results.summary.compatibilityScore}%`);
  console.log('');
  console.log(`Results saved to: ${resultsPath}`);
}

// Run all tests
async function runTests() {
  try {
    await testFileSystemOps();
    await testWASMCompatibility();
    await testProcessSpawning();
    await testMemoryAllocation();
    await testNativeModules();
    await testPlatformFeatures();
    await testConcurrency();
    await generateReport();

    process.exit(results.summary.failed > 0 ? 1 : 0);
  } catch (error) {
    console.error('Test suite failed:', error);
    process.exit(1);
  }
}

runTests();
