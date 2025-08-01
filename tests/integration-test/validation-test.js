#!/usr/bin/env node

/**
 * Validation Test Script for ruv-swarm CLI
 * Tests all input validation scenarios to ensure proper error handling
 */

const { spawn } = require('child_process');
const path = require('path');

const CLI_PATH = path.join(__dirname, '../ruv-swarm/npm/bin/ruv-swarm-clean.js');

function runCommand(args) {
  return new Promise((resolve) => {
    const child = spawn('node', [CLI_PATH, ...args], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('close', (code) => {
      resolve({ code, stdout, stderr });
    });

    // Set a timeout to kill the process if it hangs
    setTimeout(() => {
      child.kill('SIGTERM');
      resolve({ code: -1, stdout, stderr: 'Process timeout' });
    }, 10000);
  });
}

async function runTests() {
  console.log('🧪 Running Validation Tests for ruv-swarm CLI\n');

  const tests = [
    {
      name: 'Invalid topology',
      args: ['init', 'invalid-topology', '5'],
      expectFailure: true,
      expectedMessage: 'Invalid topology',
    },
    {
      name: 'Agent count too high',
      args: ['init', 'mesh', '101'],
      expectFailure: true,
      expectedMessage: 'Invalid maxAgents',
    },
    {
      name: 'Agent count too low',
      args: ['init', 'mesh', '0'],
      expectFailure: true,
      expectedMessage: 'Invalid maxAgents',
    },
    {
      name: 'Invalid agent type',
      args: ['spawn', 'invalid-type', 'Test Agent'],
      expectFailure: true,
      expectedMessage: 'Invalid agent type',
    },
    {
      name: 'Agent name with invalid characters',
      args: ['spawn', 'researcher', 'Test@Agent!'],
      expectFailure: true,
      expectedMessage: 'Agent name can only contain',
    },
    {
      name: 'Empty task description',
      args: ['orchestrate', '   '],
      expectFailure: true,
      expectedMessage: 'Task description cannot be empty',
    },
    {
      name: 'Valid topology and agent count',
      args: ['init', 'mesh', '5'],
      expectFailure: false,
      expectedMessage: 'Swarm initialized',
    },
    {
      name: 'Valid agent spawn',
      args: ['spawn', 'researcher', 'Test Agent'],
      expectFailure: false,
      expectedMessage: 'Agent spawned',
    },
    {
      name: 'Valid task orchestration',
      args: ['orchestrate', 'Create a test application'],
      expectFailure: false,
      expectedMessage: 'Task orchestrated',
    },
  ];

  let passed = 0;
  let failed = 0;

  for (const test of tests) {
    console.log(`\n🔍 Testing: ${test.name}`);
    console.log(`   Command: ruv-swarm ${test.args.join(' ')}`);

    const result = await runCommand(test.args);
    const output = result.stdout + result.stderr;

    if (test.expectFailure) {
      if (result.code !== 0 && output.includes(test.expectedMessage)) {
        console.log(`   ✅ PASS - Correctly rejected with: "${test.expectedMessage}"`);
        passed++;
      } else {
        console.log(`   ❌ FAIL - Expected failure with "${test.expectedMessage}", got: ${output.substring(0, 100)}...`);
        failed++;
      }
    } else {
      if (result.code === 0 && output.includes(test.expectedMessage)) {
        console.log(`   ✅ PASS - Successfully executed with: "${test.expectedMessage}"`);
        passed++;
      } else {
        console.log(`   ❌ FAIL - Expected success with "${test.expectedMessage}", got: ${output.substring(0, 100)}...`);
        failed++;
      }
    }
  }

  console.log(`\n📊 Test Results:`);
  console.log(`   ✅ Passed: ${passed}`);
  console.log(`   ❌ Failed: ${failed}`);
  console.log(`   📈 Success Rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);

  if (failed === 0) {
    console.log('\n🎉 All validation tests passed! Input validation is working correctly.');
    process.exit(0);
  } else {
    console.log('\n⚠️  Some validation tests failed. Please check the implementation.');
    process.exit(1);
  }
}

runTests().catch(console.error);
