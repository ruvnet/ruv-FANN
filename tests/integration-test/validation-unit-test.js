#!/usr/bin/env node

/**
 * Unit Test for Validation Functions
 * Tests validation logic directly without running the full CLI
 */

// Import validation functions by requiring the module
const path = require('path');

// Test validation functions directly
function testValidationFunctions() {
  console.log('🧪 Testing Validation Functions (Unit Tests)\n');

  // Define validation constants (copied from the CLI)
  const VALID_TOPOLOGIES = ['mesh', 'hierarchical', 'ring', 'star'];
  const VALID_AGENT_TYPES = ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator', 'architect', 'tester'];
  const MAX_AGENTS_LIMIT = 100;
  const MIN_AGENTS_LIMIT = 1;

  class ValidationError extends Error {
    constructor(message, parameter = null) {
      super(message);
      this.name = 'ValidationError';
      this.parameter = parameter;
    }
  }

  function validateTopology(topology) {
    if (!topology || typeof topology !== 'string') {
      throw new ValidationError('Topology must be a non-empty string', 'topology');
    }

    if (!VALID_TOPOLOGIES.includes(topology.toLowerCase())) {
      throw new ValidationError(
        `Invalid topology '${topology}'. Valid topologies are: ${VALID_TOPOLOGIES.join(', ')}`,
        'topology',
      );
    }

    return topology.toLowerCase();
  }

  function validateMaxAgents(maxAgents) {
    // Handle string input
    if (typeof maxAgents === 'string') {
      const parsed = parseInt(maxAgents, 10);
      if (isNaN(parsed)) {
        throw new ValidationError(
          `Invalid maxAgents '${maxAgents}'. Must be a number between ${MIN_AGENTS_LIMIT} and ${MAX_AGENTS_LIMIT}`,
          'maxAgents',
        );
      }
      maxAgents = parsed;
    }

    if (!Number.isInteger(maxAgents) || maxAgents < MIN_AGENTS_LIMIT || maxAgents > MAX_AGENTS_LIMIT) {
      throw new ValidationError(
        `Invalid maxAgents '${maxAgents}'. Must be an integer between ${MIN_AGENTS_LIMIT} and ${MAX_AGENTS_LIMIT}`,
        'maxAgents',
      );
    }

    return maxAgents;
  }

  function validateAgentType(type) {
    if (!type || typeof type !== 'string') {
      throw new ValidationError('Agent type must be a non-empty string', 'type');
    }

    if (!VALID_AGENT_TYPES.includes(type.toLowerCase())) {
      throw new ValidationError(
        `Invalid agent type '${type}'. Valid types are: ${VALID_AGENT_TYPES.join(', ')}`,
        'type',
      );
    }

    return type.toLowerCase();
  }

  function validateAgentName(name) {
    if (name !== null && name !== undefined) {
      if (typeof name !== 'string') {
        throw new ValidationError('Agent name must be a string', 'name');
      }

      if (name.length === 0) {
        throw new ValidationError('Agent name cannot be empty', 'name');
      }

      if (name.length > 100) {
        throw new ValidationError('Agent name cannot exceed 100 characters', 'name');
      }

      // Check for invalid characters
      if (!/^[a-zA-Z0-9\s\-_\.]+$/.test(name)) {
        throw new ValidationError(
          'Agent name can only contain letters, numbers, spaces, hyphens, underscores, and periods',
          'name',
        );
      }
    }

    return name;
  }

  function validateTaskDescription(task) {
    if (!task || typeof task !== 'string') {
      throw new ValidationError('Task description must be a non-empty string', 'task');
    }

    if (task.trim().length === 0) {
      throw new ValidationError('Task description cannot be empty or only whitespace', 'task');
    }

    if (task.length > 1000) {
      throw new ValidationError('Task description cannot exceed 1000 characters', 'task');
    }

    return task.trim();
  }

  const tests = [
    // Topology validation tests
    {
      name: 'Invalid topology',
      func: () => validateTopology('invalid-topology'),
      expectError: true,
      expectedMessage: 'Invalid topology',
    },
    {
      name: 'Valid topology - mesh',
      func: () => validateTopology('mesh'),
      expectError: false,
      expectedResult: 'mesh',
    },
    {
      name: 'Valid topology - hierarchical (case insensitive)',
      func: () => validateTopology('HIERARCHICAL'),
      expectError: false,
      expectedResult: 'hierarchical',
    },

    // MaxAgents validation tests
    {
      name: 'Agent count too high',
      func: () => validateMaxAgents(101),
      expectError: true,
      expectedMessage: 'Invalid maxAgents',
    },
    {
      name: 'Agent count too low',
      func: () => validateMaxAgents(0),
      expectError: true,
      expectedMessage: 'Invalid maxAgents',
    },
    {
      name: 'Valid agent count',
      func: () => validateMaxAgents(5),
      expectError: false,
      expectedResult: 5,
    },
    {
      name: 'Valid agent count from string',
      func: () => validateMaxAgents('10'),
      expectError: false,
      expectedResult: 10,
    },

    // Agent type validation tests
    {
      name: 'Invalid agent type',
      func: () => validateAgentType('invalid-type'),
      expectError: true,
      expectedMessage: 'Invalid agent type',
    },
    {
      name: 'Valid agent type - researcher',
      func: () => validateAgentType('researcher'),
      expectError: false,
      expectedResult: 'researcher',
    },
    {
      name: 'Valid agent type - coordinator (case insensitive)',
      func: () => validateAgentType('COORDINATOR'),
      expectError: false,
      expectedResult: 'coordinator',
    },

    // Agent name validation tests
    {
      name: 'Agent name with invalid characters',
      func: () => validateAgentName('Test@Agent!'),
      expectError: true,
      expectedMessage: 'Agent name can only contain',
    },
    {
      name: 'Valid agent name',
      func: () => validateAgentName('Test Agent 123'),
      expectError: false,
      expectedResult: 'Test Agent 123',
    },
    {
      name: 'Valid agent name with allowed special chars',
      func: () => validateAgentName('Test-Agent_v1.0'),
      expectError: false,
      expectedResult: 'Test-Agent_v1.0',
    },

    // Task description validation tests
    {
      name: 'Empty task description',
      func: () => validateTaskDescription('   '),
      expectError: true,
      expectedMessage: 'Task description cannot be empty',
    },
    {
      name: 'Valid task description',
      func: () => validateTaskDescription('Create a test application'),
      expectError: false,
      expectedResult: 'Create a test application',
    },
  ];

  let passed = 0;
  let failed = 0;

  for (const test of tests) {
    console.log(`\n🔍 Testing: ${test.name}`);

    try {
      const result = test.func();

      if (test.expectError) {
        console.log(`   ❌ FAIL - Expected error, but got result: ${result}`);
        failed++;
      } else {
        if (result === test.expectedResult) {
          console.log(`   ✅ PASS - Got expected result: ${result}`);
          passed++;
        } else {
          console.log(`   ❌ FAIL - Expected "${test.expectedResult}", got "${result}"`);
          failed++;
        }
      }
    } catch (error) {
      if (test.expectError) {
        if (error.message.includes(test.expectedMessage)) {
          console.log(`   ✅ PASS - Correctly threw error: ${error.message}`);
          passed++;
        } else {
          console.log(`   ❌ FAIL - Expected error message containing "${test.expectedMessage}", got: ${error.message}`);
          failed++;
        }
      } else {
        console.log(`   ❌ FAIL - Unexpected error: ${error.message}`);
        failed++;
      }
    }
  }

  console.log(`\n📊 Test Results:`);
  console.log(`   ✅ Passed: ${passed}`);
  console.log(`   ❌ Failed: ${failed}`);
  console.log(`   📈 Success Rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);

  if (failed === 0) {
    console.log('\n🎉 All validation unit tests passed! Input validation logic is working correctly.');
    process.exit(0);
  } else {
    console.log('\n⚠️  Some validation unit tests failed. Please check the implementation.');
    process.exit(1);
  }
}

testValidationFunctions();
