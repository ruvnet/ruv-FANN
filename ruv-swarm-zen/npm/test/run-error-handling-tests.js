#!/usr/bin/env node

/**
 * Comprehensive Error Handling Test Runner
 * Tests all aspects of the new error handling system
 */

import { jest } from '@jest/globals';

// Test suite configuration
const testConfig = {
  verbose: true,
  testTimeout: 30000,
  setupFilesAfterEnv: ['<rootDir>/test/setup.js'],
  testMatch: [
    '**/test/error-handling-validation.test.js',
    '**/test/mcp-integration.test.js',
  ],
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  collectCoverageFrom: [
    'src/errors.js',
    'src/schemas.js',
    'src/mcp-tools-enhanced.js',
  ],
};

// Mock console to capture error logs during testing
const originalConsole = { ...console };
const consoleCapture = [];

const mockConsole = {
  ...originalConsole,
  error: (...args) => {
    consoleCapture.push({ level: 'error', args });
    originalConsole.error(...args);
  },
  warn: (...args) => {
    consoleCapture.push({ level: 'warn', args });
    originalConsole.warn(...args);
  },
  log: (...args) => {
    consoleCapture.push({ level: 'log', args });
    originalConsole.log(...args);
  },
};

// Test result collector
class TestResults {
  constructor() {
    this.results = {
      errorClasses: { passed: 0, failed: 0, tests: [] },
      validation: { passed: 0, failed: 0, tests: [] },
      integration: { passed: 0, failed: 0, tests: [] },
      performance: { passed: 0, failed: 0, tests: [] },
    };
    this.startTime = Date.now();
  }

  addResult(category, testName, passed, error = null, duration = 0) {
    const result = {
      name: testName,
      passed,
      error: error ? error.message : null,
      duration,
    };

    this.results[category].tests.push(result);
    if (passed) {
      this.results[category].passed++;
    } else {
      this.results[category].failed++;
    }
  }

  getSummary() {
    const totalTime = Date.now() - this.startTime;
    const totalPassed = Object.values(this.results).reduce((sum, cat) => sum + cat.passed, 0);
    const totalFailed = Object.values(this.results).reduce((sum, cat) => sum + cat.failed, 0);
    const totalTests = totalPassed + totalFailed;

    return {
      totalTests,
      totalPassed,
      totalFailed,
      totalTime,
      passRate: totalTests > 0 ? (totalPassed / totalTests * 100).toFixed(2) : 0,
      categories: this.results,
    };
  }
}

// Main test runner
async function runErrorHandlingTests() {
  console.log('🧪 Starting Comprehensive Error Handling Tests');
  console.log('=' * 60);

  const results = new TestResults();

  // Replace console for testing
  Object.assign(console, mockConsole);

  try {
    // Test 1: Error Class Functionality
    console.log('\\n📋 Testing Error Classes...');
    await testErrorClasses(results);

    // Test 2: Validation System
    console.log('\\n🔍 Testing Validation System...');
    await testValidationSystem(results);

    // Test 3: MCP Integration
    console.log('\\n🔗 Testing MCP Integration...');
    await testMCPIntegration(results);

    // Test 4: Performance and Edge Cases
    console.log('\\n⚡ Testing Performance and Edge Cases...');
    await testPerformanceAndEdgeCases(results);

  } catch (error) {
    console.error('❌ Test runner error:', error);
  } finally {
    // Restore console
    Object.assign(console, originalConsole);
  }

  // Generate final report
  generateReport(results);
}

async function testErrorClasses(results) {
  const {
    ValidationError,
    SwarmError,
    AgentError,
    TaskError,
    NeuralError,
    WasmError,
    ErrorFactory,
    ErrorContext,
  } = await import('../src/errors.js');

  // Test ValidationError
  await runTest(results, 'errorClasses', 'ValidationError creation and properties', () => {
    const error = new ValidationError('Test error', 'testField', 'badValue', 'string');

    if (error.name !== 'ValidationError') {
      throw new Error('Wrong error name');
    }
    if (error.code !== 'VALIDATION_ERROR') {
      throw new Error('Wrong error code');
    }
    if (error.field !== 'testField') {
      throw new Error('Wrong field');
    }
    if (error.value !== 'badValue') {
      throw new Error('Wrong value');
    }
    if (error.expectedType !== 'string') {
      throw new Error('Wrong expected type');
    }

    const suggestions = error.getSuggestions();
    if (!suggestions.some(s => s.includes('testField'))) {
      throw new Error('Missing field-specific suggestion');
    }
  });

  // Test SwarmError
  await runTest(results, 'errorClasses', 'SwarmError with context', () => {
    const error = new SwarmError('Swarm failed', 'test-swarm-id', 'initialization');

    if (error.name !== 'SwarmError') {
      throw new Error('Wrong error name');
    }
    if (error.swarmId !== 'test-swarm-id') {
      throw new Error('Wrong swarm ID');
    }
    if (error.operation !== 'initialization') {
      throw new Error('Wrong operation');
    }

    const suggestions = error.getSuggestions();
    if (suggestions.length === 0) {
      throw new Error('No suggestions provided');
    }
  });

  // Test ErrorFactory
  await runTest(results, 'errorClasses', 'ErrorFactory creates correct error types', () => {
    const validationError = ErrorFactory.createError('validation', 'Test validation', {
      field: 'test',
      value: 'bad',
      expectedType: 'number',
    });

    if (!(validationError instanceof ValidationError)) {
      throw new Error('Factory did not create ValidationError');
    }

    const swarmError = ErrorFactory.createError('swarm', 'Test swarm error', {
      swarmId: 'test',
      operation: 'test',
    });

    if (!(swarmError instanceof SwarmError)) {
      throw new Error('Factory did not create SwarmError');
    }
  });

  // Test ErrorContext
  await runTest(results, 'errorClasses', 'ErrorContext management', () => {
    const context = new ErrorContext();
    context.set('operation', 'test');
    context.set('timestamp', '2023-01-01');

    if (context.get('operation') !== 'test') {
      throw new Error('Context get failed');
    }

    const obj = context.toObject();
    if (obj.operation !== 'test' || obj.timestamp !== '2023-01-01') {
      throw new Error('Context toObject failed');
    }

    context.clear();
    if (Object.keys(context.toObject()).length > 0) {
      throw new Error('Context clear failed');
    }
  });
}

async function testValidationSystem(results) {
  const { ValidationUtils } = await import('../src/schemas.js');
  const { ValidationError } = await import('../src/errors.js');

  // Test parameter validation
  await runTest(results, 'validation', 'Parameter validation with valid inputs', () => {
    const params = {
      topology: 'mesh',
      maxAgents: 10,
      strategy: 'balanced',
    };

    const result = ValidationUtils.validateParams(params, 'swarm_init');
    if (result.topology !== 'mesh') {
      throw new Error('Topology validation failed');
    }
    if (result.maxAgents !== 10) {
      throw new Error('MaxAgents validation failed');
    }
    if (result.strategy !== 'balanced') {
      throw new Error('Strategy validation failed');
    }
  });

  // Test validation with defaults
  await runTest(results, 'validation', 'Parameter validation with defaults', () => {
    const params = {};
    const result = ValidationUtils.validateParams(params, 'swarm_init');

    if (result.topology !== 'mesh') {
      throw new Error('Default topology not applied');
    }
    if (result.maxAgents !== 5) {
      throw new Error('Default maxAgents not applied');
    }
    if (result.strategy !== 'balanced') {
      throw new Error('Default strategy not applied');
    }
  });

  // Test validation errors
  await runTest(results, 'validation', 'Parameter validation with invalid inputs', () => {
    const params = {
      topology: 'invalid-topology',
      maxAgents: 200, // Over limit
    };

    let errorCaught = false;
    try {
      ValidationUtils.validateParams(params, 'swarm_init');
    } catch (error) {
      errorCaught = true;
      if (!(error instanceof ValidationError)) {
        throw new Error('Expected ValidationError');
      }
    }

    if (!errorCaught) {
      throw new Error('Expected validation to fail');
    }
  });

  // Test input sanitization
  await runTest(results, 'validation', 'Input sanitization', () => {
    const malicious = '<script>alert(\"xss\")</script>';
    const sanitized = ValidationUtils.sanitizeInput(malicious);

    if (sanitized.includes('<script>')) {
      throw new Error('Sanitization failed to remove script tags');
    }

    if (sanitized.includes('alert')) {
      throw new Error('Sanitization incomplete');
    }
  });

  // Test schema documentation
  await runTest(results, 'validation', 'Schema documentation generation', () => {
    const doc = ValidationUtils.getSchemaDoc('swarm_init');

    if (!doc || !doc.parameters) {
      throw new Error('Schema doc missing');
    }
    if (!doc.parameters.topology) {
      throw new Error('Topology parameter missing');
    }
    if (!doc.parameters.topology.allowedValues) {
      throw new Error('Allowed values missing');
    }

    const allowedValues = doc.parameters.topology.allowedValues;
    if (!allowedValues.includes('mesh')) {
      throw new Error('Expected allowed values missing');
    }
  });
}

async function testMCPIntegration(results) {
  // Mock the dependencies
  const mockRuvSwarm = {
    initialize: jest.fn(),
    createSwarm: jest.fn(),
    features: {
      neural_networks: true,
      forecasting: true,
      cognitive_diversity: true,
      simd_support: true,
    },
    wasmLoader: {
      getTotalMemoryUsage: jest.fn(() => 1024 * 1024),
      getModuleStatus: jest.fn(() => ({ core: { loaded: true } })),
    },
  };

  const mockPersistence = {
    createSwarm: jest.fn(),
    createAgent: jest.fn(),
    getActiveSwarms: jest.fn(() => []),
    getSwarmAgents: jest.fn(() => []),
  };

  // Test MCP tools initialization
  await runTest(results, 'integration', 'MCP Tools initialization', async() => {
    const { EnhancedMCPTools } = await import('../src/mcp-tools-enhanced.js');
    const tools = new EnhancedMCPTools(mockRuvSwarm);

    if (!tools.errorContext) {
      throw new Error('Error context not initialized');
    }
    if (!tools.errorLog) {
      throw new Error('Error log not initialized');
    }
    if (!Array.isArray(tools.errorLog)) {
      throw new Error('Error log not array');
    }
  });

  // Test error handling in swarm_init
  await runTest(results, 'integration', 'swarm_init error handling', async() => {
    const { EnhancedMCPTools } = await import('../src/mcp-tools-enhanced.js');
    const tools = new EnhancedMCPTools();

    // Mock to throw error
    tools.validateToolParams = jest.fn().mockImplementation(() => {
      throw new ValidationError('Invalid topology', 'topology', 'invalid', 'string');
    });

    let errorCaught = false;
    try {
      await tools.swarm_init({ topology: 'invalid' });
    } catch (error) {
      errorCaught = true;
      if (!(error instanceof ValidationError)) {
        throw new Error('Expected ValidationError to be thrown');
      }
    }

    if (!errorCaught) {
      throw new Error('Expected swarm_init to throw error');
    }

    // Check error was logged
    if (tools.errorLog.length === 0) {
      throw new Error('Error not logged');
    }
  });

  // Test error statistics
  await runTest(results, 'integration', 'Error statistics tracking', async() => {
    const { EnhancedMCPTools } = await import('../src/mcp-tools-enhanced.js');
    const { ValidationError, WasmError } = await import('../src/errors.js');

    const tools = new EnhancedMCPTools();

    // Simulate some errors
    tools.handleError(new ValidationError('Error 1'), 'swarm_init', 'test');
    tools.handleError(new WasmError('Error 2'), 'agent_spawn', 'test');

    const stats = tools.getErrorStats();

    if (stats.total !== 2) {
      throw new Error('Wrong total error count');
    }
    if (stats.bySeverity.medium !== 1) {
      throw new Error('Wrong medium severity count');
    }
    if (stats.bySeverity.high !== 1) {
      throw new Error('Wrong high severity count');
    }
    if (stats.byTool.swarm_init !== 1) {
      throw new Error('Wrong tool count');
    }
  });
}

async function testPerformanceAndEdgeCases(results) {
  const { ValidationUtils } = await import('../src/schemas.js');
  const { ErrorFactory } = await import('../src/errors.js');

  // Test performance with large inputs
  await runTest(results, 'performance', 'Large input validation performance', () => {
    const startTime = Date.now();

    // Test with large capability array
    const params = {
      type: 'researcher',
      capabilities: Array.from({ length: 1000 }, (_, i) => `capability_${i}`),
    };

    const result = ValidationUtils.validateParams(params, 'agent_spawn');
    const endTime = Date.now();

    if (endTime - startTime > 100) {
      throw new Error(`Validation too slow: ${endTime - startTime}ms`);
    }

    if (result.capabilities.length !== 1000) {
      throw new Error('Large array validation failed');
    }
  });

  // Test edge case: empty parameters
  await runTest(results, 'performance', 'Empty parameters handling', () => {
    const result = ValidationUtils.validateParams({}, 'swarm_init');

    // Should apply defaults
    if (!result.topology) {
      throw new Error('Default not applied for empty params');
    }
  });

  // Test edge case: null parameters
  await runTest(results, 'performance', 'Null parameters handling', () => {
    const result = ValidationUtils.validateParams(null, 'swarm_init');

    // Should treat as empty object and apply defaults
    if (!result.topology) {
      throw new Error('Default not applied for null params');
    }
  });

  // Test memory usage with many errors
  await runTest(results, 'performance', 'Memory usage with error log limit', async() => {
    const { EnhancedMCPTools } = await import('../src/mcp-tools-enhanced.js');
    const tools = new EnhancedMCPTools();

    // Set a small limit for testing
    tools.maxErrorLogSize = 10;

    // Add more errors than the limit
    for (let i = 0; i < 15; i++) {
      tools.handleError(new Error(`Error ${i}`), 'test', 'test');
    }

    // Should not exceed limit
    if (tools.errorLog.length > 10) {
      throw new Error(`Error log exceeded limit: ${tools.errorLog.length}`);
    }

    // Should contain most recent errors
    const lastError = tools.errorLog[tools.errorLog.length - 1];
    if (!lastError.error.message.includes('Error 14')) {
      throw new Error('Most recent error not preserved');
    }
  });

  // Test error wrapping
  await runTest(results, 'performance', 'Error wrapping functionality', () => {
    const originalError = new Error('Original error message');
    originalError.stack = 'Original stack trace';

    const wrappedError = ErrorFactory.wrapError(originalError, 'wasm', {
      module: 'core',
      operation: 'load',
    });

    if (!wrappedError.message.includes('WASM')) {
      throw new Error('Error type not added to message');
    }

    if (!wrappedError.details.originalError) {
      throw new Error('Original error not preserved');
    }

    if (wrappedError.details.originalError.message !== 'Original error message') {
      throw new Error('Original error message not preserved');
    }
  });
}

// Helper function to run individual tests
async function runTest(results, category, testName, testFunction) {
  const startTime = Date.now();
  try {
    console.log(`  ▶ ${testName}...`);
    await testFunction();
    const duration = Date.now() - startTime;
    console.log(`  ✅ ${testName} (${duration}ms)`);
    results.addResult(category, testName, true, null, duration);
  } catch (error) {
    const duration = Date.now() - startTime;
    console.log(`  ❌ ${testName} - ${error.message} (${duration}ms)`);
    results.addResult(category, testName, false, error, duration);
  }
}

// Generate comprehensive test report
function generateReport(results) {
  const summary = results.getSummary();

  console.log(`\\n${ '='.repeat(60)}`);
  console.log('🧪 ERROR HANDLING TEST RESULTS');
  console.log('='.repeat(60));

  console.log('\\n📊 Summary:');
  console.log(`   Total Tests: ${summary.totalTests}`);
  console.log(`   Passed: ${summary.totalPassed} (${summary.passRate}%)`);
  console.log(`   Failed: ${summary.totalFailed}`);
  console.log(`   Total Time: ${summary.totalTime}ms`);

  // Category breakdown
  Object.entries(summary.categories).forEach(([category, data]) => {
    console.log(`\\n📋 ${category.toUpperCase()}:`);
    console.log(`   Passed: ${data.passed}`);
    console.log(`   Failed: ${data.failed}`);

    if (data.failed > 0) {
      console.log('   Failed Tests:');
      data.tests.filter(t => !t.passed).forEach(test => {
        console.log(`     ❌ ${test.name}: ${test.error}`);
      });
    }
  });

  // Console capture analysis
  if (consoleCapture.length > 0) {
    console.log('\\n📝 Console Output Analysis:');
    const errorLogs = consoleCapture.filter(c => c.level === 'error').length;
    const warnLogs = consoleCapture.filter(c => c.level === 'warn').length;
    const infoLogs = consoleCapture.filter(c => c.level === 'log').length;

    console.log(`   Error logs: ${errorLogs}`);
    console.log(`   Warning logs: ${warnLogs}`);
    console.log(`   Info logs: ${infoLogs}`);
  }

  // Final assessment
  console.log('\\n🎯 Assessment:');
  if (summary.totalFailed === 0) {
    console.log('   ✅ All tests passed! Error handling system is working correctly.');
  } else if (summary.passRate >= 90) {
    console.log('   ⚠️  Most tests passed, but some issues need attention.');
  } else {
    console.log('   ❌ Multiple failures detected. Error handling system needs fixes.');
  }

  // Performance insights
  const avgDuration = summary.totalTests > 0 ? (summary.totalTime / summary.totalTests).toFixed(2) : 0;
  console.log('\\n⚡ Performance:');
  console.log(`   Average test duration: ${avgDuration}ms`);

  if (avgDuration > 50) {
    console.log('   ⚠️  Tests are running slowly, consider optimization');
  } else {
    console.log('   ✅ Test performance is good');
  }

  console.log(`\\n${ '='.repeat(60)}`);
}

// Run the tests
if (import.meta.url === `file://${process.argv[1]}`) {
  runErrorHandlingTests().catch(console.error);
}

export { runErrorHandlingTests };
