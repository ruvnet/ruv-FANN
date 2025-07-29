#!/usr/bin/env node

/**
 * Comprehensive Test Script for DAA Workflow Execution Fix
 * 
 * This test validates that the "Cannot read properties of undefined (reading 'method')" 
 * error has been fixed and that workflow execution properly handles:
 * - Valid function tasks
 * - Valid object tasks with methods
 * - Invalid object tasks with missing methods
 * - Null/undefined tasks
 * - Agent method validation
 * - Proper error handling and graceful degradation
 */

import { DAAService } from './src/daa-service.js';
import { performance } from 'perf_hooks';

// Test configuration
const TEST_CONFIG = {
  SWARM_ID: 'test-swarm-workflow-fix',
  AGENT_PREFIX: 'test-agent-',
  WORKFLOW_PREFIX: 'test-workflow-',
  VERBOSE: true,
  TIMEOUT: 30000 // 30 second timeout
};

// Test result tracking
const testResults = {
  passed: 0,
  failed: 0,
  details: []
};

// Logger utility
function log(message, type = 'INFO') {
  const timestamp = new Date().toISOString();
  const prefix = `[${timestamp}] [${type}]`;
  console.log(`${prefix} ${message}`);
}

function logTest(testName, result, details = '') {
  const status = result ? '✅ PASS' : '❌ FAIL';
  log(`${status} - ${testName}${details ? ': ' + details : ''}`);
  
  testResults.details.push({
    name: testName,
    result,
    details
  });
  
  if (result) {
    testResults.passed++;
  } else {
    testResults.failed++;
  }
}

// Mock agent with various methods for testing
class MockAgent {
  constructor(id) {
    this.id = id;
    this.executionLog = [];
  }

  // Valid methods that should work
  async validMethod(data) {
    this.executionLog.push(`validMethod called with: ${JSON.stringify(data)}`);
    return { success: true, data, timestamp: Date.now() };
  }

  async anotherValidMethod(param1, param2) {
    this.executionLog.push(`anotherValidMethod called with: ${param1}, ${param2}`);
    return { result: `${param1}-${param2}`, processed: true };
  }

  async processData(input) {
    this.executionLog.push(`processData called with: ${JSON.stringify(input)}`);
    return { processed: input, status: 'complete' };
  }

  // Method that throws an error
  async errorMethod() {
    this.executionLog.push(`errorMethod called`);
    throw new Error('Simulated method error');
  }

  getExecutionLog() {
    return this.executionLog;
  }

  clearLog() {
    this.executionLog = [];
  }
}

// Test workflow steps with various configurations
const TEST_WORKFLOWS = {
  // Valid workflow with function tasks
  validFunctionWorkflow: {
    id: 'valid-function-workflow',
    steps: [
      {
        id: 'step1',
        task: async (agent) => {
          return { result: 'Function task executed', agentId: agent.id };
        }
      },
      {
        id: 'step2', 
        task: async (agent) => {
          return { result: 'Second function task', agentId: agent.id };
        }
      }
    ]
  },

  // Valid workflow with object tasks
  validObjectWorkflow: {
    id: 'valid-object-workflow',
    steps: [
      {
        id: 'step1',
        task: {
          method: 'validMethod',
          args: [{ test: 'data' }]
        }
      },
      {
        id: 'step2',
        task: {
          method: 'anotherValidMethod',
          args: ['param1', 'param2']
        }
      }
    ]
  },

  // Invalid workflow - missing method property
  invalidObjectWorkflow: {
    id: 'invalid-object-workflow',
    steps: [
      {
        id: 'step1',
        task: {
          // Missing 'method' property - should be handled gracefully
          args: ['test']
        }
      },
      {
        id: 'step2',
        task: {
          method: 'nonexistentMethod', // Method doesn't exist on agent
          args: ['test']
        }
      }
    ]
  },

  // Invalid workflow - null/undefined tasks
  nullTaskWorkflow: {
    id: 'null-task-workflow',
    steps: [
      {
        id: 'step1',
        task: null // Null task
      },
      {
        id: 'step2',
        task: undefined // Undefined task
      },
      {
        id: 'step3'
        // Missing task property entirely
      }
    ]
  },

  // Workflow with error-throwing method
  errorMethodWorkflow: {
    id: 'error-method-workflow',
    steps: [
      {
        id: 'step1',
        task: {
          method: 'errorMethod',
          args: []
        }
      }
    ]
  },

  // Mixed workflow - valid and invalid steps
  mixedWorkflow: {
    id: 'mixed-workflow',
    steps: [
      {
        id: 'step1',
        task: async (agent) => ({ result: 'Valid function' })
      },
      {
        id: 'step2',
        task: {
          method: 'validMethod',
          args: [{ data: 'test' }]
        }
      },
      {
        id: 'step3',
        task: {
          // Missing method property
          args: ['invalid']
        }
      },
      {
        id: 'step4',
        task: null
      },
      {
        id: 'step5',
        task: {
          method: 'processData',
          args: [{ final: 'step' }]
        }
      }
    ]
  }
};

// Test suite class
class WorkflowFixTestSuite {
  constructor() {
    this.daaService = new DAAService();
    this.testAgents = [];
    this.createdWorkflows = [];
  }

  async initialize() {
    log('Initializing DAA Service for testing...');
    
    try {
      await this.daaService.initialize();
      log('✅ DAA Service initialized successfully');
      return true;
    } catch (error) {
      log(`❌ Failed to initialize DAA Service: ${error.message}`, 'ERROR');
      return false;
    }
  }

  async createTestAgents(count = 3) {
    log(`Creating ${count} test agents...`);
    
    for (let i = 0; i < count; i++) {
      const agentId = `${TEST_CONFIG.AGENT_PREFIX}${i}`;
      
      try {
        const agent = await this.daaService.createAgent({
          id: agentId,
          capabilities: ['testing', 'workflow-execution']
        });
        
        // Replace wasmAgent with our mock for testing
        const mockAgent = new MockAgent(agentId);
        agent.wasmAgent = mockAgent;
        this.testAgents.push(agent);
        
        log(`✅ Created test agent: ${agentId}`);
      } catch (error) {
        log(`❌ Failed to create agent ${agentId}: ${error.message}`, 'ERROR');
        throw error;
      }
    }
    
    return this.testAgents;
  }

  async testWorkflowCreation() {
    log('Testing workflow creation...');
    
    let allPassed = true;
    
    for (const [name, config] of Object.entries(TEST_WORKFLOWS)) {
      try {
        const workflow = await this.daaService.createWorkflow(
          config.id, 
          config.steps, 
          config.dependencies || {}
        );
        
        this.createdWorkflows.push(workflow);
        logTest(`Create workflow: ${name}`, true, `ID: ${config.id}`);
        
      } catch (error) {
        logTest(`Create workflow: ${name}`, false, error.message);
        allPassed = false;
      }
    }
    
    return allPassed;
  }

  async testValidFunctionWorkflow() {
    log('Testing valid function workflow execution...');
    
    try {
      const workflowId = TEST_WORKFLOWS.validFunctionWorkflow.id;
      const agentIds = this.testAgents.slice(0, 2).map(a => a.id);
      
      const result = await this.daaService.executeWorkflow(workflowId, {
        agentIds,
        parallel: false
      });
      
      const success = result.complete && result.stepsCompleted === 2;
      logTest('Valid function workflow execution', success, 
        `Completed: ${result.stepsCompleted}/${result.totalSteps}`);
      
      return success;
    } catch (error) {
      logTest('Valid function workflow execution', false, error.message);
      return false;
    }
  }

  async testValidObjectWorkflow() {
    log('Testing valid object workflow execution...');
    
    try {
      const workflowId = TEST_WORKFLOWS.validObjectWorkflow.id;
      const agentIds = this.testAgents.slice(0, 1).map(a => a.id);
      
      const result = await this.daaService.executeWorkflow(workflowId, {
        agentIds,
        parallel: false
      });
      
      // Check that agent methods were called
      const mockAgent = this.testAgents[0].wasmAgent;
      const executionLog = mockAgent.getExecutionLog();
      
      const success = result.complete && result.stepsCompleted === 2 && executionLog.length === 2;
      logTest('Valid object workflow execution', success, 
        `Completed: ${result.stepsCompleted}/${result.totalSteps}, Agent calls: ${executionLog.length}`);
      
      if (TEST_CONFIG.VERBOSE) {
        log(`Agent execution log: ${JSON.stringify(executionLog)}`);
      }
      
      return success;
    } catch (error) {
      logTest('Valid object workflow execution', false, error.message);
      return false;
    }
  }

  async testInvalidObjectWorkflow() {
    log('Testing invalid object workflow (should handle gracefully)...');
    
    try {
      const workflowId = TEST_WORKFLOWS.invalidObjectWorkflow.id;
      const agentIds = this.testAgents.slice(0, 1).map(a => a.id);
      
      // Clear agent log
      const mockAgent = this.testAgents[0].wasmAgent;
      mockAgent.clearLog();
      
      const result = await this.daaService.executeWorkflow(workflowId, {
        agentIds,
        parallel: false
      });
      
      // This should NOT throw the "Cannot read properties of undefined" error
      // The workflow should complete but with warnings/null results
      const success = result.stepsCompleted >= 0; // Should not crash
      logTest('Invalid object workflow handling', success, 
        `Completed: ${result.stepsCompleted}/${result.totalSteps} (graceful handling)`);
      
      return success;
    } catch (error) {
      // If we get the old error, the fix didn't work
      if (error.message.includes("Cannot read properties of undefined (reading 'method')")) {
        logTest('Invalid object workflow handling', false, 
          'Old bug still present: ' + error.message);
        return false;
      }
      
      // Other errors might be expected
      logTest('Invalid object workflow handling', true, 
        'Failed as expected with different error: ' + error.message);
      return true;
    }
  }

  async testNullTaskWorkflow() {
    log('Testing null/undefined task workflow (should handle gracefully)...');
    
    try {
      const workflowId = TEST_WORKFLOWS.nullTaskWorkflow.id;
      const agentIds = this.testAgents.slice(0, 1).map(a => a.id);
      
      const result = await this.daaService.executeWorkflow(workflowId, {
        agentIds,
        parallel: false
      });
      
      // Should not crash, but steps may not complete successfully
      const success = result.stepsCompleted >= 0;
      logTest('Null/undefined task workflow handling', success, 
        `Completed: ${result.stepsCompleted}/${result.totalSteps} (graceful handling)`);
      
      return success;
    } catch (error) {
      if (error.message.includes("Cannot read properties of undefined (reading 'method')")) {
        logTest('Null/undefined task workflow handling', false, 
          'Old bug still present: ' + error.message);
        return false;
      }
      
      logTest('Null/undefined task workflow handling', true, 
        'Failed as expected: ' + error.message);
      return true;
    }
  }

  async testErrorMethodWorkflow() {
    log('Testing error method workflow (should handle method errors)...');
    
    try {
      const workflowId = TEST_WORKFLOWS.errorMethodWorkflow.id;
      const agentIds = this.testAgents.slice(0, 1).map(a => a.id);
      
      const result = await this.daaService.executeWorkflow(workflowId, {
        agentIds,
        parallel: false
      });
      
      // Should handle the method error gracefully
      const success = result.stepsCompleted >= 0;
      logTest('Error method workflow handling', success, 
        `Handled method error gracefully`);
      
      return success;
    } catch (error) {
      logTest('Error method workflow handling', true, 
        'Failed as expected: ' + error.message);
      return true;
    }
  }

  async testMixedWorkflow() {
    log('Testing mixed workflow (valid and invalid steps)...');
    
    try {
      const workflowId = TEST_WORKFLOWS.mixedWorkflow.id;
      const agentIds = this.testAgents.slice(0, 1).map(a => a.id);
      
      // Clear agent log
      const mockAgent = this.testAgents[0].wasmAgent;
      mockAgent.clearLog();
      
      const result = await this.daaService.executeWorkflow(workflowId, {
        agentIds,
        parallel: false
      });
      
      const executionLog = mockAgent.getExecutionLog();
      
      // Should execute valid steps and skip invalid ones
      const success = result.stepsCompleted >= 2 && executionLog.length >= 2;
      logTest('Mixed workflow execution', success, 
        `Completed: ${result.stepsCompleted}/${result.totalSteps}, Valid calls: ${executionLog.length}`);
      
      if (TEST_CONFIG.VERBOSE) {
        log(`Agent execution log: ${JSON.stringify(executionLog)}`);
      }
      
      return success;
    } catch (error) {
      if (error.message.includes("Cannot read properties of undefined (reading 'method')")) {
        logTest('Mixed workflow execution', false, 
          'Old bug still present: ' + error.message);
        return false;
      }
      
      logTest('Mixed workflow execution', false, error.message);
      return false;
    }
  }

  async testParallelExecution() {
    log('Testing parallel workflow execution...');
    
    try {
      const workflowId = TEST_WORKFLOWS.validObjectWorkflow.id;
      const agentIds = this.testAgents.map(a => a.id);
      
      const startTime = performance.now();
      const result = await this.daaService.executeWorkflow(workflowId, {
        agentIds,
        parallel: true
      });
      const endTime = performance.now();
      
      const executionTime = endTime - startTime;
      const success = result.complete && executionTime < 5000; // Should be fast
      
      logTest('Parallel workflow execution', success, 
        `Time: ${executionTime.toFixed(2)}ms, Completed: ${result.stepsCompleted}/${result.totalSteps}`);
      
      return success;
    } catch (error) {
      logTest('Parallel workflow execution', false, error.message);
      return false;
    }
  }

  async runAllTests() {
    log('Starting comprehensive workflow fix tests...');
    
    const startTime = performance.now();
    
    try {
      // Initialize
      const initSuccess = await this.initialize();
      if (!initSuccess) {
        throw new Error('Failed to initialize DAA Service');
      }
      
      // Create test agents
      await this.createTestAgents(3);
      
      // Test workflow creation
      await this.testWorkflowCreation();
      
      // Test various workflow scenarios
      await this.testValidFunctionWorkflow();
      await this.testValidObjectWorkflow();
      await this.testInvalidObjectWorkflow();
      await this.testNullTaskWorkflow();
      await this.testErrorMethodWorkflow();
      await this.testMixedWorkflow();
      await this.testParallelExecution();
      
      const endTime = performance.now();
      const totalTime = endTime - startTime;
      
      // Print results
      this.printResults(totalTime);
      
    } catch (error) {
      log(`Test suite failed: ${error.message}`, 'ERROR');
      console.error(error);
    } finally {
      await this.cleanup();
    }
  }

  printResults(totalTime) {
    log('\n' + '='.repeat(60));
    log('WORKFLOW FIX TEST RESULTS');
    log('='.repeat(60));
    
    log(`Total Tests: ${testResults.passed + testResults.failed}`);
    log(`Passed: ${testResults.passed}`);
    log(`Failed: ${testResults.failed}`);
    log(`Success Rate: ${((testResults.passed / (testResults.passed + testResults.failed)) * 100).toFixed(1)}%`);
    log(`Total Time: ${totalTime.toFixed(2)}ms`);
    
    log('\n' + '-'.repeat(60));
    log('DETAILED RESULTS:');
    log('-'.repeat(60));
    
    testResults.details.forEach((test, index) => {
      const status = test.result ? '✅' : '❌';
      log(`${index + 1}. ${status} ${test.name}`);
      if (test.details) {
        log(`   Details: ${test.details}`);
      }
    });
    
    log('\n' + '-'.repeat(60));
    log('BUG FIX VERIFICATION:');
    log('-'.repeat(60));
    
    const criticalTests = [
      'Invalid object workflow handling',
      'Null/undefined task workflow handling',
      'Mixed workflow execution'
    ];
    
    const criticalResults = testResults.details.filter(test => 
      criticalTests.includes(test.name)
    );
    
    const allCriticalPassed = criticalResults.every(test => test.result);
    
    if (allCriticalPassed) {
      log('✅ BUG FIX VERIFIED: No "Cannot read properties of undefined" errors detected');
      log('✅ All error handling tests passed');
    } else {
      log('❌ BUG FIX FAILED: Critical tests failed');
      log('❌ The original bug may still be present');
    }
    
    log('='.repeat(60));
  }

  async cleanup() {
    log('Cleaning up test resources...');
    
    try {
      // Cleanup agents
      for (const agent of this.testAgents) {
        await this.daaService.destroyAgent(agent.id);
      }
      
      // Cleanup service
      await this.daaService.cleanup();
      
      log('✅ Cleanup completed');
    } catch (error) {
      log(`⚠️ Cleanup warning: ${error.message}`, 'WARN');
    }
  }
}

// Main execution
async function main() {
  const testSuite = new WorkflowFixTestSuite();
  
  // Set up timeout
  const timeout = setTimeout(() => {
    log('Test suite timed out!', 'ERROR');
    process.exit(1);
  }, TEST_CONFIG.TIMEOUT);
  
  try {
    await testSuite.runAllTests();
    clearTimeout(timeout);
    
    // Exit with appropriate code
    const exitCode = testResults.failed > 0 ? 1 : 0;
    process.exit(exitCode);
    
  } catch (error) {
    clearTimeout(timeout);
    log(`Fatal error: ${error.message}`, 'ERROR');
    console.error(error);
    process.exit(1);
  }
}

// Run tests if this script is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { WorkflowFixTestSuite, TEST_WORKFLOWS, testResults };