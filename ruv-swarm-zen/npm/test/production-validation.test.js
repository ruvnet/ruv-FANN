#!/usr/bin/env node
/**
 * Production Validation Test
 * Tests the recent global state management changes for production readiness
 */

import { RuvSwarm } from '../src/index-enhanced.js';
import { EnhancedMCPTools } from '../src/mcp-tools-enhanced.js';
import { daaMcpTools } from '../src/mcp-daa-tools.js';
import { Logger } from '../src/logger.js';

class ProductionValidationTest {
  constructor() {
    this.logger = new Logger({ name: 'production-test', level: 'INFO' });
    this.testResults = [];
    this.passed = 0;
    this.failed = 0;
  }

  async runTest(testName, testFn) {
    const startTime = Date.now();
    try {
      await testFn();
      const duration = Date.now() - startTime;
      this.testResults.push({
        name: testName,
        status: 'PASSED',
        duration: `${duration}ms`,
        error: null,
      });
      this.passed++;
      console.log(`✅ ${testName} - ${duration}ms`);
    } catch (error) {
      const duration = Date.now() - startTime;
      this.testResults.push({
        name: testName,
        status: 'FAILED',
        duration: `${duration}ms`,
        error: error.message,
      });
      this.failed++;
      console.log(`❌ ${testName} - ${duration}ms - ${error.message}`);
    }
  }

  async testGlobalStateManagement() {
    // Test 1: Verify singleton behavior
    await this.runTest('Global State Singleton Behavior', async () => {
      const instance1 = await RuvSwarm.initialize();
      const instance2 = await RuvSwarm.initialize();

      if (instance1 !== instance2) {
        throw new Error('Multiple instances created - singleton pattern broken');
      }

      // Test memory isolation
      instance1.testProperty = 'test-value';
      if (instance2.testProperty !== 'test-value') {
        throw new Error('Memory isolation broken - instances not sharing state');
      }
    });

    // Test 2: Concurrent access safety
    await this.runTest('Concurrent Access Safety', async () => {
      const promises = [];
      for (let i = 0; i < 10; i++) {
        promises.push(RuvSwarm.initialize());
      }

      const instances = await Promise.all(promises);
      const firstInstance = instances[0];

      // All instances should be the same reference
      for (let i = 1; i < instances.length; i++) {
        if (instances[i] !== firstInstance) {
          throw new Error(`Instance ${i} is not the same reference as first instance`);
        }
      }
    });

    // Test 3: Memory leak prevention
    await this.runTest('Memory Leak Prevention', async () => {
      const initialMemory = process.memoryUsage().heapUsed;

      // Create multiple swarms and agents
      const mcpTools = new EnhancedMCPTools();
      await mcpTools.initialize();

      for (let i = 0; i < 5; i++) {
        await mcpTools.swarm_init({ topology: 'mesh', maxAgents: 5 });
        await mcpTools.agent_spawn({ type: 'researcher', name: `Test Agent ${i}` });
      }

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;

      // Memory increase should be reasonable (less than 50MB)
      if (memoryIncrease > 50 * 1024 * 1024) {
        throw new Error(`Excessive memory usage: ${Math.round(memoryIncrease / 1024 / 1024)}MB increase`);
      }
    });

    // Test 4: MCP Tools Integration
    await this.runTest('MCP Tools Integration', async () => {
      const mcpTools = new EnhancedMCPTools();
      await mcpTools.initialize();

      // Test swarm initialization
      const swarmResult = await mcpTools.swarm_init({
        topology: 'hierarchical',
        maxAgents: 3,
        strategy: 'balanced',
      });

      if (!swarmResult.id || !swarmResult.topology) {
        throw new Error('Swarm initialization failed');
      }

      // Test agent spawning
      const agentResult = await mcpTools.agent_spawn({
        type: 'coder',
        name: 'Production Test Agent',
      });

      if (!agentResult.agent || !agentResult.agent.id) {
        throw new Error('Agent spawning failed');
      }

      // Test task orchestration
      const taskResult = await mcpTools.task_orchestrate({
        task: 'Test production readiness validation',
        strategy: 'adaptive',
      });

      if (!taskResult.taskId || taskResult.status !== 'orchestrated') {
        throw new Error('Task orchestration failed');
      }
    });

    // Test 5: DAA Service Integration
    await this.runTest('DAA Service Integration', async () => {
      // Initialize DAA service
      const daaResult = await daaMcpTools.daa_init({
        enableLearning: true,
        enableCoordination: true,
        persistenceMode: 'memory',
      });

      if (!daaResult.success) {
        throw new Error('DAA initialization failed');
      }

      // Create DAA agent
      const agentResult = await daaMcpTools.daa_agent_create({
        id: 'production-test-agent',
        capabilities: ['learning', 'coordination'],
        cognitivePattern: 'adaptive',
      });

      if (!agentResult.agent || agentResult.agent.id !== 'production-test-agent') {
        throw new Error('DAA agent creation failed');
      }

      // Test knowledge sharing
      const knowledgeResult = await daaMcpTools.daa_knowledge_share({
        sourceAgentId: 'production-test-agent',
        targetAgentIds: ['production-test-agent'],
        knowledgeDomain: 'production-testing',
        knowledgeContent: { test: 'data' },
      });

      if (!knowledgeResult.sharing_complete) {
        throw new Error('Knowledge sharing failed');
      }
    });

    // Test 6: Error Handling and Recovery
    await this.runTest('Error Handling and Recovery', async () => {
      const mcpTools = new EnhancedMCPTools();
      await mcpTools.initialize();

      // Test invalid topology
      try {
        await mcpTools.swarm_init({ topology: 'invalid' });
        throw new Error('Should have thrown error for invalid topology');
      } catch (error) {
        if (!error.message.includes('must be one of')) {
          throw new Error(`Wrong error message for invalid topology: ${error.message}`);
        }
      }

      // Test invalid agent type
      try {
        await mcpTools.agent_spawn({ type: 'invalid' });
        throw new Error('Should have thrown error for invalid agent type');
      } catch (error) {
        if (!error.message.includes('must be one of')) {
          throw new Error(`Wrong error message for invalid agent type: ${error.message}`);
        }
      }

      // Test system recovery after errors
      const validResult = await mcpTools.swarm_init({ topology: 'mesh', maxAgents: 2 });
      if (!validResult.id) {
        throw new Error('System failed to recover after errors');
      }
    });

    // Test 7: Performance Benchmarks
    await this.runTest('Performance Benchmarks', async () => {
      const mcpTools = new EnhancedMCPTools();
      await mcpTools.initialize();

      // Test benchmark execution
      const benchmarkResult = await mcpTools.benchmark_run({
        type: 'swarm',
        iterations: 5,
      });

      if (!benchmarkResult.results || !benchmarkResult.results.swarm) {
        throw new Error('Benchmark execution failed');
      }

      // Validate performance metrics
      const swarmResults = benchmarkResult.results.swarm;
      if (swarmResults.swarm_creation.avg_ms > 1000) {
        throw new Error('Swarm creation too slow');
      }

      if (swarmResults.agent_spawning.avg_ms > 100) {
        throw new Error('Agent spawning too slow');
      }
    });

    // Test 8: Neural Network Integration
    await this.runTest('Neural Network Integration', async () => {
      const mcpTools = new EnhancedMCPTools();
      await mcpTools.initialize();

      // Test neural status
      const statusResult = await mcpTools.neural_status({});
      if (!statusResult.available) {
        throw new Error('Neural networks not available');
      }

      // Test neural patterns
      const patternsResult = await mcpTools.neural_patterns({ pattern: 'all' });
      if (!patternsResult || Object.keys(patternsResult).length === 0) {
        throw new Error('Neural patterns not available');
      }
    });
  }

  async runAllTests() {
    console.log('🧪 Starting Production Validation Tests\n');

    const startTime = Date.now();
    await this.testGlobalStateManagement();
    const totalTime = Date.now() - startTime;

    console.log('\n📊 Test Results Summary:');
    console.log(`✅ Passed: ${this.passed}`);
    console.log(`❌ Failed: ${this.failed}`);
    console.log(`⏱️  Total Time: ${totalTime}ms`);
    console.log(`📈 Success Rate: ${((this.passed / (this.passed + this.failed)) * 100).toFixed(1)}%`);

    if (this.failed > 0) {
      console.log('\n❌ Failed Tests:');
      this.testResults
        .filter(result => result.status === 'FAILED')
        .forEach(result => {
          console.log(`   • ${result.name}: ${result.error}`);
        });
    }

    console.log('\n🎯 Production Readiness:', this.failed === 0 ? '✅ READY' : '❌ NOT READY');

    return {
      passed: this.passed,
      failed: this.failed,
      total: this.passed + this.failed,
      totalTime,
      results: this.testResults,
      productionReady: this.failed === 0,
    };
  }
}

// Run tests if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const test = new ProductionValidationTest();
  try {
    const results = await test.runAllTests();
    process.exit(results.productionReady ? 0 : 1);
  } catch (error) {
    console.error('❌ Test execution failed:', error);
    process.exit(1);
  }
}

export default ProductionValidationTest;
