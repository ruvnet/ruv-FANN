#!/usr/bin/env node

/**
 * Comprehensive Performance Validation Framework
 * Tests all performance targets and validates DAA integration
 */

import { RuvSwarm } from '../src/index-enhanced.js';
import { performanceCLI } from '../src/performance.js';
import { promises as fs } from 'fs';
import path from 'path';
import { spawn } from 'child_process';

class PerformanceValidator {
  constructor() {
    this.testResults = {
      timestamp: new Date().toISOString(),
      tests: [],
      performance: {
        simd: { target: '6-10x', actual: null, passed: false },
        speed: { target: '2.8-4.4x', actual: null, passed: false },
        loadTesting: { target: '50+ agents', actual: null, passed: false },
        memoryEfficiency: { target: '<500MB@50agents', actual: null, passed: false },
        daaIntegration: { target: 'seamless', actual: null, passed: false },
      },
      coverage: {
        lines: 0,
        branches: 0,
        functions: 0,
        statements: 0,
      },
      recommendations: [],
    };
    this.baselines = {};
  }

  async runComprehensiveValidation() {
    console.log('🚀 Starting Comprehensive Performance Validation\n');

    // 1. Establish baselines
    await this.establishBaselines();

    // 2. SIMD Performance Tests
    await this.validateSIMDPerformance();

    // 3. Speed Optimization Tests
    await this.validateSpeedOptimizations();

    // 4. Load Testing with 50+ Agents
    await this.validateLoadTesting();

    // 5. Memory Efficiency Tests
    await this.validateMemoryEfficiency();

    // 6. DAA Integration Tests
    await this.validateDAAIntegration();

    // 7. Cross-Platform Compatibility
    await this.validateCrossPlatform();

    // 8. Generate Comprehensive Report
    await this.generateValidationReport();

    return this.testResults;
  }

  async establishBaselines() {
    console.log('📊 Establishing Performance Baselines...');

    const startTime = Date.now();

    try {
      // Initialize RuvSwarm for baseline measurements
      const ruvSwarm = await RuvSwarm.initialize({
        enableNeuralNetworks: true,
        enableForecasting: true,
        loadingStrategy: 'progressive',
      });

      // Baseline: Single agent task execution
      const singleAgentStart = Date.now();
      const swarm = await ruvSwarm.createSwarm({
        topology: 'mesh',
        maxAgents: 1,
        strategy: 'balanced',
      });
      const agent = await swarm.spawn({ type: 'coder' });
      await agent.execute({ task: 'Simple arithmetic: 2+2', timeout: 5000 });
      const singleAgentTime = Date.now() - singleAgentStart;

      // Baseline: Memory usage
      const memUsage = process.memoryUsage();

      // Baseline: WASM loading time
      const wasmStart = Date.now();
      const wasmSupport = await ruvSwarm.detectSIMDSupport();
      const wasmLoadTime = Date.now() - wasmStart;

      this.baselines = {
        singleAgentExecution: singleAgentTime,
        baseMemoryUsage: memUsage.heapUsed,
        wasmLoadTime,
        simdSupport: wasmSupport,
      };

      console.log(`✅ Baselines established in ${Date.now() - startTime}ms`);
      console.log(`   Single Agent: ${singleAgentTime}ms`);
      console.log(`   Memory: ${(memUsage.heapUsed / 1024 / 1024).toFixed(1)}MB`);
      console.log(`   WASM Load: ${wasmLoadTime}ms`);
      console.log(`   SIMD Support: ${wasmSupport}\n`);

    } catch (error) {
      console.error('❌ Failed to establish baselines:', error.message);
      throw error;
    }
  }

  async validateSIMDPerformance() {
    console.log('⚡ Validating SIMD Performance (Target: 6-10x improvement)...');

    const testResult = {
      test: 'SIMD Performance',
      target: '6-10x improvement',
      startTime: Date.now(),
      passed: false,
      metrics: {},
    };

    try {
      const ruvSwarm = await RuvSwarm.initialize({
        enableNeuralNetworks: true,
        enableSIMD: false,
      });

      // Test without SIMD
      const noSIMDStart = Date.now();
      const swarmNoSIMD = await ruvSwarm.createSwarm({ topology: 'mesh', maxAgents: 4 });
      for (let i = 0; i < 4; i++) {
        const agent = await swarmNoSIMD.spawn({ type: 'optimizer' });
        await agent.execute({
          task: 'Matrix multiplication: 100x100',
          timeout: 10000,
        });
      }
      const noSIMDTime = Date.now() - noSIMDStart;

      // Test with SIMD
      const ruvSwarmSIMD = await RuvSwarm.initialize({
        enableNeuralNetworks: true,
        enableSIMD: true,
      });

      const simdStart = Date.now();
      const swarmSIMD = await ruvSwarmSIMD.createSwarm({ topology: 'mesh', maxAgents: 4 });
      for (let i = 0; i < 4; i++) {
        const agent = await swarmSIMD.spawn({ type: 'optimizer' });
        await agent.execute({
          task: 'Matrix multiplication: 100x100 (SIMD)',
          timeout: 10000,
        });
      }
      const simdTime = Date.now() - simdStart;

      const improvement = noSIMDTime / simdTime;
      testResult.metrics = {
        noSIMDTime,
        simdTime,
        improvement: `${improvement.toFixed(2) }x`,
      };

      testResult.passed = improvement >= 6.0 && improvement <= 10.0;
      this.testResults.performance.simd.actual = `${improvement.toFixed(2) }x`;
      this.testResults.performance.simd.passed = testResult.passed;

      console.log(`   No SIMD: ${noSIMDTime}ms`);
      console.log(`   With SIMD: ${simdTime}ms`);
      console.log(`   Improvement: ${improvement.toFixed(2)}x`);
      console.log(`   ${testResult.passed ? '✅ PASSED' : '❌ FAILED'} (Target: 6-10x)\n`);

    } catch (error) {
      testResult.error = error.message;
      console.error(`❌ SIMD test failed: ${error.message}\n`);
    }

    testResult.duration = Date.now() - testResult.startTime;
    this.testResults.tests.push(testResult);
  }

  async validateSpeedOptimizations() {
    console.log('🏃 Validating Speed Optimizations (Target: 2.8-4.4x improvement)...');

    const testResult = {
      test: 'Speed Optimizations',
      target: '2.8-4.4x improvement',
      startTime: Date.now(),
      passed: false,
      metrics: {},
    };

    try {
      // Test baseline speed (conservative settings)
      const baselineTime = await this.measureExecutionTime({
        topology: 'star',
        maxAgents: 1,
        strategy: 'sequential',
        optimizations: false,
      });

      // Test optimized speed
      const optimizedTime = await this.measureExecutionTime({
        topology: 'mesh',
        maxAgents: 6,
        strategy: 'parallel',
        optimizations: true,
      });

      const speedup = baselineTime / optimizedTime;
      testResult.metrics = {
        baselineTime,
        optimizedTime,
        speedup: `${speedup.toFixed(2) }x`,
      };

      testResult.passed = speedup >= 2.8 && speedup <= 4.4;
      this.testResults.performance.speed.actual = `${speedup.toFixed(2) }x`;
      this.testResults.performance.speed.passed = testResult.passed;

      console.log(`   Baseline: ${baselineTime}ms`);
      console.log(`   Optimized: ${optimizedTime}ms`);
      console.log(`   Speedup: ${speedup.toFixed(2)}x`);
      console.log(`   ${testResult.passed ? '✅ PASSED' : '❌ FAILED'} (Target: 2.8-4.4x)\n`);

    } catch (error) {
      testResult.error = error.message;
      console.error(`❌ Speed optimization test failed: ${error.message}\n`);
    }

    testResult.duration = Date.now() - testResult.startTime;
    this.testResults.tests.push(testResult);
  }

  async validateLoadTesting() {
    console.log('🔥 Validating Load Testing (Target: 50+ concurrent agents)...');

    const testResult = {
      test: 'Load Testing',
      target: '50+ concurrent agents',
      startTime: Date.now(),
      passed: false,
      metrics: {},
    };

    try {
      const ruvSwarm = await RuvSwarm.initialize({
        enableNeuralNetworks: true,
        enableForecasting: true,
        loadingStrategy: 'progressive',
      });

      const swarm = await ruvSwarm.createSwarm({
        topology: 'hierarchical',
        maxAgents: 60,
        strategy: 'parallel',
      });

      const agents = [];
      const startTime = Date.now();

      // Spawn 55 agents in parallel
      const spawnPromises = [];
      for (let i = 0; i < 55; i++) {
        spawnPromises.push(
          swarm.spawn({
            type: i % 5 === 0 ? 'coordinator' : 'coder',
            name: `agent-${i}`,
          }),
        );
      }

      const spawnedAgents = await Promise.all(spawnPromises);
      agents.push(...spawnedAgents);

      // Execute tasks concurrently
      const taskPromises = agents.map((agent, i) =>
        agent.execute({
          task: `Task ${i}: Calculate fibonacci(20)`,
          timeout: 15000,
        }),
      );

      await Promise.all(taskPromises);
      const totalTime = Date.now() - startTime;

      const memUsage = process.memoryUsage();
      const memoryMB = memUsage.heapUsed / 1024 / 1024;

      testResult.metrics = {
        agentsSpawned: agents.length,
        executionTime: totalTime,
        memoryUsage: `${memoryMB.toFixed(1) }MB`,
        avgTimePerAgent: `${(totalTime / agents.length).toFixed(1) }ms`,
      };

      testResult.passed = agents.length >= 50 && totalTime < 30000; // 30 second limit
      this.testResults.performance.loadTesting.actual = `${agents.length} agents`;
      this.testResults.performance.loadTesting.passed = testResult.passed;

      console.log(`   Agents spawned: ${agents.length}`);
      console.log(`   Total time: ${totalTime}ms`);
      console.log(`   Memory usage: ${memoryMB.toFixed(1)}MB`);
      console.log(`   Avg per agent: ${(totalTime / agents.length).toFixed(1)}ms`);
      console.log(`   ${testResult.passed ? '✅ PASSED' : '❌ FAILED'} (Target: 50+ agents)\n`);

    } catch (error) {
      testResult.error = error.message;
      console.error(`❌ Load testing failed: ${error.message}\n`);
    }

    testResult.duration = Date.now() - testResult.startTime;
    this.testResults.tests.push(testResult);
  }

  async validateMemoryEfficiency() {
    console.log('💾 Validating Memory Efficiency (Target: <500MB @ 50 agents)...');

    const testResult = {
      test: 'Memory Efficiency',
      target: '<500MB @ 50 agents',
      startTime: Date.now(),
      passed: false,
      metrics: {},
    };

    try {
      const ruvSwarm = await RuvSwarm.initialize({
        enableNeuralNetworks: true,
        memoryOptimization: true,
      });

      const initialMemory = process.memoryUsage().heapUsed / 1024 / 1024;

      const swarm = await ruvSwarm.createSwarm({
        topology: 'mesh',
        maxAgents: 50,
        strategy: 'balanced',
      });

      const agents = [];
      for (let i = 0; i < 50; i++) {
        const agent = await swarm.spawn({ type: 'coder' });
        agents.push(agent);
      }

      const peakMemory = process.memoryUsage().heapUsed / 1024 / 1024;
      const memoryIncrease = peakMemory - initialMemory;

      testResult.metrics = {
        initialMemory: `${initialMemory.toFixed(1) }MB`,
        peakMemory: `${peakMemory.toFixed(1) }MB`,
        memoryIncrease: `${memoryIncrease.toFixed(1) }MB`,
        memoryPerAgent: `${(memoryIncrease / 50).toFixed(1) }MB`,
      };

      testResult.passed = peakMemory < 500;
      this.testResults.performance.memoryEfficiency.actual = `${peakMemory.toFixed(1) }MB`;
      this.testResults.performance.memoryEfficiency.passed = testResult.passed;

      console.log(`   Initial memory: ${initialMemory.toFixed(1)}MB`);
      console.log(`   Peak memory: ${peakMemory.toFixed(1)}MB`);
      console.log(`   Memory increase: ${memoryIncrease.toFixed(1)}MB`);
      console.log(`   Per agent: ${(memoryIncrease / 50).toFixed(1)}MB`);
      console.log(`   ${testResult.passed ? '✅ PASSED' : '❌ FAILED'} (Target: <500MB)\n`);

    } catch (error) {
      testResult.error = error.message;
      console.error(`❌ Memory efficiency test failed: ${error.message}\n`);
    }

    testResult.duration = Date.now() - testResult.startTime;
    this.testResults.tests.push(testResult);
  }

  async validateDAAIntegration() {
    console.log('🔗 Validating DAA Integration (Target: seamless integration)...');

    const testResult = {
      test: 'DAA Integration',
      target: 'seamless integration',
      startTime: Date.now(),
      passed: false,
      metrics: {},
    };

    try {
      // Test DAA AI module integration
      const daaPath = '/workspaces/ruv-FANN/daa-repository';
      const daaExists = await this.checkPathExists(daaPath);

      if (!daaExists) {
        throw new Error('DAA repository not found');
      }

      // Test Rust integration
      const cargoTest = await this.runCommand('cargo test --manifest-path /workspaces/ruv-FANN/daa-repository/Cargo.toml');

      // Test MCP integration
      const mcpTest = await this.testMCPIntegration();

      testResult.metrics = {
        daaRepositoryExists: daaExists,
        cargoTestsPassed: cargoTest.success,
        mcpIntegrationWorking: mcpTest.success,
        integrationPoints: ['AI module', 'MCP server', 'WASM bindings'],
      };

      testResult.passed = daaExists && cargoTest.success && mcpTest.success;
      this.testResults.performance.daaIntegration.actual = testResult.passed ? 'integrated' : 'partial';
      this.testResults.performance.daaIntegration.passed = testResult.passed;

      console.log(`   DAA Repository: ${daaExists ? '✅' : '❌'}`);
      console.log(`   Cargo Tests: ${cargoTest.success ? '✅' : '❌'}`);
      console.log(`   MCP Integration: ${mcpTest.success ? '✅' : '❌'}`);
      console.log(`   ${testResult.passed ? '✅ PASSED' : '❌ FAILED'} (Target: seamless)\n`);

    } catch (error) {
      testResult.error = error.message;
      console.error(`❌ DAA integration test failed: ${error.message}\n`);
    }

    testResult.duration = Date.now() - testResult.startTime;
    this.testResults.tests.push(testResult);
  }

  async validateCrossPlatform() {
    console.log('🌐 Validating Cross-Platform Compatibility...');

    const testResult = {
      test: 'Cross-Platform Compatibility',
      target: 'Linux, macOS, Windows support',
      startTime: Date.now(),
      passed: false,
      metrics: {},
    };

    try {
      const platform = process.platform;
      const arch = process.arch;
      const nodeVersion = process.version;

      // Test WASM compatibility
      const wasmCompatible = await this.testWASMCompatibility();

      // Test SQLite compatibility
      const sqliteCompatible = await this.testSQLiteCompatibility();

      // Test Node.js version compatibility
      const nodeCompatible = this.checkNodeCompatibility(nodeVersion);

      testResult.metrics = {
        platform,
        architecture: arch,
        nodeVersion,
        wasmCompatible,
        sqliteCompatible,
        nodeCompatible,
      };

      testResult.passed = wasmCompatible && sqliteCompatible && nodeCompatible;

      console.log(`   Platform: ${platform} ${arch}`);
      console.log(`   Node.js: ${nodeVersion} ${nodeCompatible ? '✅' : '❌'}`);
      console.log(`   WASM: ${wasmCompatible ? '✅' : '❌'}`);
      console.log(`   SQLite: ${sqliteCompatible ? '✅' : '❌'}`);
      console.log(`   ${testResult.passed ? '✅ PASSED' : '❌ FAILED'}\n`);

    } catch (error) {
      testResult.error = error.message;
      console.error(`❌ Cross-platform test failed: ${error.message}\n`);
    }

    testResult.duration = Date.now() - testResult.startTime;
    this.testResults.tests.push(testResult);
  }

  async generateValidationReport() {
    console.log('📄 Generating Comprehensive Validation Report...');

    const passedTests = this.testResults.tests.filter(t => t.passed).length;
    const totalTests = this.testResults.tests.length;
    const successRate = ((passedTests / totalTests) * 100).toFixed(1);

    const report = {
      ...this.testResults,
      summary: {
        totalTests,
        passedTests,
        failedTests: totalTests - passedTests,
        successRate: `${successRate }%`,
        overallPassed: successRate >= 90,
      },
      recommendations: this.generateRecommendations(),
    };

    // Save detailed report
    const reportPath = '/workspaces/ruv-FANN/ruv-swarm/npm/test/validation-report.json';
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));

    // Generate readable summary
    console.log('\n📊 VALIDATION SUMMARY');
    console.log('=====================');
    console.log(`Tests Passed: ${passedTests}/${totalTests} (${successRate}%)`);
    console.log(`Overall Status: ${report.summary.overallPassed ? '✅ PASSED' : '❌ FAILED'}`);

    console.log('\n🎯 Performance Targets:');
    Object.entries(this.testResults.performance).forEach(([key, value]) => {
      console.log(`   ${key}: ${value.actual || 'N/A'} ${value.passed ? '✅' : '❌'} (Target: ${value.target})`);
    });

    if (report.recommendations.length > 0) {
      console.log('\n💡 Recommendations:');
      report.recommendations.forEach((rec, i) => {
        console.log(`   ${i + 1}. ${rec}`);
      });
    }

    console.log(`\n📄 Detailed report saved to: ${reportPath}`);

    return report;
  }

  // Helper methods
  async measureExecutionTime(config) {
    const ruvSwarm = await RuvSwarm.initialize({
      enableNeuralNetworks: true,
      enableOptimizations: config.optimizations,
    });

    const start = Date.now();
    const swarm = await ruvSwarm.createSwarm(config);

    const agents = [];
    for (let i = 0; i < config.maxAgents; i++) {
      agents.push(await swarm.spawn({ type: 'coder' }));
    }

    const tasks = agents.map(agent =>
      agent.execute({ task: 'Calculate: sum(1..1000)', timeout: 10000 }),
    );

    if (config.strategy === 'parallel') {
      await Promise.all(tasks);
    } else {
      for (const task of tasks) {
        await task;
      }
    }

    return Date.now() - start;
  }

  async checkPathExists(path) {
    try {
      await fs.access(path);
      return true;
    } catch {
      return false;
    }
  }

  async runCommand(command) {
    return new Promise((resolve) => {
      const [cmd, ...args] = command.split(' ');
      const process = spawn(cmd, args, { stdio: 'pipe' });

      let output = '';
      process.stdout.on('data', (data) => output += data.toString());
      process.stderr.on('data', (data) => output += data.toString());

      process.on('close', (code) => {
        resolve({ success: code === 0, output });
      });

      setTimeout(() => {
        process.kill();
        resolve({ success: false, output: 'Timeout' });
      }, 30000);
    });
  }

  async testMCPIntegration() {
    try {
      // Test basic MCP functionality
      const { mcp } = require('../src/mcp-tools-enhanced');
      return { success: true };
    } catch {
      return { success: false };
    }
  }

  async testWASMCompatibility() {
    try {
      const ruvSwarm = await RuvSwarm.initialize();
      return await ruvSwarm.detectSIMDSupport() !== undefined;
    } catch {
      return false;
    }
  }

  async testSQLiteCompatibility() {
    try {
      const { PersistenceManager } = require('../src/persistence');
      const pm = new PersistenceManager(':memory:');
      await pm.initialize();
      return true;
    } catch {
      return false;
    }
  }

  checkNodeCompatibility(version) {
    const major = parseInt(version.slice(1).split('.')[0], 10);
    return major >= 14; // Minimum Node.js 14
  }

  generateRecommendations() {
    const recommendations = [];

    this.testResults.tests.forEach(test => {
      if (!test.passed) {
        switch (test.test) {
          case 'SIMD Performance':
            recommendations.push('Enable SIMD optimizations and verify WASM module compilation');
            break;
          case 'Speed Optimizations':
            recommendations.push('Review parallel execution strategy and agent coordination');
            break;
          case 'Load Testing':
            recommendations.push('Optimize memory usage and consider agent pooling');
            break;
          case 'Memory Efficiency':
            recommendations.push('Implement memory pooling and garbage collection tuning');
            break;
          case 'DAA Integration':
            recommendations.push('Verify DAA repository setup and MCP server configuration');
            break;
        }
      }
    });

    return recommendations;
  }
}

// Main execution
async function runValidation() {
  try {
    const validator = new PerformanceValidator();
    const results = await validator.runComprehensiveValidation();

    process.exit(results.summary.overallPassed ? 0 : 1);
  } catch (error) {
    console.error('💥 Validation failed:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  runValidation();
}

module.exports = { PerformanceValidator };
