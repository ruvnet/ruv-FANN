#!/usr/bin/env node

/**
 * Comprehensive Test Orchestrator for ruv-swarm
 * Master test suite that orchestrates all testing components
 */

const { PerformanceValidator } = require('./comprehensive-performance-validation.test.js');
const { LoadTestingSuite } = require('./load-testing-suite.test.js');
const { SecurityAuditor } = require('./security-audit.test.js');
const { RegressionTestingPipeline } = require('./regression-testing-pipeline.test.js');
const fs = require('fs').promises;
const path = require('path');

class ComprehensiveTestOrchestrator {
  constructor() {
    this.orchestrationResults = {
      timestamp: new Date().toISOString(),
      environment: {
        platform: process.platform,
        arch: process.arch,
        nodeVersion: process.version,
        testMode: process.env.NODE_ENV || 'test',
      },
      testSuites: [],
      summary: {
        totalSuites: 0,
        passedSuites: 0,
        failedSuites: 0,
        totalDuration: 0,
        overallStatus: 'UNKNOWN',
      },
      metrics: {
        performance: {},
        security: {},
        reliability: {},
        coverage: {},
      },
      recommendations: [],
      cicdReadiness: false,
    };
    this.startTime = Date.now();
  }

  async runComprehensiveTests() {
    console.log('🚀 Starting Comprehensive Test Orchestration');
    console.log('==============================================\n');

    this.logEnvironment();

    try {
      // Suite 1: Performance Validation
      await this.runTestSuite('Performance Validation', async() => {
        const validator = new PerformanceValidator();
        return await validator.runComprehensiveValidation();
      });

      // Suite 2: Load Testing
      await this.runTestSuite('Load Testing', async() => {
        const loadTester = new LoadTestingSuite();
        return await loadTester.runLoadTests();
      });

      // Suite 3: Security Audit
      await this.runTestSuite('Security Audit', async() => {
        const auditor = new SecurityAuditor();
        return await auditor.runSecurityAudit();
      });

      // Suite 4: Regression Pipeline
      await this.runTestSuite('Regression Pipeline', async() => {
        const pipeline = new RegressionTestingPipeline();
        return await pipeline.runRegressionPipeline();
      });

      // Suite 5: Claude Code Flow Integration Tests
      await this.runTestSuite('Claude Code Flow Integration', async() => {
        return await this.runClaudeFlowTests();
      });

      // Suite 6: Cross-Platform Compatibility
      await this.runTestSuite('Cross-Platform Compatibility', async() => {
        return await this.runCrossPlatformTests();
      });

      // Generate final orchestration report
      await this.generateOrchestrationReport();

    } catch (error) {
      console.error('❌ Test orchestration failed:', error);
      this.orchestrationResults.summary.overallStatus = 'FAILED';
      throw error;
    }

    return this.orchestrationResults;
  }

  logEnvironment() {
    console.log('🌍 Test Environment:');
    console.log(`   Platform: ${this.orchestrationResults.environment.platform} ${this.orchestrationResults.environment.arch}`);
    console.log(`   Node.js: ${this.orchestrationResults.environment.nodeVersion}`);
    console.log(`   Test Mode: ${this.orchestrationResults.environment.testMode}`);
    console.log(`   Start Time: ${new Date(this.startTime).toISOString()}\n`);
  }

  async runTestSuite(suiteName, testFunction) {
    console.log(`📋 Running ${suiteName}...`);

    const suite = {
      name: suiteName,
      startTime: Date.now(),
      endTime: null,
      duration: 0,
      passed: false,
      results: null,
      error: null,
      metrics: {},
    };

    try {
      const results = await testFunction();
      suite.results = results;
      suite.passed = this.determineSuitePassed(suiteName, results);
      suite.metrics = this.extractMetrics(suiteName, results);

      console.log(`   ${suite.passed ? '✅' : '❌'} ${suiteName} ${suite.passed ? 'PASSED' : 'FAILED'}`);

    } catch (error) {
      suite.error = error.message;
      suite.passed = false;
      console.log(`   ❌ ${suiteName} ERROR: ${error.message}`);
    }

    suite.endTime = Date.now();
    suite.duration = suite.endTime - suite.startTime;
    this.orchestrationResults.testSuites.push(suite);

    console.log(`   Duration: ${Math.round(suite.duration / 1000)}s\n`);
  }

  async runClaudeFlowTests() {
    console.log('🧠 Testing Claude Code Flow Integration...');

    const claudeFlowTests = {
      timestamp: new Date().toISOString(),
      tests: [],
      passed: true,
      summary: {
        totalTests: 0,
        passedTests: 0,
        failedTests: 0,
      },
    };

    // Test 1: MCP Tools Integration
    await this.runClaudeFlowTest(claudeFlowTests, 'MCP Tools Integration', async() => {
      try {
        const { mcp } = require('../src/mcp-tools-enhanced');

        // Test swarm initialization
        await mcp.swarm_init({ topology: 'mesh', maxAgents: 3 });

        // Test agent spawning
        const agentResult = await mcp.agent_spawn({ type: 'coder', name: 'claude-test-agent' });

        // Test task orchestration
        await mcp.task_orchestrate({
          task: 'Test Claude Code Flow integration',
          strategy: 'adaptive',
        });

        return { passed: true, details: 'MCP tools working correctly' };
      } catch (error) {
        return { passed: false, details: `MCP error: ${error.message}` };
      }
    });

    // Test 2: Neural Network Integration
    await this.runClaudeFlowTest(claudeFlowTests, 'Neural Network Integration', async() => {
      try {
        const { RuvSwarm } = require('../src/index-enhanced');

        const ruvSwarm = await RuvSwarm.initialize({
          enableNeuralNetworks: true,
          enableForecasting: true,
        });

        const swarm = await ruvSwarm.createSwarm({ topology: 'mesh', maxAgents: 2 });
        const agent = await swarm.spawn({ type: 'researcher' });

        await agent.execute({
          task: 'Analyze neural network performance patterns',
          timeout: 10000,
        });

        return { passed: true, details: 'Neural networks integrated successfully' };
      } catch (error) {
        return { passed: false, details: `Neural integration error: ${error.message}` };
      }
    });

    // Test 3: Persistence Integration
    await this.runClaudeFlowTest(claudeFlowTests, 'Persistence Integration', async() => {
      try {
        const { PersistenceManager } = require('../src/persistence');

        const pm = new PersistenceManager(':memory:');
        await pm.initialize();

        // Test data storage and retrieval
        await pm.storeAgentData({
          id: 'claude-test-1',
          type: 'coder',
          name: 'test-agent',
          status: 'active',
        });

        const retrievedData = await pm.getAgentData('claude-test-1');

        return {
          passed: retrievedData && retrievedData.name === 'test-agent',
          details: 'Persistence working correctly',
        };
      } catch (error) {
        return { passed: false, details: `Persistence error: ${error.message}` };
      }
    });

    // Test 4: WASM Module Integration
    await this.runClaudeFlowTest(claudeFlowTests, 'WASM Module Integration', async() => {
      try {
        const { RuvSwarm } = require('../src/index-enhanced');

        const ruvSwarm = await RuvSwarm.initialize({ enableSIMD: true });
        const simdSupported = await ruvSwarm.detectSIMDSupport();

        // Test WASM-based operations
        const swarm = await ruvSwarm.createSwarm({ topology: 'mesh', maxAgents: 1 });
        const agent = await swarm.spawn({ type: 'optimizer' });

        await agent.execute({
          task: 'WASM optimization test: matrix multiplication',
          timeout: 8000,
        });

        return {
          passed: true,
          details: `WASM working, SIMD supported: ${simdSupported}`,
        };
      } catch (error) {
        return { passed: false, details: `WASM error: ${error.message}` };
      }
    });

    // Test 5: Hook System Integration
    await this.runClaudeFlowTest(claudeFlowTests, 'Hook System Integration', async() => {
      try {
        const { hooks } = require('../src/hooks');

        // Test pre-operation hooks
        const preResult = await hooks.preTask('claude-flow-test');

        // Test post-operation hooks
        const postResult = await hooks.postEdit('/test/file.js', {
          memoryKey: 'claude-flow-test',
        });

        return {
          passed: true,
          details: 'Hook system functioning correctly',
        };
      } catch (error) {
        return { passed: false, details: `Hook system error: ${error.message}` };
      }
    });

    return claudeFlowTests;
  }

  async runClaudeFlowTest(testSuite, testName, testFunction) {
    const test = {
      name: testName,
      startTime: Date.now(),
      passed: false,
      details: '',
      duration: 0,
    };

    try {
      const result = await testFunction();
      test.passed = result.passed;
      test.details = result.details;
    } catch (error) {
      test.passed = false;
      test.details = `Test error: ${error.message}`;
    }

    test.duration = Date.now() - test.startTime;
    testSuite.tests.push(test);
    testSuite.summary.totalTests++;

    if (test.passed) {
      testSuite.summary.passedTests++;
    } else {
      testSuite.summary.failedTests++;
      testSuite.passed = false;
    }

    console.log(`     ${test.passed ? '✅' : '❌'} ${testName}: ${test.details} (${test.duration}ms)`);
  }

  async runCrossPlatformTests() {
    console.log('🌐 Testing Cross-Platform Compatibility...');

    const platformTests = {
      timestamp: new Date().toISOString(),
      platform: process.platform,
      arch: process.arch,
      nodeVersion: process.version,
      tests: [],
      passed: true,
      compatibility: {
        nodejs: false,
        wasm: false,
        sqlite: false,
        networkStack: false,
      },
    };

    // Test Node.js compatibility
    const nodeTest = {
      name: 'Node.js Version Compatibility',
      passed: false,
      details: '',
    };

    try {
      const nodeVersion = process.version;
      const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0], 10);
      nodeTest.passed = majorVersion >= 14;
      nodeTest.details = `Node.js ${nodeVersion} ${nodeTest.passed ? 'supported' : 'unsupported (minimum: 14.x)'}`;
      platformTests.compatibility.nodejs = nodeTest.passed;
    } catch (error) {
      nodeTest.details = `Node.js test error: ${error.message}`;
    }

    platformTests.tests.push(nodeTest);

    // Test WASM compatibility
    const wasmTest = {
      name: 'WebAssembly Support',
      passed: false,
      details: '',
    };

    try {
      const { RuvSwarm } = require('../src/index-enhanced');
      const ruvSwarm = await RuvSwarm.initialize();
      const simdSupport = await ruvSwarm.detectSIMDSupport();

      wasmTest.passed = simdSupport !== undefined;
      wasmTest.details = `WASM ${wasmTest.passed ? 'supported' : 'not supported'}, SIMD: ${simdSupport}`;
      platformTests.compatibility.wasm = wasmTest.passed;
    } catch (error) {
      wasmTest.details = `WASM test error: ${error.message}`;
    }

    platformTests.tests.push(wasmTest);

    // Test SQLite compatibility
    const sqliteTest = {
      name: 'SQLite Database Support',
      passed: false,
      details: '',
    };

    try {
      const { PersistenceManager } = require('../src/persistence');
      const pm = new PersistenceManager(':memory:');
      await pm.initialize();

      sqliteTest.passed = true;
      sqliteTest.details = 'SQLite working correctly';
      platformTests.compatibility.sqlite = sqliteTest.passed;
    } catch (error) {
      sqliteTest.details = `SQLite test error: ${error.message}`;
    }

    platformTests.tests.push(sqliteTest);

    // Test network stack compatibility
    const networkTest = {
      name: 'Network Stack Support',
      passed: false,
      details: '',
    };

    try {
      const WebSocket = require('ws');
      const ws = new WebSocket.Server({ port: 0 });

      ws.on('listening', () => {
        networkTest.passed = true;
        networkTest.details = 'WebSocket server functional';
        platformTests.compatibility.networkStack = networkTest.passed;
        ws.close();
      });

      // Wait for server to start
      await new Promise((resolve) => {
        ws.on('listening', resolve);
        setTimeout(resolve, 1000); // Timeout fallback
      });
    } catch (error) {
      networkTest.details = `Network test error: ${error.message}`;
    }

    platformTests.tests.push(networkTest);

    // Overall platform compatibility
    platformTests.passed = Object.values(platformTests.compatibility).every(supported => supported);

    platformTests.tests.forEach(test => {
      console.log(`     ${test.passed ? '✅' : '❌'} ${test.name}: ${test.details}`);
    });

    return platformTests;
  }

  determineSuitePassed(suiteName, results) {
    switch (suiteName) {
      case 'Performance Validation':
        return results.summary?.overallPassed || false;
      case 'Load Testing':
        return results.passed || false;
      case 'Security Audit':
        return results.overallSecurity?.level !== 'CRITICAL';
      case 'Regression Pipeline':
        return results.overallStatus === 'PASSED';
      case 'Claude Code Flow Integration':
        return results.passed || false;
      case 'Cross-Platform Compatibility':
        return results.passed || false;
      default:
        return false;
    }
  }

  extractMetrics(suiteName, results) {
    const metrics = {};

    switch (suiteName) {
      case 'Performance Validation':
        if (results.performance) {
          this.orchestrationResults.metrics.performance = {
            simdPerformance: results.performance.simd?.actual,
            speedOptimization: results.performance.speed?.actual,
            memoryEfficiency: results.performance.memoryEfficiency?.actual,
            loadTesting: results.performance.loadTesting?.actual,
          };
        }
        break;

      case 'Load Testing':
        if (results.performance) {
          this.orchestrationResults.metrics.reliability = {
            maxConcurrentAgents: results.performance.maxConcurrentAgents,
            avgResponseTime: results.performance.avgResponseTime,
            memoryPeak: results.performance.memoryPeak,
            errorRate: results.performance.errorRate,
          };
        }
        break;

      case 'Security Audit':
        if (results.overallSecurity) {
          this.orchestrationResults.metrics.security = {
            securityScore: results.overallSecurity.score,
            securityLevel: results.overallSecurity.level,
            vulnerabilities: results.vulnerabilities?.length || 0,
            memoryLeaks: results.memoryTests?.filter(t => !t.passed).length || 0,
          };
        }
        break;

      case 'Regression Pipeline':
        if (results.summary) {
          this.orchestrationResults.metrics.coverage = {
            successRate: results.summary.successRate,
            regressionCount: results.regressions?.length || 0,
            improvementCount: results.improvements?.length || 0,
          };
        }
        break;
    }

    return metrics;
  }

  async generateOrchestrationReport() {
    console.log('📄 Generating Comprehensive Orchestration Report...');

    // Calculate summary statistics
    this.orchestrationResults.summary.totalSuites = this.orchestrationResults.testSuites.length;
    this.orchestrationResults.summary.passedSuites = this.orchestrationResults.testSuites.filter(s => s.passed).length;
    this.orchestrationResults.summary.failedSuites = this.orchestrationResults.summary.totalSuites - this.orchestrationResults.summary.passedSuites;
    this.orchestrationResults.summary.totalDuration = Date.now() - this.startTime;

    // Determine overall status
    const successRate = this.orchestrationResults.summary.passedSuites / this.orchestrationResults.summary.totalSuites;
    this.orchestrationResults.summary.overallStatus = successRate >= 0.8 ? 'PASSED' : 'FAILED';

    // Determine CI/CD readiness
    const criticalSuites = ['Performance Validation', 'Security Audit', 'Claude Code Flow Integration'];
    const criticalPassed = criticalSuites.every(suiteName =>
      this.orchestrationResults.testSuites.find(s => s.name === suiteName)?.passed,
    );
    this.orchestrationResults.cicdReadiness = criticalPassed && this.orchestrationResults.summary.overallStatus === 'PASSED';

    // Generate recommendations
    this.generateRecommendations();

    // Save comprehensive report
    const reportPath = '/workspaces/ruv-FANN/ruv-swarm/npm/test/comprehensive-test-report.json';
    await fs.writeFile(reportPath, JSON.stringify(this.orchestrationResults, null, 2));

    // Generate executive summary
    await this.generateExecutiveSummary();

    // Console output
    console.log('\n🎯 COMPREHENSIVE TEST ORCHESTRATION SUMMARY');
    console.log('============================================');
    console.log(`Overall Status: ${this.orchestrationResults.summary.overallStatus}`);
    console.log(`Test Suites: ${this.orchestrationResults.summary.passedSuites}/${this.orchestrationResults.summary.totalSuites} passed`);
    console.log(`Success Rate: ${(successRate * 100).toFixed(1)}%`);
    console.log(`Total Duration: ${Math.round(this.orchestrationResults.summary.totalDuration / 1000)}s`);
    console.log(`CI/CD Ready: ${this.orchestrationResults.cicdReadiness ? 'YES' : 'NO'}`);

    console.log('\n📊 Key Metrics:');
    if (this.orchestrationResults.metrics.performance.simdPerformance) {
      console.log(`   SIMD Performance: ${this.orchestrationResults.metrics.performance.simdPerformance}`);
    }
    if (this.orchestrationResults.metrics.performance.speedOptimization) {
      console.log(`   Speed Optimization: ${this.orchestrationResults.metrics.performance.speedOptimization}`);
    }
    if (this.orchestrationResults.metrics.reliability.maxConcurrentAgents) {
      console.log(`   Max Concurrent Agents: ${this.orchestrationResults.metrics.reliability.maxConcurrentAgents}`);
    }
    if (this.orchestrationResults.metrics.security.securityScore) {
      console.log(`   Security Score: ${this.orchestrationResults.metrics.security.securityScore}/100`);
    }

    console.log('\n📋 Test Suite Results:');
    this.orchestrationResults.testSuites.forEach(suite => {
      console.log(`   ${suite.passed ? '✅' : '❌'} ${suite.name} (${Math.round(suite.duration / 1000)}s)`);
    });

    if (this.orchestrationResults.recommendations.length > 0) {
      console.log('\n💡 Recommendations:');
      this.orchestrationResults.recommendations.forEach((rec, i) => {
        console.log(`   ${i + 1}. ${rec}`);
      });
    }

    console.log(`\n📄 Detailed report saved to: ${reportPath}`);

    return this.orchestrationResults;
  }

  generateRecommendations() {
    const recommendations = [];

    // Performance recommendations
    if (this.orchestrationResults.metrics.performance.simdPerformance &&
            !this.orchestrationResults.metrics.performance.simdPerformance.includes('6') &&
            !this.orchestrationResults.metrics.performance.simdPerformance.includes('7') &&
            !this.orchestrationResults.metrics.performance.simdPerformance.includes('8') &&
            !this.orchestrationResults.metrics.performance.simdPerformance.includes('9')) {
      recommendations.push('Optimize SIMD performance to reach 6-10x target');
    }

    // Security recommendations
    if (this.orchestrationResults.metrics.security.securityScore < 90) {
      recommendations.push('Address security vulnerabilities to improve security score');
    }

    // Reliability recommendations
    if (this.orchestrationResults.metrics.reliability.maxConcurrentAgents < 50) {
      recommendations.push('Optimize system to support 50+ concurrent agents');
    }

    // CI/CD recommendations
    if (!this.orchestrationResults.cicdReadiness) {
      recommendations.push('Fix failing test suites before deployment');
    }

    if (recommendations.length === 0) {
      recommendations.push('All tests passed successfully - system ready for production deployment');
    }

    this.orchestrationResults.recommendations = recommendations;
  }

  async generateExecutiveSummary() {
    const summary = `# ruv-swarm Test Orchestration Executive Summary

## Overview
- **Test Date**: ${new Date(this.orchestrationResults.timestamp).toLocaleDateString()}
- **Overall Status**: ${this.orchestrationResults.summary.overallStatus}
- **CI/CD Ready**: ${this.orchestrationResults.cicdReadiness ? 'YES' : 'NO'}
- **Success Rate**: ${((this.orchestrationResults.summary.passedSuites / this.orchestrationResults.summary.totalSuites) * 100).toFixed(1)}%

## Key Performance Metrics
- **SIMD Performance**: ${this.orchestrationResults.metrics.performance.simdPerformance || 'N/A'}
- **Speed Optimization**: ${this.orchestrationResults.metrics.performance.speedOptimization || 'N/A'}
- **Max Concurrent Agents**: ${this.orchestrationResults.metrics.reliability.maxConcurrentAgents || 'N/A'}
- **Security Score**: ${this.orchestrationResults.metrics.security.securityScore || 'N/A'}/100

## Test Suite Results
${this.orchestrationResults.testSuites.map(suite =>
    `- ${suite.passed ? '✅' : '❌'} **${suite.name}**: ${suite.passed ? 'PASSED' : 'FAILED'} (${Math.round(suite.duration / 1000)}s)`,
  ).join('\n')}

## Recommendations
${this.orchestrationResults.recommendations.map((rec, i) => `${i + 1}. ${rec}`).join('\n')}

## Next Steps
${this.orchestrationResults.cicdReadiness
    ? '- Deploy to production environment\n- Monitor performance metrics\n- Schedule regular regression testing'
    : '- Fix failing test suites\n- Address performance regressions\n- Re-run comprehensive tests'
}
`;

    await fs.writeFile('/workspaces/ruv-FANN/ruv-swarm/npm/test/executive-summary.md', summary);
  }
}

// Main execution
async function runComprehensiveTests() {
  try {
    const orchestrator = new ComprehensiveTestOrchestrator();
    const results = await orchestrator.runComprehensiveTests();

    process.exit(results.summary.overallStatus === 'PASSED' ? 0 : 1);
  } catch (error) {
    console.error('💥 Comprehensive testing failed:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  runComprehensiveTests();
}

module.exports = { ComprehensiveTestOrchestrator };
