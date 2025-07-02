#!/usr/bin/env node

/**
 * Integration Test Runner for ruv-swarm
 * Comprehensive end-to-end testing suite
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const chalk = require('chalk');

class IntegrationTestRunner {
  constructor() {
    this.testSuites = [
      {
        name: 'Lifecycle Tests',
        path: 'scenarios/lifecycle/full-workflow.test.js',
        timeout: 60000,
        parallel: false,
        critical: true,
      },
      {
        name: 'Resilience Tests',
        path: 'scenarios/resilience/error-recovery.test.js',
        timeout: 45000,
        parallel: false,
        critical: true,
      },
      {
        name: 'Performance Tests',
        path: 'scenarios/performance/load-testing.test.js',
        timeout: 120000,
        parallel: true,
        critical: false,
      },
      {
        name: 'Cross-Feature Integration',
        path: 'scenarios/cross-feature/system-integration.test.js',
        timeout: 90000,
        parallel: false,
        critical: true,
      },
    ];

    this.results = {
      total: 0,
      passed: 0,
      failed: 0,
      skipped: 0,
      duration: 0,
      suites: [],
    };

    this.config = {
      parallel: process.env.PARALLEL_TESTS === 'true',
      verbose: process.env.VERBOSE === 'true',
      bail: process.env.BAIL_ON_FAILURE === 'true',
      coverage: process.env.COVERAGE === 'true',
      environment: process.env.NODE_ENV || 'test',
    };
  }

  async run() {
    console.log(chalk.blue.bold('\n🧪 ruv-swarm Integration Test Suite'));
    console.log(chalk.gray('=====================================\n'));

    this.logConfig();
    await this.setupEnvironment();

    const startTime = Date.now();

    try {
      if (this.config.parallel) {
        await this.runParallel();
      } else {
        await this.runSequential();
      }
    } catch (error) {
      console.error(chalk.red.bold('\n❌ Test execution failed:'), error.message);
      process.exit(1);
    }

    this.results.duration = Date.now() - startTime;
    this.generateReport();

    process.exit(this.results.failed > 0 ? 1 : 0);
  }

  logConfig() {
    console.log(chalk.cyan('Configuration:'));
    console.log(chalk.gray(`  Environment: ${this.config.environment}`));
    console.log(chalk.gray(`  Parallel: ${this.config.parallel}`));
    console.log(chalk.gray(`  Verbose: ${this.config.verbose}`));
    console.log(chalk.gray(`  Coverage: ${this.config.coverage}`));
    console.log(chalk.gray(`  Bail on failure: ${this.config.bail}\n`));
  }

  async setupEnvironment() {
    console.log(chalk.yellow('🔧 Setting up test environment...'));

    // Ensure test database is clean
    try {
      const dbPath = path.join(__dirname, '../../data/test-ruv-swarm.db');
      if (fs.existsSync(dbPath)) {
        fs.unlinkSync(dbPath);
      }
    } catch (error) {
      console.warn(chalk.yellow('Warning: Could not clean test database'));
    }

    // Set test environment variables
    process.env.NODE_ENV = 'test';
    process.env.RUV_SWARM_TEST_MODE = 'true';
    process.env.RUV_SWARM_LOG_LEVEL = this.config.verbose ? 'debug' : 'error';

    console.log(chalk.green('✅ Environment ready\n'));
  }

  async runSequential() {
    console.log(chalk.blue('Running tests sequentially...\n'));

    for (const suite of this.testSuites) {
      if (this.config.bail && this.results.failed > 0) {
        console.log(chalk.yellow('⏭️  Bailing out due to previous failures'));
        break;
      }

      await this.runSuite(suite);
    }
  }

  async runParallel() {
    console.log(chalk.blue('Running tests in parallel...\n'));

    const parallelSuites = this.testSuites.filter(s => s.parallel);
    const sequentialSuites = this.testSuites.filter(s => !s.parallel);

    // Run parallel suites first
    if (parallelSuites.length > 0) {
      const parallelPromises = parallelSuites.map(suite => this.runSuite(suite));
      await Promise.all(parallelPromises);
    }

    // Run sequential suites
    for (const suite of sequentialSuites) {
      if (this.config.bail && this.results.failed > 0) {
        break;
      }
      await this.runSuite(suite);
    }
  }

  async runSuite(suite) {
    console.log(chalk.cyan(`📋 Running ${suite.name}...`));

    const startTime = Date.now();
    const suitePath = path.join(__dirname, suite.path);

    try {
      const result = await this.executeMocha(suitePath, suite);

      const duration = Date.now() - startTime;
      const status = result.exitCode === 0 ? 'PASSED' : 'FAILED';
      const statusColor = result.exitCode === 0 ? 'green' : 'red';

      console.log(chalk[statusColor](`  ${status} in ${duration}ms`));

      if (this.config.verbose && result.output) {
        console.log(chalk.gray('  Output:'));
        console.log(chalk.gray(result.output.split('\n').map(line => `    ${line}`).join('\n')));
      }

      this.results.suites.push({
        name: suite.name,
        status,
        duration,
        exitCode: result.exitCode,
        output: result.output,
        critical: suite.critical,
      });

      if (result.exitCode === 0) {
        this.results.passed++;
      } else {
        this.results.failed++;
        if (suite.critical) {
          console.log(chalk.red.bold('  ⚠️  Critical test suite failed!'));
        }
      }

      this.results.total++;

    } catch (error) {
      console.log(chalk.red(`  ERROR: ${error.message}`));

      this.results.suites.push({
        name: suite.name,
        status: 'ERROR',
        duration: Date.now() - startTime,
        error: error.message,
        critical: suite.critical,
      });

      this.results.failed++;
      this.results.total++;
    }

    console.log(''); // Empty line for readability
  }

  executeMocha(testPath, suite) {
    return new Promise((resolve) => {
      const args = [
        testPath,
        '--timeout', suite.timeout.toString(),
        '--reporter', this.config.verbose ? 'spec' : 'json',
      ];

      if (this.config.coverage) {
        args.unshift('--require', 'nyc/index.js');
      }

      const mocha = spawn('npx', ['mocha', ...args], {
        cwd: path.join(__dirname, '../..'),
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      let output = '';
      let error = '';

      mocha.stdout.on('data', (data) => {
        output += data.toString();
      });

      mocha.stderr.on('data', (data) => {
        error += data.toString();
      });

      mocha.on('close', (exitCode) => {
        resolve({
          exitCode,
          output: output || error,
          error: exitCode !== 0 ? error : null,
        });
      });

      mocha.on('error', (err) => {
        resolve({
          exitCode: 1,
          output: '',
          error: err.message,
        });
      });
    });
  }

  generateReport() {
    console.log(chalk.blue.bold('\n📊 Integration Test Results'));
    console.log(chalk.gray('============================\n'));

    // Summary
    const successRate = this.results.total > 0 ? (this.results.passed / this.results.total * 100).toFixed(1) : 0;
    const durationSeconds = (this.results.duration / 1000).toFixed(2);

    console.log(chalk.cyan('Summary:'));
    console.log(chalk.gray(`  Total Suites: ${this.results.total}`));
    console.log(chalk.green(`  Passed: ${this.results.passed}`));
    console.log(chalk.red(`  Failed: ${this.results.failed}`));
    console.log(chalk.yellow(`  Skipped: ${this.results.skipped}`));
    console.log(chalk.blue(`  Success Rate: ${successRate}%`));
    console.log(chalk.gray(`  Duration: ${durationSeconds}s\n`));

    // Suite details
    console.log(chalk.cyan('Suite Details:'));
    this.results.suites.forEach(suite => {
      const icon = suite.status === 'PASSED' ? '✅' : suite.status === 'ERROR' ? '💥' : '❌';
      const critical = suite.critical ? ' [CRITICAL]' : '';
      const duration = `${suite.duration}ms`;

      console.log(`  ${icon} ${suite.name}${critical} - ${duration}`);

      if (suite.error && this.config.verbose) {
        console.log(chalk.red(`    Error: ${suite.error}`));
      }
    });

    // Critical failures
    const criticalFailures = this.results.suites.filter(s => s.critical && s.status !== 'PASSED');
    if (criticalFailures.length > 0) {
      console.log(chalk.red.bold('\n⚠️  Critical Test Failures:'));
      criticalFailures.forEach(suite => {
        console.log(chalk.red(`  • ${suite.name}`));
      });
    }

    // Coverage information
    if (this.config.coverage) {
      console.log(chalk.cyan('\n📈 Coverage report will be generated in ./coverage/'));
    }

    // Recommendations
    this.generateRecommendations();

    // Save results
    this.saveResults();
  }

  generateRecommendations() {
    console.log(chalk.cyan('\n💡 Recommendations:'));

    if (this.results.failed === 0) {
      console.log(chalk.green('  🎉 All tests passed! System is ready for production.'));
      return;
    }

    const criticalFailures = this.results.suites.filter(s => s.critical && s.status !== 'PASSED').length;

    if (criticalFailures > 0) {
      console.log(chalk.red('  🚨 Critical failures detected - do not deploy to production'));
      console.log(chalk.yellow('  📋 Review failed critical test suites immediately'));
    }

    const performanceFailures = this.results.suites.filter(s =>
      s.name.includes('Performance') && s.status !== 'PASSED',
    ).length;

    if (performanceFailures > 0) {
      console.log(chalk.yellow('  ⚡ Performance issues detected - review system capacity'));
    }

    const resilienceFailures = this.results.suites.filter(s =>
      s.name.includes('Resilience') && s.status !== 'PASSED',
    ).length;

    if (resilienceFailures > 0) {
      console.log(chalk.yellow('  🛡️  Resilience issues detected - strengthen error handling'));
    }

    console.log(chalk.gray('  📖 Check individual test logs for detailed failure information'));
  }

  saveResults() {
    const resultsPath = path.join(__dirname, '../../test-results/integration-results.json');
    const resultsDir = path.dirname(resultsPath);

    try {
      if (!fs.existsSync(resultsDir)) {
        fs.mkdirSync(resultsDir, { recursive: true });
      }

      const fullResults = {
        ...this.results,
        timestamp: new Date().toISOString(),
        environment: this.config.environment,
        configuration: this.config,
      };

      fs.writeFileSync(resultsPath, JSON.stringify(fullResults, null, 2));
      console.log(chalk.gray(`\n📄 Results saved to: ${resultsPath}`));

    } catch (error) {
      console.warn(chalk.yellow(`Warning: Could not save results - ${error.message}`));
    }
  }
}

// Run integration tests if called directly
if (require.main === module) {
  const runner = new IntegrationTestRunner();
  runner.run().catch(error => {
    console.error(chalk.red.bold('Fatal error:'), error);
    process.exit(1);
  });
}

module.exports = IntegrationTestRunner;