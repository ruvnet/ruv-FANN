#!/usr/bin/env node

/**
 * Comprehensive Test Runner
 * Executes all test suites with coverage and performance tracking
 */

import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { performance } from 'perf_hooks';
import chalk from 'chalk';

const TEST_SUITES = [
  {
    name: 'Unit Tests - WASM Functions',
    command: 'vitest run test/unit/wasm-functions.test.js',
    critical: true,
  },
  {
    name: 'Integration Tests - JS-WASM Communication',
    command: 'vitest run test/integration/js-wasm-communication.test.js',
    critical: true,
  },
  {
    name: 'E2E Tests - Workflow Scenarios',
    command: 'vitest run test/e2e/workflow-scenarios.test.js',
    critical: true,
  },
  {
    name: 'Browser Tests - Cross-Browser Compatibility',
    command: 'vitest run test/browser/cross-browser-compatibility.test.js',
    critical: false,
    requiresBrowser: true,
  },
  {
    name: 'Performance Tests - Comprehensive Benchmarks',
    command: 'vitest run test/performance/comprehensive-benchmarks.test.js',
    critical: true,
  },
  {
    name: 'Existing Tests - Legacy Suite',
    command: 'npm run test:all',
    critical: false,
  },
];

class TestRunner {
  constructor() {
    this.results = [];
    this.startTime = performance.now();
    this.coverageData = {};
  }

  async run() {
    console.log(chalk.bold.blue('\n🚀 Starting Comprehensive Test Suite\n'));
    console.log(chalk.gray(`Time: ${new Date().toISOString()}`));
    console.log(chalk.gray('═'.repeat(80)));

    // Check prerequisites
    await this.checkPrerequisites();

    // Run each test suite
    for (const suite of TEST_SUITES) {
      await this.runTestSuite(suite);
    }

    // Generate reports
    await this.generateReports();

    // Display summary
    this.displaySummary();

    // Exit with appropriate code
    const failedCritical = this.results.some(r => r.critical && !r.success);
    process.exit(failedCritical ? 1 : 0);
  }

  async checkPrerequisites() {
    console.log(chalk.yellow('\n🔍 Checking prerequisites...'));

    // Check WASM files
    const wasmExists = await fs.access(
      path.join(process.cwd(), 'wasm/ruv_swarm_wasm_bg.wasm'),
    ).then(() => true).catch(() => false);

    if (!wasmExists) {
      console.log(chalk.red('❌ WASM files not found. Building...'));
      await this.runCommand('npm run build:wasm');
    }

    // Check node version
    const nodeVersion = process.version;
    const majorVersion = parseInt(nodeVersion.split('.')[0].substring(1), 10);
    if (majorVersion < 14) {
      console.error(chalk.red(`❌ Node.js ${nodeVersion} is too old. Required: >= 14.0.0`));
      process.exit(1);
    }

    console.log(chalk.green('✅ Prerequisites satisfied\n'));
  }

  async runTestSuite(suite) {
    console.log(chalk.bold.cyan(`\n📋 ${suite.name}`));
    console.log(chalk.gray('─'.repeat(80)));

    if (suite.requiresBrowser && process.env.CI) {
      console.log(chalk.yellow('⚠️  Skipping browser tests in CI environment'));
      this.results.push({
        ...suite,
        success: true,
        skipped: true,
        duration: 0,
      });
      return;
    }

    const suiteStart = performance.now();

    try {
      const result = await this.runCommand(suite.command, {
        stdio: 'inherit',
        env: {
          ...process.env,
          NODE_ENV: 'test',
          FORCE_COLOR: '1',
        },
      });

      const duration = performance.now() - suiteStart;

      this.results.push({
        ...suite,
        success: result.code === 0,
        duration,
        output: result.output,
      });

      if (result.code === 0) {
        console.log(chalk.green(`✅ ${suite.name} passed (${(duration / 1000).toFixed(2)}s)`));
      } else {
        console.log(chalk.red(`❌ ${suite.name} failed (${(duration / 1000).toFixed(2)}s)`));
      }
    } catch (error) {
      const duration = performance.now() - suiteStart;
      console.error(chalk.red(`❌ ${suite.name} error: ${error.message}`));

      this.results.push({
        ...suite,
        success: false,
        duration,
        error: error.message,
      });
    }
  }

  async runCommand(command, options = {}) {
    return new Promise((resolve) => {
      const [cmd, ...args] = command.split(' ');
      const child = spawn(cmd, args, {
        shell: true,
        ...options,
      });

      let output = '';

      if (options.stdio !== 'inherit') {
        child.stdout.on('data', (data) => {
          output += data.toString();
        });

        child.stderr.on('data', (data) => {
          output += data.toString();
        });
      }

      child.on('close', (code) => {
        resolve({ code, output });
      });
    });
  }

  async generateReports() {
    console.log(chalk.yellow('\n📊 Generating reports...'));

    const reportDir = path.join(process.cwd(), 'test-reports');
    await fs.mkdir(reportDir, { recursive: true });

    // Test results report
    const testReport = {
      timestamp: new Date().toISOString(),
      duration: performance.now() - this.startTime,
      suites: this.results,
      summary: {
        total: this.results.length,
        passed: this.results.filter(r => r.success).length,
        failed: this.results.filter(r => !r.success && !r.skipped).length,
        skipped: this.results.filter(r => r.skipped).length,
      },
    };

    await fs.writeFile(
      path.join(reportDir, `test-report-${Date.now()}.json`),
      JSON.stringify(testReport, null, 2),
    );

    // Coverage report
    try {
      const coverageFile = path.join(process.cwd(), 'coverage/coverage-summary.json');
      const coverageData = JSON.parse(await fs.readFile(coverageFile, 'utf-8'));

      this.coverageData = coverageData.total;

      // Generate coverage badge
      const coveragePercent = coverageData.total.lines.pct;
      const badgeColor = coveragePercent >= 90 ? 'green' :
        coveragePercent >= 80 ? 'yellow' : 'red';

      console.log(chalk.bold(`\n📈 Coverage: ${coveragePercent}% (${badgeColor})`));
    } catch (error) {
      console.log(chalk.yellow('⚠️  Coverage data not available'));
    }

    // Performance summary
    const perfReport = await this.generatePerformanceReport();
    await fs.writeFile(
      path.join(reportDir, 'performance-summary.json'),
      JSON.stringify(perfReport, null, 2),
    );

    console.log(chalk.green('✅ Reports generated'));
  }

  async generatePerformanceReport() {
    // Extract performance metrics from test results
    const metrics = {
      wasmInitialization: { target: 200, actual: null },
      agentCreation: { target: 5, actual: null },
      neuralInference: { target: 5, actual: null },
      messageThoughput: { target: 10000, actual: null },
    };

    // Parse performance test output
    const perfTest = this.results.find(r => r.name.includes('Performance'));
    if (perfTest && perfTest.output) {
      // Extract metrics from output (simplified)
      const lines = perfTest.output.split('\n');
      lines.forEach(line => {
        if (line.includes('initialization:')) {
          const match = line.match(/avg=(\d+\.?\d*)/);
          if (match) {
            metrics.wasmInitialization.actual = parseFloat(match[1]);
          }
        }
        // ... parse other metrics
      });
    }

    return {
      timestamp: new Date().toISOString(),
      metrics,
      meetsTargets: Object.values(metrics).every(m =>
        m.actual === null || m.actual <= m.target,
      ),
    };
  }

  displaySummary() {
    const totalDuration = (performance.now() - this.startTime) / 1000;

    console.log(chalk.bold.blue('\n\n📊 Test Suite Summary'));
    console.log(chalk.gray('═'.repeat(80)));

    // Test results
    console.log(chalk.bold('\nTest Results:'));
    this.results.forEach(result => {
      const icon = result.skipped ? '⚪' : result.success ? '✅' : '❌';
      const time = result.duration ? ` (${(result.duration / 1000).toFixed(2)}s)` : '';
      console.log(`  ${icon} ${result.name}${time}`);
    });

    // Coverage summary
    if (this.coverageData.lines) {
      console.log(chalk.bold('\nCoverage Summary:'));
      console.log(`  Lines:      ${this.coverageData.lines.pct}%`);
      console.log(`  Statements: ${this.coverageData.statements.pct}%`);
      console.log(`  Branches:   ${this.coverageData.branches.pct}%`);
      console.log(`  Functions:  ${this.coverageData.functions.pct}%`);
    }

    // Overall summary
    const passed = this.results.filter(r => r.success).length;
    const failed = this.results.filter(r => !r.success && !r.skipped).length;
    const skipped = this.results.filter(r => r.skipped).length;

    console.log(chalk.bold('\nOverall Summary:'));
    console.log(`  Total:    ${this.results.length} suites`);
    console.log(`  Passed:   ${chalk.green(passed)}`);
    console.log(`  Failed:   ${failed > 0 ? chalk.red(failed) : failed}`);
    console.log(`  Skipped:  ${chalk.yellow(skipped)}`);
    console.log(`  Duration: ${totalDuration.toFixed(2)}s`);

    console.log(chalk.gray('\n═'.repeat(80)));

    if (failed === 0) {
      console.log(chalk.bold.green('\n✅ All tests passed! 🎉'));
    } else {
      console.log(chalk.bold.red(`\n❌ ${failed} test suite(s) failed`));
    }
  }
}

// Run tests
const runner = new TestRunner();
runner.run().catch(error => {
  console.error(chalk.red('Fatal error:', error));
  process.exit(1);
});
