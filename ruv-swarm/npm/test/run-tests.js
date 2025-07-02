#!/usr/bin/env node

/**
 * Comprehensive test runner for ruv-swarm
 */

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Test categories
const testSuites = {
  unit: {
    name: 'Unit Tests',
    pattern: 'test/unit/**/*.test.js',
    timeout: 30000,
  },
  integration: {
    name: 'Integration Tests',
    pattern: 'test/integration/**/*.test.js',
    timeout: 60000,
  },
  performance: {
    name: 'Performance Benchmarks',
    pattern: 'test/performance/**/*.test.js',
    timeout: 300000,
  },
  existing: {
    name: 'Existing Tests',
    pattern: 'test/*.test.js',
    timeout: 60000,
  },
};

// Colors for output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function logSection(title) {
  console.log(`\n${ '='.repeat(80)}`);
  log(title, 'bright');
  console.log(`${'='.repeat(80) }\n`);
}

async function runJest(suite, coverage = false) {
  return new Promise((resolve, reject) => {
    const args = [
      '--testMatch', `**/${suite.pattern}`,
      '--testTimeout', suite.timeout.toString(),
      '--forceExit',
    ];

    if (coverage) {
      args.push(
        '--coverage',
        '--coverageDirectory', 'coverage',
        '--collectCoverageFrom', 'src/**/*.js',
        '--coveragePathIgnorePatterns', '/node_modules/',
        '--coverageReporters', 'text', 'lcov', 'html',
      );
    }

    const jest = spawn('npx', ['jest', ...args], {
      stdio: 'inherit',
      cwd: path.join(__dirname, '..'),
    });

    jest.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Jest exited with code ${code}`));
      }
    });

    jest.on('error', (err) => {
      reject(err);
    });
  });
}

async function runTestSuite(suiteName, suite, options = {}) {
  logSection(`Running ${suite.name}`);

  try {
    const startTime = Date.now();
    await runJest(suite, options.coverage);
    const duration = Date.now() - startTime;

    log(`✓ ${suite.name} completed in ${(duration / 1000).toFixed(2)}s`, 'green');
    return { success: true, duration };
  } catch (error) {
    log(`✗ ${suite.name} failed: ${error.message}`, 'red');
    return { success: false, error: error.message };
  }
}

async function generateTestReport(results) {
  const report = {
    timestamp: new Date().toISOString(),
    summary: {
      total: Object.keys(results).length,
      passed: 0,
      failed: 0,
      totalDuration: 0,
    },
    suites: results,
  };

  for (const [name, result] of Object.entries(results)) {
    if (result.success) {
      report.summary.passed++;
      report.summary.totalDuration += result.duration;
    } else {
      report.summary.failed++;
    }
  }

  const reportPath = path.join(__dirname, `test-report-${Date.now()}.json`);
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

  return report;
}

async function main() {
  const args = process.argv.slice(2);
  const runAll = args.length === 0 || args.includes('--all');
  const runCoverage = args.includes('--coverage');
  const runUnit = runAll || args.includes('--unit');
  const runIntegration = runAll || args.includes('--integration');
  const runPerformance = args.includes('--performance'); // Only run on demand
  const runExisting = runAll || args.includes('--existing');

  logSection('ruv-swarm Test Suite');
  log('Starting comprehensive test run...', 'cyan');

  const results = {};

  // Run selected test suites
  if (runUnit) {
    results.unit = await runTestSuite('unit', testSuites.unit, { coverage: runCoverage });
  }

  if (runIntegration) {
    results.integration = await runTestSuite('integration', testSuites.integration, { coverage: runCoverage });
  }

  if (runExisting) {
    results.existing = await runTestSuite('existing', testSuites.existing, { coverage: runCoverage });
  }

  if (runPerformance) {
    results.performance = await runTestSuite('performance', testSuites.performance);
  }

  // Generate report
  const report = await generateTestReport(results);

  // Display summary
  logSection('Test Summary');
  log(`Total Suites: ${report.summary.total}`, 'bright');
  log(`Passed: ${report.summary.passed}`, 'green');
  log(`Failed: ${report.summary.failed}`, report.summary.failed > 0 ? 'red' : 'green');
  log(`Total Duration: ${(report.summary.totalDuration / 1000).toFixed(2)}s`, 'cyan');

  if (runCoverage) {
    log('\nCoverage report generated in ./coverage/', 'yellow');
  }

  log(`\nDetailed report saved to: ${path.basename(Object.keys(results)[0])}`, 'magenta');

  // Exit with appropriate code
  process.exit(report.summary.failed > 0 ? 1 : 0);
}

// Handle errors
process.on('unhandledRejection', (error) => {
  log(`Unhandled rejection: ${error.message}`, 'red');
  process.exit(1);
});

// Show usage
if (process.argv.includes('--help')) {
  console.log(`
Usage: node run-tests.js [options]

Options:
  --all          Run all test suites (default)
  --unit         Run unit tests only
  --integration  Run integration tests only
  --existing     Run existing tests only
  --performance  Run performance benchmarks
  --coverage     Generate code coverage report
  --help         Show this help message

Examples:
  node run-tests.js                    # Run all tests except performance
  node run-tests.js --unit --coverage  # Run unit tests with coverage
  node run-tests.js --performance      # Run performance benchmarks only
`);
  process.exit(0);
}

// Run tests
main().catch((error) => {
  log(`Test runner error: ${error.message}`, 'red');
  process.exit(1);
});