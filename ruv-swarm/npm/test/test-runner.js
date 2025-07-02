/**
 * Test Runner for Coverage Tests
 * Properly handles ES modules and async tests
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import assert from 'assert';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Track test results
const results = {
  total: 0,
  passed: 0,
  failed: 0,
  errors: []
};

// Custom test runner for describe/it pattern
const suites = [];
let currentSuite = null;
let currentTest = null;

global.describe = (name, fn) => {
  const suite = {
    name,
    tests: [],
    beforeEach: null,
    afterEach: null
  };
  suites.push(suite);
  currentSuite = suite;
  fn();
  currentSuite = null;
};

global.it = (name, fn) => {
  if (!currentSuite) throw new Error('it() must be inside describe()');
  currentSuite.tests.push({ name, fn });
};

global.beforeEach = (fn) => {
  if (!currentSuite) throw new Error('beforeEach() must be inside describe()');
  currentSuite.beforeEach = fn;
};

global.afterEach = (fn) => {
  if (!currentSuite) throw new Error('afterEach() must be inside describe()');
  currentSuite.afterEach = fn;
};

// Enhanced assert with better error messages
global.assert = new Proxy(assert, {
  get(target, prop) {
    if (prop === 'rejects') {
      return async (promise, expectedError) => {
        try {
          await promise;
          throw new Error(`Expected promise to reject with: ${expectedError}`);
        } catch (error) {
          if (expectedError instanceof RegExp) {
            if (!expectedError.test(error.message)) {
              throw new Error(`Error message "${error.message}" does not match ${expectedError}`);
            }
          } else if (typeof expectedError === 'string') {
            if (!error.message.includes(expectedError)) {
              throw new Error(`Error message "${error.message}" does not include "${expectedError}"`);
            }
          }
        }
      };
    }
    return target[prop];
  }
});

// Run all suites
async function runSuites() {
  for (const suite of suites) {
    console.log(`\n  ${suite.name}`);
    
    for (const test of suite.tests) {
      currentTest = test;
      results.total++;
      
      try {
        // Run beforeEach if exists
        if (suite.beforeEach) {
          await suite.beforeEach();
        }
        
        // Run the test
        await test.fn();
        
        // Run afterEach if exists
        if (suite.afterEach) {
          await suite.afterEach();
        }
        
        console.log(`    ✓ ${test.name}`);
        results.passed++;
      } catch (error) {
        console.log(`    ✗ ${test.name}`);
        console.log(`      ${error.message}`);
        results.failed++;
        results.errors.push({
          suite: suite.name,
          test: test.name,
          error: error.message,
          stack: error.stack
        });
      }
    }
  }
}

// Run a specific test file
export async function run(testFile) {
  try {
    // Clear previous suites
    suites.length = 0;
    
    // Import the test file
    await import(testFile);
    
    // Run all suites
    await runSuites();
    
    // Print summary
    console.log('\n' + '='.repeat(50));
    console.log(`Total: ${results.total}`);
    console.log(`Passed: ${results.passed}`);
    console.log(`Failed: ${results.failed}`);
    
    if (results.failed > 0) {
      console.log('\nFailed Tests:');
      results.errors.forEach(error => {
        console.log(`\n${error.suite} > ${error.test}`);
        console.log(error.error);
        if (process.env.VERBOSE) {
          console.log(error.stack);
        }
      });
    }
    
    return results;
  } catch (error) {
    console.error('Test runner error:', error);
    throw error;
  }
}

// Run all coverage test files
export async function runAll() {
  const testFiles = [
    './coverage-edge-cases.test.js',
    './neural-models-coverage.test.js',
    './hooks-coverage.test.js'
  ];
  
  console.log('Running all coverage tests...\n');
  
  const allResults = {
    total: 0,
    passed: 0,
    failed: 0
  };
  
  for (const file of testFiles) {
    console.log(`\nRunning ${file}...`);
    console.log('='.repeat(50));
    
    const fileResults = await run(join(__dirname, file));
    allResults.total += fileResults.total;
    allResults.passed += fileResults.passed;
    allResults.failed += fileResults.failed;
    
    // Reset results for next file
    results.total = 0;
    results.passed = 0;
    results.failed = 0;
    results.errors = [];
  }
  
  // Print overall summary
  console.log('\n' + '='.repeat(50));
  console.log('OVERALL SUMMARY');
  console.log('='.repeat(50));
  console.log(`Total Tests: ${allResults.total}`);
  console.log(`Passed: ${allResults.passed}`);
  console.log(`Failed: ${allResults.failed}`);
  console.log(`Success Rate: ${((allResults.passed / allResults.total) * 100).toFixed(2)}%`);
  
  return allResults;
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAll()
    .then(results => {
      process.exit(results.failed > 0 ? 1 : 0);
    })
    .catch(error => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}