#!/bin/bash

# Comprehensive Docker Test Suite for ruv-swarm v1.0.6
# Tests all functionality across different Node.js versions

set -e

echo "================================================"
echo "ruv-swarm v1.0.6 Docker Test Suite"
echo "================================================"
echo "Node Version: $(node --version)"
echo "NPM Version: $(npm --version)"
echo "Environment: $NODE_VERSION"
echo "Date: $(date)"
echo ""

# Create results directory
RESULTS_DIR="/app/test-results"
mkdir -p $RESULTS_DIR

# Run all test suites
echo "1. Running Unit Tests"
echo "===================="
npm run test:jest -- --json --outputFile=$RESULTS_DIR/unit-tests.json || true

echo ""
echo "2. Running Integration Tests"
echo "==========================="
npm run test:integration -- --json --outputFile=$RESULTS_DIR/integration-tests.json || true

echo ""
echo "3. Running WASM Tests"
echo "===================="
node test/validate-wasm-loading.js > $RESULTS_DIR/wasm-tests.log 2>&1 || true

echo ""
echo "4. Running MCP Tests"
echo "==================="
npm run test:mcp > $RESULTS_DIR/mcp-tests.log 2>&1 || true

echo ""
echo "5. Running Performance Tests"
echo "==========================="
npm run test:performance > $RESULTS_DIR/performance-tests.log 2>&1 || true

echo ""
echo "6. Running Neural Tests"
echo "======================"
npm run test:neural > $RESULTS_DIR/neural-tests.log 2>&1 || true

echo ""
echo "7. Running Coverage Analysis"
echo "==========================="
npm run test:coverage > $RESULTS_DIR/coverage.log 2>&1 || true

# Generate summary report
echo ""
echo "Generating Test Summary Report..."
node -e "
const fs = require('fs');
const path = require('path');

const resultsDir = '$RESULTS_DIR';
const summary = {
  environment: {
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch,
    timestamp: new Date().toISOString()
  },
  tests: {}
};

// Read test results
const files = fs.readdirSync(resultsDir);
files.forEach(file => {
  const filePath = path.join(resultsDir, file);
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    summary.tests[file] = {
      exists: true,
      size: fs.statSync(filePath).size
    };
  } catch (err) {
    summary.tests[file] = { error: err.message };
  }
});

fs.writeFileSync(path.join(resultsDir, 'summary.json'), JSON.stringify(summary, null, 2));
console.log('Test summary generated:', path.join(resultsDir, 'summary.json'));
"

echo ""
echo "================================================"
echo "Docker Test Suite Complete"
echo "Results saved to: $RESULTS_DIR"
echo "================================================"