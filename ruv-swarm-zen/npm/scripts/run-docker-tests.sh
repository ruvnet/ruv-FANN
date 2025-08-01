#!/bin/bash

# Comprehensive Docker Test Runner for ruv-swarm v1.0.6
# Orchestrates all Docker-based tests and generates a final report

set -e

echo "================================================"
echo "ruv-swarm v1.0.6 Docker Test Runner"
echo "================================================"
echo "Date: $(date)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create test results directory
mkdir -p test-results
rm -rf test-results/*

# Build all Docker images
echo "Building Docker test images..."
docker-compose -f docker-compose.test.yml build --parallel

# Function to run a test and capture results
run_test() {
    local service=$1
    local description=$2
    
    echo ""
    echo "Running: $description"
    echo "========================================"
    
    if docker-compose -f docker-compose.test.yml run --rm $service; then
        echo -e "${GREEN}✅ $description: PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ $description: FAILED${NC}"
        return 1
    fi
}

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Run each test
tests=(
    "test-node18:Node.js 18 Compatibility"
    "test-node20:Node.js 20 Compatibility"
    "test-node22:Node.js 22 Compatibility"
    "test-npx:NPX Integration"
    "test-wasm:WASM Loading Validation"
    "test-performance:Performance Benchmarks"
    "test-mcp:MCP Server Functionality"
    "test-cross-platform:Cross-Platform Compatibility"
)

for test in "${tests[@]}"; do
    IFS=':' read -r service description <<< "$test"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if run_test "$service" "$description"; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
done

# Generate final report
echo ""
echo "================================================"
echo "Generating Final Test Report..."
echo "================================================"

# Combine all test results
node -e "
const fs = require('fs');
const path = require('path');

const resultsDir = 'test-results';
const finalReport = {
    testSuite: 'docker-comprehensive',
    version: '1.0.6',
    timestamp: new Date().toISOString(),
    summary: {
        total: $TOTAL_TESTS,
        passed: $PASSED_TESTS,
        failed: $FAILED_TESTS,
        passRate: ($PASSED_TESTS / $TOTAL_TESTS * 100).toFixed(2) + '%'
    },
    testRuns: []
};

// Read all test result files
const resultFiles = fs.readdirSync(resultsDir);
resultFiles.forEach(file => {
    if (file.endsWith('.json')) {
        try {
            const content = fs.readFileSync(path.join(resultsDir, file), 'utf8');
            const data = JSON.parse(content);
            finalReport.testRuns.push({
                file: file,
                suite: data.testSuite || 'unknown',
                summary: data.summary || {}
            });
        } catch (err) {
            console.error('Error reading', file, err.message);
        }
    }
});

// Save final report
const reportPath = path.join(resultsDir, 'docker-test-report.json');
fs.writeFileSync(reportPath, JSON.stringify(finalReport, null, 2));
console.log('Final report saved to:', reportPath);

// Generate markdown report
const markdown = \`# Docker Test Report for ruv-swarm v1.0.6

## Summary
- **Date**: \${finalReport.timestamp}
- **Total Tests**: \${finalReport.summary.total}
- **Passed**: \${finalReport.summary.passed}
- **Failed**: \${finalReport.summary.failed}
- **Pass Rate**: \${finalReport.summary.passRate}

## Test Results
\${finalReport.testRuns.map(run => \`
### \${run.suite}
- File: \${run.file}
- Total: \${run.summary.total || 'N/A'}
- Passed: \${run.summary.passed || 'N/A'}
- Failed: \${run.summary.failed || 'N/A'}
\`).join('')}

## Recommendations
\${finalReport.summary.failed > 0 ? '- ⚠️ Some tests failed. Please review the individual test reports.' : '- ✅ All tests passed! Ready for release.'}
\`;

fs.writeFileSync(path.join(resultsDir, 'docker-test-report.md'), markdown);
console.log('Markdown report saved to:', path.join(resultsDir, 'docker-test-report.md'));
"

# Cleanup
echo ""
echo "Cleaning up Docker resources..."
docker-compose -f docker-compose.test.yml down -v

# Display final summary
echo ""
echo "================================================"
echo "Docker Test Summary"
echo "================================================"
echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
echo "Pass Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✅ All Docker tests passed!${NC}"
    echo "Ready for v1.0.6 release."
    exit 0
else
    echo -e "${RED}❌ Some tests failed.${NC}"
    echo "Please review the test results in test-results/"
    exit 1
fi