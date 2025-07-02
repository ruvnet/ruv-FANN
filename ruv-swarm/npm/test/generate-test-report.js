#!/usr/bin/env node

/**
 * Generate comprehensive test report for ruv-swarm
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class TestReportGenerator {
  constructor() {
    this.reportData = {
      generated: new Date().toISOString(),
      project: 'ruv-swarm',
      version: require('../package.json').version,
      coverage: null,
      testSuites: {},
      summary: {
        totalFiles: 0,
        totalTests: 0,
        totalPassed: 0,
        totalFailed: 0,
        totalSkipped: 0,
        coverage: {
          lines: 0,
          statements: 0,
          functions: 0,
          branches: 0,
        },
      },
    };
  }

  async generate() {
    console.log('Generating comprehensive test report...\n');

    // Count test files
    this.countTestFiles();

    // Run tests with coverage
    await this.runTestsWithCoverage();

    // Parse coverage report
    this.parseCoverageReport();

    // Generate report files
    this.generateMarkdownReport();
    this.generateJSONReport();
    this.generateHTMLReport();

    console.log('\n✅ Test report generation complete!');
    console.log('   - Markdown: test-report.md');
    console.log('   - JSON: test-report.json');
    console.log('   - HTML: test-report.html');
    console.log('   - Coverage: coverage/lcov-report/index.html');
  }

  countTestFiles() {
    const testDirs = [
      'test/unit',
      'test/integration',
      'test/performance',
      'test',
    ];

    let totalFiles = 0;
    for (const dir of testDirs) {
      const fullPath = path.join(__dirname, '..', dir);
      if (fs.existsSync(fullPath)) {
        const files = this.getTestFiles(fullPath);
        totalFiles += files.length;
        this.reportData.testSuites[dir] = {
          files: files.length,
          fileList: files.map(f => path.relative(path.join(__dirname, '..'), f)),
        };
      }
    }

    this.reportData.summary.totalFiles = totalFiles;
  }

  getTestFiles(dir) {
    const files = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        files.push(...this.getTestFiles(fullPath));
      } else if (entry.name.endsWith('.test.js') || entry.name.endsWith('.spec.js')) {
        files.push(fullPath);
      }
    }

    return files;
  }

  async runTestsWithCoverage() {
    console.log('Running tests with coverage...');

    try {
      const output = execSync('npm test -- --coverage --json --outputFile=test-results.json', {
        cwd: path.join(__dirname, '..'),
        encoding: 'utf8',
        stdio: 'pipe',
      });

      // Parse test results
      if (fs.existsSync(path.join(__dirname, '..', 'test-results.json'))) {
        const results = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'test-results.json'), 'utf8'));
        this.parseTestResults(results);
      }
    } catch (error) {
      console.error('Error running tests:', error.message);
      // Continue with report generation even if tests fail
    }
  }

  parseTestResults(results) {
    if (results.testResults) {
      for (const suite of results.testResults) {
        this.reportData.summary.totalTests += suite.numTotalTests || 0;
        this.reportData.summary.totalPassed += suite.numPassedTests || 0;
        this.reportData.summary.totalFailed += suite.numFailedTests || 0;
        this.reportData.summary.totalSkipped += suite.numPendingTests || 0;
      }
    }
  }

  parseCoverageReport() {
    const coveragePath = path.join(__dirname, '..', 'coverage', 'coverage-summary.json');

    if (fs.existsSync(coveragePath)) {
      const coverage = JSON.parse(fs.readFileSync(coveragePath, 'utf8'));

      if (coverage.total) {
        this.reportData.coverage = coverage;
        this.reportData.summary.coverage = {
          lines: coverage.total.lines.pct,
          statements: coverage.total.statements.pct,
          functions: coverage.total.functions.pct,
          branches: coverage.total.branches.pct,
        };
      }
    }
  }

  generateMarkdownReport() {
    const reportPath = path.join(__dirname, '..', 'test-report.md');

    let markdown = `# Test Report for ruv-swarm

Generated: ${this.reportData.generated}  
Version: ${this.reportData.version}

## Summary

- **Total Test Files**: ${this.reportData.summary.totalFiles}
- **Total Tests**: ${this.reportData.summary.totalTests}
- **Passed**: ${this.reportData.summary.totalPassed} ✅
- **Failed**: ${this.reportData.summary.totalFailed} ❌
- **Skipped**: ${this.reportData.summary.totalSkipped} ⏭️
- **Success Rate**: ${this.calculateSuccessRate()}%

## Code Coverage

| Metric | Coverage | Status |
|--------|----------|--------|
| Lines | ${this.reportData.summary.coverage.lines.toFixed(2)}% | ${this.getCoverageStatus(this.reportData.summary.coverage.lines)} |
| Statements | ${this.reportData.summary.coverage.statements.toFixed(2)}% | ${this.getCoverageStatus(this.reportData.summary.coverage.statements)} |
| Functions | ${this.reportData.summary.coverage.functions.toFixed(2)}% | ${this.getCoverageStatus(this.reportData.summary.coverage.functions)} |
| Branches | ${this.reportData.summary.coverage.branches.toFixed(2)}% | ${this.getCoverageStatus(this.reportData.summary.coverage.branches)} |

## Test Suites

`;

    for (const [suite, data] of Object.entries(this.reportData.testSuites)) {
      markdown += `### ${suite}\n\n`;
      markdown += `- Files: ${data.files}\n`;
      markdown += '- Test files:\n';
      for (const file of data.fileList) {
        markdown += `  - ${file}\n`;
      }
      markdown += '\n';
    }

    markdown += `## Coverage Details

`;

    if (this.reportData.coverage) {
      markdown += '### File Coverage\n\n';
      markdown += '| File | Lines | Statements | Functions | Branches |\n';
      markdown += '|------|-------|------------|-----------|----------|\n';

      for (const [file, data] of Object.entries(this.reportData.coverage)) {
        if (file !== 'total' && data.lines) {
          const relPath = file.replace(process.cwd(), '.');
          markdown += `| ${relPath} | ${data.lines.pct}% | ${data.statements.pct}% | ${data.functions.pct}% | ${data.branches.pct}% |\n`;
        }
      }
    }

    markdown += `\n## Recommendations

`;

    if (this.reportData.summary.coverage.lines < 80) {
      markdown += '- ⚠️ Line coverage is below 80%. Consider adding more unit tests.\n';
    }
    if (this.reportData.summary.coverage.branches < 80) {
      markdown += '- ⚠️ Branch coverage is below 80%. Ensure all code paths are tested.\n';
    }
    if (this.reportData.summary.totalFailed > 0) {
      markdown += `- ❌ There are ${this.reportData.summary.totalFailed} failing tests that need to be fixed.\n`;
    }
    if (this.reportData.summary.coverage.lines >= 80) {
      markdown += '- ✅ Good job! Line coverage meets the 80% threshold.\n';
    }

    fs.writeFileSync(reportPath, markdown);
  }

  generateJSONReport() {
    const reportPath = path.join(__dirname, '..', 'test-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(this.reportData, null, 2));
  }

  generateHTMLReport() {
    const reportPath = path.join(__dirname, '..', 'test-report.html');

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ruv-swarm Test Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2196F3;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .metric {
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 4px;
            font-weight: bold;
        }
        .metric.success { background-color: #4CAF50; color: white; }
        .metric.warning { background-color: #FF9800; color: white; }
        .metric.error { background-color: #f44336; color: white; }
        .coverage-bar {
            background-color: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin: 5px 0;
        }
        .coverage-fill {
            height: 20px;
            background-color: #4CAF50;
            text-align: center;
            color: white;
            font-size: 12px;
            line-height: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>ruv-swarm Test Report</h1>
        <p>Generated: ${this.reportData.generated}</p>
        <p>Version: ${this.reportData.version}</p>
    </div>

    <div class="card">
        <h2>Test Summary</h2>
        <div>
            <span class="metric">Total Files: ${this.reportData.summary.totalFiles}</span>
            <span class="metric">Total Tests: ${this.reportData.summary.totalTests}</span>
            <span class="metric success">Passed: ${this.reportData.summary.totalPassed}</span>
            <span class="metric error">Failed: ${this.reportData.summary.totalFailed}</span>
            <span class="metric warning">Skipped: ${this.reportData.summary.totalSkipped}</span>
        </div>
    </div>

    <div class="card">
        <h2>Code Coverage</h2>
        ${this.generateCoverageHTML()}
    </div>

    <div class="card">
        <h2>Test Suites</h2>
        ${this.generateSuitesHTML()}
    </div>
</body>
</html>`;

    fs.writeFileSync(reportPath, html);
  }

  generateCoverageHTML() {
    const metrics = ['lines', 'statements', 'functions', 'branches'];
    let html = '';

    for (const metric of metrics) {
      const value = this.reportData.summary.coverage[metric];
      const status = this.getCoverageStatus(value);
      const color = status === '✅' ? '#4CAF50' : '#FF9800';

      html += `
        <div>
            <strong>${metric.charAt(0).toUpperCase() + metric.slice(1)}: ${value.toFixed(2)}%</strong>
            <div class="coverage-bar">
                <div class="coverage-fill" style="width: ${value}%; background-color: ${color};">
                    ${value.toFixed(2)}%
                </div>
            </div>
        </div>`;
    }

    return html;
  }

  generateSuitesHTML() {
    let html = '<table><tr><th>Suite</th><th>Files</th></tr>';

    for (const [suite, data] of Object.entries(this.reportData.testSuites)) {
      html += `<tr><td>${suite}</td><td>${data.files}</td></tr>`;
    }

    html += '</table>';
    return html;
  }

  calculateSuccessRate() {
    if (this.reportData.summary.totalTests === 0) {
      return 0;
    }
    return ((this.reportData.summary.totalPassed / this.reportData.summary.totalTests) * 100).toFixed(2);
  }

  getCoverageStatus(percentage) {
    return percentage >= 80 ? '✅' : '⚠️';
  }
}

// Run report generation
// Direct execution block
{
  const generator = new TestReportGenerator();
  generator.generate().catch(error => {
    console.error('Error generating report:', error);
    process.exit(1);
  });
}