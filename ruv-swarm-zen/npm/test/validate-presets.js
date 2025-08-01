#!/usr/bin/env node

import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';

const execAsync = promisify(exec);

const presets = ['default', 'minGPT', 'stateOfArt'];

async function validateAllPresets() {
  console.log('🧬 Validating All Neural Presets...\n');

  const results = {
    timestamp: new Date().toISOString(),
    presets: {},
  };

  for (const preset of presets) {
    console.log(`\n📋 Testing ${preset} preset...`);

    try {
      // Test with the preset
      const { stdout: testOutput } = await execAsync(`npm test -- --preset=${preset}`);

      // Extract test results
      const passed = testOutput.match(/(\d+) passed/)?.[1] || 0;
      const failed = testOutput.match(/(\d+) failed/)?.[1] || 0;

      // Run coverage for this preset
      const { stdout: coverageOutput } = await execAsync(`npx nyc --reporter=json-summary npm test -- --preset=${preset}`);

      // Read coverage data
      const coverageData = JSON.parse(
        await fs.readFile('coverage/coverage-summary.json', 'utf8'),
      );

      results.presets[preset] = {
        success: parseInt(failed, 10) === 0,
        tests: {
          passed: parseInt(passed, 10),
          failed: parseInt(failed, 10),
        },
        coverage: {
          lines: coverageData.total.lines.pct,
          branches: coverageData.total.branches.pct,
          functions: coverageData.total.functions.pct,
          statements: coverageData.total.statements.pct,
        },
        performance: await testPresetPerformance(preset),
      };

      console.log(`  ✅ Tests: ${passed} passed, ${failed} failed`);
      console.log(`  📊 Coverage: ${coverageData.total.lines.pct.toFixed(2)}% lines`);

    } catch (error) {
      console.log(`  ❌ Error: ${error.message}`);
      results.presets[preset] = {
        success: false,
        error: error.message,
      };
    }
  }

  // Save results
  await fs.writeFile(
    'preset-validation-results.json',
    JSON.stringify(results, null, 2),
  );

  // Generate report
  await generatePresetReport(results);

  return results;
}

async function testPresetPerformance(preset) {
  try {
    const { stdout } = await execAsync(`node test/benchmarks/benchmark-neural-models.js --preset=${preset} --iterations=3`);

    // Extract performance metrics
    const metrics = {
      initialized: stdout.includes('initialized'),
      avgTime: parseFloat(stdout.match(/Average time: ([\d.]+)ms/)?.[1] || 0),
      memory: stdout.match(/Memory: ([\d.]+)MB/)?.[1] || 'N/A',
    };

    return metrics;
  } catch (error) {
    return { error: error.message };
  }
}

async function generatePresetReport(results) {
  const report = `# 🧬 Neural Preset Validation Report

**Generated**: ${results.timestamp}

## 📊 Summary

| Preset | Tests | Coverage | Performance | Status |
|--------|-------|----------|-------------|---------|
${presets.map(preset => {
    const data = results.presets[preset];
    if (!data.success) {
      return `| ${preset} | ❌ Error | - | - | Failed |`;
    }
    return `| ${preset} | ✅ ${data.tests.passed}/${data.tests.passed + data.tests.failed} | ${data.coverage.lines.toFixed(1)}% | ${data.performance.avgTime?.toFixed(2) || 'N/A'}ms | ${data.success ? 'Pass' : 'Fail'} |`;
  }).join('\n')}

## 📈 Detailed Results

${presets.map(preset => {
    const data = results.presets[preset];
    if (!data.success) {
      return `### ❌ ${preset}
- Error: ${data.error}`;
    }

    return `### ${data.success ? '✅' : '❌'} ${preset}
- **Tests**: ${data.tests.passed} passed, ${data.tests.failed} failed
- **Coverage**:
  - Lines: ${data.coverage.lines.toFixed(2)}%
  - Branches: ${data.coverage.branches.toFixed(2)}%
  - Functions: ${data.coverage.functions.toFixed(2)}%
  - Statements: ${data.coverage.statements.toFixed(2)}%
- **Performance**:
  - Avg Time: ${data.performance.avgTime?.toFixed(2) || 'N/A'}ms
  - Memory: ${data.performance.memory || 'N/A'}
  - Status: ${data.performance.initialized ? 'Initialized' : 'Failed'}`;
  }).join('\n\n')}

## 🎯 Recommendations

1. Ensure all presets maintain 100% test coverage
2. Monitor performance regression between presets
3. Add preset-specific test cases where needed
4. Document preset differences in test behavior
`;

  await fs.writeFile('PRESET_VALIDATION_REPORT.md', report);
  console.log('\n📄 Report saved to PRESET_VALIDATION_REPORT.md');
}

// Run validation
validateAllPresets().catch(console.error);
