#!/usr/bin/env node

/**
 * Comprehensive test script for validating neural pattern fixes
 * Tests: Pattern parsing, memory optimization, and persistence indicators
 */

import { exec } from 'child_process';
import util from 'util';
const execPromise = util.promisify(exec);
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Color codes for output
const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  reset: '\x1b[0m',
  bold: '\x1b[1m',
};

// Test result tracking
const testResults = {
  passed: [],
  failed: [],
  warnings: [],
};

// Helper function to run command and capture output
async function runCommand(command) {
  try {
    const { stdout, stderr } = await execPromise(command, {
      cwd: path.join(__dirname, '..'),
    });
    return { success: true, stdout, stderr };
  } catch (error) {
    return {
      success: false,
      stdout: error.stdout || '',
      stderr: error.stderr || error.message,
      error: error.message,
    };
  }
}

// Helper function to check if output contains expected text
function assertContains(output, expected, testName) {
  if (output.includes(expected)) {
    testResults.passed.push(`${testName}: Found "${expected}"`);
    return true;
  }
  testResults.failed.push(`${testName}: Missing "${expected}"`);
  return false;

}

// Helper function to check pattern in output
function assertPattern(output, pattern, testName) {
  if (pattern.test(output)) {
    testResults.passed.push(`${testName}: Pattern matched`);
    return true;
  }
  testResults.failed.push(`${testName}: Pattern not matched`);
  return false;

}

// Test 1: Pattern Parsing Fix Validation
async function testPatternParsing() {
  console.log(`\n${colors.bold}${colors.blue}Test 1: Pattern Parsing Fix Validation${colors.reset}`);
  console.log('─'.repeat(50));

  // Test 1.1: All patterns
  console.log(`\n${colors.yellow}Test 1.1: Testing --pattern all${colors.reset}`);
  const result1 = await runCommand('npx ruv-swarm neural patterns --pattern all');

  if (result1.success) {
    assertContains(result1.stdout, 'All Patterns', 'All patterns header');
    assertContains(result1.stdout, 'Cognitive Patterns:', 'Cognitive patterns section');
    assertContains(result1.stdout, 'Neural Model Patterns:', 'Neural model patterns section');

    // Check all 6 cognitive patterns
    const cognitivePatterns = ['Convergent', 'Divergent', 'Lateral', 'Systems', 'Critical', 'Abstract'];
    for (const pattern of cognitivePatterns) {
      assertContains(result1.stdout, `${pattern} Pattern:`, `${pattern} pattern presence`);
    }

    // Check neural models
    const neuralModels = ['Attention', 'Lstm', 'Transformer'];
    for (const model of neuralModels) {
      assertContains(result1.stdout, `${model} Model:`, `${model} model presence`);
    }
  } else {
    testResults.failed.push('All patterns command failed to execute');
  }

  // Test 1.2: Specific pattern (convergent)
  console.log(`\n${colors.yellow}Test 1.2: Testing --pattern convergent${colors.reset}`);
  const result2 = await runCommand('npx ruv-swarm neural patterns --pattern convergent');

  if (result2.success) {
    assertContains(result2.stdout, 'Convergent Pattern', 'Convergent pattern header');
    assertContains(result2.stdout, 'Cognitive Patterns:', 'Cognitive patterns category');
    assertContains(result2.stdout, 'Focused problem-solving', 'Convergent pattern content');

    // Should NOT contain other patterns
    if (!result2.stdout.includes('Divergent Pattern:')) {
      testResults.passed.push('Convergent: Correctly excludes other patterns');
    } else {
      testResults.failed.push('Convergent: Incorrectly includes other patterns');
    }
  } else {
    testResults.failed.push('Convergent pattern command failed');
  }

  // Test 1.3: Invalid pattern
  console.log(`\n${colors.yellow}Test 1.3: Testing --pattern invalid${colors.reset}`);
  const result3 = await runCommand('npx ruv-swarm neural patterns --pattern invalid');

  if (result3.success) {
    assertContains(result3.stdout, 'Unknown pattern type:', 'Invalid pattern error message');
    assertContains(result3.stdout, 'Available patterns:', 'Available patterns list');
    assertContains(result3.stdout, 'Cognitive: convergent, divergent, lateral, systems, critical, abstract', 'Cognitive patterns list');
    assertContains(result3.stdout, 'Models: attention, lstm, transformer', 'Model patterns list');
  } else {
    testResults.failed.push('Invalid pattern handling failed');
  }
}

// Test 2: Memory Optimization Validation
async function testMemoryOptimization() {
  console.log(`\n${colors.bold}${colors.blue}Test 2: Memory Optimization Validation${colors.reset}`);
  console.log('─'.repeat(50));

  const patterns = ['convergent', 'divergent', 'lateral', 'systems', 'critical', 'abstract'];
  const memoryValues = [];

  for (const pattern of patterns) {
    console.log(`\n${colors.yellow}Testing memory for ${pattern} pattern${colors.reset}`);
    const result = await runCommand(`npx ruv-swarm neural patterns --pattern ${pattern}`);

    if (result.success) {
      // Extract memory usage
      const memoryMatch = result.stdout.match(/Memory Usage:\s*(\d+)\s*MB/);
      if (memoryMatch) {
        const memoryUsage = parseInt(memoryMatch[1], 10);
        memoryValues.push({ pattern, memory: memoryUsage });

        // Check if memory is in optimized range (250-300 MB)
        if (memoryUsage >= 250 && memoryUsage <= 300) {
          testResults.passed.push(`${pattern}: Memory optimized (${memoryUsage} MB)`);
        } else if (memoryUsage >= 200 && memoryUsage <= 350) {
          testResults.warnings.push(`${pattern}: Memory slightly outside target (${memoryUsage} MB)`);
        } else {
          testResults.failed.push(`${pattern}: Memory not optimized (${memoryUsage} MB)`);
        }
      } else {
        testResults.failed.push(`${pattern}: Could not extract memory usage`);
      }
    } else {
      testResults.failed.push(`${pattern}: Pattern command failed`);
    }
  }

  // Calculate memory variance
  if (memoryValues.length > 0) {
    const memoryNumbers = memoryValues.map(v => v.memory);
    const minMemory = Math.min(...memoryNumbers);
    const maxMemory = Math.max(...memoryNumbers);
    const variance = maxMemory - minMemory;

    console.log(`\n${colors.yellow}Memory Variance Analysis:${colors.reset}`);
    console.log(`Min: ${minMemory} MB, Max: ${maxMemory} MB, Variance: ${variance} MB`);

    if (variance < 100) {
      testResults.passed.push(`Memory variance under 100 MB (${variance} MB)`);
    } else {
      testResults.failed.push(`Memory variance exceeds 100 MB (${variance} MB)`);
    }
  }
}

// Test 3: Persistence Indicators Validation
async function testPersistenceIndicators() {
  console.log(`\n${colors.bold}${colors.blue}Test 3: Persistence Indicators Validation${colors.reset}`);
  console.log('─'.repeat(50));

  // First, create some training data to ensure we have persistence
  console.log(`\n${colors.yellow}Creating training data...${colors.reset}`);
  await runCommand('npx ruv-swarm neural train --model attention --iterations 5');

  // Now test neural status
  console.log(`\n${colors.yellow}Testing neural status persistence indicators${colors.reset}`);
  const result = await runCommand('npx ruv-swarm neural status');

  if (result.success) {
    // Check for training session count
    assertPattern(result.stdout, /Training Sessions:\s*\d+\s*sessions/, 'Training sessions display');

    // Check for saved models count with 📁 indicator
    assertPattern(result.stdout, /📁\s*\d+\s*saved models/, 'Saved models indicator');

    // Check for model status indicators
    assertContains(result.stdout, '✅ Trained', 'Trained indicator');

    // Check for persistence indicators in model lines
    const persistenceIndicators = ['✅', '📁', '🔄'];
    let foundIndicators = 0;
    for (const indicator of persistenceIndicators) {
      if (result.stdout.includes(indicator)) {
        foundIndicators++;
      }
    }

    if (foundIndicators >= 2) {
      testResults.passed.push(`Found ${foundIndicators}/3 persistence indicators`);
    } else {
      testResults.failed.push(`Only found ${foundIndicators}/3 persistence indicators`);
    }

    // Check for session continuity section
    if (result.stdout.includes('Session Continuity:')) {
      testResults.passed.push('Session continuity section present');
      assertContains(result.stdout, 'Models loaded from previous session:', 'Session loading info');
      assertContains(result.stdout, 'Persistent memory:', 'Persistent memory info');
    }

    // Check for performance metrics
    assertContains(result.stdout, 'Performance Metrics:', 'Performance metrics section');
    assertContains(result.stdout, 'Total Training Time:', 'Training time display');
    assertContains(result.stdout, 'Average Accuracy:', 'Average accuracy display');
    assertContains(result.stdout, 'Best Model:', 'Best model display');
  } else {
    testResults.failed.push('Neural status command failed');
  }
}

// Additional test: Pattern switching memory efficiency
async function testPatternSwitching() {
  console.log(`\n${colors.bold}${colors.blue}Additional Test: Pattern Switching Memory Efficiency${colors.reset}`);
  console.log('─'.repeat(50));

  const patterns = ['convergent', 'divergent', 'lateral'];
  const memorySamples = [];

  console.log(`\n${colors.yellow}Testing rapid pattern switching...${colors.reset}`);

  // Switch between patterns multiple times
  for (let i = 0; i < 3; i++) {
    for (const pattern of patterns) {
      const result = await runCommand(`npx ruv-swarm neural patterns --pattern ${pattern}`);
      if (result.success) {
        const memoryMatch = result.stdout.match(/Memory Usage:\s*(\d+)\s*MB/);
        if (memoryMatch) {
          memorySamples.push({
            iteration: i,
            pattern,
            memory: parseInt(memoryMatch[1], 10),
          });
        }
      }
    }
  }

  // Analyze memory stability
  if (memorySamples.length > 0) {
    const patternMemoryAvg = {};

    for (const pattern of patterns) {
      const samples = memorySamples.filter(s => s.pattern === pattern);
      if (samples.length > 0) {
        const avg = samples.reduce((sum, s) => sum + s.memory, 0) / samples.length;
        const variance = Math.max(...samples.map(s => s.memory)) - Math.min(...samples.map(s => s.memory));
        patternMemoryAvg[pattern] = { avg, variance };

        if (variance < 50) {
          testResults.passed.push(`${pattern}: Stable memory across switches (variance: ${variance} MB)`);
        } else {
          testResults.warnings.push(`${pattern}: Higher memory variance (${variance} MB)`);
        }
      }
    }
  }
}

// Main test runner
async function runAllTests() {
  console.log(`${colors.bold}${colors.green}🧪 Neural Pattern Fixes Validation Test Suite${colors.reset}`);
  console.log(`${'='.repeat(60)}`);
  console.log(`Started: ${new Date().toLocaleString()}`);

  try {
    // Run all tests
    await testPatternParsing();
    await testMemoryOptimization();
    await testPersistenceIndicators();
    await testPatternSwitching();

    // Display results
    console.log(`\n${colors.bold}${colors.blue}📊 Test Results Summary${colors.reset}`);
    console.log('─'.repeat(50));

    console.log(`\n${colors.green}✅ Passed Tests (${testResults.passed.length}):${colors.reset}`);
    testResults.passed.forEach(test => console.log(`   ✓ ${test}`));

    if (testResults.warnings.length > 0) {
      console.log(`\n${colors.yellow}⚠️  Warnings (${testResults.warnings.length}):${colors.reset}`);
      testResults.warnings.forEach(test => console.log(`   ⚠ ${test}`));
    }

    if (testResults.failed.length > 0) {
      console.log(`\n${colors.red}❌ Failed Tests (${testResults.failed.length}):${colors.reset}`);
      testResults.failed.forEach(test => console.log(`   ✗ ${test}`));
    }

    // Overall status
    const totalTests = testResults.passed.length + testResults.failed.length;
    const passRate = totalTests > 0 ? (testResults.passed.length / totalTests * 100).toFixed(1) : 0;

    console.log(`\n${colors.bold}Overall Status:${colors.reset}`);
    console.log(`Total Tests: ${totalTests}`);
    console.log(`Pass Rate: ${passRate}%`);

    if (testResults.failed.length === 0) {
      console.log(`\n${colors.green}${colors.bold}🎉 All tests passed! All fixes are working correctly.${colors.reset}`);
      process.exit(0);
    } else {
      console.log(`\n${colors.red}${colors.bold}❌ Some tests failed. Please review the issues above.${colors.reset}`);
      process.exit(1);
    }

  } catch (error) {
    console.error(`\n${colors.red}${colors.bold}Fatal Error: ${error.message}${colors.reset}`);
    process.exit(1);
  }
}

// Run the tests
runAllTests();