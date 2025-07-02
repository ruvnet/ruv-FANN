#!/usr/bin/env node

/**
 * Test Setup Validation Script
 * Ensures all test dependencies and files are properly configured
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('🔍 Validating Test Setup...\n');

const checks = [];
let passed = 0;
let failed = 0;

function check(name, condition, error = null) {
  if (condition) {
    console.log(`✅ ${name}`);
    passed++;
  } else {
    console.log(`❌ ${name}`);
    if (error) {
      console.log(`   ${error}`);
    }
    failed++;
  }
  checks.push({ name, passed: condition, error });
}

// Check test files exist
console.log('📁 Checking test files:');
const testFiles = [
  'mcp-integration.test.js',
  'persistence.test.js',
  'neural-integration.test.js',
  'run-all-tests.js',
  'test.js',
  'README.md',
];

testFiles.forEach(file => {
  const filePath = path.join(__dirname, file);
  check(`  ${file}`, fs.existsSync(filePath), `File not found: ${filePath}`);
});

// Check example files
console.log('\n📁 Checking example files:');
const examplePath = path.join(__dirname, '..', 'examples', 'mcp-workflows.js');
check('  mcp-workflows.js', fs.existsSync(examplePath), `File not found: ${examplePath}`);

// Check dependencies
console.log('\n📦 Checking dependencies:');
try {
  const packageJson = require('../package.json');
  const requiredDeps = ['ws', 'uuid', 'better-sqlite3'];

  requiredDeps.forEach(dep => {
    check(`  ${dep}`,
      packageJson.dependencies[dep] || packageJson.devDependencies[dep],
      'Missing dependency in package.json');
  });
} catch (error) {
  check('  package.json', false, error.message);
}

// Check Node.js version
console.log('\n🔧 Checking environment:');
const nodeVersion = process.version;
const majorVersion = parseInt(nodeVersion.split('.')[0].substring(1), 10);
check(`  Node.js version (${nodeVersion})`, majorVersion >= 14, 'Node.js 14+ required');

// Check if we can import required modules
console.log('\n📚 Checking module imports:');
const modules = ['ws', 'uuid', 'sqlite3'];
modules.forEach(mod => {
  try {
    require(mod);
    check(`  ${mod}`, true);
  } catch (error) {
    check(`  ${mod}`, false, 'Module not installed. Run: npm install');
  }
});

// Check MCP server availability
console.log('\n🌐 Checking MCP server:');
const mcpServerPath = path.join(__dirname, '..', '..', 'crates', 'ruv-swarm-mcp', 'Cargo.toml');
check('  MCP server crate', fs.existsSync(mcpServerPath), 'MCP server crate not found');

// Summary
console.log(`\n${ '='.repeat(50)}`);
console.log('📊 Validation Summary:');
console.log(`  ✅ Passed: ${passed}`);
console.log(`  ❌ Failed: ${failed}`);
console.log('='.repeat(50));

if (failed > 0) {
  console.log('\n⚠️  Some checks failed. Please address the issues above.');
  console.log('💡 Try running: npm install');
  process.exit(1);
} else {
  console.log('\n✅ All checks passed! Test environment is ready.');
  console.log('\n📚 Next steps:');
  console.log('  1. Start MCP server: npm run mcp:server');
  console.log('  2. Run all tests: npm run test:all');
  console.log('  3. Run examples: npm run examples');
  process.exit(0);
}