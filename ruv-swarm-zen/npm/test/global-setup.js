/**
 * Global test setup
 * Initializes test environment and performs pre-test checks
 */

import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default async function globalSetup() {
  console.log('🔧 Setting up global test environment...\n');

  // Create test directories
  const testDirs = [
    'test-outputs',
    'test-reports',
    'coverage',
    'test-data',
  ];

  for (const dir of testDirs) {
    const dirPath = path.join(__dirname, '..', dir);
    await fs.mkdir(dirPath, { recursive: true });
  }

  // Check WASM files exist
  console.log('🔍 Checking WASM files...');
  const wasmFiles = [
    '../wasm/ruv_swarm_wasm_bg.wasm',
    '../wasm/ruv_swarm_wasm.js',
  ];

  for (const file of wasmFiles) {
    const filePath = path.join(__dirname, file);
    try {
      await fs.access(filePath);
      console.log(`  ✅ ${path.basename(filePath)}`);
    } catch (error) {
      console.error(`  ❌ Missing: ${path.basename(filePath)}`);
      console.error('     Run npm run build:wasm to generate WASM files');
      process.exit(1);
    }
  }

  // Check for required dependencies
  console.log('\n🔍 Checking dependencies...');
  const requiredDeps = [
    'vitest',
    'playwright',
    'better-sqlite3',
    'ws',
  ];

  const packageJson = JSON.parse(
    await fs.readFile(path.join(__dirname, '../package.json'), 'utf-8'),
  );

  const allDeps = {
    ...packageJson.dependencies,
    ...packageJson.devDependencies,
  };

  for (const dep of requiredDeps) {
    if (allDeps[dep]) {
      console.log(`  ✅ ${dep}`);
    } else {
      console.error(`  ❌ Missing: ${dep}`);
    }
  }

  // Start MCP server if needed
  console.log('\n🚀 Starting MCP server...');
  const mcpProcess = spawn('npm', ['run', 'mcp:server'], {
    cwd: path.join(__dirname, '..'),
    detached: true,
    stdio: 'ignore',
  });

  mcpProcess.unref();

  // Store process ID for cleanup
  await fs.writeFile(
    path.join(__dirname, '.mcp-server.pid'),
    mcpProcess.pid.toString(),
  );

  // Wait for MCP server to start
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Initialize test database
  console.log('\n💾 Initializing test database...');
  const dbPath = path.join(__dirname, '../test-data/test.db');

  // Clean up existing test database
  try {
    await fs.unlink(dbPath);
  } catch (error) {
    // Ignore if doesn't exist
  }

  // Create coverage tracking
  console.log('\n📊 Setting up coverage tracking...');
  const coverageData = {
    startTime: Date.now(),
    testRun: new Date().toISOString(),
    environment: {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
    },
  };

  await fs.writeFile(
    path.join(__dirname, '../coverage/coverage-run.json'),
    JSON.stringify(coverageData, null, 2),
  );

  console.log('\n✅ Global setup complete!\n');

  // Return cleanup function
  return async() => {
    console.log('\n🧹 Cleaning up test environment...');

    // Stop MCP server
    try {
      const pid = await fs.readFile(
        path.join(__dirname, '.mcp-server.pid'),
        'utf-8',
      );
      process.kill(parseInt(pid), 'SIGTERM');
      await fs.unlink(path.join(__dirname, '.mcp-server.pid'));
    } catch (error) {
      // Ignore if already stopped
    }

    // Clean up test data
    const testDataDir = path.join(__dirname, '../test-data');
    await fs.rm(testDataDir, { recursive: true, force: true });

    console.log('✅ Cleanup complete!\n');
  };
}
