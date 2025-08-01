#!/usr/bin/env node

/**
 * Test script to validate MCP server startup fix
 * Tests Issue #155: MCP server startup binary selection
 */

import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';

const TEST_TIMEOUT = 10000; // 10 seconds

async function testMcpServerStartup() {
  console.log('🧪 Testing MCP Server Startup Fix (Issue #155)');
  console.log('='.repeat(50));

  try {
    // Test 1: Verify npm script runs without binary ambiguity
    console.log('📋 Test 1: Binary Selection Fix');

    const result = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        // Kill the process after timeout - this is expected for stdio server
        child.kill('SIGTERM');
        resolve({ success: hasStarted, stdout, stderr, timeout: true });
      }, TEST_TIMEOUT);

      const child = spawn('npm', ['run', 'mcp:server'], {
        cwd: process.cwd(),
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';
      let hasStarted = false;

      child.stdout.on('data', (data) => {
        stdout += data.toString();
        // Check for successful startup message
        if (data.toString().includes('Starting ruv-swarm-mcp stdio server')) {
          hasStarted = true;
        }
      });

      child.stderr.on('data', (data) => {
        stderr += data.toString();
        // Check for binary ambiguity error (should not occur)
        if (data.toString().includes('could not determine which binary to run')) {
          clearTimeout(timeout);
          child.kill('SIGTERM');
          resolve({ success: false, stdout, stderr, error: 'binary_ambiguity' });
        }
      });

      child.on('close', (code) => {
        clearTimeout(timeout);
        resolve({ success: hasStarted, stdout, stderr, code, hasStarted });
      });

      child.on('error', (error) => {
        clearTimeout(timeout);
        reject(error);
      });
    });

    // Analyze results
    if (result.error === 'binary_ambiguity') {
      console.log('❌ FAILED: Binary ambiguity error still occurs');
      console.log('   The fix was not successful');
      return false;
    }

    if (result.hasStarted) {
      console.log('✅ PASSED: MCP server started successfully');
      console.log('   Binary selection fix is working');
      if (result.timeout) {
        console.log('   Server running continuously (terminated by timeout - expected)');
      }
    } else {
      console.log('❌ FAILED: Server did not start or startup message not detected');
      console.log('   stdout:', result.stdout.substring(0, 200));
      console.log('   stderr:', result.stderr.substring(0, 200));
      return false;
    }

    // Test 2: Verify the fix in package.json
    console.log('\n📋 Test 2: Package.json Configuration');

    const packagePath = path.join(process.cwd(), 'package.json');
    const packageContent = await fs.readFile(packagePath, 'utf8');
    const packageJson = JSON.parse(packageContent);

    const mcpServerScript = packageJson.scripts['mcp:server'];
    const expectedScript = 'cd ../crates/ruv-swarm-mcp && cargo run --bin ruv-swarm-mcp-stdio';

    if (mcpServerScript === expectedScript) {
      console.log('✅ PASSED: npm script correctly specifies binary');
      console.log(`   Script: ${mcpServerScript}`);
    } else {
      console.log('❌ FAILED: npm script does not match expected fix');
      console.log(`   Expected: ${expectedScript}`);
      console.log(`   Actual:   ${mcpServerScript}`);
      return false;
    }

    // Test 3: Verify development script consistency
    console.log('\n📋 Test 3: Development Script Consistency');

    const mcpServerDevScript = packageJson.scripts['mcp:server:dev'];
    const expectedDevScript = 'cd ../crates/ruv-swarm-mcp && cargo watch -x \'run --bin ruv-swarm-mcp-stdio\'';

    if (mcpServerDevScript === expectedDevScript) {
      console.log('✅ PASSED: Development script is consistent');
      console.log(`   Script: ${mcpServerDevScript}`);
    } else {
      console.log('⚠️  WARNING: Development script may need updating');
      console.log(`   Expected: ${expectedDevScript}`);
      console.log(`   Actual:   ${mcpServerDevScript}`);
    }

    console.log('\n' + '='.repeat(50));
    console.log('🎉 MCP Server Startup Fix Validation Complete');
    console.log('✅ Issue #155 has been resolved');
    console.log('\nFix Summary:');
    console.log('- Updated npm script to specify --bin ruv-swarm-mcp-stdio');
    console.log('- Binary ambiguity error eliminated');
    console.log('- MCP server starts successfully');

    return true;

  } catch (error) {
    console.error('❌ Test failed with error:', error.message);
    return false;
  }
}

// Run the test if this script is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  testMcpServerStartup()
    .then(success => {
      process.exit(success ? 0 : 1);
    })
    .catch(error => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}

export { testMcpServerStartup };
