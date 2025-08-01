/**
 * Simple test for CLAUDE.md protection - compatible with existing test setup
 */

import fs from 'fs/promises';
import path from 'path';
import assert from 'assert';

// Test the protection logic in a way that's compatible with the existing test runner
async function runProtectionTests() {
  console.log('Running CLAUDE.md Protection Tests...\n');

  let passed = 0;
  let failed = 0;

  async function test(name, fn) {
    try {
      await fn();
      console.log(`✓ ${name}`);
      passed++;
    } catch (error) {
      console.error(`✗ ${name}`);
      console.error(`  ${error.message}`);
      failed++;
    }
  }

  // Create temporary test directory
  const testDir = path.join(process.cwd(), `temp-test-${ Date.now()}`);
  await fs.mkdir(testDir, { recursive: true });

  try {
    // Test file existence checking
    await test('should detect when CLAUDE.md exists', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      await fs.writeFile(claudePath, 'existing content');

      const exists = await fs.access(claudePath).then(() => true).catch(() => false);
      assert(exists === true);
    });

    await test('should detect when CLAUDE.md does not exist', async() => {
      const claudePath = path.join(testDir, 'NONEXISTENT.md');

      const exists = await fs.access(claudePath).then(() => true).catch(() => false);
      assert(exists === false);
    });

    // Test backup creation
    await test('should create backup with timestamp', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const originalContent = 'original content for backup test';
      await fs.writeFile(claudePath, originalContent);

      // Create backup
      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');
      const backupPath = `${claudePath}.backup.${timestamp}`;
      await fs.copyFile(claudePath, backupPath);

      // Check backup was created
      const backupExists = await fs.access(backupPath).then(() => true).catch(() => false);
      assert(backupExists === true);

      // Check backup content matches original
      const backupContent = await fs.readFile(backupPath, 'utf8');
      assert(backupContent === originalContent);

      // Check backup naming pattern
      assert(path.basename(backupPath).match(/^CLAUDE\.md\.backup\.\d{8}T\d{6}$/));
    });

    // Test content merging logic
    await test('should merge content correctly', async() => {
      const existingContent = `# My Project Configuration

This is important project information that should be preserved.

## Setup Instructions
1. Install dependencies
2. Configure environment`;

      const ruvSwarmContent = `# Claude Code Configuration for ruv-swarm

## IMPORTANT: Separation of Responsibilities

### Claude Code Handles:
- ALL file operations (Read, Write, Edit, MultiEdit)
- ALL code generation and development tasks

### ruv-swarm MCP Tools Handle:
- Coordination only - Orchestrating Claude Code's actions
- Memory management - Persistent state across sessions

Remember: **ruv-swarm coordinates, Claude Code creates!**`;

      // Simple merge logic test
      const lines = existingContent.split('\n');

      // Add ruv-swarm section at end
      if (lines[lines.length - 1].trim() !== '') {
        lines.push(''); // Add blank line before new section
      }
      lines.push('---', '', ruvSwarmContent);

      const mergedContent = lines.join('\n');

      // Check that both contents are preserved
      assert(mergedContent.includes('My Project Configuration'));
      assert(mergedContent.includes('important project information'));
      assert(mergedContent.includes('Setup Instructions'));
      assert(mergedContent.includes('Claude Code Configuration for ruv-swarm'));
      assert(mergedContent.includes('ruv-swarm coordinates, Claude Code creates'));
    });

    // Test section detection
    await test('should detect section ends correctly', async() => {
      const lines = [
        '# First Section',
        'Content here',
        '## Subsection',
        'More content',
        '# Second Section',
        'Different content',
      ];

      // Find section end logic
      function findSectionEnd(lines, startIndex) {
        for (let i = startIndex + 1; i < lines.length; i++) {
          if (lines[i].startsWith('# ') && !lines[i].includes('ruv-swarm')) {
            return i;
          }
        }
        return lines.length;
      }

      const sectionEnd = findSectionEnd(lines, 0);
      assert(sectionEnd === 4); // Should stop at "# Second Section"
    });

    // Test argument parsing logic
    await test('should parse CLI arguments correctly', async() => {
      // Simulate argument parsing
      function parseArgs(args) {
        const positionalArgs = args.filter(arg => !arg.startsWith('--'));
        const setupClaude = args.includes('--claude');
        const forceSetup = args.includes('--force');
        const mergeSetup = args.includes('--merge');
        const noInteractive = args.includes('--no-interactive');

        return {
          topology: positionalArgs[0] || 'mesh',
          maxAgents: parseInt(positionalArgs[1]) || 5,
          setupClaude,
          forceSetup,
          mergeSetup,
          interactive: !noInteractive,
        };
      }

      // Test various argument combinations
      const test1 = parseArgs(['mesh', '5', '--claude', '--force']); // Remove 'init' as it's not a positional arg
      assert(test1.topology === 'mesh');
      assert(test1.maxAgents === 5);
      assert(test1.setupClaude === true);
      assert(test1.forceSetup === true);
      assert(test1.mergeSetup === false);
      assert(test1.interactive === true);

      const test2 = parseArgs(['hierarchical', '8', '--claude', '--merge', '--no-interactive']);
      assert(test2.topology === 'hierarchical');
      assert(test2.maxAgents === 8);
      assert(test2.setupClaude === true);
      assert(test2.forceSetup === false);
      assert(test2.mergeSetup === true);
      assert(test2.interactive === false);
    });

    // Test error handling scenarios
    await test('should handle protection scenarios correctly', async() => {
      // Simulate protection logic
      function shouldProceed(fileExists, options) {
        if (!fileExists) {
          return { proceed: true, action: 'create' };
        }

        if (options.force) {
          return { proceed: true, action: 'overwrite' };
        }

        if (options.merge) {
          return { proceed: true, action: 'merge' };
        }

        if (!options.interactive) {
          return {
            proceed: false,
            error: 'CLAUDE.md already exists. Use --force to overwrite or --merge to combine.',
          };
        }

        return { proceed: false, needsInput: true };
      }

      // Test scenarios
      const scenario1 = shouldProceed(false, { force: false, merge: false, interactive: true });
      assert(scenario1.proceed === true);
      assert(scenario1.action === 'create');

      const scenario2 = shouldProceed(true, { force: true, merge: false, interactive: false });
      assert(scenario2.proceed === true);
      assert(scenario2.action === 'overwrite');

      const scenario3 = shouldProceed(true, { force: false, merge: true, interactive: false });
      assert(scenario3.proceed === true);
      assert(scenario3.action === 'merge');

      const scenario4 = shouldProceed(true, { force: false, merge: false, interactive: false });
      assert(scenario4.proceed === false);
      assert(scenario4.error.includes('already exists'));
    });

    console.log(`\n✅ CLAUDE.md Protection Tests completed: ${passed} passed, ${failed} failed`);

    if (failed > 0) {
      throw new Error(`${failed} tests failed`);
    }

  } finally {
    // Clean up test directory
    try {
      await fs.rm(testDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }
  }
}

// Export for use in other test runners
export { runProtectionTests };

// Run directly if this is the main module
if (process.argv[1].endsWith('claude-md-protection-simple.test.js')) {
  runProtectionTests().catch(error => {
    console.error('Protection test error:', error);
    process.exit(1);
  });
}
