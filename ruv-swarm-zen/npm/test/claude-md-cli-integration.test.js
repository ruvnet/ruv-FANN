/**
 * Integration tests for CLAUDE.md protection in CLI
 * Tests the actual CLI behavior with different flags
 */

import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('CLAUDE.md CLI Protection Integration', () => {
  let testDir;
  let binPath;

  beforeEach(async() => {
    // Create temporary test directory
    testDir = path.join(__dirname, 'temp', `cli-test-${Date.now()}`);
    await fs.mkdir(testDir, { recursive: true });

    // Path to the CLI script
    binPath = path.join(__dirname, '..', 'bin', 'ruv-swarm-clean.js');
  });

  afterEach(async() => {
    // Clean up test directory
    try {
      await fs.rm(testDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }
  });

  /**
     * Helper function to run CLI command
     */
  function runCLI(args, options = {}) {
    return new Promise((resolve, reject) => {
      const child = spawn('node', [binPath, ...args], {
        cwd: testDir,
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: 10000, // 10 second timeout
        ...options,
      });

      let stdout = '';
      let stderr = '';

      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      child.on('close', (code) => {
        resolve({
          code,
          stdout,
          stderr,
          success: code === 0,
        });
      });

      child.on('error', (error) => {
        reject(error);
      });

      // For interactive tests, we might need to send input
      if (options.input) {
        child.stdin.write(options.input);
        child.stdin.end();
      }
    });
  }

  describe('Basic Protection Behavior', () => {
    test('should create CLAUDE.md when it does not exist', async() => {
      const result = await runCLI(['init', 'mesh', '5', '--claude', '--no-interactive']);

      expect(result.success).toBe(true);
      expect(result.stdout).toContain('Swarm initialized');
      expect(result.stdout).toContain('Documentation generated successfully');

      // Check file was created
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const exists = await fs.access(claudePath).then(() => true).catch(() => false);
      expect(exists).toBe(true);

      const content = await fs.readFile(claudePath, 'utf8');
      expect(content).toContain('Claude Code Configuration for ruv-swarm');
    });

    test('should fail when CLAUDE.md exists without force or merge', async() => {
      // Create existing CLAUDE.md
      const claudePath = path.join(testDir, 'CLAUDE.md');
      await fs.writeFile(claudePath, 'existing content');

      const result = await runCLI(['init', 'mesh', '5', '--claude', '--no-interactive']);

      expect(result.success).toBe(false);
      expect(result.stderr).toContain('already exists');
      expect(result.stderr).toContain('Use --force to overwrite or --merge to combine');
    });

    test('should overwrite with --force flag', async() => {
      // Create existing CLAUDE.md
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const originalContent = 'original content that should be replaced';
      await fs.writeFile(claudePath, originalContent);

      const result = await runCLI(['init', 'mesh', '5', '--claude', '--force', '--no-interactive']);

      expect(result.success).toBe(true);
      expect(result.stdout).toContain('Backing up existing CLAUDE.md');
      expect(result.stdout).toContain('Files regenerated with --force flag');

      // Check original file was overwritten
      const newContent = await fs.readFile(claudePath, 'utf8');
      expect(newContent).toContain('Claude Code Configuration for ruv-swarm');
      expect(newContent).not.toContain('original content that should be replaced');

      // Check backup was created
      const files = await fs.readdir(testDir);
      const backupFiles = files.filter(f => f.startsWith('CLAUDE.md.backup.'));
      expect(backupFiles.length).toBeGreaterThan(0);

      const backupContent = await fs.readFile(path.join(testDir, backupFiles[0]), 'utf8');
      expect(backupContent).toBe(originalContent);
    });

    test('should merge with --merge flag', async() => {
      // Create existing CLAUDE.md
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const originalContent = `# My Project Configuration

This is important project information that should be preserved.

## Setup Instructions
1. Install dependencies
2. Configure environment
3. Run tests`;

      await fs.writeFile(claudePath, originalContent);

      const result = await runCLI(['init', 'mesh', '5', '--claude', '--merge', '--no-interactive']);

      expect(result.success).toBe(true);
      expect(result.stdout).toContain('Merging ruv-swarm configuration');
      expect(result.stdout).toContain('Configuration merged with existing files');

      // Check merged content
      const mergedContent = await fs.readFile(claudePath, 'utf8');
      expect(mergedContent).toContain('My Project Configuration');
      expect(mergedContent).toContain('important project information');
      expect(mergedContent).toContain('Setup Instructions');
      expect(mergedContent).toContain('Claude Code Configuration for ruv-swarm');

      // Check backup was created
      const files = await fs.readdir(testDir);
      const backupFiles = files.filter(f => f.startsWith('CLAUDE.md.backup.'));
      expect(backupFiles.length).toBeGreaterThan(0);
    });
  });

  describe('Help Documentation', () => {
    test('should show updated help with new options', async() => {
      const result = await runCLI(['help']);

      expect(result.success).toBe(true);
      expect(result.stdout).toContain('--force');
      expect(result.stdout).toContain('--merge');
      expect(result.stdout).toContain('--no-interactive');
      expect(result.stdout).toContain('Overwrite existing CLAUDE.md (creates backup)');
      expect(result.stdout).toContain('Merge with existing CLAUDE.md content');
      expect(result.stdout).toContain('Skip interactive prompts (fail on conflicts)');
    });

    test('should show examples with different flags', async() => {
      const result = await runCLI(['help']);

      expect(result.success).toBe(true);
      expect(result.stdout).toContain('ruv-swarm init mesh 5 --claude');
      expect(result.stdout).toContain('ruv-swarm init mesh 5 --claude --force');
      expect(result.stdout).toContain('ruv-swarm init mesh 5 --claude --merge');
      expect(result.stdout).toContain('ruv-swarm init mesh 5 --claude --no-interactive');
    });
  });

  describe('Error Handling', () => {
    test('should handle permission errors gracefully', async() => {
      // Create directory without write permissions
      const readOnlyDir = path.join(testDir, 'readonly');
      await fs.mkdir(readOnlyDir);
      await fs.chmod(readOnlyDir, 0o444); // Read-only

      try {
        const result = await runCLI(['init', 'mesh', '5', '--claude', '--no-interactive'], {
          cwd: readOnlyDir,
        });

        expect(result.success).toBe(false);
        expect(result.stderr).toContain('Error');
      } finally {
        // Restore permissions for cleanup
        await fs.chmod(readOnlyDir, 0o755);
      }
    });

    test('should handle invalid command combinations', async() => {
      const result = await runCLI(['init', 'mesh', '5', '--claude', '--force', '--merge', '--no-interactive']);

      // Should prioritize --force over --merge
      expect(result.success).toBe(true);
      expect(result.stdout).toContain('Files regenerated with --force flag');
    });
  });

  describe('Backup System Integration', () => {
    test('should create multiple backups and clean up old ones', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');

      // Create initial file
      await fs.writeFile(claudePath, 'version 1');

      // Run init with force multiple times
      for (let i = 2; i <= 8; i++) {
        await fs.writeFile(claudePath, `version ${i}`);
        await runCLI(['init', 'mesh', '5', '--claude', '--force', '--no-interactive']);

        // Add small delay to ensure different timestamps
        await new Promise(resolve => setTimeout(resolve, 10));
      }

      // Check that only 5 backups remain
      const files = await fs.readdir(testDir);
      const backupFiles = files.filter(f => f.startsWith('CLAUDE.md.backup.'));
      expect(backupFiles.length).toBeLessThanOrEqual(5);
    });
  });

  describe('Real-world Scenarios', () => {
    test('should handle Lion system integration project', async() => {
      // Simulate existing Lion system CLAUDE.md
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const lionContent = `# Claude Code Configuration

## lion (Language Interoperable Network)

### Central Orchestrator: lion
The lion agent serves as the central orchestrator and decision-making hub.

### Multi-Hat Hierarchy
- lion: Central orchestrator and decision maker
- Task Agents: Specialized agents for specific domains

## Configuration
This project uses lion for multi-agent coordination.`;

      await fs.writeFile(claudePath, lionContent);

      const result = await runCLI(['init', 'mesh', '5', '--claude', '--merge', '--no-interactive']);

      expect(result.success).toBe(true);

      const mergedContent = await fs.readFile(claudePath, 'utf8');

      // Should preserve Lion content
      expect(mergedContent).toContain('lion (Language Interoperable Network)');
      expect(mergedContent).toContain('Central Orchestrator: lion');
      expect(mergedContent).toContain('Multi-Hat Hierarchy');

      // Should add ruv-swarm content
      expect(mergedContent).toContain('Claude Code Configuration for ruv-swarm');
      expect(mergedContent).toContain('BATCH EVERYTHING');
      expect(mergedContent).toContain('ruv-swarm coordinates, Claude Code creates');
    });

    test('should handle complex existing project structure', async() => {
      // Create complex project structure
      await fs.mkdir(path.join(testDir, '.claude'), { recursive: true });
      await fs.mkdir(path.join(testDir, '.claude', 'commands'), { recursive: true });

      const claudePath = path.join(testDir, 'CLAUDE.md');
      const complexContent = `# My Complex Project

## Architecture Overview
This is a complex multi-service architecture.

### Services
- API Gateway
- User Service  
- Payment Service
- Notification Service

## Development Workflow
1. Feature branches
2. Code review
3. Automated testing
4. Deployment

## Important Notes
⚠️ Never deploy to production on Fridays!

## Legacy Configuration
Some old configuration that might conflict.`;

      await fs.writeFile(claudePath, complexContent);

      const result = await runCLI(['init', 'mesh', '5', '--claude', '--merge', '--no-interactive']);

      expect(result.success).toBe(true);

      const mergedContent = await fs.readFile(claudePath, 'utf8');

      // Should preserve all original content
      expect(mergedContent).toContain('My Complex Project');
      expect(mergedContent).toContain('Architecture Overview');
      expect(mergedContent).toContain('multi-service architecture');
      expect(mergedContent).toContain('User Service');
      expect(mergedContent).toContain('Payment Service');
      expect(mergedContent).toContain('Development Workflow');
      expect(mergedContent).toContain('Never deploy to production on Fridays!');

      // Should add ruv-swarm without disrupting structure
      expect(mergedContent).toContain('Claude Code Configuration for ruv-swarm');

      // Check that content is properly separated
      const sections = mergedContent.split('\n---\n');
      expect(sections.length).toBeGreaterThan(1);
    });
  });
});
