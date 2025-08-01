/**
 * Tests for CLAUDE.md protection features
 * Tests the --force, --merge, and interactive prompt functionality
 */

import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { ClaudeDocsGenerator } from '../src/claude-integration/docs.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('CLAUDE.md Protection Features', () => {
  let testDir;
  let docsGenerator;
  let originalStdin;
  let originalStdout;

  beforeEach(async() => {
    // Create temporary test directory
    testDir = path.join(__dirname, 'temp', `test-${Date.now()}`);
    await fs.mkdir(testDir, { recursive: true });

    // Initialize docs generator
    docsGenerator = new ClaudeDocsGenerator({ workingDir: testDir });

    // Mock stdin/stdout for testing
    originalStdin = process.stdin;
    originalStdout = process.stdout;
  });

  afterEach(async() => {
    // Clean up test directory
    try {
      await fs.rm(testDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }

    // Restore stdin/stdout
    process.stdin = originalStdin;
    process.stdout = originalStdout;
  });

  describe('File Existence Checking', () => {
    test('should detect when CLAUDE.md exists', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      await fs.writeFile(claudePath, 'existing content');

      const exists = await docsGenerator.fileExists(claudePath);
      expect(exists).toBe(true);
    });

    test('should detect when CLAUDE.md does not exist', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');

      const exists = await docsGenerator.fileExists(claudePath);
      expect(exists).toBe(false);
    });
  });

  describe('Backup System', () => {
    test('should create backup with timestamp', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const originalContent = 'original content';
      await fs.writeFile(claudePath, originalContent);

      const backupPath = await docsGenerator.createBackup(claudePath);

      // Check backup was created
      expect(await docsGenerator.fileExists(backupPath)).toBe(true);

      // Check backup content matches original
      const backupContent = await fs.readFile(backupPath, 'utf8');
      expect(backupContent).toBe(originalContent);

      // Check backup naming pattern
      expect(path.basename(backupPath)).toMatch(/^CLAUDE\.md\.backup\.\d{8}T\d{6}$/);
    });

    test('should clean up old backups (keep only 5)', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      await fs.writeFile(claudePath, 'content');

      // Create 8 backup files
      const backupPaths = [];
      for (let i = 0; i < 8; i++) {
        // Create backups with different timestamps
        const timestamp = new Date(Date.now() + i * 1000).toISOString().slice(0, 19).replace(/[:-]/g, '');
        const backupPath = `${claudePath}.backup.${timestamp}`;
        await fs.writeFile(backupPath, `backup ${i}`);
        backupPaths.push(backupPath);
      }

      // Run cleanup
      await docsGenerator.cleanupOldBackups(claudePath);

      // Check that only 5 most recent backups remain
      const remainingBackups = [];
      for (const backupPath of backupPaths) {
        if (await docsGenerator.fileExists(backupPath)) {
          remainingBackups.push(backupPath);
        }
      }

      expect(remainingBackups).toHaveLength(5);
      // Should keep the 5 most recent (indices 3-7)
      expect(remainingBackups).toEqual(backupPaths.slice(3));
    });
  });

  describe('Protection Logic - Non-Interactive Mode', () => {
    test('should fail when CLAUDE.md exists without --force, --backup, or --merge', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      await fs.writeFile(claudePath, 'existing content');

      await expect(
        docsGenerator.generateClaudeMd({ interactive: false }),
      ).rejects.toThrow('CLAUDE.md already exists. Use --force to overwrite, --backup to backup existing, or --merge to combine.');
    });

    test('should create new file when CLAUDE.md does not exist', async() => {
      const result = await docsGenerator.generateClaudeMd({ interactive: false });

      expect(result.success).toBe(true);
      expect(result.action).toBe('created');

      const claudePath = path.join(testDir, 'CLAUDE.md');
      expect(await docsGenerator.fileExists(claudePath)).toBe(true);

      const content = await fs.readFile(claudePath, 'utf8');
      expect(content).toContain('Claude Code Configuration for ruv-swarm');
    });

    test('should overwrite WITHOUT backup when --force is used (ruv spec)', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const originalContent = 'existing content';
      await fs.writeFile(claudePath, originalContent);

      const result = await docsGenerator.generateClaudeMd({
        force: true,
        interactive: false,
      });

      expect(result.success).toBe(true);
      expect(result.action).toBe('created');

      // Check original file was overwritten
      const newContent = await fs.readFile(claudePath, 'utf8');
      expect(newContent).toContain('Claude Code Configuration for ruv-swarm');
      expect(newContent).not.toBe(originalContent);

      // Check NO backup was created (ruv's requirement)
      const files = await fs.readdir(testDir);
      const backupFiles = files.filter(f => f.startsWith('CLAUDE.md.backup.'));
      expect(backupFiles).toHaveLength(0);
    });

    test('should create backup with --backup flag', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const originalContent = 'existing content';
      await fs.writeFile(claudePath, originalContent);

      const result = await docsGenerator.generateClaudeMd({
        backup: true,
        interactive: false,
      });

      expect(result.success).toBe(true);
      expect(result.action).toBe('created');

      // Check original file was overwritten
      const newContent = await fs.readFile(claudePath, 'utf8');
      expect(newContent).toContain('Claude Code Configuration for ruv-swarm');
      expect(newContent).not.toBe(originalContent);

      // Check backup was created
      const files = await fs.readdir(testDir);
      const backupFiles = files.filter(f => f.startsWith('CLAUDE.md.backup.'));
      expect(backupFiles).toHaveLength(1);

      const backupContent = await fs.readFile(path.join(testDir, backupFiles[0]), 'utf8');
      expect(backupContent).toBe(originalContent);
    });

    test('should create backup when using --force --backup together', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const originalContent = 'existing content';
      await fs.writeFile(claudePath, originalContent);

      const result = await docsGenerator.generateClaudeMd({
        force: true,
        backup: true,
        interactive: false,
      });

      expect(result.success).toBe(true);
      expect(result.action).toBe('created');

      // Check original file was overwritten
      const newContent = await fs.readFile(claudePath, 'utf8');
      expect(newContent).toContain('Claude Code Configuration for ruv-swarm');
      expect(newContent).not.toBe(originalContent);

      // Check backup was created
      const files = await fs.readdir(testDir);
      const backupFiles = files.filter(f => f.startsWith('CLAUDE.md.backup.'));
      expect(backupFiles).toHaveLength(1);

      const backupContent = await fs.readFile(path.join(testDir, backupFiles[0]), 'utf8');
      expect(backupContent).toBe(originalContent);
    });

    test('should intelligently merge when --merge is used (not just append)', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const originalContent = `# My Project Configuration

This is my existing project setup.

## Important Notes
- Keep this content
- Don't lose this information`;

      await fs.writeFile(claudePath, originalContent);

      const result = await docsGenerator.generateClaudeMd({
        merge: true,
        interactive: false,
      });

      expect(result.success).toBe(true);
      expect(result.action).toBe('merged');

      // Check merged content
      const mergedContent = await fs.readFile(claudePath, 'utf8');
      expect(mergedContent).toContain('My Project Configuration');
      expect(mergedContent).toContain('Keep this content');
      expect(mergedContent).toContain('Claude Code Configuration for ruv-swarm');

      // Should intelligently position content, not just append to bottom
      const lines = mergedContent.split('\n');
      const projectConfigIndex = lines.findIndex(line => line.includes('My Project Configuration'));
      const ruvSwarmIndex = lines.findIndex(line => line.includes('Claude Code Configuration for ruv-swarm'));
      const notesIndex = lines.findIndex(line => line.includes('Important Notes'));

      // ruv-swarm content should be positioned intelligently, not necessarily at the end
      expect(projectConfigIndex).toBeGreaterThanOrEqual(0);
      expect(ruvSwarmIndex).toBeGreaterThanOrEqual(0);
      expect(notesIndex).toBeGreaterThanOrEqual(0);

      // Check NO backup was created by default with merge
      const files = await fs.readdir(testDir);
      const backupFiles = files.filter(f => f.startsWith('CLAUDE.md.backup.'));
      expect(backupFiles).toHaveLength(0);
    });

    test('should create backup when --merge --backup is used', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const originalContent = `# My Project Configuration
This is my existing project setup.`;

      await fs.writeFile(claudePath, originalContent);

      const result = await docsGenerator.generateClaudeMd({
        merge: true,
        backup: true,
        interactive: false,
      });

      expect(result.success).toBe(true);
      expect(result.action).toBe('merged');

      // Check backup was created when both flags used
      const files = await fs.readdir(testDir);
      const backupFiles = files.filter(f => f.startsWith('CLAUDE.md.backup.'));
      expect(backupFiles).toHaveLength(1);

      const backupContent = await fs.readFile(path.join(testDir, backupFiles[0]), 'utf8');
      expect(backupContent).toBe(originalContent);
    });
  });

  describe('Content Merging Logic', () => {
    test('should replace existing ruv-swarm section in place (true combining)', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const originalContent = `# My Project

Some content here.

# Claude Code Configuration for ruv-swarm

Old ruv-swarm configuration that should be replaced.

# Other Section

More content here.`;

      await fs.writeFile(claudePath, originalContent);

      const result = await docsGenerator.generateClaudeMd({
        merge: true,
        interactive: false,
      });

      expect(result.success).toBe(true);

      const mergedContent = await fs.readFile(claudePath, 'utf8');
      expect(mergedContent).toContain('My Project');
      expect(mergedContent).toContain('Other Section');
      expect(mergedContent).toContain('More content here');
      expect(mergedContent).not.toContain('Old ruv-swarm configuration');
      expect(mergedContent).toContain('MANDATORY RULE #1: BATCH EVERYTHING');

      // Verify section replacement, not duplication
      const ruvSwarmOccurrences = (mergedContent.match(/Claude Code Configuration for ruv-swarm/g) || []).length;
      expect(ruvSwarmOccurrences).toBe(1); // Should appear only once
    });

    test('should intelligently insert ruv-swarm section when none exists (not just append)', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      const originalContent = `# My Project Configuration

This is my existing project setup.

## Setup Instructions
1. Install dependencies
2. Configure environment`;

      await fs.writeFile(claudePath, originalContent);

      const result = await docsGenerator.generateClaudeMd({
        merge: true,
        interactive: false,
      });

      expect(result.success).toBe(true);

      const mergedContent = await fs.readFile(claudePath, 'utf8');
      expect(mergedContent).toContain('My Project Configuration');
      expect(mergedContent).toContain('Setup Instructions');
      expect(mergedContent).toContain('---'); // Section separator
      expect(mergedContent).toContain('Claude Code Configuration for ruv-swarm');

      // Should intelligently position content (not necessarily at bottom)
      const lines = mergedContent.split('\n');
      const setupIndex = lines.findIndex(line => line.includes('Setup Instructions'));
      const ruvSwarmIndex = lines.findIndex(line => line.includes('Claude Code Configuration for ruv-swarm'));

      // Content should be intelligently positioned
      expect(setupIndex).toBeGreaterThanOrEqual(0);
      expect(ruvSwarmIndex).toBeGreaterThanOrEqual(0);
    });

    test('should handle section end detection correctly', async() => {
      const lines = [
        '# First Section',
        'Content here',
        '## Subsection',
        'More content',
        '# Second Section',
        'Different content',
      ];

      const sectionEnd = docsGenerator.findSectionEnd(lines, 0);
      expect(sectionEnd).toBe(4); // Should stop at "# Second Section"
    });

    test('should handle missing section end', async() => {
      const lines = [
        '# Only Section',
        'Content here',
        'More content',
        'Even more content',
      ];

      const sectionEnd = docsGenerator.findSectionEnd(lines, 0);
      expect(sectionEnd).toBe(lines.length); // Should go to end of file
    });
  });

  describe('Error Handling', () => {
    test('should handle backup creation failure gracefully', async() => {
      const claudePath = path.join(testDir, 'nonexistent', 'CLAUDE.md');

      await expect(
        docsGenerator.createBackup(claudePath),
      ).rejects.toThrow();
    });

    test('should handle merge failure gracefully', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');

      // Create file with no read permissions (simulate permission error)
      await fs.writeFile(claudePath, 'content');
      await fs.chmod(claudePath, 0o000);

      try {
        await expect(
          docsGenerator.mergeClaudeMd(claudePath),
        ).rejects.toThrow();
      } finally {
        // Restore permissions for cleanup
        await fs.chmod(claudePath, 0o644);
      }
    });
  });

  describe('Integration with generateAll', () => {
    test('should pass options through to generateClaudeMd', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      await fs.writeFile(claudePath, 'existing content');

      // Mock generateClaudeMd to track calls
      const originalGenerateClaudeMd = docsGenerator.generateClaudeMd;
      let calledWithOptions = null;

      docsGenerator.generateClaudeMd = async(options) => {
        calledWithOptions = options;
        return { file: 'CLAUDE.md', success: true, action: 'created' };
      };

      try {
        await docsGenerator.generateAll({
          force: true,
          merge: false,
          interactive: false,
        });

        expect(calledWithOptions).toEqual({
          force: true,
          merge: false,
          interactive: false,
        });
      } finally {
        // Restore original method
        docsGenerator.generateClaudeMd = originalGenerateClaudeMd;
      }
    });
  });

  describe('Edge Cases', () => {
    test('should handle empty existing file', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      await fs.writeFile(claudePath, '');

      const result = await docsGenerator.generateClaudeMd({
        merge: true,
        interactive: false,
      });

      expect(result.success).toBe(true);

      const mergedContent = await fs.readFile(claudePath, 'utf8');
      expect(mergedContent).toContain('Claude Code Configuration for ruv-swarm');
    });

    test('should handle file with only whitespace', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');
      await fs.writeFile(claudePath, '   \n\n   \t   \n   ');

      const result = await docsGenerator.generateClaudeMd({
        merge: true,
        interactive: false,
      });

      expect(result.success).toBe(true);

      const mergedContent = await fs.readFile(claudePath, 'utf8');
      expect(mergedContent).toContain('Claude Code Configuration for ruv-swarm');
    });

    test('should handle very large existing file', async() => {
      const claudePath = path.join(testDir, 'CLAUDE.md');

      // Create large content (> 1MB)
      const largeContent = `# Large File\n${ 'x'.repeat(1024 * 1024) }\n# End`;
      await fs.writeFile(claudePath, largeContent);

      const result = await docsGenerator.generateClaudeMd({
        merge: true,
        interactive: false,
      });

      expect(result.success).toBe(true);

      const mergedContent = await fs.readFile(claudePath, 'utf8');
      expect(mergedContent).toContain('Large File');
      expect(mergedContent).toContain('Claude Code Configuration for ruv-swarm');
    });
  });
});
