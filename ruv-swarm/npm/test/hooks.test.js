/**
 * Test suite for ruv-swarm hooks implementation
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { fileURLToPath } from 'url';
import path from 'path';
import { promises as fs } from 'fs';
import { execSync } from 'child_process';

// Mock modules
jest.mock('fs', () => ({
  promises: {
    readFile: jest.fn(),
    writeFile: jest.fn(),
    access: jest.fn(),
    mkdir: jest.fn(),
    readdir: jest.fn(),
  },
}));

jest.mock('child_process', () => ({
  execSync: jest.fn(),
}));

// Import the module to test
import RuvSwarmHooks from '../src/hooks/index.js';

describe('RuvSwarmHooks', () => {
  let hooks;
  let mockFs;
  let mockExecSync;

  beforeEach(() => {
    hooks = new RuvSwarmHooks();
    mockFs = fs;
    mockExecSync = execSync;
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('constructor', () => {
    it('should initialize session data correctly', () => {
      const newHooks = new RuvSwarmHooks();
      expect(newHooks.sessionData).toBeDefined();
      expect(newHooks.sessionData.operations).toEqual([]);
      expect(newHooks.sessionData.agents).toBeInstanceOf(Map);
      expect(newHooks.sessionData.learnings).toEqual([]);
      expect(newHooks.sessionData.metrics).toEqual({
        tokensSaved: 0,
        tasksCompleted: 0,
        patternsImproved: 0,
      });
    });
  });

  describe('handleHook', () => {
    it('should route pre-edit hook correctly', async() => {
      hooks.preEditHook = jest.fn().mockResolvedValue({ continue: true });
      const result = await hooks.handleHook('pre-edit', { file: 'test.js' });
      expect(hooks.preEditHook).toHaveBeenCalledWith({ file: 'test.js' });
      expect(result).toEqual({ continue: true });
    });

    it('should route post-edit hook correctly', async() => {
      hooks.postEditHook = jest.fn().mockResolvedValue({ success: true });
      const result = await hooks.handleHook('post-edit', { file: 'test.js' });
      expect(hooks.postEditHook).toHaveBeenCalledWith({ file: 'test.js' });
      expect(result).toEqual({ success: true });
    });

    it('should handle unknown hook types', async() => {
      const result = await hooks.handleHook('unknown-hook', {});
      expect(result).toEqual({
        continue: true,
        reason: 'Hook type not implemented: unknown-hook',
      });
    });

    it('should handle errors gracefully', async() => {
      hooks.preEditHook = jest.fn().mockRejectedValue(new Error('Test error'));
      const result = await hooks.handleHook('pre-edit', {});
      expect(result).toEqual({
        continue: true,
        error: 'Test error',
      });
    });
  });

  describe('preEditHook', () => {
    it('should auto-assign agent for JavaScript files', async() => {
      const args = { file: '/path/to/test.js', autoAssignAgent: true };
      mockExecSync.mockReturnValue('{"id": "agent-123"}');

      const result = await hooks.preEditHook(args);

      expect(result.continue).toBe(true);
      expect(result.assignedAgent).toBeDefined();
      expect(mockExecSync).toHaveBeenCalledWith(
        expect.stringContaining('ruv-swarm agent spawn'),
        expect.any(Object),
      );
    });

    it('should skip agent assignment when disabled', async() => {
      const args = { file: '/path/to/test.js', autoAssignAgent: false };

      const result = await hooks.preEditHook(args);

      expect(result.continue).toBe(true);
      expect(result.assignedAgent).toBeUndefined();
      expect(mockExecSync).not.toHaveBeenCalled();
    });

    it('should handle file type detection correctly', async() => {
      const testCases = [
        { file: 'test.js', expectedType: 'js' },
        { file: 'test.ts', expectedType: 'ts' },
        { file: 'test.py', expectedType: 'py' },
        { file: 'test.rs', expectedType: 'rs' },
        { file: 'test.go', expectedType: 'go' },
        { file: 'test.unknown', expectedType: null },
      ];

      for (const { file, expectedType } of testCases) {
        const result = await hooks.preEditHook({ file });
        if (expectedType) {
          expect(result.fileType).toBe(expectedType);
        }
      }
    });
  });

  describe('postEditHook', () => {
    it('should format JavaScript files', async() => {
      const args = { file: '/path/to/test.js', autoFormat: true };
      mockFs.access.mockResolvedValue(undefined); // File exists
      mockExecSync.mockReturnValue('');

      const result = await hooks.postEditHook(args);

      expect(result.continue).toBe(true);
      expect(result.formatted).toBe(true);
      expect(mockExecSync).toHaveBeenCalledWith(
        expect.stringContaining('prettier'),
        expect.any(Object),
      );
    });

    it('should skip formatting when disabled', async() => {
      const args = { file: '/path/to/test.js', autoFormat: false };

      const result = await hooks.postEditHook(args);

      expect(result.continue).toBe(true);
      expect(result.formatted).toBeUndefined();
      expect(mockExecSync).not.toHaveBeenCalled();
    });

    it('should store operation in memory when key provided', async() => {
      const args = {
        file: '/path/to/test.js',
        memoryKey: 'test/operation',
        trainNeural: false,
      };
      mockExecSync.mockReturnValue('{"stored": true}');

      const result = await hooks.postEditHook(args);

      expect(result.memoryStored).toBe(true);
      expect(mockExecSync).toHaveBeenCalledWith(
        expect.stringContaining('memory store'),
        expect.any(Object),
      );
    });
  });

  describe('preTaskHook', () => {
    it('should analyze task complexity', async() => {
      const args = { description: 'Complex refactoring task' };

      const result = await hooks.preTaskHook(args);

      expect(result.continue).toBe(true);
      expect(result.complexity).toBeDefined();
      expect(result.complexity.score).toBeGreaterThan(5);
    });

    it('should auto-spawn agents for complex tasks', async() => {
      const args = {
        description: 'Build complete authentication system with JWT',
        autoSpawnAgents: true,
      };
      mockExecSync.mockReturnValue('{"id": "swarm-123"}');

      const result = await hooks.preTaskHook(args);

      expect(result.swarmInitialized).toBe(true);
      expect(result.agentsSpawned).toBeGreaterThan(0);
    });

    it('should restore session when sessionId provided', async() => {
      const args = {
        description: 'Continue task',
        sessionId: 'session-123',
      };
      mockExecSync.mockReturnValue('{"restored": true}');

      const result = await hooks.preTaskHook(args);

      expect(result.sessionRestored).toBe(true);
    });
  });

  describe('postTaskHook', () => {
    it('should analyze performance when enabled', async() => {
      const args = {
        taskId: 'task-123',
        analyzePerformance: true,
      };
      hooks.sessionData.operations = [
        { type: 'edit', timestamp: Date.now() - 1000 },
        { type: 'edit', timestamp: Date.now() - 500 },
      ];

      const result = await hooks.postTaskHook(args);

      expect(result.continue).toBe(true);
      expect(result.performance).toBeDefined();
      expect(result.performance.totalOperations).toBe(2);
    });

    it('should generate report when requested', async() => {
      const args = {
        taskId: 'task-123',
        generateReport: true,
      };
      mockFs.writeFile.mockResolvedValue(undefined);

      const result = await hooks.postTaskHook(args);

      expect(result.reportGenerated).toBe(true);
      expect(mockFs.writeFile).toHaveBeenCalled();
    });
  });

  describe('preBashHook', () => {
    it('should validate safe commands', async() => {
      const args = { command: 'npm install express' };

      const result = await hooks.preBashHook(args);

      expect(result.continue).toBe(true);
      expect(result.validated).toBe(true);
    });

    it('should block dangerous commands', async() => {
      const dangerousCommands = [
        'rm -rf /',
        'dd if=/dev/zero of=/dev/sda',
        'fork() { fork|fork& }; fork',
      ];

      for (const command of dangerousCommands) {
        const result = await hooks.preBashHook({ command, validateSafety: true });
        expect(result.continue).toBe(false);
        expect(result.reason).toContain('potentially dangerous');
      }
    });

    it('should optimize package installations', async() => {
      const args = {
        command: 'npm install express body-parser cors',
        optimizeInstalls: true,
      };

      const result = await hooks.preBashHook(args);

      expect(result.optimized).toBe(true);
      expect(result.command).toContain('--prefer-offline');
    });
  });

  describe('sessionEndHook', () => {
    it('should export metrics when requested', async() => {
      const args = { exportMetrics: true };
      hooks.sessionData.metrics = {
        tokensSaved: 1500,
        tasksCompleted: 5,
        patternsImproved: 3,
      };
      mockFs.writeFile.mockResolvedValue(undefined);

      const result = await hooks.sessionEndHook(args);

      expect(result.metricsExported).toBe(true);
      expect(mockFs.writeFile).toHaveBeenCalledWith(
        expect.stringContaining('metrics.json'),
        expect.any(String),
      );
    });

    it('should generate summary when requested', async() => {
      const args = { generateSummary: true };
      hooks.sessionData.operations = [
        { type: 'edit', file: 'test1.js' },
        { type: 'edit', file: 'test2.js' },
        { type: 'bash', command: 'npm test' },
      ];

      const result = await hooks.sessionEndHook(args);

      expect(result.summary).toBeDefined();
      expect(result.summary.totalOperations).toBe(3);
      expect(result.summary.fileEdits).toBe(2);
    });

    it('should persist session state', async() => {
      const args = { persistState: true };
      mockExecSync.mockReturnValue('{"persisted": true}');

      const result = await hooks.sessionEndHook(args);

      expect(result.statePersisted).toBe(true);
      expect(mockExecSync).toHaveBeenCalledWith(
        expect.stringContaining('memory store'),
        expect.any(Object),
      );
    });
  });

  describe('helper methods', () => {
    describe('getFileType', () => {
      it('should correctly identify file types', () => {
        const testCases = [
          { file: 'test.js', expected: 'js' },
          { file: 'test.jsx', expected: 'jsx' },
          { file: 'test.ts', expected: 'ts' },
          { file: 'test.tsx', expected: 'tsx' },
          { file: 'test.py', expected: 'py' },
          { file: 'test.rs', expected: 'rs' },
          { file: 'test.go', expected: 'go' },
          { file: 'test.java', expected: 'java' },
          { file: 'test.cpp', expected: 'cpp' },
          { file: 'test.c', expected: 'c' },
          { file: 'test.rb', expected: 'rb' },
          { file: 'test.php', expected: 'php' },
          { file: 'test.unknown', expected: null },
        ];

        for (const { file, expected } of testCases) {
          expect(hooks.getFileType(file)).toBe(expected);
        }
      });
    });

    describe('analyzeComplexity', () => {
      it('should calculate complexity scores correctly', () => {
        const testCases = [
          {
            description: 'Fix typo',
            expectedScore: 1,
            expectedLevel: 'simple',
          },
          {
            description: 'Implement authentication system',
            expectedScore: 8,
            expectedLevel: 'complex',
          },
          {
            description: 'Refactor database schema and optimize queries',
            expectedScore: 7,
            expectedLevel: 'complex',
          },
        ];

        for (const { description, expectedScore, expectedLevel } of testCases) {
          const complexity = hooks.analyzeComplexity(description);
          expect(complexity.score).toBeCloseTo(expectedScore, 0);
          expect(complexity.level).toBe(expectedLevel);
        }
      });
    });
  });

  describe('error handling', () => {
    it('should handle file system errors gracefully', async() => {
      mockFs.access.mockRejectedValue(new Error('File not found'));

      const result = await hooks.postEditHook({
        file: 'nonexistent.js',
        autoFormat: true,
      });

      expect(result.continue).toBe(true);
      expect(result.formatted).toBeUndefined();
    });

    it('should handle command execution errors', async() => {
      mockExecSync.mockImplementation(() => {
        throw new Error('Command failed');
      });

      const result = await hooks.preEditHook({
        file: 'test.js',
        autoAssignAgent: true,
      });

      expect(result.continue).toBe(true);
      expect(result.assignedAgent).toBeUndefined();
    });
  });

  describe('integration scenarios', () => {
    it('should handle complete edit workflow', async() => {
      // Pre-edit
      const preResult = await hooks.preEditHook({
        file: 'test.js',
        autoAssignAgent: true,
      });
      expect(preResult.continue).toBe(true);

      // Simulate edit
      hooks.sessionData.operations.push({
        type: 'edit',
        file: 'test.js',
        timestamp: Date.now(),
      });

      // Post-edit
      const postResult = await hooks.postEditHook({
        file: 'test.js',
        autoFormat: true,
        trainNeural: true,
        memoryKey: 'edits/test',
      });
      expect(postResult.continue).toBe(true);
    });

    it('should handle complete task workflow', async() => {
      // Pre-task
      const preResult = await hooks.preTaskHook({
        description: 'Build REST API with authentication',
        autoSpawnAgents: true,
      });
      expect(preResult.continue).toBe(true);

      // Simulate operations
      for (let i = 0; i < 5; i++) {
        hooks.sessionData.operations.push({
          type: 'edit',
          timestamp: Date.now() + i * 100,
        });
      }

      // Post-task
      const postResult = await hooks.postTaskHook({
        taskId: 'task-123',
        analyzePerformance: true,
        generateReport: true,
      });
      expect(postResult.continue).toBe(true);
    });
  });
});