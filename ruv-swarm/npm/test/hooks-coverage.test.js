/**
 * Hooks System Coverage Tests
 * Tests all hook implementations for 100% coverage
 */

import assert from 'assert';
import { HooksManager } from '../src/hooks/index.js';
import { CLIHooks } from '../src/hooks/cli.js';
import {
  ClaudeIntegration,
  AdvancedCommands,
  RemoteIntegration,
} from '../src/claude-integration/index.js';
import {
  GitHubCoordinator,
  ClaudeHooks,
} from '../src/github-coordinator/index.js';

describe('Hooks System 100% Coverage', () => {
  let hooks;

  beforeEach(() => {
    hooks = new HooksManager();
  });

  describe('HooksManager Core', () => {
    it('should handle hook registration edge cases', () => {
      // Null hook
      assert.throws(
        () => hooks.register(null, () => {}),
        /Invalid hook name/,
      );

      // Non-function handler
      assert.throws(
        () => hooks.register('test-hook', 'not-a-function'),
        /Handler must be a function/,
      );

      // Duplicate registration
      hooks.register('duplicate', () => {});
      assert.throws(
        () => hooks.register('duplicate', () => {}),
        /Hook already registered/,
      );
    });

    it('should handle hook execution failures', async() => {
      hooks.register('failing-hook', async() => {
        throw new Error('Hook failed');
      });

      await assert.rejects(
        hooks.execute('failing-hook', { data: 'test' }),
        /Hook failed/,
      );
    });

    it('should handle hook timeout', async() => {
      hooks.register('slow-hook', async() => {
        await new Promise(resolve => setTimeout(resolve, 5000));
      });

      hooks.setTimeout(100);

      await assert.rejects(
        hooks.execute('slow-hook', {}),
        /Hook timeout/,
      );
    });

    it('should handle hook context validation', async() => {
      hooks.register('validated-hook', async(context) => {
        if (!context.required) {
          throw new Error('Missing required field');
        }
        return context.required;
      });

      await assert.rejects(
        hooks.execute('validated-hook', { optional: 'value' }),
        /Missing required field/,
      );
    });

    it('should handle hook chain execution', async() => {
      const results = [];

      hooks.register('chain-1', async() => {
        results.push(1);
        return { step: 1 };
      });

      hooks.register('chain-2', async(context) => {
        results.push(2);
        return { ...context, step: 2 };
      });

      await hooks.executeChain(['chain-1', 'chain-2'], {});
      assert.deepEqual(results, [1, 2]);
    });

    it('should handle hook middleware errors', async() => {
      hooks.addMiddleware(async(hookName, context, next) => {
        if (hookName === 'restricted') {
          throw new Error('Access denied');
        }
        return next();
      });

      hooks.register('restricted', async() => ({ success: true }));

      await assert.rejects(
        hooks.execute('restricted', {}),
        /Access denied/,
      );
    });
  });

  describe('CLI Hooks', () => {
    let cliHooks;

    beforeEach(() => {
      cliHooks = new CLIHooks();
    });

    it('should handle invalid command parsing', () => {
      assert.throws(
        () => cliHooks.parseCommand(null),
        /Invalid command/,
      );

      assert.throws(
        () => cliHooks.parseCommand(''),
        /Empty command/,
      );
    });

    it('should handle command execution with missing arguments', async() => {
      await assert.rejects(
        cliHooks.execute('swarm', []), // Missing required args
        /Missing required arguments/,
      );
    });

    it('should handle command validation failures', async() => {
      await assert.rejects(
        cliHooks.execute('swarm', ['init', '--invalid-flag']),
        /Unknown flag/,
      );
    });

    it('should handle interactive mode edge cases', async() => {
      cliHooks.setInteractive(true);

      // Simulate no TTY
      const originalIsTTY = process.stdin.isTTY;
      process.stdin.isTTY = false;

      await assert.rejects(
        cliHooks.prompt('Enter value:'),
        /Not in interactive terminal/,
      );

      process.stdin.isTTY = originalIsTTY;
    });
  });

  describe('Claude Integration', () => {
    let claude;

    beforeEach(() => {
      claude = new ClaudeIntegration();
    });

    it('should handle API key validation', async() => {
      claude.setApiKey(''); // Empty API key

      await assert.rejects(
        claude.complete({ prompt: 'test' }),
        /Invalid API key/,
      );
    });

    it('should handle rate limiting', async() => {
      claude.setRateLimit(1); // 1 request per second

      const promises = [];
      for (let i = 0; i < 5; i++) {
        promises.push(claude.complete({ prompt: `test ${i}` }));
      }

      const results = await Promise.allSettled(promises);
      const rejected = results.filter(r => r.status === 'rejected');

      assert(rejected.length > 0, 'Some requests should be rate limited');
    });

    it('should handle response parsing errors', async() => {
      claude._mockResponse = 'invalid-json';

      await assert.rejects(
        claude.complete({ prompt: 'test' }),
        /Failed to parse response/,
      );
    });

    it('should handle context window overflow', async() => {
      const hugePrompt = 'x'.repeat(200000); // Exceeds context window

      await assert.rejects(
        claude.complete({ prompt: hugePrompt }),
        /Context window exceeded/,
      );
    });
  });

  describe('Advanced Commands', () => {
    let commands;

    beforeEach(() => {
      commands = new AdvancedCommands();
    });

    it('should handle command registration conflicts', () => {
      commands.register('test', () => {});

      assert.throws(
        () => commands.register('test', () => {}),
        /Command already exists/,
      );
    });

    it('should handle command alias conflicts', () => {
      commands.register('test', () => {}, { aliases: ['t'] });

      assert.throws(
        () => commands.register('test2', () => {}, { aliases: ['t'] }),
        /Alias already in use/,
      );
    });

    it('should handle command permission errors', async() => {
      commands.register('admin', () => {}, {
        requiresAdmin: true,
      });

      await assert.rejects(
        commands.execute('admin', {}, { isAdmin: false }),
        /Insufficient permissions/,
      );
    });

    it('should handle command validation schemas', async() => {
      commands.register('validated', (args) => args, {
        schema: {
          name: { type: 'string', required: true },
          age: { type: 'number', min: 0 },
        },
      });

      await assert.rejects(
        commands.execute('validated', { age: -5 }),
        /Validation failed/,
      );
    });
  });

  describe('Remote Integration', () => {
    let remote;

    beforeEach(() => {
      remote = new RemoteIntegration();
    });

    it('should handle connection failures', async() => {
      await assert.rejects(
        remote.connect('invalid://url'),
        /Failed to connect/,
      );
    });

    it('should handle authentication failures', async() => {
      remote.setCredentials({ token: 'invalid' });

      await assert.rejects(
        remote.authenticate(),
        /Authentication failed/,
      );
    });

    it('should handle message serialization errors', async() => {
      await remote.connect('ws://localhost:3000');

      const circularRef = {};
      circularRef.self = circularRef;

      await assert.rejects(
        remote.send(circularRef),
        /Failed to serialize message/,
      );
    });

    it('should handle reconnection logic', async() => {
      remote.setAutoReconnect(true, { maxRetries: 3 });

      let attempts = 0;
      remote.onReconnectAttempt = () => attempts++;

      await remote.connect('ws://invalid-host');

      // Wait for reconnection attempts
      await new Promise(resolve => setTimeout(resolve, 1000));

      assert(attempts >= 1, 'Should attempt reconnection');
    });
  });

  describe('GitHub Coordinator', () => {
    let coordinator;

    beforeEach(() => {
      coordinator = new GitHubCoordinator();
    });

    it('should handle PR creation failures', async() => {
      await assert.rejects(
        coordinator.createPR({
          title: '', // Empty title
          body: 'Test PR',
        }),
        /PR title required/,
      );
    });

    it('should handle branch protection violations', async() => {
      await assert.rejects(
        coordinator.push({
          branch: 'main',
          files: [{ path: 'test.js', content: 'test' }],
          force: true,
        }),
        /Branch protection/,
      );
    });

    it('should handle merge conflict detection', async() => {
      const result = await coordinator.checkMergeability({
        base: 'main',
        head: 'feature-branch',
      });

      if (!result.mergeable) {
        assert(result.conflicts.length > 0, 'Should detect conflicts');
      }
    });

    it('should handle webhook signature validation', () => {
      const payload = { event: 'push' };
      const invalidSignature = 'invalid';

      assert.throws(
        () => coordinator.validateWebhook(payload, invalidSignature),
        /Invalid webhook signature/,
      );
    });
  });

  describe('Claude Hooks Integration', () => {
    let claudeHooks;

    beforeEach(() => {
      claudeHooks = new ClaudeHooks();
    });

    it('should handle pre-task hook failures', async() => {
      claudeHooks.register('pre-task', async(context) => {
        if (!context.taskId) {
          throw new Error('Task ID required');
        }
      });

      await assert.rejects(
        claudeHooks.runPreTask({}),
        /Task ID required/,
      );
    });

    it('should handle post-edit hook validation', async() => {
      claudeHooks.register('post-edit', async(context) => {
        if (context.file.endsWith('.py') && !context.formatted) {
          throw new Error('Python files must be formatted');
        }
      });

      await assert.rejects(
        claudeHooks.runPostEdit({
          file: 'test.py',
          formatted: false,
        }),
        /Python files must be formatted/,
      );
    });

    it('should handle session persistence errors', async() => {
      claudeHooks._storage = null; // Simulate storage failure

      await assert.rejects(
        claudeHooks.saveSession({ id: 'test', data: {} }),
        /Failed to save session/,
      );
    });

    it('should handle hook execution order', async() => {
      const order = [];

      claudeHooks.register('ordered', async() => order.push(1), { priority: 10 });
      claudeHooks.register('ordered', async() => order.push(2), { priority: 5 });
      claudeHooks.register('ordered', async() => order.push(3), { priority: 15 });

      await claudeHooks.execute('ordered', {});

      assert.deepEqual(order, [2, 1, 3], 'Hooks should execute by priority');
    });
  });
});

// Run tests when executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  console.log('Running hooks coverage tests...');

  // Run all tests
  const { run } = await import('./test-runner.js');
  await run(__filename);
}