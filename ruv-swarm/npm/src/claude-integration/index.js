/**
 * Main Claude Code integration orchestrator
 * Coordinates all integration modules for modular, remote-capable setup
 */

import { ClaudeIntegrationCore } from './core.js';
import { ClaudeDocsGenerator } from './docs.js';
import { RemoteWrapperGenerator } from './remote.js';

class ClaudeIntegrationOrchestrator {
  constructor(options = {}) {
    this.options = {
      autoSetup: options.autoSetup || false,
      forceSetup: options.forceSetup || false,
      workingDir: options.workingDir || process.cwd(),
      packageName: options.packageName || 'ruv-swarm',
      ...options,
    };

    // Initialize modules
    this.core = new ClaudeIntegrationCore(this.options);
    this.docs = new ClaudeDocsGenerator(this.options);
    this.remote = new RemoteWrapperGenerator(this.options);
  }

  /**
     * Setup complete Claude Code integration
     */
  async setupIntegration() {
    console.log('🚀 Setting up modular Claude Code integration...');
    console.log('   Working directory:', this.options.workingDir);
    console.log('   Force setup:', this.options.forceSetup);
    console.log('   Auto setup MCP:', this.options.autoSetup);

    try {
      const results = {
        timestamp: new Date().toISOString(),
        workingDir: this.options.workingDir,
        success: true,
        modules: {},
      };

      // Step 1: Generate documentation
      console.log('\n📚 Step 1: Documentation Generation');
      results.modules.docs = await this.docs.generateAll();

      // Step 2: Setup remote capabilities
      console.log('\n🌐 Step 2: Remote Execution Setup');
      results.modules.remote = await this.remote.createAll();

      // Step 3: Initialize core integration (if auto setup enabled)
      if (this.options.autoSetup) {
        console.log('\n🔧 Step 3: Core Integration Setup');
        try {
          results.modules.core = await this.core.initialize();
        } catch (error) {
          console.log('⚠️  Core integration setup failed (manual setup required)');
          console.log('   Error:', error.message);
          results.modules.core = {
            success: false,
            error: error.message,
            manualSetup: true,
          };
        }
      } else {
        console.log('\n💡 Step 3: Manual Core Setup Required');
        results.modules.core = {
          success: true,
          manualSetup: true,
          instructions: [
            'Run: claude mcp add ruv-swarm npx ruv-swarm mcp start',
            'Test with: mcp__ruv-swarm__agent_spawn',
          ],
        };
      }

      // Summary
      console.log('\n✅ Modular Claude Code integration setup complete!');
      console.log('\n📋 What was created:');
      console.log('   📄 claude.md - Main configuration guide');
      console.log('   📁 .claude/commands/ - Command documentation');
      console.log('   🔧 Cross-platform wrapper scripts');
      console.log('   🤖 Claude helper scripts');
      console.log('   🌐 Remote execution support');

      console.log('\n🔗 Next steps:');
      if (results.modules.core.manualSetup) {
        console.log('   1. claude mcp add ruv-swarm npx ruv-swarm mcp start');
        console.log('   2. Test with MCP tools: mcp__ruv-swarm__agent_spawn');
      } else {
        console.log('   1. Test with MCP tools: mcp__ruv-swarm__agent_spawn');
      }
      console.log('   3. Check .claude/commands/ for detailed usage guides');
      console.log('   4. Use wrapper scripts for remote execution');

      return results;

    } catch (error) {
      console.error('❌ Integration setup failed:', error.message);
      throw error;
    }
  }

  /**
     * Invoke Claude with a prompt using the core module
     */
  async invokeClaudeWithPrompt(prompt) {
    return await this.core.invokeClaudeWithPrompt(prompt);
  }

  /**
     * Check integration status
     */
  async checkStatus() {
    console.log('🔍 Checking Claude Code integration status...');

    try {
      const status = {
        claudeAvailable: await this.core.isClaudeAvailable(),
        filesExist: await this.core.checkExistingFiles(),
        workingDir: this.options.workingDir,
        timestamp: new Date().toISOString(),
      };

      console.log('Claude CLI available:', status.claudeAvailable ? '✅' : '❌');
      console.log('Integration files exist:', status.filesExist ? '✅' : '❌');

      return status;
    } catch (error) {
      console.error('❌ Status check failed:', error.message);
      throw error;
    }
  }

  /**
     * Clean up integration files
     */
  async cleanup() {
    console.log('🧹 Cleaning up Claude Code integration files...');

    const fs = require('fs').promises;
    const path = require('path');

    try {
      const filesToRemove = [
        'claude.md',
        '.claude',
        this.options.packageName,
        `${this.options.packageName }.bat`,
        `${this.options.packageName }.ps1`,
        'claude-swarm.sh',
        'claude-swarm.bat',
      ];

      const removedFiles = [];

      for (const file of filesToRemove) {
        try {
          const filePath = path.join(this.options.workingDir, file);
          await fs.rm(filePath, { recursive: true, force: true });
          removedFiles.push(file);
        } catch {
          // File doesn't exist, continue
        }
      }

      console.log('✅ Cleanup complete. Removed:', removedFiles.join(', '));
      return { success: true, removedFiles };

    } catch (error) {
      console.error('❌ Cleanup failed:', error.message);
      throw error;
    }
  }
}

// Convenience function for simple setup
async function setupClaudeIntegration(options = {}) {
  const orchestrator = new ClaudeIntegrationOrchestrator(options);
  return await orchestrator.setupIntegration();
}

// Convenience function for Claude invocation
async function invokeClaudeWithSwarm(prompt, options = {}) {
  const orchestrator = new ClaudeIntegrationOrchestrator(options);
  return await orchestrator.invokeClaudeWithPrompt(prompt);
}

export {
  ClaudeIntegrationOrchestrator,
  setupClaudeIntegration,
  invokeClaudeWithSwarm,
  // Export individual modules for advanced usage
  ClaudeIntegrationCore,
  ClaudeDocsGenerator,
  RemoteWrapperGenerator,
};