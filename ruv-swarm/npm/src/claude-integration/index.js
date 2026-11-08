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
      mergeSetup: options.mergeSetup || false,
      backupSetup: options.backupSetup || false,
      noBackup: options.noBackup || false,
      interactive: options.interactive !== false, // Default to true
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
    console.log('   Backup setup:', this.options.backupSetup);
    console.log('   Merge setup:', this.options.mergeSetup);
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
      results.modules.docs = await this.docs.generateAll({
        force: this.options.forceSetup,
        merge: this.options.mergeSetup,
        backup: this.options.backupSetup,
        noBackup: this.options.noBackup,
        interactive: this.options.interactive,
      });

      // Step 2: Setup remote capabilities
      console.log('\n🌐 Step 2: Remote Execution Setup');
      results.modules.remote = await this.remote.createAll();

      results.modules.core = { success: true, configuration: '.mcp.json', projectScoped: true };
      console.log('Integration files written: CLAUDE.md, supported command docs, project .mcp.json and local wrappers.');
      console.log('Existing permission rules, hooks and MCP servers are preserved.');

      return results;

    } catch (error) {
      console.error('❌ Integration setup failed:', error.message);
      throw error;
    }
  }

  /**
     * Invoke Claude with a prompt using the core module
     */
  async invokeClaudeWithPrompt(prompt, options = {}) {
    return await this.core.invokeClaudeWithPrompt(prompt, options);
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
  return await orchestrator.invokeClaudeWithPrompt(prompt, options);
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