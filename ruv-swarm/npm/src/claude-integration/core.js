/**
 * Core Claude Code integration module
 * Handles MCP server setup and basic integration
 */

import { execSync } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';

class ClaudeIntegrationCore {
  constructor(options = {}) {
    this.autoSetup = options.autoSetup || false;
    this.forceSetup = options.forceSetup || false;
    this.workingDir = options.workingDir || process.cwd();
  }

  /**
     * Check if Claude CLI is available
     */
  async isClaudeAvailable() {
    try {
      execSync('claude --version', { stdio: 'ignore' });
      return true;
    } catch {
      return false;
    }
  }

  /**
     * Add ruv-swarm MCP server to Claude Code
     */
  async addMcpServer() {
    if (!await this.isClaudeAvailable()) {
      throw new Error('Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code');
    }

    try {
      // Add ruv-swarm MCP server using stdio (no port needed)
      const mcpCommand = 'claude mcp add ruv-swarm npx ruv-swarm mcp start';
      execSync(mcpCommand, { stdio: 'inherit', cwd: this.workingDir });
      return { success: true, message: 'Added ruv-swarm MCP server to Claude Code (stdio)' };
    } catch (error) {
      throw new Error(`Failed to add MCP server: ${error.message}`);
    }
  }

  /**
     * Check if integration files already exist
     */
  async checkExistingFiles() {
    try {
      await fs.access(path.join(this.workingDir, 'claude.md'));
      await fs.access(path.join(this.workingDir, '.claude/commands'));
      return true;
    } catch {
      return false;
    }
  }

  /**
     * Initialize Claude integration
     */
  async initialize() {
    console.log('🔧 Initializing Claude Code integration...');

    // Check if files exist (unless force setup)
    if (!this.forceSetup && await this.checkExistingFiles()) {
      console.log('   ℹ️  Claude integration files already exist (use --force to regenerate)');
      return { success: true, message: 'Integration files already exist' };
    }

    try {
      const results = {
        core: await this.addMcpServer(),
        success: true,
      };

      console.log('✅ Claude integration initialized successfully');
      return results;
    } catch (error) {
      console.error('❌ Failed to initialize Claude integration:', error.message);
      throw error;
    }
  }

  /**
     * Invoke Claude with a prompt (automatically includes --dangerously-skip-permissions)
     */
  async invokeClaudeWithPrompt(prompt) {
    if (!prompt || !prompt.trim()) {
      throw new Error('No prompt provided');
    }

    if (!await this.isClaudeAvailable()) {
      throw new Error('Claude Code CLI not found');
    }

    const claudeCommand = `claude "${ prompt.trim() }" --dangerously-skip-permissions`;

    try {
      execSync(claudeCommand, { stdio: 'inherit', cwd: this.workingDir });
      return { success: true, message: 'Claude invocation completed' };
    } catch (error) {
      throw new Error(`Claude invocation failed: ${ error.message}`);
    }
  }
}

export { ClaudeIntegrationCore };