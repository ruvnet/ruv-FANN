/**
 * Core Claude Code integration module
 * Handles MCP server setup and basic integration
 */

import { execFileSync } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';

export function parseClaudeInvocationArgs(args) {
  const delimiter = args.indexOf('--');
  const options = delimiter < 0 ? args : args.slice(0, delimiter);
  const literal = delimiter < 0 ? [] : args.slice(delimiter + 1);
  const skipPermissions = options.includes('--dangerously-skip-permissions');
  const prompt = [...options.filter(arg => arg !== '--dangerously-skip-permissions'), ...literal].join(' ').trim();
  return { skipPermissions, prompt };
}

class ClaudeIntegrationCore {
  constructor(options = {}) {
    this.autoSetup = options.autoSetup || false;
    this.forceSetup = options.forceSetup || false;
    this.workingDir = options.workingDir || process.cwd();
    this.run = options.run || execFileSync;
  }

  /**
     * Check if Claude CLI is available
     */
  async isClaudeAvailable() {
    try {
      this.run('claude', ['--version'], { stdio: 'ignore' });
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
      this.run('claude', ['mcp', 'add', 'ruv-swarm', '--', 'npx', 'ruv-swarm', 'mcp', 'start'], {
        stdio: 'inherit', cwd: this.workingDir,
      });
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
      await fs.access(path.join(this.workingDir, 'CLAUDE.md'));
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
     * Invoke Claude with normal permission checks unless explicitly disabled
     */
  async invokeClaudeWithPrompt(prompt, options = {}) {
    if (typeof prompt !== 'string' || !prompt.trim()) {
      throw new Error('No prompt provided');
    }

    if (!await this.isClaudeAvailable()) {
      throw new Error('Claude Code CLI not found');
    }

    // Bypassing permission checks requires an explicit opt-in.
    const addPermissions = options.secure === false;
    const args = addPermissions ? ['--dangerously-skip-permissions'] : [];
    // Treat the prompt as one literal positional argument, even if it begins with '-'.
    args.push('--', prompt.trim());

    try {
      this.run('claude', args, { stdio: 'inherit', cwd: this.workingDir });
      return { success: true, message: 'Claude invocation completed' };
    } catch (error) {
      throw new Error(`Claude invocation failed: ${error.message}`);
    }
  }
}

export { ClaudeIntegrationCore };
