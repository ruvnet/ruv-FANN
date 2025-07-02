/**
 * Claude Code Hooks for GitHub Coordination
 * Automatically coordinates swarm activities with GitHub
 */

const GHCoordinator = require('./gh-cli-coordinator');
// const fs = require('fs').promises; // Unused - will be used in future implementation
const path = require('path');

class ClaudeGitHubHooks {
  constructor(options = {}) {
    this.coordinator = new GHCoordinator(options);
    this.swarmId = options.swarmId || this.generateSwarmId();
    this.activeTask = null;
  }

  generateSwarmId() {
    // Generate swarm ID from environment or random
    return process.env.CLAUDE_SWARM_ID || `claude-${Date.now().toString(36)}`;
  }

  /**
   * Pre-task hook: Claim a GitHub issue before starting work
   */
  async preTask(taskDescription) {
    console.log(`🎯 Pre-task: Looking for GitHub issues related to: ${taskDescription}`);

    try {
      // Search for related issues
      const tasks = await this.coordinator.getAvailableTasks({ state: 'open' });

      // Find best matching task (simple keyword matching for now)
      const keywords = taskDescription.toLowerCase().split(' ');
      const matchedTask = tasks.find(task => {
        const taskText = `${task.title} ${task.body || ''}`.toLowerCase();
        return keywords.some(keyword => taskText.includes(keyword));
      });

      if (matchedTask) {
        const claimed = await this.coordinator.claimTask(this.swarmId, matchedTask.number);
        if (claimed) {
          this.activeTask = matchedTask.number;
          console.log(`✅ Claimed GitHub issue #${matchedTask.number}: ${matchedTask.title}`);
          return { claimed: true, issue: matchedTask.number };
        }
      }

      console.log('ℹ️ No matching GitHub issue found, proceeding without claim');
      return { claimed: false };
    } catch (error) {
      console.error('❌ Pre-task hook error:', error.message);
      return { error: error.message };
    }
  }

  /**
   * Post-edit hook: Update GitHub issue with progress
   */
  async postEdit(filePath, changes) {
    if (!this.activeTask) {
      return;
    }

    try {
      const message = `Updated \`${path.basename(filePath)}\`\n\n${changes.summary || 'File modified'}`;
      await this.coordinator.updateTaskProgress(this.swarmId, this.activeTask, message);
      console.log(`📝 Updated GitHub issue #${this.activeTask} with edit progress`);
    } catch (error) {
      console.error('❌ Post-edit hook error:', error.message);
    }
  }

  /**
   * Post-task hook: Complete or release the GitHub issue
   */
  async postTask(taskId, result) {
    if (!this.activeTask) {
      return;
    }

    try {
      if (result.completed) {
        const message = `✅ **Task Completed**\n\n${result.summary || 'Task completed successfully'}`;
        await this.coordinator.updateTaskProgress(this.swarmId, this.activeTask, message);

        // Option to auto-close issue (disabled by default)
        if (result.autoClose) {
          console.log(`🏁 Closing GitHub issue #${this.activeTask}`);
          // Use gh CLI to close issue
        }
      } else {
        await this.coordinator.releaseTask(this.swarmId, this.activeTask);
        console.log(`🔓 Released GitHub issue #${this.activeTask}`);
      }

      this.activeTask = null;
    } catch (error) {
      console.error('❌ Post-task hook error:', error.message);
    }
  }

  /**
   * Conflict detection hook
   */
  async detectConflicts() {
    try {
      const status = await this.coordinator.getCoordinationStatus();

      // Check if multiple swarms are working on similar files
      // const conflicts = []; // Unused - will be used in future implementation

      // Simple conflict detection based on swarm count
      if (Object.keys(status.swarmStatus).length > 1) {
        console.log('⚠️ Multiple swarms detected, checking for conflicts...');

        // More sophisticated conflict detection could be added here
        // For now, just warn about multiple active swarms
        return {
          hasConflicts: false,
          warningCount: Object.keys(status.swarmStatus).length - 1,
          message: 'Multiple swarms active, coordinate through GitHub issues',
        };
      }

      return { hasConflicts: false };
    } catch (error) {
      console.error('❌ Conflict detection error:', error.message);
      return { error: error.message };
    }
  }

  /**
   * Get coordination dashboard URL
   */
  async getDashboardUrl() {
    const baseUrl = `https://github.com/${this.coordinator.config.owner}/${this.coordinator.config.repo}`;
    return {
      issues: `${baseUrl}/issues?q=is:issue+is:open+label:${this.coordinator.config.labelPrefix}${this.swarmId}`,
      allSwarms: `${baseUrl}/issues?q=is:issue+is:open+label:${this.coordinator.config.labelPrefix}`,
      board: `${baseUrl}/projects`,
    };
  }
}

// Hook registration for Claude Code
async function registerHooks() {
  const hooks = new ClaudeGitHubHooks({
    owner: process.env.GITHUB_OWNER || 'ruvnet',
    repo: process.env.GITHUB_REPO || 'ruv-FANN',
  });

  // Register with Claude Code's hook system
  return {
    'pre-task': (args) => hooks.preTask(args.description),
    'post-edit': (args) => hooks.postEdit(args.file, args.changes),
    'post-task': (args) => hooks.postTask(args.taskId, args.result),
    'check-conflicts': () => hooks.detectConflicts(),
    'get-dashboard': () => hooks.getDashboardUrl(),
  };
}

module.exports = { ClaudeGitHubHooks, registerHooks };