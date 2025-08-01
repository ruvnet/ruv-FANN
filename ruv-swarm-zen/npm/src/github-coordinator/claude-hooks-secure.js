/**
 * SECURE Claude Code Hooks for GitHub Coordination
 * Addresses security vulnerabilities from Issue #115
 *
 * SECURITY FIXES:
 * - Uses secure GitHub coordinator implementation
 * - Implements input validation and sanitization
 * - Prevents prompt injection attacks
 * - Adds comprehensive security logging
 */

const SecureGHCoordinator = require('./gh-cli-coordinator-secure');
const CommandSanitizer = require('../security/command-sanitizer');
const path = require('path');
const crypto = require('crypto');

class SecureClaudeGitHubHooks {
  constructor(options = {}) {
    // Validate environment and options
    try {
      CommandSanitizer.validateEnvironment();
      this.coordinator = new SecureGHCoordinator(options);
      this.swarmId = this.generateSecureSwarmId(options.swarmId);
      this.activeTask = null;
      this.sessionStartTime = Date.now();
      this.operationLog = [];
    } catch (error) {
      throw new Error(`Failed to initialize secure hooks: ${error.message}`);
    }
  }

  /**
   * Generate secure swarm ID with validation
   */
  generateSecureSwarmId(providedId) {
    if (providedId) {
      // Validate provided swarm ID
      return CommandSanitizer.sanitizeSwarmId(providedId);
    }

    // Generate secure ID from environment or create cryptographically random one
    const envId = process.env.CLAUDE_SWARM_ID;
    if (envId) {
      return CommandSanitizer.sanitizeSwarmId(envId);
    }

    // Generate cryptographically secure random ID
    const randomBytes = crypto.randomBytes(8);
    const timestamp = Date.now().toString(36);
    const randomHex = randomBytes.toString('hex');
    return `claude-${timestamp}-${randomHex}`;
  }

  /**
   * Log operation for security audit
   */
  logOperation(operation, details, metadata = {}) {
    const logEntry = {
      timestamp: Date.now(),
      operation,
      details,
      swarmId: this.swarmId,
      activeTask: this.activeTask,
      ...metadata,
    };

    this.operationLog.push(logEntry);

    // Keep only last 100 operations to prevent memory issues
    if (this.operationLog.length > 100) {
      this.operationLog.shift();
    }

    console.log(`🔐 Security Log [${operation}]: ${details}`);
  }

  /**
   * Sanitize task description to prevent prompt injection
   */
  sanitizeTaskDescription(taskDescription) {
    if (typeof taskDescription !== 'string') {
      throw new Error('Task description must be a string');
    }

    // Remove potential prompt injection patterns
    let sanitized = taskDescription
      // Remove common injection starters
      .replace(/^\s*(ignore|disregard|forget|override)\s+(previous|above|all)\s+(instructions?|prompts?|rules?)/gi, '')
      // Remove system prompt attempts
      .replace(/\b(system|assistant|user|human):\s*/gi, '')
      // Remove instruction keywords
      .replace(/\b(execute|run|eval|system|shell|cmd|command)[\s(]/gi, '')
      // Remove dangerous characters but preserve normal punctuation
      .replace(/[`${}\\]/g, '')
      // Limit length to prevent DoS
      .slice(0, 500);

    // Ensure we still have meaningful content
    sanitized = sanitized.trim();
    if (sanitized.length < 3) {
      throw new Error('Task description too short or contained only unsafe content');
    }

    return sanitized;
  }

  /**
   * Secure keyword matching to prevent injection
   */
  performSecureMatching(taskDescription, taskText) {
    try {
      const sanitizedDescription = this.sanitizeTaskDescription(taskDescription);
      const sanitizedTaskText = CommandSanitizer.sanitizeMessage(taskText);

      // Simple, safe keyword matching
      const keywords = sanitizedDescription.toLowerCase()
        .split(/\s+/)
        .filter(word => word.length > 2 && /^[a-zA-Z0-9-_]+$/.test(word))
        .slice(0, 10); // Limit keywords to prevent DoS

      return keywords.some(keyword =>
        sanitizedTaskText.toLowerCase().includes(keyword),
      );
    } catch (error) {
      this.logOperation('matching_error', `Safe matching failed: ${error.message}`);
      return false;
    }
  }

  /**
   * SECURE Pre-task hook: Claim a GitHub issue before starting work
   */
  async preTask(taskDescription, metadata = {}) {
    this.logOperation('pre_task_start', 'Starting pre-task hook', {
      description: taskDescription?.slice(0, 100) + '...',
      ...metadata,
    });

    try {
      // CRITICAL: Sanitize task description to prevent prompt injection
      const sanitizedDescription = this.sanitizeTaskDescription(taskDescription);

      console.log(`🎯 Pre-task: Looking for GitHub issues related to: ${sanitizedDescription}`);

      // Search for related issues using secure coordinator
      const tasks = await this.coordinator.getAvailableTasks({ state: 'open' });

      // Perform secure matching
      let matchedTask = null;
      for (const task of tasks) {
        const taskText = `${task.title || ''} ${task.body || ''}`;
        if (this.performSecureMatching(sanitizedDescription, taskText)) {
          matchedTask = task;
          break;
        }
      }

      if (matchedTask) {
        // CRITICAL: Validate issue number before claiming
        const validatedIssueNumber = CommandSanitizer.validateIssueNumber(matchedTask.number);

        const claimed = await this.coordinator.claimTask(this.swarmId, validatedIssueNumber, metadata);
        if (claimed) {
          this.activeTask = validatedIssueNumber;
          this.logOperation('task_claimed', `Claimed issue #${validatedIssueNumber}`, { title: matchedTask.title });
          console.log(`✅ Claimed GitHub issue #${validatedIssueNumber}: ${matchedTask.title}`);
          return { claimed: true, issue: validatedIssueNumber };
        }
      }

      this.logOperation('no_match', 'No matching GitHub issue found');
      console.log('ℹ️ No matching GitHub issue found, proceeding without claim');
      return { claimed: false };

    } catch (error) {
      this.logOperation('pre_task_error', `Pre-task hook failed: ${error.message}`, metadata);
      console.error('❌ Pre-task hook error:', error.message);
      return { error: error.message };
    }
  }

  /**
   * SECURE Post-edit hook: Update GitHub issue with progress
   */
  async postEdit(filePath, changes, metadata = {}) {
    if (!this.activeTask) {
      return;
    }

    try {
      // CRITICAL: Validate and sanitize file path
      const safePath = path.basename(filePath); // Only use filename, not full path
      const sanitizedPath = safePath.replace(/[^a-zA-Z0-9._-]/g, '');

      // CRITICAL: Sanitize changes content
      const changesSummary = changes?.summary || 'File modified';
      const sanitizedSummary = CommandSanitizer.sanitizeMessage(changesSummary);

      const message = `Updated \`${sanitizedPath}\`\n\n${sanitizedSummary}`;

      await this.coordinator.updateTaskProgress(this.swarmId, this.activeTask, message, metadata);

      this.logOperation('progress_updated', `Updated issue #${this.activeTask}`, {
        file: sanitizedPath,
        ...metadata,
      });
      console.log(`📝 Updated GitHub issue #${this.activeTask} with edit progress`);

    } catch (error) {
      this.logOperation('post_edit_error', `Post-edit hook failed: ${error.message}`, metadata);
      console.error('❌ Post-edit hook error:', error.message);
    }
  }

  /**
   * SECURE Post-task hook: Complete or release the GitHub issue
   */
  async postTask(taskId, result, metadata = {}) {
    if (!this.activeTask) {
      return;
    }

    try {
      // CRITICAL: Sanitize task result content
      if (result?.completed) {
        const resultSummary = result.summary || 'Task completed successfully';
        const sanitizedSummary = CommandSanitizer.sanitizeMessage(resultSummary);

        const message = `✅ **Task Completed**\n\n${sanitizedSummary}`;
        await this.coordinator.updateTaskProgress(this.swarmId, this.activeTask, message, metadata);

        this.logOperation('task_completed', `Task completed for issue #${this.activeTask}`, metadata);

        // Note: Auto-close functionality removed for security
        // Issues should be manually closed after review
      } else {
        await this.coordinator.releaseTask(this.swarmId, this.activeTask, metadata);
        this.logOperation('task_released', `Released issue #${this.activeTask}`, metadata);
        console.log(`🔓 Released GitHub issue #${this.activeTask}`);
      }

      this.activeTask = null;

    } catch (error) {
      this.logOperation('post_task_error', `Post-task hook failed: ${error.message}`, metadata);
      console.error('❌ Post-task hook error:', error.message);
    }
  }

  /**
   * SECURE Conflict detection hook
   */
  async detectConflicts(metadata = {}) {
    try {
      const status = await this.coordinator.getCoordinationStatus();

      // Simple conflict detection based on swarm count
      if (Object.keys(status.swarmStatus).length > 1) {
        console.log('⚠️ Multiple swarms detected, checking for conflicts...');

        this.logOperation('conflict_check', `Multiple swarms detected: ${Object.keys(status.swarmStatus).length}`, metadata);

        return {
          hasConflicts: false,
          warningCount: Object.keys(status.swarmStatus).length - 1,
          message: 'Multiple swarms active, coordinate through GitHub issues',
          swarmCount: Object.keys(status.swarmStatus).length,
        };
      }

      return { hasConflicts: false };

    } catch (error) {
      this.logOperation('conflict_error', `Conflict detection failed: ${error.message}`, metadata);
      console.error('❌ Conflict detection error:', error.message);
      return { error: error.message };
    }
  }

  /**
   * SECURE Get coordination dashboard URL
   */
  async getDashboardUrl() {
    try {
      // Validate repository configuration
      const owner = CommandSanitizer.validateRepoIdentifier(this.coordinator.config.owner);
      const repo = CommandSanitizer.validateRepoIdentifier(this.coordinator.config.repo);
      const labelPrefix = CommandSanitizer.sanitizeLabel(this.coordinator.config.labelPrefix);
      const swarmId = CommandSanitizer.sanitizeSwarmId(this.swarmId);

      const baseUrl = `https://github.com/${owner}/${repo}`;

      return {
        issues: `${baseUrl}/issues?q=is:issue+is:open+label:${labelPrefix}${swarmId}`,
        allSwarms: `${baseUrl}/issues?q=is:issue+is:open+label:${labelPrefix}`,
        board: `${baseUrl}/projects`,
        security: `${baseUrl}/security`,
      };

    } catch (error) {
      this.logOperation('dashboard_error', `Dashboard URL generation failed: ${error.message}`);
      throw new Error('Failed to generate dashboard URLs');
    }
  }

  /**
   * Get security audit log
   */
  getSecurityAuditLog(limit = 50) {
    const limitedLimit = Math.min(limit, 100);
    return {
      sessionStartTime: this.sessionStartTime,
      swarmId: this.swarmId,
      operationCount: this.operationLog.length,
      operations: this.operationLog.slice(-limitedLimit),
      databaseLog: this.coordinator.getSecurityLog(limitedLimit),
    };
  }

  /**
   * Cleanup resources
   */
  cleanup() {
    this.logOperation('cleanup', 'Security hooks cleanup initiated');
    if (this.coordinator) {
      this.coordinator.close();
    }
  }
}

/**
 * SECURE Hook registration for Claude Code
 */
async function registerSecureHooks(options = {}) {
  try {
    const hooks = new SecureClaudeGitHubHooks({
      owner: options.owner || process.env.GITHUB_OWNER || 'ruvnet',
      repo: options.repo || process.env.GITHUB_REPO || 'ruv-FANN',
      ...options,
    });

    // Register with Claude Code's hook system with security validation
    return {
      'pre-task': (args) => hooks.preTask(args.description, { source: 'claude-code' }),
      'post-edit': (args) => hooks.postEdit(args.file, args.changes, { source: 'claude-code' }),
      'post-task': (args) => hooks.postTask(args.taskId, args.result, { source: 'claude-code' }),
      'check-conflicts': () => hooks.detectConflicts({ source: 'claude-code' }),
      'get-dashboard': () => hooks.getDashboardUrl(),
      'get-security-log': (args) => hooks.getSecurityAuditLog(args?.limit),
      'cleanup': () => hooks.cleanup(),
    };

  } catch (error) {
    console.error('❌ Failed to register secure hooks:', error.message);
    throw error;
  }
}

module.exports = { SecureClaudeGitHubHooks, registerSecureHooks };
