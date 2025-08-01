/**
 * SECURE GitHub CLI-based Coordinator for ruv-swarm
 * Addresses critical security vulnerabilities from Issue #115
 *
 * SECURITY FIXES:
 * - Replaced execSync with safe spawn-based execution
 * - Implemented comprehensive input validation and sanitization
 * - Added command injection prevention
 * - Implemented proper error handling and logging
 */

const fs = require('fs').promises;
const path = require('path');
const Database = require('better-sqlite3');
const CommandSanitizer = require('../security/command-sanitizer');

class SecureGHCoordinator {
  constructor(options = {}) {
    // Validate environment before proceeding
    CommandSanitizer.validateEnvironment();

    this.config = {
      owner: CommandSanitizer.validateRepoIdentifier(options.owner || process.env.GITHUB_OWNER),
      repo: CommandSanitizer.validateRepoIdentifier(options.repo || process.env.GITHUB_REPO),
      dbPath: options.dbPath || path.join(__dirname, '..', '..', 'data', 'gh-coordinator-secure.db'),
      labelPrefix: CommandSanitizer.sanitizeLabel(options.labelPrefix || 'swarm-'),
      ...options,
    };

    this.db = null;
    this.initialize();
  }

  async initialize() {
    // Check if gh CLI is available using safe execution
    try {
      await CommandSanitizer.safeExec('gh', ['--version'], { stdio: 'ignore' });
    } catch {
      throw new Error('GitHub CLI (gh) is not installed. Install it from https://cli.github.com/');
    }

    // Setup database for local coordination state
    await this.setupDatabase();
  }

  async setupDatabase() {
    const dataDir = path.dirname(this.config.dbPath);
    await fs.mkdir(dataDir, { recursive: true });

    this.db = new Database(this.config.dbPath);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS swarm_tasks (
        issue_number INTEGER PRIMARY KEY,
        swarm_id TEXT,
        locked_at INTEGER,
        lock_expires INTEGER
      );

      CREATE TABLE IF NOT EXISTS swarm_registry (
        swarm_id TEXT PRIMARY KEY,
        user TEXT,
        capabilities TEXT,
        last_seen INTEGER DEFAULT (strftime('%s', 'now'))
      );

      CREATE TABLE IF NOT EXISTS security_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER DEFAULT (strftime('%s', 'now')),
        event_type TEXT,
        details TEXT,
        ip_address TEXT,
        user_agent TEXT
      );
    `);
  }

  /**
   * Log security events
   */
  logSecurityEvent(eventType, details, metadata = {}) {
    try {
      this.db.prepare(`
        INSERT INTO security_log (event_type, details, ip_address, user_agent)
        VALUES (?, ?, ?, ?)
      `).run(eventType, details, metadata.ip || 'unknown', metadata.userAgent || 'unknown');
    } catch (error) {
      console.error('Failed to log security event:', error);
    }
  }

  /**
   * Get available tasks from GitHub issues - SECURE VERSION
   */
  async getAvailableTasks(filters = {}) {
    try {
      // Sanitize and validate filters
      const params = {
        repo: `${this.config.owner}/${this.config.repo}`,
        state: filters.state || 'open',
        limit: Math.min(parseInt(filters.limit) || 100, 1000), // Cap at 1000
      };

      if (filters.label) {
        params.label = CommandSanitizer.sanitizeLabel(filters.label);
      }

      // Create safe GitHub CLI arguments
      const args = CommandSanitizer.createGitHubArgs('issue-list', params);

      // Execute command safely
      const result = await CommandSanitizer.safeExec('gh', args, { encoding: 'utf8' });
      const issues = JSON.parse(result.stdout);

      // Filter out already assigned tasks
      const availableIssues = issues.filter(issue => {
        // Check if issue has swarm assignment label
        const hasSwarmLabel = issue.labels.some(l => l.name.startsWith(this.config.labelPrefix));
        // Check if issue is assigned
        const isAssigned = issue.assignees.length > 0;

        return !hasSwarmLabel && !isAssigned;
      });

      this.logSecurityEvent('task_list', `Retrieved ${availableIssues.length} available tasks`);
      return availableIssues;

    } catch (error) {
      this.logSecurityEvent('task_list_error', error.message);
      console.error('Failed to get available tasks:', error.message);
      throw new Error('Failed to retrieve tasks from GitHub');
    }
  }

  /**
   * Claim a task for a swarm - SECURE VERSION
   */
  async claimTask(swarmId, issueNumber, metadata = {}) {
    try {
      // CRITICAL: Validate and sanitize all inputs
      const sanitizedSwarmId = CommandSanitizer.sanitizeSwarmId(swarmId);
      const validatedIssueNumber = CommandSanitizer.validateIssueNumber(issueNumber);
      const label = `${this.config.labelPrefix}${sanitizedSwarmId}`;

      // Add swarm label to issue using safe execution
      const editArgs = CommandSanitizer.createGitHubArgs('issue-edit', {
        repo: `${this.config.owner}/${this.config.repo}`,
        issueNumber: validatedIssueNumber,
        addLabel: label,
      });

      await CommandSanitizer.safeExec('gh', editArgs, { stdio: 'ignore' });

      // Add comment to issue using safe execution
      const comment = `🐝 Task claimed by swarm: ${sanitizedSwarmId}\n\nThis task is being worked on by an automated swarm agent. Updates will be posted as progress is made.`;
      const commentArgs = CommandSanitizer.createGitHubArgs('issue-comment', {
        repo: `${this.config.owner}/${this.config.repo}`,
        issueNumber: validatedIssueNumber,
        body: comment,
      });

      await CommandSanitizer.safeExec('gh', commentArgs, { stdio: 'ignore' });

      // Record in local database
      this.db.prepare(`
        INSERT OR REPLACE INTO swarm_tasks (issue_number, swarm_id, locked_at, lock_expires)
        VALUES (?, ?, strftime('%s', 'now'), strftime('%s', 'now', '+1 hour'))
      `).run(validatedIssueNumber, sanitizedSwarmId);

      this.logSecurityEvent('task_claimed', `Task ${validatedIssueNumber} claimed by swarm ${sanitizedSwarmId}`, metadata);
      return true;

    } catch (error) {
      this.logSecurityEvent('task_claim_error', `Failed to claim task ${issueNumber}: ${error.message}`, metadata);
      console.error(`Failed to claim task ${issueNumber}:`, error.message);
      return false;
    }
  }

  /**
   * Release a task - SECURE VERSION
   */
  async releaseTask(swarmId, issueNumber, metadata = {}) {
    try {
      // CRITICAL: Validate and sanitize all inputs
      const sanitizedSwarmId = CommandSanitizer.sanitizeSwarmId(swarmId);
      const validatedIssueNumber = CommandSanitizer.validateIssueNumber(issueNumber);
      const label = `${this.config.labelPrefix}${sanitizedSwarmId}`;

      // Remove label using safe execution
      const editArgs = CommandSanitizer.createGitHubArgs('issue-edit', {
        repo: `${this.config.owner}/${this.config.repo}`,
        issueNumber: validatedIssueNumber,
        removeLabel: label,
      });

      await CommandSanitizer.safeExec('gh', editArgs, { stdio: 'ignore' });

      // Remove from database
      this.db.prepare('DELETE FROM swarm_tasks WHERE issue_number = ?').run(validatedIssueNumber);

      this.logSecurityEvent('task_released', `Task ${validatedIssueNumber} released by swarm ${sanitizedSwarmId}`, metadata);
      return true;

    } catch (error) {
      this.logSecurityEvent('task_release_error', `Failed to release task ${issueNumber}: ${error.message}`, metadata);
      console.error(`Failed to release task ${issueNumber}:`, error.message);
      return false;
    }
  }

  /**
   * Update task progress - SECURE VERSION
   */
  async updateTaskProgress(swarmId, issueNumber, message, metadata = {}) {
    try {
      // CRITICAL: Validate and sanitize all inputs
      const sanitizedSwarmId = CommandSanitizer.sanitizeSwarmId(swarmId);
      const validatedIssueNumber = CommandSanitizer.validateIssueNumber(issueNumber);
      const sanitizedMessage = CommandSanitizer.sanitizeMessage(message);

      const comment = `🔄 **Progress Update from swarm ${sanitizedSwarmId}**\n\n${sanitizedMessage}`;

      const commentArgs = CommandSanitizer.createGitHubArgs('issue-comment', {
        repo: `${this.config.owner}/${this.config.repo}`,
        issueNumber: validatedIssueNumber,
        body: comment,
      });

      await CommandSanitizer.safeExec('gh', commentArgs, { stdio: 'ignore' });

      this.logSecurityEvent('task_progress', `Progress updated for task ${validatedIssueNumber} by swarm ${sanitizedSwarmId}`, metadata);
      return true;

    } catch (error) {
      this.logSecurityEvent('task_progress_error', `Failed to update task ${issueNumber}: ${error.message}`, metadata);
      console.error(`Failed to update task ${issueNumber}:`, error.message);
      return false;
    }
  }

  /**
   * Create a task allocation PR - SECURE VERSION
   */
  async createAllocationPR(allocations, metadata = {}) {
    try {
      // Validate allocations
      if (!Array.isArray(allocations) || allocations.length === 0) {
        throw new Error('Invalid allocations: must be non-empty array');
      }

      // Validate each allocation
      const validatedAllocations = allocations.map(allocation => ({
        issue: CommandSanitizer.validateIssueNumber(allocation.issue),
        swarm_id: CommandSanitizer.sanitizeSwarmId(allocation.swarm_id),
      }));

      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const branch = CommandSanitizer.sanitizeBranchName(`swarm-allocation-${timestamp}`);

      // Create allocation file content
      const allocationContent = {
        timestamp: new Date().toISOString(),
        allocations: validatedAllocations,
      };

      const allocationPath = CommandSanitizer.validateFilePath('.github/swarm-allocations.json');
      await fs.writeFile(allocationPath, JSON.stringify(allocationContent, null, 2));

      // Create PR using safe git commands
      const checkoutArgs = CommandSanitizer.createGitArgs('checkout', {
        createBranch: true,
        branch: branch,
      });
      await CommandSanitizer.safeExec('git', checkoutArgs, { stdio: 'ignore' });

      const addArgs = CommandSanitizer.createGitArgs('add', {
        file: allocationPath,
      });
      await CommandSanitizer.safeExec('git', addArgs, { stdio: 'ignore' });

      const commitArgs = CommandSanitizer.createGitArgs('commit', {
        message: 'Update swarm task allocations',
      });
      await CommandSanitizer.safeExec('git', commitArgs, { stdio: 'ignore' });

      const pushArgs = CommandSanitizer.createGitArgs('push', {
        remote: 'origin',
        branch: branch,
      });
      await CommandSanitizer.safeExec('git', pushArgs, { stdio: 'ignore' });

      // Create PR body
      const prBody = `## Swarm Task Allocation Update

This PR updates the task allocation for active swarms.

### Allocations:
${validatedAllocations.map(a => `- Issue #${a.issue}: Assigned to swarm ${a.swarm_id}`).join('\n')}

This is an automated update from the secure swarm coordinator.`;

      const prArgs = CommandSanitizer.createGitHubArgs('pr-create', {
        repo: `${this.config.owner}/${this.config.repo}`,
        title: 'Update swarm task allocations',
        body: prBody,
        base: 'main',
        head: branch,
      });

      const result = await CommandSanitizer.safeExec('gh', prArgs, { encoding: 'utf8' });

      this.logSecurityEvent('pr_created', `Allocation PR created for ${validatedAllocations.length} tasks`, metadata);
      return result.stdout.trim();

    } catch (error) {
      this.logSecurityEvent('pr_creation_error', `Failed to create allocation PR: ${error.message}`, metadata);
      console.error('Failed to create allocation PR:', error.message);
      return null;
    }
  }

  /**
   * Get swarm coordination status - SECURE VERSION
   */
  async getCoordinationStatus() {
    try {
      // Get issues with swarm labels using safe execution
      const args = CommandSanitizer.createGitHubArgs('issue-list', {
        repo: `${this.config.owner}/${this.config.repo}`,
        limit: 100,
      });

      const result = await CommandSanitizer.safeExec('gh', args, { encoding: 'utf8' });
      const issues = JSON.parse(result.stdout);

      const swarmTasks = issues.filter(issue =>
        issue.labels.some(l => l.name.startsWith(this.config.labelPrefix)),
      );

      // Group by swarm
      const swarmStatus = {};
      swarmTasks.forEach(issue => {
        const swarmLabel = issue.labels.find(l => l.name.startsWith(this.config.labelPrefix));
        if (swarmLabel) {
          const swarmId = swarmLabel.name.replace(this.config.labelPrefix, '');
          if (!swarmStatus[swarmId]) {
            swarmStatus[swarmId] = [];
          }
          swarmStatus[swarmId].push({
            number: issue.number,
            title: issue.title,
          });
        }
      });

      this.logSecurityEvent('status_check', `Retrieved status for ${Object.keys(swarmStatus).length} swarms`);

      return {
        totalIssues: issues.length,
        swarmTasks: swarmTasks.length,
        availableTasks: issues.length - swarmTasks.length,
        swarmStatus,
      };

    } catch (error) {
      this.logSecurityEvent('status_error', `Failed to get coordination status: ${error.message}`);
      console.error('Failed to get coordination status:', error.message);
      throw new Error('Failed to retrieve coordination status');
    }
  }

  /**
   * Clean up stale locks - SECURE VERSION
   */
  async cleanupStaleLocks() {
    const staleTasks = this.db.prepare(`
      SELECT issue_number, swarm_id FROM swarm_tasks 
      WHERE lock_expires < strftime('%s', 'now')
    `).all();

    let cleanedCount = 0;
    for (const task of staleTasks) {
      try {
        await this.releaseTask(task.swarm_id, task.issue_number);
        cleanedCount++;
      } catch (error) {
        console.error(`Failed to clean up stale task ${task.issue_number}:`, error.message);
      }
    }

    this.logSecurityEvent('cleanup', `Cleaned up ${cleanedCount} stale locks`);
    return cleanedCount;
  }

  /**
   * Get security log entries
   */
  getSecurityLog(limit = 100) {
    return this.db.prepare(`
      SELECT * FROM security_log 
      ORDER BY timestamp DESC 
      LIMIT ?
    `).all(Math.min(limit, 1000));
  }

  /**
   * Close database connection
   */
  close() {
    if (this.db) {
      this.db.close();
    }
  }
}

module.exports = SecureGHCoordinator;
