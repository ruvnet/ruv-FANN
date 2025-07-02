/**
 * GitHub CLI-based Coordinator for ruv-swarm
 * Uses gh CLI for all GitHub operations - simpler and more reliable
 */

const { execSync } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const Database = require('better-sqlite3');

class GHCoordinator {
  constructor(options = {}) {
    this.config = {
      owner: options.owner || process.env.GITHUB_OWNER,
      repo: options.repo || process.env.GITHUB_REPO,
      dbPath: options.dbPath || path.join(__dirname, '..', '..', 'data', 'gh-coordinator.db'),
      labelPrefix: options.labelPrefix || 'swarm-',
      ...options,
    };

    this.db = null;
    this.initialize();
  }

  async initialize() {
    // Check if gh CLI is available
    try {
      execSync('gh --version', { stdio: 'ignore' });
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
    `);
  }

  /**
   * Get available tasks from GitHub issues
   */
  async getAvailableTasks(filters = {}) {
    let cmd = `gh issue list --repo ${this.config.owner}/${this.config.repo} --json number,title,labels,assignees,state,body --limit 100`;

    if (filters.label) {
      cmd += ` --label "${filters.label}"`;
    }
    if (filters.state) {
      cmd += ` --state ${filters.state}`;
    }

    const output = execSync(cmd, { encoding: 'utf8' });
    const issues = JSON.parse(output);

    // Filter out already assigned tasks
    const availableIssues = issues.filter(issue => {
      // Check if issue has swarm assignment label
      const hasSwarmLabel = issue.labels.some(l => l.name.startsWith(this.config.labelPrefix));
      // Check if issue is assigned
      const isAssigned = issue.assignees.length > 0;

      return !hasSwarmLabel && !isAssigned;
    });

    return availableIssues;
  }

  /**
   * Claim a task for a swarm
   */
  async claimTask(swarmId, issueNumber) {
    try {
      // Add swarm label to issue
      const label = `${this.config.labelPrefix}${swarmId}`;
      execSync(`gh issue edit ${issueNumber} --repo ${this.config.owner}/${this.config.repo} --add-label "${label}"`, { stdio: 'ignore' });

      // Add comment to issue
      const comment = `🐝 Task claimed by swarm: ${swarmId}\n\nThis task is being worked on by an automated swarm agent. Updates will be posted as progress is made.`;
      execSync(`gh issue comment ${issueNumber} --repo ${this.config.owner}/${this.config.repo} --body "${comment}"`, { stdio: 'ignore' });

      // Record in local database
      this.db.prepare(`
        INSERT OR REPLACE INTO swarm_tasks (issue_number, swarm_id, locked_at, lock_expires)
        VALUES (?, ?, strftime('%s', 'now'), strftime('%s', 'now', '+1 hour'))
      `).run(issueNumber, swarmId);

      return true;
    } catch (error) {
      console.error(`Failed to claim task ${issueNumber}:`, error.message);
      return false;
    }
  }

  /**
   * Release a task
   */
  async releaseTask(swarmId, issueNumber) {
    try {
      const label = `${this.config.labelPrefix}${swarmId}`;
      execSync(`gh issue edit ${issueNumber} --repo ${this.config.owner}/${this.config.repo} --remove-label "${label}"`, { stdio: 'ignore' });

      this.db.prepare('DELETE FROM swarm_tasks WHERE issue_number = ?').run(issueNumber);
      return true;
    } catch (error) {
      console.error(`Failed to release task ${issueNumber}:`, error.message);
      return false;
    }
  }

  /**
   * Update task progress
   */
  async updateTaskProgress(swarmId, issueNumber, message) {
    try {
      const comment = `🔄 **Progress Update from swarm ${swarmId}**\n\n${message}`;
      execSync(`gh issue comment ${issueNumber} --repo ${this.config.owner}/${this.config.repo} --body "${comment}"`, { stdio: 'ignore' });
      return true;
    } catch (error) {
      console.error(`Failed to update task ${issueNumber}:`, error.message);
      return false;
    }
  }

  /**
   * Create a task allocation PR
   */
  async createAllocationPR(allocations) {
    const branch = `swarm-allocation-${Date.now()}`;

    // Create allocation file
    const allocationContent = {
      timestamp: new Date().toISOString(),
      allocations,
    };

    const allocationPath = '.github/swarm-allocations.json';
    await fs.writeFile(allocationPath, JSON.stringify(allocationContent, null, 2));

    // Create PR using gh CLI
    try {
      execSync(`git checkout -b ${branch}`, { stdio: 'ignore' });
      execSync(`git add ${allocationPath}`, { stdio: 'ignore' });
      execSync('git commit -m "Update swarm task allocations"', { stdio: 'ignore' });
      execSync(`git push origin ${branch}`, { stdio: 'ignore' });

      const prBody = `## Swarm Task Allocation Update

This PR updates the task allocation for active swarms.

### Allocations:
${allocations.map(a => `- Issue #${a.issue}: Assigned to swarm ${a.swarm_id}`).join('\n')}

This is an automated update from the swarm coordinator.`;

      const output = execSync(`gh pr create --repo ${this.config.owner}/${this.config.repo} --title "Update swarm task allocations" --body "${prBody}" --base main --head ${branch}`, { encoding: 'utf8' });

      return output.trim();
    } catch (error) {
      console.error('Failed to create allocation PR:', error.message);
      return null;
    }
  }

  /**
   * Get swarm coordination status
   */
  async getCoordinationStatus() {
    // Get issues with swarm labels
    const cmd = `gh issue list --repo ${this.config.owner}/${this.config.repo} --json number,title,labels,assignees --limit 100`;
    const output = execSync(cmd, { encoding: 'utf8' });
    const issues = JSON.parse(output);

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

    return {
      totalIssues: issues.length,
      swarmTasks: swarmTasks.length,
      availableTasks: issues.length - swarmTasks.length,
      swarmStatus,
    };
  }

  /**
   * Clean up stale locks
   */
  async cleanupStaleLocks() {
    const staleTasks = this.db.prepare(`
      SELECT issue_number, swarm_id FROM swarm_tasks 
      WHERE lock_expires < strftime('%s', 'now')
    `).all();

    for (const task of staleTasks) {
      await this.releaseTask(task.swarm_id, task.issue_number);
    }

    return staleTasks.length;
  }
}

// Example usage with gh CLI - commented out to avoid no-unused-vars warning
// async function example() {
//   const coordinator = new GHCoordinator({
//     owner: 'ruvnet',
//     repo: 'ruv-FANN',
//   });
//
//   // Get available tasks
//   const tasks = await coordinator.getAvailableTasks({ state: 'open' });
//   console.log(`Found ${tasks.length} available tasks`);
//
//   // Claim a task for a swarm
//   if (tasks.length > 0) {
//     const claimed = await coordinator.claimTask('swarm-123', tasks[0].number);
//     console.log(`Claimed task #${tasks[0].number}: ${claimed}`);
//   }
//
//   // Get coordination status
//   const status = await coordinator.getCoordinationStatus();
//   console.log('Coordination status:', status);
// }

module.exports = GHCoordinator;