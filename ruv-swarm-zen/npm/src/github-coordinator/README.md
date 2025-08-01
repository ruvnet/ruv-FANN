# GitHub Coordinator for ruv-swarm

Simple GitHub-based coordination for multiple Claude Code swarms working on the same repository.

## Quick Start

### Option 1: GitHub CLI (Recommended)

The simplest approach uses the GitHub CLI (`gh`) that you already have:

```bash
# Install gh CLI if not already installed
# https://cli.github.com/

# Use the coordinator
npx ruv-swarm github-coordinate

# Or use directly in your swarm
const GHCoordinator = require('./gh-cli-coordinator');
const coordinator = new GHCoordinator({
  owner: 'ruvnet',
  repo: 'ruv-FANN'
});

# Get available tasks
const tasks = await coordinator.getAvailableTasks();

# Claim a task
await coordinator.claimTask('my-swarm-id', issueNumber);

# Update progress
await coordinator.updateTaskProgress('my-swarm-id', issueNumber, 'Working on authentication...');
```

### Option 2: Claude Code Hooks

Add automatic GitHub coordination to your Claude Code sessions:

1. Add to `.claude/settings.json`:
```json
{
  "hooks": {
    "pre-task": "npx ruv-swarm hook github pre-task",
    "post-edit": "npx ruv-swarm hook github post-edit",
    "post-task": "npx ruv-swarm hook github post-task"
  }
}
```

2. Set environment variables:
```bash
export GITHUB_OWNER=ruvnet
export GITHUB_REPO=ruv-FANN
export CLAUDE_SWARM_ID=my-swarm-123  # Optional, auto-generated if not set
```

3. Now Claude Code will automatically:
   - Claim GitHub issues when starting tasks
   - Update issues with progress
   - Release issues when done

### Option 3: GitHub MCP Server

If you have a GitHub MCP server, you can use it directly:

```javascript
// Use GitHub MCP tools
mcp__github__issues_list({ state: 'open', labels: ['swarm-available'] })
mcp__github__issues_update({ number: 123, labels: ['swarm-claimed'] })
mcp__github__issues_comment({ number: 123, body: 'Progress update...' })
```

## How It Works

### Task Coordination via GitHub Labels

The system uses GitHub labels to coordinate work:

- `swarm-available` - Tasks ready to be claimed
- `swarm-{id}` - Task claimed by a specific swarm
- `swarm-conflict` - Multiple swarms trying to work on same area

### Workflow

1. **Task Discovery**: Swarms check for open issues without swarm labels
2. **Task Claiming**: Add a `swarm-{id}` label to claim an issue
3. **Progress Updates**: Post comments with progress updates
4. **Task Completion**: Remove label or close issue when done

### Conflict Prevention

- Each swarm has a unique ID
- Issues can only have one `swarm-*` label
- Comments show which swarm is working on what
- Dashboard shows all active swarms and their tasks

## Examples

### Basic Coordination

```javascript
// Initialize coordinator
const coordinator = new GHCoordinator({
  owner: 'ruvnet',
  repo: 'ruv-FANN',
  labelPrefix: 'swarm-'  // default
});

// Get coordination status
const status = await coordinator.getCoordinationStatus();
console.log(`${status.swarmTasks} tasks being worked on`);
console.log(`${status.availableTasks} tasks available`);

// Show swarm assignments
for (const [swarmId, tasks] of Object.entries(status.swarmStatus)) {
  console.log(`Swarm ${swarmId} is working on:`);
  tasks.forEach(task => console.log(`  - #${task.number}: ${task.title}`));
}
```

### With Claude Code Integration

When using the hooks, Claude Code will automatically coordinate:

```bash
# Start a task - automatically claims related GitHub issue
claude> Can you fix the authentication bug?
🎯 Pre-task: Looking for GitHub issues related to: fix authentication bug
✅ Claimed GitHub issue #123: Fix JWT token validation

# Make changes - automatically updates issue
claude> [edits auth.js]
📝 Updated GitHub issue #123 with edit progress

# Complete task - automatically updates/releases issue
claude> I've fixed the authentication issue
✅ Task Completed on GitHub issue #123
```

### Manual Coordination Commands

```bash
# Check coordination status
npx ruv-swarm github status

# List available tasks
npx ruv-swarm github tasks --available

# Claim a specific task
npx ruv-swarm github claim 123

# Release a task
npx ruv-swarm github release 123

# Show dashboard URLs
npx ruv-swarm github dashboard
```

## Configuration

### Environment Variables

- `GITHUB_OWNER` - Repository owner
- `GITHUB_REPO` - Repository name
- `GITHUB_TOKEN` - Personal access token (optional, uses gh CLI auth)
- `CLAUDE_SWARM_ID` - Unique swarm identifier

### Label Configuration

Customize the label prefix in the coordinator:

```javascript
const coordinator = new GHCoordinator({
  labelPrefix: 'ai-swarm-'  // Use custom prefix
});
```

## Benefits

1. **Simple**: Uses existing GitHub features (labels, comments)
2. **Visible**: All coordination visible in GitHub UI
3. **No Extra Infrastructure**: Just needs `gh` CLI
4. **Conflict-Free**: Labels prevent duplicate work
5. **Auditable**: Full history in GitHub

## Limitations

- Requires GitHub CLI or API access
- Limited to GitHub's rate limits
- Basic conflict detection (label-based)
- No real-time updates (polling-based)

For more advanced coordination, consider using the full GitHubCoordinator with a database backend or a dedicated MCP server.