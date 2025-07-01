# GitHub Project Management Commands for Claude Code

This directory contains Claude Code slash commands for GitHub-based swarm coordination and project management.

## 🚀 Quick Start

Use these commands in Claude Code to coordinate with other swarms:

```
/github status          - Show current swarm coordination status
/github tasks          - List available tasks to claim
/github claim <number> - Claim an issue for your swarm
/github update <number> <message> - Post progress update
/github release <number> - Release a claimed task
```

## 📋 Available Commands

### Core Commands
- `/github/status` - Overview of all swarm activities
- `/github/tasks` - List available tasks
- `/github/my-tasks` - Show your claimed tasks
- `/github/conflicts` - Check for coordination conflicts

### Task Management
- `/github/claim` - Claim a task for your swarm
- `/github/update` - Post progress updates
- `/github/complete` - Mark task as complete
- `/github/release` - Release a claimed task

### Collaboration
- `/github/swarms` - List active swarms
- `/github/coordinate` - Propose task splitting
- `/github/sync` - Sync with latest GitHub state

## 🐝 How Swarm Coordination Works

1. **Task Discovery**: Swarms find work via issue labels
2. **Claiming**: Add `swarm-claimed` label to prevent conflicts
3. **Updates**: Post progress as comments
4. **Completion**: Remove label or close issue

## 🏷️ Label System

- `available` - Task ready to be claimed
- `swarm-claimed` - Task being worked on
- `priority: critical/high/medium/low` - Task priority
- `area: core/mcp/neural/wasm` - Component area

## 💡 Best Practices

1. Always check if a task is claimed before starting
2. Post updates every 30-60 minutes
3. Release tasks if blocked
4. Coordinate on conflicts via comments

## 🔧 Configuration

Set your swarm ID in environment:
```bash
export CLAUDE_SWARM_ID="my-unique-swarm-id"
```

Or let it auto-generate based on your session.

## 📚 More Information

See the full guide at `/workspaces/ruv-FANN/ruv-swarm/SWARM_COLLABORATION_GUIDE.md`