# /github status

Shows the current swarm coordination status including available tasks, active swarms, and potential conflicts.

## Usage

```
/github status
```

## What it shows

- Total available tasks by priority
- Currently claimed tasks and their swarms
- Active swarms and what they're working on
- Any coordination conflicts
- Recent activity summary

## Implementation

This command uses the GitHub MCP to:
1. List all open issues with coordination labels
2. Group by swarm assignment
3. Identify potential conflicts
4. Show summary statistics

## Example Output

```
🐝 Swarm Coordination Status

📊 Overview:
- Available Tasks: 12 (3 critical, 5 high, 4 medium)
- Claimed Tasks: 8
- Active Swarms: 4
- Conflicts: 0

🚀 Active Swarms:
- swarm-alpha-123: Working on #45 (Memory optimization)
- swarm-beta-456: Working on #52 (Neural agent fixes)
- swarm-gamma-789: Working on #48 (Documentation update)
- swarm-delta-012: Working on #50 (WASM performance)

📋 Available High Priority:
- #53: [BUG] Memory leak in pattern matching
- #54: [FEATURE] Add real-time metrics
- #55: [TASK] Refactor coordination system

⚠️ Stale Claims (>24h):
- #38: Last updated 2 days ago by swarm-old-999
```

## Related Commands

- `/github tasks` - List only available tasks
- `/github my-tasks` - Show your swarm's tasks
- `/github swarms` - Detailed swarm information