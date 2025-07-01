# /github my-tasks

Shows all GitHub issues currently claimed by your swarm, including status and time tracking.

## Usage

```
/github my-tasks [options]
```

## Options

- `--verbose` - Show detailed information
- `--include-completed` - Include recently completed tasks

## Examples

```
/github my-tasks
/github my-tasks --verbose
/github my-tasks --include-completed
```

## What it shows

- Currently claimed issues
- Time since claimed
- Last update time
- Progress status
- Staleness warnings

## Implementation

Uses GitHub MCP to:
1. Find issues with your swarm's claim label
2. Check last comment timestamps
3. Calculate time tracking
4. Identify stale tasks

## Example Output

### Basic View
```
📋 Your Active Tasks (3)

#45 [BUG] Memory leak in neural agents
    Claimed: 2 hours ago
    Last update: 30 min ago
    Status: 🔄 In progress

#52 [FEATURE] Add persistent storage
    Claimed: 5 hours ago
    Last update: 4 hours ago
    Status: ⚠️ Needs update

#48 [TASK] Update documentation
    Claimed: 1 day ago
    Last update: 1 day ago
    Status: 🔴 Stale - consider releasing
```

### Verbose View
```
📋 Your Active Tasks (3)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#45 [BUG] Memory leak in neural agents
    URL: https://github.com/ruvnet/ruv-FANN/issues/45
    Claimed: 2 hours ago (2024-01-07 14:30)
    Last update: 30 min ago
    Progress: Implementing fixes (70% complete)
    Time spent: ~2.5 hours
    Status: 🔄 In progress

    Recent updates:
    - "Identified 3 memory leak sources"
    - "Implementing cleanup handlers"
    
    Next: Run performance tests

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[... more tasks ...]
```

## Status Indicators

- 🟢 Active - Recently updated
- 🔄 In progress - Working normally  
- ⚠️ Needs update - No update in 2+ hours
- 🔴 Stale - No update in 12+ hours
- ⏰ Overdue - Claimed 24+ hours ago

## Time Tracking

Shows:
- Time since claimed
- Time since last update
- Estimated time remaining
- Total time spent (verbose mode)

## Warnings

The command will warn about:
- Stale tasks (no updates)
- Overdue tasks (>24 hours)
- Multiple claimed tasks
- Blocked tasks

## Best Practices

1. Keep tasks updated regularly
2. Release stale tasks promptly
3. Work on one task at a time
4. Complete within 24 hours
5. Check this before claiming new tasks

## Related Commands

- `/github update <number>` - Update a task
- `/github complete <number>` - Mark complete
- `/github release <number>` - Release a task
- `/github status` - Overall status