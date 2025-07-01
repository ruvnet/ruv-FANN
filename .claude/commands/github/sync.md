# /github sync

Synchronizes your local swarm state with the latest GitHub repository state, updating task statuses and detecting changes.

## Usage

```
/github sync [options]
```

## Options

- `--full` - Complete resync of all data
- `--tasks` - Sync only task states
- `--conflicts` - Check for new conflicts

## Examples

```
/github sync
/github sync --full
/github sync --conflicts
```

## What it syncs

1. Task availability changes
2. New swarm activities
3. Issue updates and comments
4. PR status changes
5. Label modifications
6. Conflict emergence

## Implementation

Uses GitHub MCP to:
1. Fetch latest issue states
2. Compare with local cache
3. Update coordination data
4. Alert on important changes

## Example Output

### Basic Sync
```
🔄 Syncing with GitHub...

📥 Updates Found:

NEW TASKS (2):
- #64: [BUG] WebSocket connection drops
- #65: [FEATURE] Add dark mode

CLAIMED (3):
- #45: Claimed by swarm-zeta-999
- #52: Claimed by swarm-eta-111
- #48: Released by swarm-old-555

COMPLETED (1):
- #38: Completed by swarm-alpha-123

YOUR CHANGES:
- #55: New comment from maintainer
- #55: Priority changed to critical

✅ Sync complete - 6 changes
```

### Conflict Detection
```
🔄 Syncing for conflicts...

⚠️ New Conflicts Detected:

1. TASK OVERLAP
   swarm-new-777 claimed #66
   Related to your #55 (same component)
   Suggest: Coordinate approach

2. FILE CONFLICT  
   PR #125 modifies src/neural.js
   You're editing same file for #55
   Suggest: Check PR changes first

📋 No conflicts: Your other 2 tasks
```

### Full Sync
```
🔄 Full sync initiated...

📊 Repository State:
- Total issues: 89 (12 new)
- Open tasks: 34 (8 available)
- Active swarms: 7 (2 new)
- Your tasks: 3 (1 needs update)

🔄 Updating local state...
- Task cache refreshed
- Swarm registry updated
- Conflict analysis complete
- Coordination rules updated

✅ Full sync complete
💡 3 new high-priority tasks available
```

## Change Notifications

Alerts you about:
- New high-priority tasks
- Changes to your issues
- New conflicts
- Released tasks in your area
- Maintainer comments

## Auto-Sync

Happens automatically when:
- Running any command
- Every 5 minutes (if active)
- On conflict detection
- Before major operations

## State Management

Maintains:
- Task availability cache
- Swarm activity log
- Conflict history
- Coordination state
- Your task status

## Best Practices

1. Sync before claiming tasks
2. Full sync daily
3. Check after breaks
4. Sync before coordinating
5. Monitor your task changes

## Related Commands

- `/github status` - View synced state
- `/github conflicts` - Check conflicts
- `/github my-tasks` - Your current work
- `/github tasks` - Available tasks