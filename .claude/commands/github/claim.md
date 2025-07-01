# /github claim

Claims a GitHub issue for your swarm to work on, preventing conflicts with other swarms.

## Usage

```
/github claim <issue_number> [message]
```

## Arguments

- `<issue_number>` - The GitHub issue number to claim (required)
- `[message]` - Optional message about your plan (default: auto-generated)

## Examples

```
/github claim 45
/github claim 45 "Will focus on memory optimization using new WASM features"
```

## What it does

1. Checks if the issue is available (not already claimed)
2. Adds `swarm-claimed` label to the issue
3. Posts a comment with:
   - Your swarm ID
   - Work plan (if provided)
   - Estimated completion time
4. Updates local tracking

## Implementation

Uses GitHub MCP to:
1. Get current issue state
2. Verify no existing `swarm-claimed` label
3. Update labels to include `swarm-claimed`
4. Add comment with claim details

## Example Interaction

```
> /github claim 45

🎯 Claiming issue #45...

✅ Successfully claimed issue #45: "Implement memory optimization"

Posted comment:
🐝 Swarm ID: `claude-abc-123` claiming this task.

Plan:
1. Analyze current memory usage patterns
2. Implement optimization strategies
3. Add memory benchmarks
4. Update documentation

ETA: 2-3 hours

The issue is now locked to your swarm. Remember to post updates!
```

## Error Cases

- Issue already claimed: Shows current owner and suggests coordination
- Issue closed: Suggests finding another task
- Issue not found: Verifies issue number

## Best Practices

1. Only claim tasks you can complete
2. Release tasks if blocked
3. Post updates regularly
4. Complete or release within 24 hours

## Related Commands

- `/github tasks` - Find available tasks
- `/github update <number>` - Post progress
- `/github release <number>` - Release if needed
- `/github complete <number>` - Mark as done