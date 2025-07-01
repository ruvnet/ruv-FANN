# /github update

Posts a progress update to a claimed GitHub issue, keeping other swarms informed of your progress.

## Usage

```
/github update <issue_number> <message>
```

## Arguments

- `<issue_number>` - The issue you're updating (required)
- `<message>` - Progress update message (required)

## Examples

```
/github update 45 "Completed memory analysis, found 3 leak sources"
/github update 45 "50% done - implemented fixes, testing now"
/github update 45 "Blocked by #52, waiting for neural agent fixes"
```

## Progress Indicators

Use these prefixes for clarity:
- `✅` - Completed items
- `🔄` - In progress
- `⏳` - Pending
- `❌` - Blocked
- `⚠️` - Issues found

## Implementation

Uses GitHub MCP to:
1. Verify you own the claimed task
2. Post formatted comment with progress
3. Update last activity timestamp
4. Check for staleness warnings

## Example Interactions

### Simple Update
```
> /github update 45 "Implemented first optimization pass"

📝 Posted update to issue #45
```

### Detailed Update
```
> /github update 45 "Major progress on memory optimization:
✅ Analyzed memory patterns
✅ Identified 3 leak sources
🔄 Implementing fixes (70% done)
⏳ Tests pending
⏳ Documentation update"

📝 Posted detailed update to issue #45
```

### Blocked Update
```
> /github update 45 "Blocked: Need PR #52 merged first"

⚠️ Posted blocked status to issue #45
Consider releasing if blocked for long?
```

## Auto-Generated Format

Updates are posted as:
```
🔄 Progress Update from swarm `<your-id>`:

<your message>

---
Updated at: <timestamp>
```

## Best Practices

1. Update at least every hour when working
2. Be specific about what you've done
3. Mention blockers immediately
4. Reference related issues/PRs
5. Give completion estimates

## Related Commands

- `/github claim <number>` - Claim a task first
- `/github complete <number>` - Mark as done
- `/github release <number>` - Release if blocked
- `/github my-tasks` - See your tasks