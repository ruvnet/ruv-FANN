# /github complete

Marks a GitHub issue as completed by your swarm, optionally linking to the PR with changes.

## Usage

```
/github complete <issue_number> [options]
```

## Options

- `--pr <number>` - Link to pull request with changes
- `--message <text>` - Custom completion message
- `--close` - Also close the issue (default: false)

## Examples

```
/github complete 45
/github complete 45 --pr 123
/github complete 45 --pr 123 --close
/github complete 45 --message "Fixed with new caching strategy"
```

## What it does

1. Removes `swarm-claimed` label
2. Posts completion comment with:
   - Summary of work done
   - Link to PR (if provided)
   - Any relevant notes
3. Optionally closes the issue
4. Updates coordination tracking

## Implementation

Uses GitHub MCP to:
1. Verify task ownership
2. Remove claim label
3. Post completion comment
4. Optionally close issue
5. Link to PR if provided

## Example Interactions

### Basic Completion
```
> /github complete 45

✅ Marked issue #45 as complete
Removed swarm claim
Posted completion notice
```

### With PR Link
```
> /github complete 45 --pr 123

✅ Marked issue #45 as complete
Linked to PR #123
Posted: "Task completed in PR #123"
```

### With Close
```
> /github complete 45 --pr 123 --close

✅ Marked issue #45 as complete
Linked to PR #123
Issue closed
Posted: "Task completed and verified in PR #123"
```

## Completion Message Format

```
✅ Task Complete by swarm `<your-id>`

Changes implemented in PR #123

Summary:
- Fixed memory leaks in neural agent
- Added cleanup handlers
- Improved performance by 40%

All tests passing. Ready for review.

---
Completed at: <timestamp>
```

## Best Practices

1. Always link to your PR if you made changes
2. Summarize what was actually done
3. Only close if you're certain it's fully resolved
4. Mention any follow-up work needed
5. Thank collaborators if applicable

## Related Commands

- `/github claim <number>` - Initial task claim
- `/github update <number>` - Progress updates
- `/github release <number>` - Release without completing
- `/github my-tasks` - View your tasks