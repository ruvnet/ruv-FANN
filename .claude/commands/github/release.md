# /github release

Releases a claimed GitHub issue back to the available pool, allowing other swarms to claim it.

## Usage

```
/github release <issue_number> [reason]
```

## Arguments

- `<issue_number>` - Issue to release (required)
- `[reason]` - Why you're releasing it (optional but recommended)

## Examples

```
/github release 45
/github release 45 "Blocked by dependency"
/github release 45 "Need expertise in WASM optimization"
/github release 45 "Higher priority task came up"
```

## When to Release

- Blocked by dependencies
- Realized you lack required expertise
- Higher priority work emerged
- Can't complete within reasonable time
- Found it's duplicate of another issue

## Implementation

Uses GitHub MCP to:
1. Verify you currently own the task
2. Remove `swarm-claimed` label
3. Post release comment with reason
4. Make task available again

## Example Interactions

### Basic Release
```
> /github release 45

🔓 Released issue #45
Task now available for other swarms
```

### With Reason
```
> /github release 45 "Need WASM expertise - better suited for specialized swarm"

🔓 Released issue #45
Posted: "Released - Need WASM expertise"
Suggested area: wasm specialists
```

### With Handoff
```
> /github release 45 "Partially complete - see PR #120 for progress so far"

🔓 Released issue #45
Posted handoff notes for next swarm
Progress preserved in PR #120
```

## Release Message Format

```
🔓 Task Released by swarm `<your-id>`

Reason: <your reason>

Progress made:
- Analyzed the issue
- Started implementation
- See branch: feature/issue-45

Available for other swarms to continue.

---
Released at: <timestamp>
```

## Best Practices

1. Always provide a reason
2. Document any progress made
3. Link to branches/PRs with partial work
4. Suggest what expertise is needed
5. Release promptly when blocked

## Partial Work

If you made progress:
1. Push your branch
2. Create draft PR
3. Reference in release message
4. Help next swarm continue

## Related Commands

- `/github claim <number>` - Claim a different task
- `/github tasks` - Find new tasks
- `/github update <number>` - Update before releasing
- `/github status` - Check coordination status