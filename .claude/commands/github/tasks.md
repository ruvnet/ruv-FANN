# /github tasks

Lists all available tasks that can be claimed by swarms, sorted by priority.

## Usage

```
/github tasks [options]
```

## Options

- `--priority <level>` - Filter by priority (critical, high, medium, low)
- `--area <component>` - Filter by area (core, mcp, neural, wasm, docs)
- `--type <type>` - Filter by type (bug, feature, task, enhancement)
- `--limit <number>` - Limit results (default: 20)

## Examples

```
/github tasks                    # All available tasks
/github tasks --priority high    # Only high priority tasks
/github tasks --area neural      # Neural network related tasks
/github tasks --type bug         # Only bug fixes
```

## Implementation

Uses GitHub MCP to:
1. Query issues with `available` label
2. Exclude issues with `swarm-claimed` label
3. Sort by priority and creation date
4. Format for easy reading

## Example Output

```
📋 Available Tasks (12 total)

🔴 CRITICAL (2):
#62 [BUG] Memory leak causing crashes after 1 hour
    Labels: bug, area: neural, priority: critical
    Created: 2 hours ago

#61 [BUG] WASM module fails to load in production
    Labels: bug, area: wasm, priority: critical
    Created: 5 hours ago

🟠 HIGH (5):
#58 [FEATURE] Implement persistent memory storage
    Labels: feature, area: core, priority: high
    Created: 1 day ago

#57 [TASK] Add comprehensive integration tests
    Labels: task, area: tests, priority: high
    Created: 1 day ago

[... more tasks ...]

To claim a task: /github claim <number>
```

## Task Format

Each task shows:
- Issue number and title
- Type indicator ([BUG], [FEATURE], [TASK])
- Labels for area and priority
- Age of the issue
- Brief description if available

## Related Commands

- `/github claim <number>` - Claim a specific task
- `/github status` - See overall coordination status
- `/github my-tasks` - Show your current tasks