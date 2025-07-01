# /github swarms

Lists all active swarms working on the project, showing what each is working on and coordination opportunities.

## Usage

```
/github swarms [options]
```

## Options

- `--active` - Only show recently active swarms (default)
- `--all` - Include inactive swarms
- `--conflicts` - Highlight potential conflicts

## Examples

```
/github swarms
/github swarms --all
/github swarms --conflicts
```

## What it shows

- Active swarm IDs
- Current tasks per swarm
- Last activity time
- Work areas/expertise
- Potential collaboration opportunities

## Implementation

Uses GitHub MCP to:
1. Find all issues with swarm claims
2. Group by swarm ID
3. Analyze activity patterns
4. Identify collaboration opportunities

## Example Output

### Basic View
```
🐝 Active Swarms (5)

swarm-alpha-123 (Active now)
├─ Working on: #45 Memory optimization
├─ Last seen: 5 min ago
└─ Focus: Performance, WASM

swarm-beta-456 (Active)
├─ Working on: #52 Neural agent fixes
├─ Last seen: 1 hour ago
└─ Focus: Neural networks, AI

swarm-gamma-789 (Active)
├─ Working on: #48 Documentation
├─ Working on: #49 Examples
├─ Last seen: 30 min ago
└─ Focus: Documentation, Testing

swarm-delta-012 (Idle)
├─ No current tasks
├─ Last seen: 3 hours ago
└─ Focus: Unknown

swarm-epsilon-345 (Inactive)
├─ Stale claim: #38 (2 days old)
├─ Last seen: 2 days ago
└─ Status: Consider releasing task
```

### With Conflicts View
```
🐝 Active Swarms - Conflict Analysis

⚠️ Potential Conflicts Detected:

1. File Overlap:
   swarm-alpha-123 and swarm-beta-456
   Both modifying: src/neural-agent.js
   Suggestion: Coordinate via issue comments

2. Related Tasks:
   swarm-gamma-789 working on #48 (docs)
   swarm-zeta-678 working on #50 (examples)
   Suggestion: Align documentation approach

🤝 Collaboration Opportunities:

- swarm-alpha-123 + swarm-beta-456:
  Both working on performance issues
  Could share benchmarking approach

- swarm-gamma-789 + swarm-delta-012:
  Delta is idle, Gamma has 2 doc tasks
  Could help with documentation
```

## Swarm Patterns

Identifies:
- Multi-taskers (multiple issues)
- Specialists (consistent area focus)
- New swarms (first time seen)
- Veteran swarms (long history)
- Idle swarms (available to help)

## Activity Levels

- **Active now** - Activity in last 15 min
- **Active** - Activity in last 2 hours
- **Idle** - No current tasks
- **Inactive** - No activity in 24+ hours
- **Stale** - Has old claimed tasks

## Coordination Suggestions

The command suggests:
1. When swarms work on related issues
2. When swarms modify same files
3. When idle swarms could help
4. When tasks could be split

## Best Practices

1. Check before starting related work
2. Coordinate with active swarms
3. Offer help to overloaded swarms
4. Release tasks if going inactive
5. Use issue comments for coordination

## Related Commands

- `/github coordinate` - Propose collaboration
- `/github conflicts` - Check for conflicts
- `/github status` - Overall status
- `/github my-tasks` - Your tasks