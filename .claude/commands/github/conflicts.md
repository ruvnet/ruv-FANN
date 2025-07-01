# /github conflicts

Detects and displays potential conflicts between swarms working on the same codebase.

## Usage

```
/github conflicts [options]
```

## Options

- `--scope <type>` - Check specific conflict types (files, tasks, areas)
- `--resolve` - Show resolution suggestions

## Examples

```
/github conflicts
/github conflicts --scope files
/github conflicts --resolve
```

## Conflict Types

1. **File Conflicts** - Multiple swarms editing same files
2. **Task Dependencies** - Blocked or related tasks
3. **Area Overlap** - Same component/module work
4. **Resource Competition** - Same test suites, benchmarks

## Implementation

Uses GitHub MCP to:
1. Analyze current swarm activities
2. Check PR files being modified
3. Compare task descriptions
4. Identify overlapping work

## Example Output

### Basic Conflict Check
```
🔍 Conflict Analysis

⚠️ 2 Potential Conflicts Detected:

1. FILE CONFLICT - High Priority
   Swarms: swarm-alpha-123, swarm-beta-456
   File: src/neural-agent.js
   
   swarm-alpha-123: Optimizing memory usage
   swarm-beta-456: Fixing pattern storage
   
   Risk: Merge conflicts likely
   
2. DEPENDENCY CONFLICT - Medium Priority
   Swarms: swarm-gamma-789, swarm-delta-012
   
   swarm-gamma-789: Working on #52 (needs #51)
   swarm-delta-012: Working on #51 (blocking)
   
   Risk: Gamma blocked until Delta completes

✅ No conflicts: 3 other active swarms
```

### With Resolution Suggestions
```
🔍 Conflict Analysis with Resolutions

⚠️ FILE CONFLICT: src/neural-agent.js

Affected Swarms:
- swarm-alpha-123 (Issue #45)
- swarm-beta-456 (Issue #52)

📋 Suggested Resolution:
1. Comment on both issues about overlap
2. Suggested split:
   - Alpha: Focus on memory optimization methods
   - Beta: Focus on storage implementation
3. Create shared branch for coordination
4. Schedule sync point in 2 hours

💬 Sample Coordination Message:
"Hi @swarm-beta-456, I'm working on memory optimization 
in neural-agent.js. I'll focus on the cleanup methods 
(lines 100-200). Can you work on storage (lines 300-400)?
Let's sync before merging."

[Copy message to clipboard]
```

## Conflict Severity

- 🔴 **High** - Direct file conflicts, will cause merge issues
- 🟠 **Medium** - Dependencies or related work
- 🟡 **Low** - Same area but different files
- 🟢 **None** - No conflicts detected

## Auto-Detection

Monitors for:
- Same file edits in different PRs
- Conflicting task descriptions
- Dependency chains
- Competition for resources

## Resolution Strategies

1. **Communication** - Issue comments
2. **Task Splitting** - Divide work clearly
3. **Sequencing** - Agree on order
4. **Pairing** - Work together
5. **Branching** - Shared feature branch

## Best Practices

1. Check conflicts before starting
2. Communicate early and often
3. Split work at clear boundaries
4. Update plans if conflicts found
5. Use shared branches when needed

## Related Commands

- `/github coordinate` - Propose splits
- `/github swarms` - See who's active
- `/github update` - Communicate plans
- `/github my-tasks` - Your current work