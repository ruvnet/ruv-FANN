# Orchestrate Tasks

## MCP Tool Usage in Claude Code

**Tool:** `mcp__ruv-swarm-zen__task_orchestrate`

## Parameters
```json
{
  "task": "Build REST API with authentication",  // Task description (required)
  "strategy": "parallel",                        // parallel, sequential, adaptive
  "priority": "high",                           // low, medium, high, critical
  "maxAgents": 5                               // Maximum agents to use (optional)
}
```

## Examples

**Research task:**
- Tool: `mcp__ruv-swarm-zen__task_orchestrate`
- Parameters: `{"task": "Research modern web frameworks", "strategy": "adaptive"}`

**Development with parallel strategy:**
- Tool: `mcp__ruv-swarm-zen__task_orchestrate`
- Parameters: `{"task": "Build REST API", "strategy": "parallel", "priority": "high"}`

**Analysis with agent limit:**
- Tool: `mcp__ruv-swarm-zen__task_orchestrate`
- Parameters: `{"task": "Analyze user behavior patterns", "maxAgents": 3, "strategy": "sequential"}`

**Critical deployment task:**
- Tool: `mcp__ruv-swarm-zen__task_orchestrate`
- Parameters: `{"task": "Deploy authentication system", "priority": "critical", "strategy": "adaptive"}`

## Strategies
- **parallel** - Execute subtasks simultaneously for speed
- **sequential** - Execute subtasks in order for dependencies
- **adaptive** - Dynamically choose based on task complexity (default)

## Priorities
- **low** - Background processing, lower resource allocation
- **medium** - Standard priority with balanced resources (default)
- **high** - Expedited processing with increased resources
- **critical** - Immediate attention with maximum resources
