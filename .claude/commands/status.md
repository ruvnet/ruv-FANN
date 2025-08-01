# Monitor Swarm Status

## MCP Tool Usage in Claude Code

### Basic Status Monitoring

**Tool:** `mcp__ruv-swarm-zen__swarm_status`

## Parameters
```json
{
  "verbose": false  // Include detailed agent information (optional)
}
```

**Example:**
- Tool: `mcp__ruv-swarm-zen__swarm_status`
- Parameters: `{"verbose": true}`

### Memory Usage Monitoring

**Tool:** `mcp__ruv-swarm-zen__memory_usage`

## Parameters
```json
{
  "detail": "summary"  // summary, detailed, by-agent
}
```

**Example:**
- Tool: `mcp__ruv-swarm-zen__memory_usage`
- Parameters: `{"detail": "detailed"}`

### Neural Network Status

**Tool:** `mcp__ruv-swarm-zen__neural_status`

## Parameters
```json
{
  "agentId": "agent-123"  // Specific agent ID (optional)
}
```

**Example:**
- Tool: `mcp__ruv-swarm-zen__neural_status`
- Parameters: `{}` (for all agents)

### Real-time Monitoring

**Tool:** `mcp__ruv-swarm-zen__swarm_monitor`

## Parameters
```json
{
  "duration": 10,  // Monitoring duration in seconds
  "interval": 1    // Update interval in seconds
}
```

**Example:**
- Tool: `mcp__ruv-swarm-zen__swarm_monitor`
- Parameters: `{"duration": 30, "interval": 2}`

## Status Information Provided
- Active agents and their current states
- Task queue and execution progress
- Memory usage and optimization metrics
- Neural network performance statistics
- Swarm topology and agent connectivity
- Performance benchmarks and metrics
