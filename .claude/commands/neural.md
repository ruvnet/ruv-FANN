# Neural Network Management

## MCP Tool Usage in Claude Code

### Neural Status Monitoring

**Tool:** `mcp__ruv-swarm-zen__neural_status`

## Parameters
```json
{
  "agentId": "agent-123"  // Specific agent ID (optional, leave empty for all)
}
```

**Examples:**
- Tool: `mcp__ruv-swarm-zen__neural_status`
- Parameters: `{}` (check all neural agents)
- Parameters: `{"agentId": "researcher-001"}` (specific agent)

### Neural Training

**Tool:** `mcp__ruv-swarm-zen__neural_train`

## Parameters
```json
{
  "agentId": "agent-123",  // Specific agent ID (optional)
  "iterations": 10         // Training iterations (1-100)
}
```

**Examples:**

**Train all neural agents:**
- Tool: `mcp__ruv-swarm-zen__neural_train`
- Parameters: `{"iterations": 20}`

**Train specific agent:**
- Tool: `mcp__ruv-swarm-zen__neural_train`
- Parameters: `{"agentId": "coder-002", "iterations": 15}`

**Intensive training session:**
- Tool: `mcp__ruv-swarm-zen__neural_train`
- Parameters: `{"iterations": 50}`

### Cognitive Pattern Analysis

**Tool:** `mcp__ruv-swarm-zen__neural_patterns`

## Parameters
```json
{
  "pattern": "convergent"  // all, convergent, divergent, lateral, systems, critical, abstract
}
```

**Examples:**

**All cognitive patterns:**
- Tool: `mcp__ruv-swarm-zen__neural_patterns`
- Parameters: `{"pattern": "all"}`

**Convergent thinking patterns:**
- Tool: `mcp__ruv-swarm-zen__neural_patterns`
- Parameters: `{"pattern": "convergent"}`

**Divergent creativity patterns:**
- Tool: `mcp__ruv-swarm-zen__neural_patterns`
- Parameters: `{"pattern": "divergent"}`

**Lateral thinking analysis:**
- Tool: `mcp__ruv-swarm-zen__neural_patterns`
- Parameters: `{"pattern": "lateral"}`

**Systems thinking patterns:**
- Tool: `mcp__ruv-swarm-zen__neural_patterns`
- Parameters: `{"pattern": "systems"}`

**Critical analysis patterns:**
- Tool: `mcp__ruv-swarm-zen__neural_patterns`
- Parameters: `{"pattern": "critical"}`

**Abstract reasoning patterns:**
- Tool: `mcp__ruv-swarm-zen__neural_patterns`
- Parameters: `{"pattern": "abstract"}`

## Cognitive Diversity Benefits
- **Enhanced problem-solving** through multiple thinking styles
- **Reduced bias** via diverse cognitive approaches
- **Improved creativity** with divergent and lateral thinking
- **Better analysis** through convergent and critical patterns
- **Holistic understanding** via systems thinking
- **Complex reasoning** through abstract pattern recognition