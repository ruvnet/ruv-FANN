# Spawn Agents

## MCP Tool Usage in Claude Code

**Tool:** `mcp__ruv-swarm-zen__agent_spawn`

## Parameters
```json
{
  "type": "researcher",           // Agent type (required)
  "name": "AI Research Specialist", // Custom name (optional)
  "capabilities": ["analysis"],    // Additional capabilities (optional)
  "enableNeuralNetwork": true      // Neural capabilities (default: true)
}
```

## Agent Types
- **researcher** - Research and analysis tasks, literature review
- **coder** - Code generation, development, implementation
- **analyst** - Data analysis, insights, pattern recognition
- **architect** - System design, planning, architecture decisions
- **reviewer** - Code review, quality assurance, validation
- **optimizer** - Performance optimization, efficiency improvements
- **coordinator** - Team coordination, project management

## Examples

**Spawn research agent:**
- Tool: `mcp__ruv-swarm-zen__agent_spawn`
- Parameters: `{"type": "researcher", "name": "AI Research Specialist"}`

**Spawn development team:**
- Tool: `mcp__ruv-swarm-zen__agent_spawn` (call multiple times)
- Frontend: `{"type": "coder", "name": "Frontend Developer", "capabilities": ["react", "typescript"]}`
- Backend: `{"type": "coder", "name": "Backend Developer", "capabilities": ["python", "fastapi"]}`

**Spawn analyst:**
- Tool: `mcp__ruv-swarm-zen__agent_spawn`
- Parameters: `{"type": "analyst", "name": "Data Scientist", "capabilities": ["ml", "statistics"]}`
