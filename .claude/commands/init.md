# Initialize ruv-swarm

## MCP Tool Usage in Claude Code

**Tool:** `mcp__ruv-swarm-zen__swarm_init`

## Parameters
```json
{
  "topology": "mesh",     // mesh, hierarchical, ring, star
  "maxAgents": 5,         // Maximum number of agents
  "strategy": "balanced"  // balanced, specialized, adaptive
}
```

## Examples

**Basic mesh topology:**
- Tool: `mcp__ruv-swarm-zen__swarm_init`
- Parameters: `{"topology": "mesh", "maxAgents": 5}`

**Hierarchical for large projects:**
- Tool: `mcp__ruv-swarm-zen__swarm_init`
- Parameters: `{"topology": "hierarchical", "maxAgents": 10, "strategy": "specialized"}`

**Research swarm:**
- Tool: `mcp__ruv-swarm-zen__swarm_init`
- Parameters: `{"topology": "mesh", "maxAgents": 8, "strategy": "balanced"}`

## Topology Types
- **mesh**: Full connectivity, best for collaboration
- **hierarchical**: Tree structure, best for large projects
- **ring**: Circular coordination, best for sequential tasks
- **star**: Central coordination, best for controlled workflows
