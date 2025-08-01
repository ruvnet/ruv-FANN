# MCP Resources in ruv-swarm

## Overview

The ruv-swarm MCP server now exposes 10 resources to provide documentation, examples, schemas, and performance data directly within Claude Code. These resources complement the 25 available tools by providing reference material and guidance.

## Available Resources

### Documentation Resources

1. **Getting Started Guide** (`swarm://docs/getting-started`)
   - Introduction to ruv-swarm concepts
   - Quick start examples
   - Key concepts and best practices

2. **Swarm Topologies** (`swarm://docs/topologies`)
   - Detailed explanation of mesh, hierarchical, ring, and star topologies
   - When to use each topology
   - Performance characteristics

3. **Agent Types Guide** (`swarm://docs/agent-types`)
   - All 7 agent types with their cognitive patterns
   - Capabilities and best use cases
   - How agents complement each other

4. **DAA Integration Guide** (`swarm://docs/daa-guide`)
   - Decentralized Autonomous Agents features
   - How to use DAA tools effectively
   - Learning and adaptation strategies

### Example Resources

5. **REST API Example** (`swarm://examples/rest-api`)
   - Complete example of building a REST API
   - Shows parallel agent coordination
   - Demonstrates best practices

6. **Neural Training Example** (`swarm://examples/neural-training`)
   - How to train neural agents
   - Monitoring and optimization
   - Advanced training techniques

### Schema Resources

7. **Swarm Configuration Schema** (`swarm://schemas/swarm-config`)
   - JSON schema for swarm configuration
   - All available options and defaults
   - Validation rules

8. **Agent Configuration Schema** (`swarm://schemas/agent-config`)
   - JSON schema for agent configuration
   - Properties and constraints
   - Cognitive pattern options

### Performance Resources

9. **Performance Benchmarks** (`swarm://performance/benchmarks`)
   - Latest performance metrics
   - Comparison against targets
   - SWE-Bench results

### Reference Resources

10. **Available Hooks** (`swarm://hooks/available`)
    - All Claude Code hooks
    - Usage examples
    - Features and capabilities

## Using Resources in Claude Code

Resources can be accessed through the MCP interface:

```typescript
// List all available resources
const resources = await mcp.resources.list();

// Read a specific resource
const gettingStarted = await mcp.resources.read({
  uri: 'swarm://docs/getting-started'
});
```

## Benefits

1. **In-context Documentation**: Access documentation without leaving Claude Code
2. **Live Examples**: See real working examples with proper syntax
3. **Schema Validation**: Use schemas to validate configurations
4. **Performance Insights**: Monitor actual vs target performance
5. **Quick Reference**: Fast access to hooks and tool documentation

## Implementation Details

The resources are implemented in the MCP server with:
- Proper MIME types for content
- Structured URIs using the `swarm://` protocol
- Dynamic content generation for up-to-date information
- Comprehensive coverage of all major features

## Future Enhancements

Planned resource additions:
- Video tutorials (when MCP supports video resources)
- Interactive examples
- Troubleshooting guides
- Migration guides
- API reference documentation