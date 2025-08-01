# ruv-swarm MCP Usage Guide

This guide provides comprehensive documentation for using ruv-swarm with Model Context Protocol (MCP) in Claude Code and other MCP-enabled tools.

## Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [MCP Tools Reference](#mcp-tools-reference)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Installation

### Quick Start with NPX
```bash
# No installation needed - run directly
npx ruv-swarm mcp start --protocol=stdio
```

### Global Installation
```bash
npm install -g ruv-swarm
ruv-swarm mcp start --protocol=stdio
```

## Configuration

### Claude Code Configuration

Create or edit `.claude/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "ruv-swarm": {
      "command": "npx",
      "args": ["ruv-swarm", "mcp", "start", "--protocol=stdio"],
      "capabilities": {
        "tools": true
      },
      "metadata": {
        "name": "ruv-swarm",
        "version": "0.1.0",
        "description": "Distributed agent orchestration with neural networks"
      }
    }
  }
}
```

### Alternative Configuration (Script Wrapper)

For better control, create `mcp-server.sh`:
```bash
#!/bin/bash
cd /path/to/your/project
exec npx ruv-swarm mcp start --protocol=stdio
```

Then update `.claude/mcp.json`:
```json
{
  "mcpServers": {
    "ruv-swarm": {
      "command": "/path/to/mcp-server.sh",
      "capabilities": {
        "tools": true
      }
    }
  }
}
```

## MCP Tools Reference

### 1. swarm_init
Initialize a new swarm with specified topology and configuration.

**Parameters:**
- `topology` (string): Network topology - "mesh", "star", "hierarchical", "ring"
- `maxAgents` (number): Maximum number of agents (1-100, default: 5)
- `strategy` (string): Distribution strategy - "balanced", "specialized", "adaptive"

**Example:**
```typescript
await swarm_init({
  topology: "mesh",
  maxAgents: 10,
  strategy: "balanced"
});
```

**Response:**
```json
{
  "id": "swarm_1234567890_abc",
  "message": "Successfully initialized mesh swarm",
  "topology": "mesh",
  "strategy": "balanced",
  "maxAgents": 10,
  "created": "2025-06-29T12:00:00.000Z",
  "features": {
    "wasm_enabled": true,
    "simd_support": false,
    "runtime_features": {}
  }
}
```

### 2. swarm_status
Get current status of all active swarms.

**Parameters:**
- `verbose` (boolean): Include detailed agent information (default: false)

**Example:**
```typescript
await swarm_status({ verbose: true });
```

**Response:**
```json
{
  "active_swarms": 1,
  "swarms": [{
    "id": "swarm_1234567890_abc",
    "name": "mesh-swarm-1234567890",
    "topology": "mesh",
    "agents": {
      "total": 5,
      "active": 3,
      "idle": 2,
      "max": 10
    },
    "tasks": {
      "total": 15,
      "completed": 12,
      "success_rate": "80.0%"
    },
    "uptime": "45.2 minutes"
  }]
}
```

### 3. swarm_monitor
Monitor swarm activity in real-time.

**Parameters:**
- `duration` (number): Monitoring duration in seconds (default: 10)
- `interval` (number): Update interval in seconds (default: 1)

**Example:**
```typescript
await swarm_monitor({ 
  duration: 30, 
  interval: 5 
});
```

### 4. agent_spawn
Create a new agent in the swarm.

**Parameters:**
- `type` (string): Agent type - "researcher", "coder", "analyst", "optimizer", "coordinator"
- `name` (string): Custom agent name (optional)
- `capabilities` (array): Additional capabilities

**Example:**
```typescript
await agent_spawn({
  type: "researcher",
  name: "data-researcher-1",
  capabilities: ["web_search", "data_mining"]
});
```

**Response:**
```json
{
  "agent": {
    "id": "agent_1234567890_xyz",
    "name": "data-researcher-1",
    "type": "researcher",
    "status": "idle",
    "capabilities": [
      "data_analysis",
      "pattern_recognition",
      "web_search",
      "data_mining"
    ],
    "neural_network": {
      "id": "nn_1234567890_abc",
      "architecture": {
        "input_size": 10,
        "hidden_layers": [64, 32],
        "output_size": 5
      }
    }
  },
  "message": "Successfully spawned researcher agent",
  "swarm_capacity": "6/10"
}
```

### 5. agent_list
List all agents in the swarm.

**Parameters:**
- `filter` (string): Filter by status - "all", "active", "idle", "busy"

**Example:**
```typescript
await agent_list({ filter: "active" });
```

### 6. agent_metrics
Get performance metrics for agents.

**Parameters:**
- `agentId` (string): Specific agent ID (optional)
- `metric` (string): Metric type - "all", "cpu", "memory", "tasks", "performance"

**Example:**
```typescript
await agent_metrics({ 
  metric: "performance" 
});
```

### 7. task_orchestrate
Orchestrate a task across the swarm.

**Parameters:**
- `task` (string): Task description
- `priority` (string): Priority level - "low", "medium", "high", "critical"
- `strategy` (string): Execution strategy - "parallel", "sequential", "adaptive"
- `maxAgents` (number): Maximum agents to use

**Example:**
```typescript
await task_orchestrate({
  task: "Analyze system performance and generate optimization report",
  priority: "high",
  strategy: "adaptive",
  maxAgents: 3
});
```

**Response:**
```json
{
  "taskId": "task_1234567890_abc",
  "status": "orchestrated",
  "priority": "high",
  "strategy": "adaptive",
  "executionTime": 523,
  "agentsUsed": 3,
  "assignedAgents": [
    "agent_1234567890_xyz",
    "agent_1234567890_def",
    "agent_1234567890_ghi"
  ],
  "summary": "Task successfully orchestrated across 3 agents"
}
```

### 8. task_status
Check the status of running tasks.

**Parameters:**
- `taskId` (string): Specific task ID (optional)
- `detailed` (boolean): Include detailed progress

**Example:**
```typescript
await task_status({ 
  taskId: "task_1234567890_abc",
  detailed: true 
});
```

### 9. task_results
Retrieve results from completed tasks.

**Parameters:**
- `taskId` (string): Task ID to retrieve results for
- `format` (string): Result format - "summary", "detailed", "raw"

**Example:**
```typescript
await task_results({
  taskId: "task_1234567890_abc",
  format: "detailed"
});
```

### 10. benchmark_run
Execute performance benchmarks.

**Parameters:**
- `type` (string): Benchmark type - "all", "wasm", "swarm", "agent", "task"
- `iterations` (number): Number of iterations (1-100, default: 10)

**Example:**
```typescript
await benchmark_run({
  type: "agent",
  iterations: 20
});
```

### 11. features_detect
Detect runtime features and capabilities.

**Parameters:**
- `category` (string): Feature category - "all", "wasm", "simd", "memory", "platform"

**Example:**
```typescript
await features_detect({ category: "all" });
```

### 12. memory_usage
Get current memory usage statistics.

**Parameters:**
- `detail` (string): Detail level - "summary", "detailed", "by-agent"

**Example:**
```typescript
await memory_usage({ detail: "by-agent" });
```

## Usage Examples

### Example 1: Complete Workflow
```typescript
// 1. Initialize swarm
const swarm = await swarm_init({
  topology: "mesh",
  maxAgents: 10,
  strategy: "balanced"
});

// 2. Spawn specialized agents
const agents = await Promise.all([
  agent_spawn({ type: "researcher", name: "research-1" }),
  agent_spawn({ type: "coder", name: "coder-1" }),
  agent_spawn({ type: "analyst", name: "analyst-1" })
]);

// 3. Orchestrate task
const task = await task_orchestrate({
  task: "Build authentication system with JWT tokens",
  priority: "high",
  strategy: "adaptive",
  maxAgents: 3
});

// 4. Monitor progress
await swarm_monitor({ duration: 30, interval: 5 });

// 5. Get results
const results = await task_results({
  taskId: task.taskId,
  format: "detailed"
});
```

### Example 2: Performance Analysis
```typescript
// Run benchmarks
const benchmarks = await benchmark_run({
  type: "all",
  iterations: 50
});

// Get memory usage
const memory = await memory_usage({
  detail: "by-agent"
});

// Get agent metrics
const metrics = await agent_metrics({
  metric: "all"
});
```

### Example 3: Real-time Monitoring
```typescript
// Start monitoring
const monitoring = await swarm_monitor({
  duration: 300, // 5 minutes
  interval: 10   // Update every 10 seconds
});

// Check swarm status periodically
setInterval(async () => {
  const status = await swarm_status({ verbose: true });
  console.log(`Active agents: ${status.swarms[0].agents.active}`);
}, 30000);
```

## Best Practices

### 1. Swarm Initialization
- Start with smaller swarms (5-10 agents) and scale as needed
- Choose topology based on task requirements:
  - **Mesh**: Best for collaborative tasks
  - **Star**: Best for centralized coordination
  - **Hierarchical**: Best for complex workflows

### 2. Agent Management
- Spawn agents based on task requirements
- Monitor agent performance and adjust capacity
- Use specialized agents for specific tasks

### 3. Task Orchestration
- Set appropriate priorities for tasks
- Use adaptive strategy for complex tasks
- Monitor task progress and handle failures

### 4. Performance Optimization
- Run benchmarks to identify bottlenecks
- Monitor memory usage regularly
- Adjust agent count based on workload

## Troubleshooting

### Common Issues

#### 1. MCP Server Not Starting
```bash
# Check if port is in use
lsof -i :3000

# Kill existing processes
pkill -f "ruv-swarm mcp"

# Restart with debug mode
npx ruv-swarm mcp start --protocol=stdio --debug
```

#### 2. Agent Spawn Failures
- Check swarm capacity with `swarm_status`
- Ensure swarm is initialized with `swarm_init`
- Verify agent type is valid

#### 3. Task Orchestration Errors
- Ensure agents are available
- Check task syntax and parameters
- Monitor swarm health with `swarm_monitor`

#### 4. Memory Issues
- Monitor with `memory_usage` tool
- Reduce agent count if needed
- Clear old tasks and data periodically

### Debug Mode
Enable debug logging:
```bash
export RUV_SWARM_DEBUG=true
npx ruv-swarm mcp start --protocol=stdio --debug
```

### Log Files
Check logs at:
- `./data/ruv-swarm.log` - General logs
- `./data/ruv-swarm.db` - SQLite database
- `stderr` output for MCP errors

## Advanced Configuration

### Environment Variables
```bash
# Maximum agents per swarm
export RUV_SWARM_MAX_AGENTS=50

# Database location
export RUV_SWARM_DB_PATH=./custom/path/swarm.db

# Enable SIMD optimizations
export RUV_SWARM_USE_SIMD=true

# Debug mode
export RUV_SWARM_DEBUG=true
```

### Custom Neural Network Configuration
```javascript
{
  "neural_config": {
    "architecture": "cascade",
    "learning_rate": 0.01,
    "momentum": 0.9,
    "hidden_layers": [128, 64, 32],
    "activation": "relu",
    "optimizer": "adam"
  }
}
```

### Performance Tuning
```javascript
{
  "performance": {
    "batch_size": 32,
    "max_concurrent_tasks": 10,
    "agent_timeout": 30000,
    "memory_limit": "512MB",
    "cpu_threshold": 0.8
  }
}
```

## Integration Examples

### With Claude Code
```javascript
// In Claude Code, use the tools directly
const swarm = await mcp.tools.ruv_swarm.swarm_init({
  topology: "mesh",
  maxAgents: 10
});

const agents = await mcp.tools.ruv_swarm.agent_list({
  filter: "all"
});
```

### With Custom Scripts
```javascript
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

async function callMcpTool(tool, params) {
  const request = {
    jsonrpc: "2.0",
    method: "tools/call",
    params: {
      name: tool,
      arguments: params
    },
    id: Date.now()
  };
  
  const { stdout } = await execAsync(
    `echo '${JSON.stringify(request)}' | npx ruv-swarm mcp start --protocol=stdio`
  );
  
  return JSON.parse(stdout);
}

// Use the tool
const result = await callMcpTool('swarm_init', {
  topology: 'mesh',
  maxAgents: 5
});
```

## Support

For issues, questions, or contributions:
- GitHub: [https://github.com/ruvnet/ruv-FANN](https://github.com/ruvnet/ruv-FANN)
- Issues: [https://github.com/ruvnet/ruv-FANN/issues](https://github.com/ruvnet/ruv-FANN/issues)
- NPM: [https://www.npmjs.com/package/ruv-swarm](https://www.npmjs.com/package/ruv-swarm)