# ruv-swarm API Reference

Complete API documentation for the ruv-swarm JavaScript/TypeScript library.

## Table of Contents
- [RuvSwarm Class](#ruvswarm-class)
- [Swarm Class](#swarm-class)
- [Agent Class](#agent-class)
- [NeuralAgentFactory](#neuralagentfactory)
- [SwarmPersistence](#swarmpersistence)
- [Types and Interfaces](#types-and-interfaces)
- [Events](#events)
- [Error Handling](#error-handling)

## RuvSwarm Class

The main entry point for interacting with ruv-swarm.

### Static Methods

#### `RuvSwarm.initialize(options?)`
Initialize the RuvSwarm module with WebAssembly.

**Parameters:**
- `options` (object, optional):
  - `wasmPath` (string): Path to WASM files (default: './wasm')
  - `useSIMD` (boolean): Enable SIMD optimizations if available
  - `debug` (boolean): Enable debug logging

**Returns:** `Promise<RuvSwarm>`

**Example:**
```javascript
const ruvSwarm = await RuvSwarm.initialize({
  wasmPath: './node_modules/ruv-swarm/wasm',
  useSIMD: true,
  debug: false
});
```

#### `RuvSwarm.detectSIMDSupport()`
Check if SIMD is supported in the current environment.

**Returns:** `boolean`

#### `RuvSwarm.getVersion()`
Get the library version.

**Returns:** `string`

#### `RuvSwarm.getMemoryUsage()`
Get current WASM memory usage in bytes.

**Returns:** `number`

#### `RuvSwarm.getRuntimeFeatures()`
Get runtime feature detection results.

**Returns:** `object`

### Instance Methods

#### `createSwarm(config)`
Create a new swarm instance.

**Parameters:**
- `config` (object):
  - `name` (string): Swarm name
  - `strategy` (string): Distribution strategy
  - `mode` (string): Coordination mode
  - `maxAgents` (number): Maximum agents allowed

**Returns:** `Promise<Swarm>`

**Example:**
```javascript
const swarm = await ruvSwarm.createSwarm({
  name: 'my-swarm',
  strategy: 'balanced',
  mode: 'distributed',
  maxAgents: 10
});
```

## Swarm Class

Represents an active swarm of agents.

### Properties

- `id` (string): Unique swarm identifier
- `name` (string): Swarm name
- `strategy` (string): Distribution strategy
- `mode` (string): Coordination mode
- `maxAgents` (number): Maximum agents allowed

### Methods

#### `spawn(config)`
Spawn a new agent in the swarm.

**Parameters:**
- `config` (object):
  - `name` (string): Agent name
  - `type` (string): Agent type
  - `capabilities` (string[]): Agent capabilities

**Returns:** `Promise<Agent>`

**Example:**
```javascript
const agent = await swarm.spawn({
  name: 'researcher-1',
  type: 'researcher',
  capabilities: ['data_analysis', 'web_search']
});
```

#### `orchestrate(task)`
Orchestrate a task across the swarm.

**Parameters:**
- `task` (object):
  - `id` (string): Task identifier
  - `description` (string): Task description
  - `priority` (string): Priority level
  - `dependencies` (string[]): Task dependencies
  - `metadata` (object): Additional metadata

**Returns:** `Promise<OrchestrationResult>`

**Example:**
```javascript
const result = await swarm.orchestrate({
  id: 'task-1',
  description: 'Analyze codebase for security vulnerabilities',
  priority: 'high',
  dependencies: [],
  metadata: {
    timeout: 30000,
    requiredAgents: ['researcher', 'analyst']
  }
});
```

#### `getStatus()`
Get current swarm status.

**Returns:** `SwarmStatus`

#### `getAgents()`
Get all agents in the swarm.

**Returns:** `Agent[]`

#### `terminate()`
Terminate the swarm and all agents.

**Returns:** `Promise<void>`

## Agent Class

Represents an individual agent in the swarm.

### Properties

- `id` (string): Unique agent identifier
- `name` (string): Agent name
- `type` (string): Agent type
- `status` (string): Current status ('idle', 'busy', 'offline')
- `capabilities` (string[]): Agent capabilities
- `neuralNetwork` (NeuralNetwork): Associated neural network

### Methods

#### `execute(task)`
Execute a task on this agent.

**Parameters:**
- `task` (object):
  - `description` (string): Task description
  - `parameters` (object): Task parameters

**Returns:** `Promise<ExecutionResult>`

**Example:**
```javascript
const result = await agent.execute({
  description: 'Search for security vulnerabilities',
  parameters: {
    scanType: 'deep',
    reportFormat: 'json'
  }
});
```

#### `getMetrics()`
Get agent performance metrics.

**Returns:** `AgentMetrics`

#### `setStatus(status)`
Update agent status.

**Parameters:**
- `status` (string): New status ('idle', 'busy', 'offline')

#### `terminate()`
Terminate this agent.

**Returns:** `Promise<void>`

## NeuralAgentFactory

Factory for creating neural-enhanced agents.

### Static Methods

#### `createNeuralAgent(baseAgent, agentType)`
Enhance a base agent with neural capabilities.

**Parameters:**
- `baseAgent` (object): Base agent configuration
- `agentType` (string): Agent type

**Returns:** `NeuralAgent`

**Example:**
```javascript
const neuralAgent = NeuralAgentFactory.createNeuralAgent(
  { id: 'agent-1', name: 'researcher-1' },
  'researcher'
);
```

#### `getCognitiveProfiles()`
Get available cognitive profiles.

**Returns:** `object`

### Cognitive Patterns

```javascript
const COGNITIVE_PATTERNS = {
  CONVERGENT: {
    name: 'Convergent Thinking',
    focus: 'optimization',
    traits: ['analytical', 'systematic', 'efficiency-focused']
  },
  DIVERGENT: {
    name: 'Divergent Thinking',
    focus: 'exploration',
    traits: ['creative', 'brainstorming', 'alternative-solutions']
  },
  LATERAL: {
    name: 'Lateral Thinking',
    focus: 'innovation',
    traits: ['cross-domain', 'analogical', 'reframing']
  },
  SYSTEMS: {
    name: 'Systems Thinking',
    focus: 'holistic',
    traits: ['interconnections', 'emergence', 'feedback-loops']
  },
  CRITICAL: {
    name: 'Critical Thinking',
    focus: 'analysis',
    traits: ['questioning', 'evidence-based', 'logical']
  }
};
```

## SwarmPersistence

SQLite-based persistence layer.

### Constructor

```javascript
const persistence = new SwarmPersistence(dbPath);
```

**Parameters:**
- `dbPath` (string): Path to SQLite database file

### Methods

#### Swarm Operations

##### `createSwarm(swarm)`
Create a new swarm record.

**Parameters:**
- `swarm` (object): Swarm configuration

**Returns:** `object`

##### `getActiveSwarms()`
Get all active swarms.

**Returns:** `Swarm[]`

#### Agent Operations

##### `createAgent(agent)`
Create a new agent record.

**Parameters:**
- `agent` (object): Agent configuration

**Returns:** `object`

##### `updateAgentStatus(agentId, status)`
Update agent status.

**Parameters:**
- `agentId` (string): Agent ID
- `status` (string): New status

##### `getAgent(id)`
Get agent by ID.

**Parameters:**
- `id` (string): Agent ID

**Returns:** `Agent | null`

##### `getSwarmAgents(swarmId, filter?)`
Get all agents in a swarm.

**Parameters:**
- `swarmId` (string): Swarm ID
- `filter` (string, optional): Status filter

**Returns:** `Agent[]`

#### Task Operations

##### `createTask(task)`
Create a new task record.

**Parameters:**
- `task` (object): Task configuration

**Returns:** `object`

##### `updateTask(taskId, updates)`
Update task information.

**Parameters:**
- `taskId` (string): Task ID
- `updates` (object): Fields to update

##### `getTask(id)`
Get task by ID.

**Parameters:**
- `id` (string): Task ID

**Returns:** `Task | null`

##### `getSwarmTasks(swarmId, status?)`
Get all tasks in a swarm.

**Parameters:**
- `swarmId` (string): Swarm ID
- `status` (string, optional): Status filter

**Returns:** `Task[]`

#### Memory Operations

##### `storeAgentMemory(agentId, key, value)`
Store agent memory.

**Parameters:**
- `agentId` (string): Agent ID
- `key` (string): Memory key
- `value` (any): Value to store

##### `getAgentMemory(agentId, key?)`
Retrieve agent memory.

**Parameters:**
- `agentId` (string): Agent ID
- `key` (string, optional): Specific key

**Returns:** `any`

#### Neural Network Operations

##### `storeNeuralNetwork(network)`
Store neural network configuration.

**Parameters:**
- `network` (object): Neural network data

**Returns:** `object`

##### `getAgentNeuralNetworks(agentId)`
Get neural networks for an agent.

**Parameters:**
- `agentId` (string): Agent ID

**Returns:** `NeuralNetwork[]`

#### Event Logging

##### `logEvent(swarmId, eventType, eventData)`
Log a swarm event.

**Parameters:**
- `swarmId` (string): Swarm ID
- `eventType` (string): Event type
- `eventData` (object): Event data

##### `getSwarmEvents(swarmId, limit?)`
Get swarm events.

**Parameters:**
- `swarmId` (string): Swarm ID
- `limit` (number, optional): Maximum events to return

**Returns:** `Event[]`

## Types and Interfaces

### SwarmConfig
```typescript
interface SwarmConfig {
  name: string;
  strategy: 'balanced' | 'specialized' | 'adaptive';
  mode: 'centralized' | 'distributed' | 'hierarchical';
  maxAgents: number;
  metadata?: Record<string, any>;
}
```

### AgentConfig
```typescript
interface AgentConfig {
  name: string;
  type: 'researcher' | 'coder' | 'analyst' | 'optimizer' | 'coordinator';
  capabilities: string[];
  neuralConfig?: NeuralConfig;
}
```

### Task
```typescript
interface Task {
  id: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  dependencies: string[];
  metadata?: Record<string, any>;
}
```

### OrchestrationResult
```typescript
interface OrchestrationResult {
  status: 'success' | 'partial' | 'failed';
  results: AgentResult[];
  metrics: {
    totalTime: number;
    agentsSpawned: number;
    tasksCompleted: number;
  };
}
```

### AgentResult
```typescript
interface AgentResult {
  agentId: string;
  agentType: string;
  status: 'completed' | 'failed' | 'timeout';
  output: any;
  executionTime: number;
  error?: string;
}
```

### NeuralConfig
```typescript
interface NeuralConfig {
  architecture: {
    input_size: number;
    hidden_layers: number[];
    output_size: number;
    activation: string;
    optimizer: string;
  };
  weights?: number[][];
  trainingData?: any;
}
```

### SwarmStatus
```typescript
interface SwarmStatus {
  id: string;
  name: string;
  agents: {
    total: number;
    active: number;
    idle: number;
    offline: number;
  };
  tasks: {
    total: number;
    completed: number;
    failed: number;
    pending: number;
  };
  uptime: number;
  memory: {
    used: number;
    total: number;
  };
}
```

### AgentMetrics
```typescript
interface AgentMetrics {
  tasksCompleted: number;
  tasksFailed: number;
  averageExecutionTime: number;
  successRate: number;
  lastActive: Date;
  memoryUsage: number;
  cpuUsage: number;
}
```

## Events

The swarm emits various events that can be subscribed to:

### Swarm Events
- `swarm:initialized` - Swarm successfully initialized
- `swarm:terminated` - Swarm terminated
- `swarm:error` - Swarm error occurred

### Agent Events
- `agent:spawned` - New agent spawned
- `agent:terminated` - Agent terminated
- `agent:status_changed` - Agent status changed
- `agent:error` - Agent error occurred

### Task Events
- `task:created` - New task created
- `task:assigned` - Task assigned to agents
- `task:completed` - Task completed
- `task:failed` - Task failed

### Example Event Handling
```javascript
swarm.on('agent:spawned', (agent) => {
  console.log(`New agent spawned: ${agent.name}`);
});

swarm.on('task:completed', (result) => {
  console.log(`Task completed: ${result.taskId}`);
});

swarm.on('swarm:error', (error) => {
  console.error(`Swarm error: ${error.message}`);
});
```

## Error Handling

### Error Types

#### `SwarmError`
Base error class for swarm-related errors.

```javascript
class SwarmError extends Error {
  constructor(message, code) {
    super(message);
    this.code = code;
  }
}
```

#### `AgentError`
Agent-specific errors.

```javascript
class AgentError extends SwarmError {
  constructor(message, agentId) {
    super(message, 'AGENT_ERROR');
    this.agentId = agentId;
  }
}
```

#### `TaskError`
Task execution errors.

```javascript
class TaskError extends SwarmError {
  constructor(message, taskId) {
    super(message, 'TASK_ERROR');
    this.taskId = taskId;
  }
}
```

### Error Handling Example
```javascript
try {
  const result = await swarm.orchestrate({
    id: 'task-1',
    description: 'Complex task',
    priority: 'high'
  });
} catch (error) {
  if (error instanceof TaskError) {
    console.error(`Task ${error.taskId} failed: ${error.message}`);
  } else if (error instanceof AgentError) {
    console.error(`Agent ${error.agentId} error: ${error.message}`);
  } else {
    console.error(`Unexpected error: ${error.message}`);
  }
}
```

## Best Practices

### 1. Resource Management
```javascript
// Always clean up resources
const swarm = await ruvSwarm.createSwarm(config);
try {
  // Use swarm
} finally {
  await swarm.terminate();
}
```

### 2. Error Handling
```javascript
// Wrap operations in try-catch
try {
  const agent = await swarm.spawn(agentConfig);
  const result = await agent.execute(task);
} catch (error) {
  logger.error('Operation failed:', error);
  // Handle gracefully
}
```

### 3. Performance Optimization
```javascript
// Batch operations when possible
const agents = await Promise.all([
  swarm.spawn({ type: 'researcher' }),
  swarm.spawn({ type: 'coder' }),
  swarm.spawn({ type: 'analyst' })
]);

// Use appropriate agent types
const task = {
  description: 'Analyze data patterns',
  requiredAgentType: 'analyst' // Match task to agent type
};
```

### 4. Memory Management
```javascript
// Monitor memory usage
setInterval(async () => {
  const memory = RuvSwarm.getMemoryUsage();
  if (memory > threshold) {
    // Take action (reduce agents, cleanup, etc.)
  }
}, 60000);

// Clean up old data
persistence.cleanup();
```

## Examples

### Complete Example: Building a Research System
```javascript
import { RuvSwarm, SwarmPersistence } from 'ruv-swarm';

async function buildResearchSystem() {
  // Initialize
  const ruvSwarm = await RuvSwarm.initialize({
    useSIMD: true,
    debug: true
  });
  
  const persistence = new SwarmPersistence('./research.db');
  
  // Create swarm
  const swarm = await ruvSwarm.createSwarm({
    name: 'research-swarm',
    strategy: 'specialized',
    mode: 'distributed',
    maxAgents: 20
  });
  
  // Spawn research team
  const researchers = await Promise.all([
    swarm.spawn({ type: 'researcher', name: 'web-researcher' }),
    swarm.spawn({ type: 'researcher', name: 'data-researcher' }),
    swarm.spawn({ type: 'analyst', name: 'data-analyst' }),
    swarm.spawn({ type: 'coordinator', name: 'team-lead' })
  ]);
  
  // Define research task
  const researchTask = {
    id: 'research-001',
    description: 'Research AI trends in 2025',
    priority: 'high',
    metadata: {
      topics: ['machine learning', 'neural networks', 'AGI'],
      depth: 'comprehensive',
      outputFormat: 'report'
    }
  };
  
  // Execute research
  const result = await swarm.orchestrate(researchTask);
  
  // Store results
  persistence.createTask({
    id: researchTask.id,
    swarmId: swarm.id,
    description: researchTask.description,
    status: 'completed',
    result: result
  });
  
  // Get insights
  const events = persistence.getSwarmEvents(swarm.id);
  const metrics = await Promise.all(
    researchers.map(r => r.getMetrics())
  );
  
  console.log('Research completed:', {
    result,
    events,
    metrics
  });
  
  // Cleanup
  await swarm.terminate();
  persistence.close();
}

// Run the system
buildResearchSystem().catch(console.error);
```

## TypeScript Support

ruv-swarm includes full TypeScript definitions. Import types:

```typescript
import { 
  RuvSwarm, 
  SwarmConfig, 
  AgentConfig, 
  Task,
  OrchestrationResult,
  SwarmStatus,
  AgentMetrics
} from 'ruv-swarm';

// Use with type safety
const config: SwarmConfig = {
  name: 'typed-swarm',
  strategy: 'balanced',
  mode: 'distributed',
  maxAgents: 10
};

const swarm = await ruvSwarm.createSwarm(config);
```

## Migration Guide

### From v0.0.x to v0.1.0
- SQLite persistence is now default (previously in-memory)
- Neural networks are automatically created for each agent
- MCP support is built-in
- API changes:
  - `Swarm.create()` → `RuvSwarm.createSwarm()`
  - `Agent.spawn()` → `swarm.spawn()`
  - `Task.execute()` → `swarm.orchestrate()`

## Support

For issues or questions:
- GitHub Issues: [https://github.com/ruvnet/ruv-FANN/issues](https://github.com/ruvnet/ruv-FANN/issues)
- Documentation: [https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm/docs](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm/docs)
- NPM: [https://www.npmjs.com/package/ruv-swarm](https://www.npmjs.com/package/ruv-swarm)