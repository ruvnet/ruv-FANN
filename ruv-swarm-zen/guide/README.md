# ruv-swarm User Guide

Welcome to the comprehensive user guide for ruv-swarm, the cognitive diversity-enabled distributed agent orchestration framework.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Installation Methods](#installation-methods)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Cognitive Patterns](#cognitive-patterns)
7. [Deployment Scenarios](#deployment-scenarios)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Getting Started

### What is ruv-swarm?

ruv-swarm is a distributed agent orchestration framework that enables multiple AI agents to work together using different cognitive patterns. Think of it as a way to create teams of AI agents where each agent thinks differently - some are analytical, others are creative, and some focus on the big picture.

### Key Benefits

- **Cognitive Diversity**: Agents with different thinking patterns solve problems better together
- **Scalability**: Deploy from 2 to 100+ agents across multiple machines
- **Flexibility**: Run natively, in browsers via WebAssembly, or through NPX
- **Integration**: Works seamlessly with Claude-Flow and other AI tools
- **Performance**: Built on high-performance Rust with neural network optimizations

## Core Concepts

### Agents
Agents are autonomous workers that can process tasks. Each agent has:
- **Capabilities**: What the agent can do (e.g., "search", "analyze", "code")
- **Cognitive Pattern**: How the agent thinks (e.g., convergent, divergent)
- **Status**: Current state (idle, working, busy)

### Swarms
A swarm is a collection of agents working together with:
- **Topology**: How agents are connected (mesh, star, hierarchical)
- **Orchestration Strategy**: How tasks are distributed
- **Communication**: How agents share information

### Tasks
Tasks are units of work that can be:
- **Simple**: Single-step operations
- **Complex**: Multi-phase workflows
- **Collaborative**: Requiring multiple agents

## Installation Methods

### Option 1: NPX (Easiest)
```bash
# Use directly without installation
npx @ruv/swarm init --topology mesh

# Install globally
npm install -g @ruv/swarm
ruv-swarm --help
```

### Option 2: Rust Crate
```toml
[dependencies]
ruv-swarm = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

### Option 3: WebAssembly
```html
<script type="module">
  import { RuvSwarm } from 'https://unpkg.com/@ruv/swarm/dist/ruv-swarm.js';
  // Use in browser
</script>
```

## Basic Usage

### Command Line Interface

#### Initialize a Swarm
```bash
# Basic mesh network with 3 agents
npx @ruv/swarm init --topology mesh --agents 3

# Star topology with persistence
npx @ruv/swarm init --topology star --persistence sqlite --db ./myswarm.db

# Hierarchical with custom configuration
npx @ruv/swarm init --config ./swarm-config.yaml
```

#### Spawn Agents
```bash
# Basic worker agent
npx @ruv/swarm spawn worker

# Specialized researcher with specific capabilities
npx @ruv/swarm spawn researcher --capabilities "search,analyze,summarize"

# Agent with specific cognitive pattern
npx @ruv/swarm spawn analyst --cognitive-pattern convergent
```

#### Orchestrate Tasks
```bash
# Simple task
npx @ruv/swarm orchestrate "Research best practices for microservices"

# Complex multi-agent task
npx @ruv/swarm orchestrate --strategy cognitive-diversity \
  "Design, implement, and test a user authentication system"

# Task with specific requirements
npx @ruv/swarm orchestrate --agents researcher,coder,tester \
  --timeline "2 weeks" \
  "Build e-commerce checkout flow"
```

### Rust API

#### Basic Swarm Setup
```rust
use ruv_swarm::{Swarm, SwarmConfig, AgentType, CognitivePattern};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create swarm configuration
    let config = SwarmConfig {
        max_agents: 10,
        topology: ruv_swarm::Topology::Mesh,
        cognitive_patterns: vec![
            CognitivePattern::Convergent,
            CognitivePattern::Divergent,
            CognitivePattern::Lateral,
        ],
        ..Default::default()
    };
    
    // Initialize swarm
    let swarm = Swarm::new(config)?;
    
    // Spawn agents with different patterns
    let researcher = swarm.spawn_agent(
        AgentType::Worker, 
        Some(CognitivePattern::Divergent)
    ).await?;
    
    let analyst = swarm.spawn_agent(
        AgentType::Worker, 
        Some(CognitivePattern::Convergent)
    ).await?;
    
    // Create collaborative task
    let tasks = vec![
        "Research market opportunities".into(),
        "Analyze competitive landscape".into(),
        "Identify strategic advantages".into(),
    ];
    
    // Execute with cognitive diversity
    let results = swarm.orchestrate(tasks).await?;
    
    println!("Results: {:?}", results);
    Ok(())
}
```

#### Advanced Task Management
```rust
use ruv_swarm::{Task, TaskPriority, OrchestrationStrategy};

// Create complex task with dependencies
let task = Task::builder()
    .id("auth-system-development")
    .priority(TaskPriority::High)
    .description("Implement OAuth2 authentication")
    .subtasks(vec![
        "Design OAuth2 flow",
        "Implement token validation",
        "Create user management API",
        "Write comprehensive tests",
    ])
    .deadline(std::time::Duration::from_days(14))
    .build();

// Execute with specific strategy
let result = swarm.orchestrate_with_strategy(
    vec![task],
    OrchestrationStrategy::CognitiveDiversity {
        convergent_weight: 0.3,
        divergent_weight: 0.4,
        lateral_weight: 0.3,
    }
).await?;
```

### JavaScript/TypeScript API

#### Browser Usage
```typescript
import { RuvSwarm, CognitivePattern, SwarmTopology } from '@ruv/swarm';

async function initializeSwarm() {
    // Initialize with browser-optimized settings
    const swarm = await RuvSwarm.initialize({
        topology: SwarmTopology.Mesh,
        maxAgents: 5,
        persistence: {
            type: 'indexeddb',
            database: 'my-swarm-app'
        },
        features: {
            simd: true,  // Use SIMD if available
            workers: true  // Use Web Workers for parallelism
        }
    });
    
    // Create cognitive diversity team
    const team = await swarm.createTeam({
        name: 'design-team',
        cognitiveProfiles: [
            { pattern: CognitivePattern.Creative, weight: 0.4 },
            { pattern: CognitivePattern.Analytical, weight: 0.3 },
            { pattern: CognitivePattern.Strategic, weight: 0.3 }
        ]
    });
    
    // Execute design task
    const result = await team.solve({
        problem: "Design mobile app onboarding flow",
        constraints: ["accessibility compliant", "under 3 screens", "conversion optimized"],
        deliverables: ["wireframes", "user flow", "copy suggestions"]
    });
    
    console.log('Design solution:', result);
}
```

#### Node.js Server Usage
```typescript
import { RuvSwarm } from '@ruv/swarm';
import express from 'express';

const app = express();

// Initialize persistent swarm
const swarm = await RuvSwarm.initialize({
    topology: 'hierarchical',
    persistence: {
        type: 'sqlite',
        path: './production-swarm.db'
    },
    transport: {
        type: 'websocket',
        port: 8080,
        secure: true
    }
});

// API endpoint for task submission
app.post('/api/tasks', async (req, res) => {
    const { description, priority, deadline } = req.body;
    
    const result = await swarm.submitTask({
        description,
        priority: priority || 'medium',
        deadline: deadline ? new Date(deadline) : undefined,
        strategy: 'cognitive-diversity'
    });
    
    res.json({ taskId: result.id, status: 'submitted' });
});

// WebSocket for real-time updates
import { WebSocketServer } from 'ws';
const wss = new WebSocketServer({ port: 8081 });

swarm.on('task-completed', (event) => {
    wss.clients.forEach(client => {
        client.send(JSON.stringify({
            type: 'task-completed',
            data: event
        }));
    });
});
```

## Advanced Features

### Persistent State Management

#### SQLite Persistence
```rust
use ruv_swarm::{Swarm, SwarmConfig, PersistenceConfig};

let config = SwarmConfig {
    persistence: Some(PersistenceConfig::Sqlite {
        path: "./swarm-state.db".into(),
        checkpoint_interval: std::time::Duration::from_secs(300), // 5 minutes
        max_history: 10000,
    }),
    ..Default::default()
};

let swarm = Swarm::new(config)?;

// State is automatically persisted
// Swarm will restore from database on restart
```

#### Custom Storage Backend
```rust
use ruv_swarm::{Storage, StorageResult};

#[derive(Debug)]
struct RedisStorage {
    client: redis::Client,
}

#[async_trait::async_trait]
impl Storage for RedisStorage {
    async fn store_agent(&self, agent: &AgentModel) -> StorageResult<()> {
        // Implementation for Redis storage
        todo!()
    }
    
    async fn get_agent(&self, id: &str) -> StorageResult<Option<AgentModel>> {
        // Implementation for Redis retrieval
        todo!()
    }
    
    // ... implement other methods
}
```

### Transport Layer Configuration

#### WebSocket Transport
```yaml
transport:
  type: websocket
  host: "0.0.0.0"
  port: 8080
  tls:
    enabled: true
    cert_path: "./certs/server.crt"
    key_path: "./certs/server.key"
  compression: true
  max_connections: 1000
```

#### Shared Memory Transport (High Performance)
```rust
use ruv_swarm::{Transport, SharedMemoryTransport};

let transport = SharedMemoryTransport::new(
    "swarm-shmem",  // Shared memory name
    1024 * 1024,    // 1MB buffer
    16              // 16 message slots
)?;

let config = SwarmConfig {
    transport: Box::new(transport),
    ..Default::default()
};
```

### Event Monitoring and Metrics

#### Event Subscription
```rust
use ruv_swarm::{SwarmEvent, EventSubscriber};

// Subscribe to specific events
swarm.subscribe(|event: SwarmEvent| {
    match event {
        SwarmEvent::AgentSpawned { agent_id, pattern } => {
            println!("New agent: {} with pattern {:?}", agent_id, pattern);
        },
        SwarmEvent::TaskCompleted { task_id, result, duration } => {
            println!("Task {} completed in {:?}: {:?}", task_id, duration, result);
        },
        SwarmEvent::CognitiveDiversityEvent { insight, agents } => {
            println!("Cognitive insight from {:?}: {}", agents, insight);
        },
        _ => {}
    }
}).await?;
```

#### Custom Metrics
```rust
use ruv_swarm::{MetricCollector, Metric};

// Custom metric collection
swarm.add_metric_collector(|metrics: &MetricCollector| {
    // Cognitive diversity score
    let diversity_score = metrics.calculate_cognitive_diversity();
    metrics.record(Metric::new("cognitive_diversity", diversity_score));
    
    // Task success rate
    let success_rate = metrics.task_success_rate(std::time::Duration::from_hours(24));
    metrics.record(Metric::new("success_rate_24h", success_rate));
    
    // Agent utilization
    for agent in metrics.active_agents() {
        let utilization = agent.utilization_percentage();
        metrics.record(Metric::new(&format!("agent_{}_utilization", agent.id()), utilization));
    }
});
```

## Cognitive Patterns

### Understanding Cognitive Diversity

Cognitive diversity is the key differentiator of ruv-swarm. Different agents think in different ways, leading to better collective problem-solving.

#### Pattern Descriptions

| Pattern | Thinking Style | Best Use Cases | Characteristics |
|---------|---------------|----------------|-----------------|
| **Convergent** | Focused, analytical | Optimization, debugging, data analysis | Systematic, detail-oriented, efficient |
| **Divergent** | Broad, exploratory | Brainstorming, creative solutions, research | Imaginative, flexible, generates alternatives |
| **Lateral** | Cross-domain | Innovation, reframing problems | Makes unexpected connections, analogical thinking |
| **Systems** | Holistic | Architecture, strategy, integration | Sees big picture, understands interactions |
| **Critical** | Evaluative | Code review, quality assurance, testing | Identifies flaws, edge cases, risks |
| **Abstract** | Conceptual | Theory, patterns, generalization | Works with concepts, finds underlying principles |

### Configuring Cognitive Teams

#### Balanced Team
```rust
// Good for complex, multifaceted problems
let team_config = CognitiveTeamConfig {
    convergent: 0.3,   // Analysis and execution
    divergent: 0.3,    // Creativity and options
    lateral: 0.2,      // Innovation and reframing
    systems: 0.2,      // Integration and architecture
    critical: 0.0,     // Added during review phase
    abstract: 0.0,     // Added for theoretical problems
};
```

#### Specialized Teams
```rust
// Research team - heavy on exploration
let research_team = CognitiveTeamConfig {
    divergent: 0.5,    // Primary: generate many ideas
    lateral: 0.3,      // Secondary: find connections
    convergent: 0.2,   // Focus the research
    ..Default::default()
};

// Development team - focused on implementation
let dev_team = CognitiveTeamConfig {
    convergent: 0.5,   // Primary: implement efficiently
    systems: 0.3,      // Secondary: integrate well
    critical: 0.2,     // Review and test
    ..Default::default()
};
```

### Adaptive Cognitive Strategies

```rust
use ruv_swarm::{AdaptiveCognition, TaskPhase};

// Strategy that changes based on task phase
let adaptive_strategy = AdaptiveCognition::new()
    .phase(TaskPhase::Research, CognitiveTeamConfig {
        divergent: 0.6,
        lateral: 0.4,
        ..Default::default()
    })
    .phase(TaskPhase::Analysis, CognitiveTeamConfig {
        convergent: 0.7,
        systems: 0.3,
        ..Default::default()
    })
    .phase(TaskPhase::Implementation, CognitiveTeamConfig {
        convergent: 0.5,
        systems: 0.3,
        critical: 0.2,
        ..Default::default()
    })
    .phase(TaskPhase::Review, CognitiveTeamConfig {
        critical: 0.8,
        convergent: 0.2,
        ..Default::default()
    });

swarm.set_cognitive_strategy(adaptive_strategy)?;
```

## Deployment Scenarios

### Single Machine Development
```bash
# Quick local development setup
npx @ruv/swarm init --topology mesh --agents 3 --persistence memory
npx @ruv/swarm spawn researcher --capabilities "search,analyze"
npx @ruv/swarm spawn coder --capabilities "implement,debug"
npx @ruv/swarm spawn tester --capabilities "test,validate"
```

### Distributed Production
```yaml
# docker-compose.yml
version: '3.8'
services:
  swarm-coordinator:
    image: ruv/swarm:latest
    command: ["ruv-swarm", "coordinator", "--bind", "0.0.0.0:8080"]
    ports:
      - "8080:8080"
    environment:
      - RUV_SWARM_PERSISTENCE=postgresql://user:pass@db:5432/swarm
      
  swarm-worker-1:
    image: ruv/swarm:latest
    command: ["ruv-swarm", "worker", "--coordinator", "coordinator:8080", "--cognitive-pattern", "convergent"]
    depends_on: [swarm-coordinator]
    
  swarm-worker-2:
    image: ruv/swarm:latest
    command: ["ruv-swarm", "worker", "--coordinator", "coordinator:8080", "--cognitive-pattern", "divergent"]
    depends_on: [swarm-coordinator]
    
  db:
    image: postgres:14
    environment:
      POSTGRES_DB: swarm
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
```

### Cloud Native (Kubernetes)
```yaml
# swarm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ruv-swarm
spec:
  replicas: 5
  selector:
    matchLabels:
      app: ruv-swarm
  template:
    metadata:
      labels:
        app: ruv-swarm
    spec:
      containers:
      - name: swarm
        image: ruv/swarm:latest
        env:
        - name: RUV_SWARM_TOPOLOGY
          value: "kubernetes"
        - name: RUV_SWARM_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: ruv-swarm-service
spec:
  selector:
    app: ruv-swarm
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

### Serverless (AWS Lambda)
```typescript
// lambda-function.ts
import { RuvSwarm } from '@ruv/swarm';

let swarm: RuvSwarm | null = null;

export const handler = async (event: any) => {
    // Initialize swarm if not already done (cold start)
    if (!swarm) {
        swarm = await RuvSwarm.initialize({
            topology: 'star',
            maxAgents: 3,
            persistence: {
                type: 'dynamodb',
                table: process.env.SWARM_TABLE!
            }
        });
    }
    
    // Process the task
    const result = await swarm.orchestrate(event.task);
    
    return {
        statusCode: 200,
        body: JSON.stringify(result)
    };
};
```

## Performance Tuning

### Optimization Strategies

#### Memory Optimization
```rust
// Reduce memory footprint for embedded systems
let config = SwarmConfig {
    optimization: OptimizationConfig {
        memory_pool_size: 1024 * 1024,  // 1MB pool
        max_message_size: 4096,         // 4KB messages
        gc_interval: Duration::from_secs(60),
        compact_threshold: 0.7,
    },
    ..Default::default()
};
```

#### CPU Optimization
```rust
// Maximize CPU utilization
let config = SwarmConfig {
    optimization: OptimizationConfig {
        worker_threads: num_cpus::get(),
        simd_enabled: true,
        batch_size: 64,
        prefetch_tasks: true,
    },
    ..Default::default()
};
```

#### Network Optimization
```yaml
transport:
  compression: true
  batch_messages: true
  max_batch_size: 100
  batch_timeout: 10ms
  tcp_nodelay: true
  keepalive: 60s
```

### Monitoring and Profiling

#### Built-in Metrics
```rust
// Enable detailed metrics collection
let config = SwarmConfig {
    metrics: MetricsConfig {
        enabled: true,
        detailed_timing: true,
        memory_tracking: true,
        cognitive_analysis: true,
        export_prometheus: true,
        export_interval: Duration::from_secs(30),
    },
    ..Default::default()
};
```

#### Custom Profiling
```rust
use ruv_swarm::{ProfilerScope, SwarmProfiler};

// Profile specific operations
{
    let _scope = ProfilerScope::new("task-orchestration");
    let result = swarm.orchestrate(tasks).await?;
}

// Get profiling report
let report = SwarmProfiler::global().generate_report();
println!("Performance report:\n{}", report);
```

## Troubleshooting

### Common Issues

#### Agent Not Responding
```bash
# Check agent status
npx @ruv/swarm status --agent <agent-id>

# Restart stuck agent
npx @ruv/swarm agent restart <agent-id>

# Check logs
npx @ruv/swarm logs --agent <agent-id> --tail 100
```

#### High Memory Usage
```rust
// Enable memory monitoring
let config = SwarmConfig {
    monitoring: MonitoringConfig {
        memory_alerts: true,
        memory_limit: 1024 * 1024 * 512, // 512MB
        auto_gc: true,
    },
    ..Default::default()
};
```

#### Network Connectivity Issues
```bash
# Test network connectivity
npx @ruv/swarm network test --peer <peer-address>

# Check transport status
npx @ruv/swarm transport status

# Reset transport layer
npx @ruv/swarm transport reset
```

### Debug Mode
```bash
# Enable debug logging
export RUV_SWARM_LOG_LEVEL=debug
npx @ruv/swarm init --debug

# Or in code
let config = SwarmConfig {
    debug: DebugConfig {
        enabled: true,
        trace_messages: true,
        log_cognitive_decisions: true,
        dump_state_on_error: true,
    },
    ..Default::default()
};
```

### Performance Issues
```rust
// Enable performance monitoring
swarm.enable_profiler(ProfilerConfig {
    sample_rate: 1000, // Sample every 1000 operations
    detailed_stacks: true,
    memory_profiling: true,
})?;

// Get performance insights
let insights = swarm.performance_insights().await?;
for insight in insights {
    println!("Performance tip: {}", insight.description);
    println!("Impact: {} ({}%)", insight.impact, insight.percentage);
}
```

## Best Practices

### Cognitive Team Design

1. **Balanced Diversity**: Include multiple cognitive patterns for complex problems
2. **Task-Specific Teams**: Match cognitive patterns to task requirements
3. **Adaptive Strategies**: Change cognitive composition based on project phase
4. **Feedback Loops**: Monitor cognitive diversity effectiveness and adjust

### Performance

1. **Right-Size Your Swarm**: More agents aren't always better
2. **Use Appropriate Topology**: Mesh for collaboration, star for coordination
3. **Persistent State**: Use SQLite for production, memory for development
4. **Monitor Resource Usage**: Set up alerts for memory and CPU usage

### Security

1. **Secure Transport**: Use TLS for network communication
2. **Input Validation**: Validate all task inputs and parameters
3. **Access Control**: Implement authentication for swarm management
4. **Audit Logging**: Log all significant swarm operations

### Development Workflow

1. **Start Small**: Begin with 2-3 agents for development
2. **Test Cognitive Patterns**: Verify different patterns work as expected
3. **Integration Testing**: Test swarm behavior under various conditions
4. **Gradual Scaling**: Increase agent count and complexity incrementally

### Production Deployment

1. **Health Checks**: Implement comprehensive health monitoring
2. **Graceful Shutdown**: Handle shutdown signals properly
3. **Backup Strategy**: Regular backups of persistent state
4. **Disaster Recovery**: Plan for agent failures and network partitions

---

This user guide covers the essential aspects of using ruv-swarm effectively. For more advanced topics, see the [API documentation](https://docs.rs/ruv-swarm) and [examples](../examples/).

## Getting Help

- **Documentation**: [docs.rs/ruv-swarm](https://docs.rs/ruv-swarm)
- **Examples**: [GitHub Examples](https://github.com/ruv-inc/ruv-swarm/tree/main/examples)
- **Issues**: [GitHub Issues](https://github.com/ruv-inc/ruv-swarm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ruv-inc/ruv-swarm/discussions)
- **Discord**: [RUV Community](https://discord.gg/ruv)