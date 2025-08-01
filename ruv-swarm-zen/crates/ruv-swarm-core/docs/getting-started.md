# Getting Started with ruv-swarm-core

## Quick Start (5 minutes)

### 1. Add Dependency

```toml
[dependencies]
ruv-swarm-core = "1.0"
tokio = { version = "1.0", features = ["full"] }
```

### 2. Basic Example

```rust
use ruv_swarm_core::{
    agent::DynamicAgent,
    swarm::{Swarm, SwarmConfig},
    task::Task,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create swarm
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Register agents
    let agent = DynamicAgent::new("worker-1", vec!["compute".to_string()]);
    swarm.register_agent(agent)?;
    
    // Submit task
    let task = Task::new("job-1", "compute")
        .require_capability("compute");
    swarm.submit_task(task)?;
    
    // Start agents and distribute tasks
    swarm.start_all_agents().await?;
    let assignments = swarm.distribute_tasks().await?;
    
    println!("Task assignments: {:?}", assignments);
    Ok(())
}
```

## Step-by-Step Tutorial

### Step 1: Understanding Agents

Agents are the workers in your swarm. They have capabilities and can process tasks:

```rust
use ruv_swarm_core::agent::DynamicAgent;

// Create agents with different capabilities
let compute_agent = DynamicAgent::new(
    "compute-worker", 
    vec!["compute".to_string(), "ml".to_string()]
);

let storage_agent = DynamicAgent::new(
    "storage-worker",
    vec!["storage".to_string(), "cache".to_string()]
);
```

### Step 2: Creating Tasks

Tasks represent work to be done. Use the builder pattern for complex tasks:

```rust
use ruv_swarm_core::task::{Task, TaskPriority, TaskPayload};

// Simple task
let simple_task = Task::new("task-1", "computation");

// Complex task with requirements
let complex_task = Task::new("ml-training", "machine-learning")
    .with_priority(TaskPriority::High)
    .require_capability("ml")
    .require_capability("gpu")
    .with_timeout(30000)  // 30 seconds
    .with_payload(TaskPayload::Json(r#"{"model": "gpt", "data": "dataset.json"}"#.to_string()));
```

### Step 3: Configuring Swarms

Choose the right configuration for your use case:

```rust
use ruv_swarm_core::{
    swarm::SwarmConfig,
    task::DistributionStrategy,
    topology::TopologyType,
};

// High-performance configuration
let config = SwarmConfig {
    topology_type: TopologyType::Mesh,
    distribution_strategy: DistributionStrategy::LeastLoaded,
    max_agents: 100,
    enable_auto_scaling: true,
    health_check_interval_ms: 5000,
};
```

### Step 4: Production AsyncSwarm

For production systems, use AsyncSwarm for better performance:

```rust
use ruv_swarm_core::async_swarm::{AsyncSwarm, AsyncSwarmConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AsyncSwarmConfig {
        max_concurrent_tasks_per_agent: 10,
        task_timeout_ms: 30000,
        ..Default::default()
    };
    
    let swarm = AsyncSwarm::new(config);
    
    // Register agents (can be done concurrently)
    let agent1 = DynamicAgent::new("worker-1", vec!["compute".to_string()]);
    let agent2 = DynamicAgent::new("worker-2", vec!["compute".to_string()]);
    
    swarm.register_agent(agent1).await?;
    swarm.register_agent(agent2).await?;
    
    // Process tasks concurrently
    let results = swarm.process_tasks_concurrently(50).await?;
    
    println!("Processed {} tasks", results.len());
    Ok(())
}
```

## Common Patterns

### Pattern 1: Multi-Phase Processing

```rust
// Phase 1: Preprocessing
for i in 0..10 {
    let task = Task::new(format!("preprocess-{i}"), "preprocessing")
        .require_capability("preprocess");
    swarm.submit_task(task)?;
}
let phase1 = swarm.distribute_tasks().await?;

// Phase 2: Main processing  
for i in 0..10 {
    let task = Task::new(format!("process-{i}"), "processing")
        .require_capability("compute");
    swarm.submit_task(task)?;
}
let phase2 = swarm.distribute_tasks().await?;
```

### Pattern 2: Error Handling

```rust
match swarm.distribute_tasks().await {
    Ok(assignments) => {
        println!("Successfully assigned {} tasks", assignments.len());
    },
    Err(SwarmError::ResourceExhausted { resource }) => {
        println!("Need more {}", resource);
    },
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
    }
}
```

### Pattern 3: Monitoring

```rust
// Get real-time metrics
let metrics = swarm.metrics().await;
println!("Active agents: {}/{}", metrics.active_agents, metrics.total_agents);
println!("Queue size: {}", metrics.queued_tasks);
println!("Avg load: {:.2}", metrics.avg_agent_load);

// Check agent health
let statuses = swarm.agent_statuses().await;
for (id, status) in statuses {
    if status == AgentStatus::Error {
        println!("Agent {} needs attention", id);
    }
}
```

## Best Practices

1. **Choose the Right Swarm Type**
   - Use `Swarm` for simple, single-threaded scenarios
   - Use `AsyncSwarm` for production, multi-threaded systems

2. **Design Good Capabilities**
   - Use specific, descriptive capability names
   - Group related capabilities logically
   - Consider capability hierarchies

3. **Handle Errors Gracefully**
   - Always handle `ResourceExhausted` errors
   - Implement retry logic for retriable errors
   - Monitor agent health regularly

4. **Optimize Performance**
   - Use appropriate distribution strategies
   - Set reasonable timeouts
   - Monitor queue sizes and agent loads

5. **Test Thoroughly**
   - Test with different agent configurations
   - Simulate failure scenarios
   - Validate task distribution logic

## Troubleshooting

### Issue: Tasks not being assigned
- Check agent capabilities match task requirements
- Verify agents are in `Running` status
- Ensure agents are registered with the swarm

### Issue: Poor performance
- Consider using `AsyncSwarm` for better concurrency
- Optimize distribution strategy (`LeastLoaded` often works best)
- Increase `max_concurrent_tasks_per_agent`

### Issue: Agents failing
- Check error logs for specific failure reasons
- Verify agent dependencies and resources
- Implement proper error recovery

## Next Steps

- [API Reference](./api-reference.md) - Detailed API documentation
- [Swarm vs AsyncSwarm](./swarm-vs-async-swarm.md) - Choose the right implementation
- [Testing Guide](./testing-guide.md) - Learn testing best practices