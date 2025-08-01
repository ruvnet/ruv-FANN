# Getting Started with ruv-swarm-persistence

## Quick Start (5 minutes)

### 1. Add Dependency

```toml
[dependencies]
ruv-swarm-persistence = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
serde_json = "1.0"

# For WASM support
[target.'cfg(target_arch = "wasm32")'.dependencies]
ruv-swarm-persistence = { version = "0.1.0", features = ["wasm"] }
```

### 2. Basic Example

```rust
use ruv_swarm_persistence::{
    SqliteStorage, Storage, 
    AgentModel, TaskModel, EventModel,
    TaskPriority
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize storage
    let storage = SqliteStorage::new("swarm.db").await?;
    
    // Create and store an agent
    let agent = AgentModel::new(
        "worker-001".to_string(),
        "researcher".to_string(),
        vec!["analysis".to_string(), "data-processing".to_string()]
    );
    storage.store_agent(&agent).await?;
    
    // Create and store a task
    let task = TaskModel::new(
        "analyze-data".to_string(),
        serde_json::json!({"dataset": "sales_data.csv"}),
        TaskPriority::High
    );
    storage.store_task(&task).await?;
    
    // Log an event
    let event = EventModel::new(
        "task_created".to_string(),
        serde_json::json!({
            "task_id": task.id,
            "agent_id": agent.id,
            "timestamp": chrono::Utc::now()
        })
    );
    storage.store_event(&event).await?;
    
    // Query recent events
    let recent_events = storage.get_events_since(
        chrono::Utc::now().timestamp() - 3600  // Last hour
    ).await?;
    
    println!("Stored agent: {}", agent.id);
    println!("Stored task: {}", task.id);
    println!("Recent events: {}", recent_events.len());
    
    Ok(())
}
```

## Step-by-Step Tutorial

### Step 1: Understanding Storage Backends

Choose the right storage backend for your use case:

```rust
use ruv_swarm_persistence::{SqliteStorage, MemoryStorage, Storage};

// SQLite - Production, persistent storage
let sqlite_storage = SqliteStorage::new("production.db").await?;

// Memory - Fast, for testing and development
let memory_storage = MemoryStorage::new();

// Both implement the same Storage trait
fn use_storage<S: Storage>(storage: &S) {
    // Same API regardless of backend
}
```

### Step 2: Working with Agents

Agents represent workers in your swarm with full lifecycle management:

```rust
use ruv_swarm_persistence::{AgentModel, AgentStatus};

// Create agents with different capabilities
let compute_agent = AgentModel::new(
    "compute-worker-001".to_string(),
    "compute".to_string(),
    vec!["ml".to_string(), "data-processing".to_string()]
);

let storage_agent = AgentModel::new(
    "storage-worker-001".to_string(),
    "storage".to_string(),
    vec!["database".to_string(), "cache".to_string()]
);

// Store agents
storage.store_agent(&compute_agent).await?;
storage.store_agent(&storage_agent).await?;

// Update agent status
let mut agent = storage.get_agent(&compute_agent.id).await?.unwrap();
agent.status = AgentStatus::Busy;
agent.updated_at = chrono::Utc::now();
storage.update_agent(&agent).await?;

// List all active agents
let active_agents = storage.list_agents().await?;
println!("Active agents: {}", active_agents.len());
```

### Step 3: Task Management

Tasks represent work to be done with priority and dependency tracking:

```rust
use ruv_swarm_persistence::{TaskModel, TaskPriority, TaskStatus};

// Create tasks with different priorities
let urgent_task = TaskModel::new(
    "urgent-analysis".to_string(),
    serde_json::json!({"priority": "critical", "deadline": "2024-01-01"}),
    TaskPriority::Critical
);

let routine_task = TaskModel::new(
    "daily-report".to_string(),
    serde_json::json!({"template": "daily", "recipients": ["team@company.com"]}),
    TaskPriority::Low
);

// Store tasks
storage.store_task(&urgent_task).await?;
storage.store_task(&routine_task).await?;

// Claim task (atomic operation)
let success = storage.claim_task(&urgent_task.id, &compute_agent.id).await?;
if success {
    println!("Task claimed successfully!");
} else {
    println!("Task already claimed by another agent");
}

// Update task status
let mut task = storage.get_task(&urgent_task.id).await?.unwrap();
task.status = TaskStatus::Completed;
task.updated_at = chrono::Utc::now();
storage.update_task(&task).await?;
```

### Step 4: ACID Transactions

Use transactions for atomic operations that must succeed or fail together:

```rust
use ruv_swarm_persistence::{SqliteStorage, Transaction};

async fn atomic_task_processing(
    storage: &SqliteStorage
) -> Result<(), Box<dyn std::error::Error>> {
    // Begin transaction
    let mut tx = storage.begin_transaction().await?;
    
    // Multiple operations in single transaction
    let task1 = TaskModel::new(
        "preprocess".to_string(),
        serde_json::json!({"input": "raw_data.csv"}),
        TaskPriority::High
    );
    
    let task2 = TaskModel::new(
        "analyze".to_string(),
        serde_json::json!({"depends_on": task1.id}),
        TaskPriority::Medium
    );
    
    // Both operations must succeed or both fail
    storage.store_task(&task1).await?;
    storage.store_task(&task2).await?;
    
    // Log the transaction
    let event = EventModel::new(
        "batch_tasks_created".to_string(),
        serde_json::json!({
            "task_ids": [task1.id, task2.id],
            "batch_size": 2
        })
    );
    storage.store_event(&event).await?;
    
    // Commit all changes atomically
    tx.commit().await?;
    
    println!("All operations committed successfully!");
    Ok(())
}
```

### Step 5: Event Sourcing

Track all system events for audit trails and debugging:

```rust
use ruv_swarm_persistence::EventModel;

// Log agent lifecycle events
let agent_started = EventModel::new(
    "agent_started".to_string(),
    serde_json::json!({
        "agent_id": "worker-001",
        "startup_time": chrono::Utc::now(),
        "capabilities": ["ml", "data-processing"]
    })
).with_agent("worker-001".to_string());

storage.store_event(&agent_started).await?;

// Log task completion
let task_completed = EventModel::new(
    "task_completed".to_string(),
    serde_json::json!({
        "task_id": "analyze-data",
        "agent_id": "worker-001",
        "duration_ms": 5000,
        "result": "success"
    })
).with_agent("worker-001".to_string());

storage.store_event(&task_completed).await?;

// Query events by type
let task_events = storage.get_events_by_type("task_completed").await?;
println!("Task completions: {}", task_events.len());

// Query events by agent
let agent_events = storage.get_events_by_agent("worker-001").await?;
println!("Agent events: {}", agent_events.len());
```

## Common Patterns

### Pattern 1: Secure Query Building

Always use parameterized queries to prevent SQL injection:

```rust
use ruv_swarm_persistence::QueryBuilder;

// ✅ SECURE: Uses parameterized queries
let (query, params) = QueryBuilder::<AgentModel>::new("agents")
    .where_eq("status", "active")
    .where_like("agent_type", "worker%")
    .order_by("created_at", true)
    .limit(50)
    .build();

// The query uses ? placeholders: "SELECT * FROM agents WHERE status = ? AND agent_type LIKE ?"
// The params contain the actual values: ["active", "worker%"]
```

### Pattern 2: Error Handling with Retries

Implement robust error handling for production systems:

```rust
use ruv_swarm_persistence::{StorageError, Storage};

async fn robust_store_agent<S: Storage>(
    storage: &S,
    agent: &AgentModel
) -> Result<(), StorageError> {
    let mut retries = 3;
    
    loop {
        match storage.store_agent(agent).await {
            Ok(_) => return Ok(()),
            Err(StorageError::Database(msg)) if retries > 0 => {
                eprintln!("Database error, retrying: {}", msg);
                retries -= 1;
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            },
            Err(e) => return Err(e),
        }
    }
}
```

### Pattern 3: Performance Monitoring

Track performance metrics for your swarm:

```rust
use ruv_swarm_persistence::MetricModel;

// Store performance metrics
let cpu_metric = MetricModel {
    id: "cpu_usage_001".to_string(),
    agent_id: Some("worker-001".to_string()),
    metric_type: "cpu_usage".to_string(),
    value: 85.5,
    unit: "percent".to_string(),
    tags: [("host".to_string(), "worker-node-01".to_string())].into_iter().collect(),
    timestamp: chrono::Utc::now(),
};

storage.store_metric(&cpu_metric).await?;

// Query aggregated metrics
let metrics = storage.get_aggregated_metrics(
    "cpu_usage",
    chrono::Utc::now().timestamp() - 3600, // Last hour
    chrono::Utc::now().timestamp()
).await?;

println!("Average CPU usage: {:.2}%", 
    metrics.iter().map(|m| m.value).sum::<f64>() / metrics.len() as f64);
```

## Best Practices

1. **Choose the Right Storage Backend**
   - Use `SqliteStorage` for production with persistence requirements
   - Use `MemoryStorage` for testing and development
   - Consider WASM storage for browser applications

2. **Use Transactions for Related Operations**
   - Group related operations in transactions
   - Always commit or rollback explicitly
   - Keep transactions short to avoid blocking

3. **Implement Proper Error Handling**
   - Handle `StorageError` variants appropriately
   - Implement retry logic for transient failures
   - Log errors for debugging and monitoring

4. **Design for Security**
   - Always use parameterized queries
   - Validate input data before storing
   - Use the QueryBuilder for safe query construction

5. **Monitor Performance**
   - Track key metrics like task completion rates
   - Monitor storage performance and connection pool health
   - Use event sourcing for audit trails

## Production Configuration

### SQLite Optimization

```rust
use ruv_swarm_persistence::SqliteStorage;

// Configure SQLite for production
let storage = SqliteStorage::builder()
    .max_connections(32)                    // Adjust based on workload
    .min_idle_connections(4)                // Keep connections warm
    .connection_timeout(std::time::Duration::from_secs(30))
    .idle_timeout(std::time::Duration::from_secs(600))
    .build("production.db")
    .await?;
```

### Connection Pool Monitoring

```rust
// Monitor connection pool health
let pool_status = storage.pool_status().await?;
println!("Active connections: {}/{}", 
    pool_status.active, pool_status.max);

if pool_status.active > pool_status.max * 0.8 {
    eprintln!("Warning: Connection pool utilization high");
}
```

## Troubleshooting

### Issue: SQL Injection Vulnerabilities
- **Solution**: Always use `QueryBuilder` with parameterized queries
- **Check**: Ensure no string concatenation in SQL queries
- **Test**: Run security tests to validate query safety

### Issue: Database Locks and Timeouts
- **Solution**: Use shorter transactions and proper connection scoping
- **Check**: Monitor connection pool utilization
- **Test**: Run concurrent tests to validate thread safety

### Issue: Poor Performance
- **Solution**: Optimize SQLite configuration and use connection pooling
- **Check**: Monitor query execution times and connection usage
- **Test**: Run performance benchmarks with realistic data volumes

### Issue: Data Corruption
- **Solution**: Use ACID transactions and proper error handling
- **Check**: Verify foreign key constraints and data validation
- **Test**: Test failure scenarios and recovery procedures

## Next Steps

- [API Reference](./api-reference.md) - Detailed API documentation
- [Storage Backends](./storage-backends.md) - SQLite vs Memory comparison
- [Security Guide](./security-guide.md) - ACID transactions and SQL injection prevention
- [Testing Guide](./testing-guide.md) - Learn testing best practices

---

*Ready to build secure, high-performance persistent swarms! 🗄️*