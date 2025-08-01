# API Reference

This document provides comprehensive documentation for the ruv-swarm-persistence API.

## Table of Contents

- [Core Traits](#core-traits)
- [Data Models](#data-models)
- [Storage Implementations](#storage-implementations)
- [Query Builder](#query-builder)
- [Error Handling](#error-handling)
- [Utility Functions](#utility-functions)

## Core Traits

### `Storage` Trait

The main persistence interface providing async operations for all data types.

```rust
#[async_trait]
pub trait Storage: Send + Sync {
    type Error: StdError + Send + Sync + 'static;
    
    // Agent operations
    async fn store_agent(&self, agent: &AgentModel) -> Result<(), Self::Error>;
    async fn get_agent(&self, id: &str) -> Result<Option<AgentModel>, Self::Error>;
    async fn update_agent(&self, agent: &AgentModel) -> Result<(), Self::Error>;
    async fn delete_agent(&self, id: &str) -> Result<(), Self::Error>;
    async fn list_agents(&self) -> Result<Vec<AgentModel>, Self::Error>;
    async fn list_agents_by_status(&self, status: &str) -> Result<Vec<AgentModel>, Self::Error>;
    
    // Task operations
    async fn store_task(&self, task: &TaskModel) -> Result<(), Self::Error>;
    async fn get_task(&self, id: &str) -> Result<Option<TaskModel>, Self::Error>;
    async fn update_task(&self, task: &TaskModel) -> Result<(), Self::Error>;
    async fn get_pending_tasks(&self) -> Result<Vec<TaskModel>, Self::Error>;
    async fn get_tasks_by_agent(&self, agent_id: &str) -> Result<Vec<TaskModel>, Self::Error>;
    async fn claim_task(&self, task_id: &str, agent_id: &str) -> Result<bool, Self::Error>;
    
    // Event operations
    async fn store_event(&self, event: &EventModel) -> Result<(), Self::Error>;
    async fn get_events_by_agent(&self, agent_id: &str, limit: usize) -> Result<Vec<EventModel>, Self::Error>;
    async fn get_events_by_type(&self, event_type: &str, limit: usize) -> Result<Vec<EventModel>, Self::Error>;
    async fn get_events_since(&self, timestamp: i64) -> Result<Vec<EventModel>, Self::Error>;
    
    // Message operations
    async fn store_message(&self, message: &MessageModel) -> Result<(), Self::Error>;
    async fn get_messages_between(&self, agent1: &str, agent2: &str, limit: usize) -> Result<Vec<MessageModel>, Self::Error>;
    async fn get_unread_messages(&self, agent_id: &str) -> Result<Vec<MessageModel>, Self::Error>;
    async fn mark_message_read(&self, message_id: &str) -> Result<(), Self::Error>;
    
    // Metric operations
    async fn store_metric(&self, metric: &MetricModel) -> Result<(), Self::Error>;
    async fn get_metrics_by_agent(&self, agent_id: &str, metric_type: &str) -> Result<Vec<MetricModel>, Self::Error>;
    async fn get_aggregated_metrics(&self, metric_type: &str, start_time: i64, end_time: i64) -> Result<Vec<MetricModel>, Self::Error>;
    
    // Transaction support
    async fn begin_transaction(&self) -> Result<Box<dyn Transaction>, Self::Error>;
    
    // Maintenance operations
    async fn vacuum(&self) -> Result<(), Self::Error>;
    async fn checkpoint(&self) -> Result<(), Self::Error>;
    async fn get_storage_size(&self) -> Result<u64, Self::Error>;
}
```

#### Agent Operations

##### `store_agent(agent: &AgentModel)`
Stores a new agent in the persistence layer.

**Example:**
```rust
let agent = AgentModel::new("worker-001".to_string(), "researcher".to_string(), vec!["analysis".to_string()]);
storage.store_agent(&agent).await?;
```

##### `get_agent(id: &str)`
Retrieves an agent by ID. Returns `None` if not found.

**Example:**
```rust
let agent = storage.get_agent("agent-123").await?;
if let Some(agent) = agent {
    println!("Found agent: {}", agent.name);
}
```

##### `update_agent(agent: &AgentModel)`
Updates an existing agent's information.

**Example:**
```rust
let mut agent = storage.get_agent("agent-123").await?.unwrap();
agent.status = AgentStatus::Busy;
agent.updated_at = Utc::now();
storage.update_agent(&agent).await?;
```

##### `list_agents()`
Returns all agents in the system.

**Example:**
```rust
let agents = storage.list_agents().await?;
println!("Total agents: {}", agents.len());
```

##### `list_agents_by_status(status: &str)`
Returns agents filtered by status.

**Example:**
```rust
let active_agents = storage.list_agents_by_status("active").await?;
println!("Active agents: {}", active_agents.len());
```

#### Task Operations

##### `claim_task(task_id: &str, agent_id: &str)`
Atomically claims a task for an agent. Returns `true` if successful, `false` if already claimed.

**Example:**
```rust
let claimed = storage.claim_task("task-123", "agent-456").await?;
if claimed {
    println!("Task claimed successfully");
} else {
    println!("Task already claimed by another agent");
}
```

##### `get_pending_tasks()`
Returns all tasks that haven't been assigned to an agent.

**Example:**
```rust
let pending = storage.get_pending_tasks().await?;
println!("Pending tasks: {}", pending.len());
```

### `Transaction` Trait

Provides ACID transaction support for atomic operations.

```rust
#[async_trait]
pub trait Transaction: Send + Sync {
    async fn commit(self: Box<Self>) -> Result<(), StorageError>;
    async fn rollback(self: Box<Self>) -> Result<(), StorageError>;
}
```

**Example:**
```rust
let mut tx = storage.begin_transaction().await?;

// Multiple operations in single transaction
storage.store_task(&task1).await?;
storage.store_task(&task2).await?;
storage.store_event(&event).await?;

// Commit all changes atomically
tx.commit().await?;
```

### `Repository<T>` Trait

Provides a repository pattern for type-safe data access.

```rust
pub trait Repository<T> {
    type Error: StdError + Send + Sync + 'static;
    
    fn find_by_id(&self, id: &str) -> Result<Option<T>, Self::Error>;
    fn find_all(&self) -> Result<Vec<T>, Self::Error>;
    fn save(&self, entity: &T) -> Result<(), Self::Error>;
    fn update(&self, entity: &T) -> Result<(), Self::Error>;
    fn delete(&self, id: &str) -> Result<(), Self::Error>;
    fn query(&self, builder: QueryBuilder<T>) -> Result<Vec<T>, Self::Error>;
}
```

## Data Models

### `AgentModel`

Represents a swarm agent with lifecycle management.

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentModel {
    pub id: String,
    pub name: String,
    pub agent_type: String,
    pub status: AgentStatus,
    pub capabilities: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub heartbeat: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
```

#### Methods

##### `new(name: String, agent_type: String, capabilities: Vec<String>)`
Creates a new agent with auto-generated ID and current timestamp.

**Example:**
```rust
let agent = AgentModel::new(
    "worker-001".to_string(),
    "researcher".to_string(),
    vec!["analysis".to_string(), "data-processing".to_string()]
);
```

##### `update_heartbeat(&mut self)`
Updates the heartbeat and updated_at timestamps.

##### `set_status(&mut self, status: AgentStatus)`
Updates the agent status and updated_at timestamp.

### `AgentStatus`

Enumeration representing agent lifecycle states.

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Initializing,
    Active,
    Idle,
    Busy,
    Paused,
    Error,
    Shutdown,
}
```

### `TaskModel`

Represents work items in the swarm with priority and dependency tracking.

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskModel {
    pub id: String,
    pub task_type: String,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub assigned_to: Option<String>,
    pub payload: serde_json::Value,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub retry_count: u32,
    pub max_retries: u32,
    pub dependencies: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}
```

#### Methods

##### `new(task_type: String, payload: serde_json::Value, priority: TaskPriority)`
Creates a new task with auto-generated ID and current timestamp.

**Example:**
```rust
let task = TaskModel::new(
    "analyze-data".to_string(),
    serde_json::json!({"dataset": "sales_data.csv"}),
    TaskPriority::High
);
```

### `TaskPriority`

Enumeration for task priority levels.

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}
```

### `TaskStatus`

Enumeration for task lifecycle states.

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Pending,
    Assigned,
    Running,
    Completed,
    Failed,
    Cancelled,
}
```

### `EventModel`

Represents system events for audit trails and debugging.

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EventModel {
    pub id: String,
    pub event_type: String,
    pub agent_id: Option<String>,
    pub data: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}
```

#### Methods

##### `new(event_type: String, data: serde_json::Value)`
Creates a new event with auto-generated ID and current timestamp.

**Example:**
```rust
let event = EventModel::new(
    "task_completed".to_string(),
    serde_json::json!({
        "task_id": "task-123",
        "duration_ms": 5000,
        "result": "success"
    })
);
```

##### `with_agent(mut self, agent_id: String)`
Builder method to associate an event with a specific agent.

**Example:**
```rust
let event = EventModel::new("agent_started".to_string(), serde_json::json!({}))
    .with_agent("agent-123".to_string());
```

### `MessageModel`

Represents inter-agent communication.

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageModel {
    pub id: String,
    pub from_agent: String,
    pub to_agent: String,
    pub message_type: String,
    pub content: serde_json::Value,
    pub read: bool,
    pub timestamp: DateTime<Utc>,
}
```

#### Methods

##### `new(from_agent: String, to_agent: String, message_type: String, content: serde_json::Value)`
Creates a new message with auto-generated ID and current timestamp.

**Example:**
```rust
let message = MessageModel::new(
    "coordinator".to_string(),
    "worker-001".to_string(),
    "task_assignment".to_string(),
    serde_json::json!({"task_id": "task-123"})
);
```

### `MetricModel`

Represents performance and monitoring data.

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricModel {
    pub id: String,
    pub agent_id: Option<String>,
    pub metric_type: String,
    pub value: f64,
    pub unit: String,
    pub tags: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}
```

## Storage Implementations

### `SqliteStorage`

Production-ready SQLite backend with ACID compliance and connection pooling.

```rust
impl SqliteStorage {
    pub async fn new(db_path: &str) -> Result<Self, StorageError>;
    pub fn builder() -> SqliteStorageBuilder;
}
```

#### Configuration

```rust
pub struct SqliteStorageBuilder {
    max_connections: u32,
    min_idle_connections: u32,
    connection_timeout: Duration,
    idle_timeout: Duration,
}

impl SqliteStorageBuilder {
    pub fn max_connections(mut self, max: u32) -> Self;
    pub fn min_idle_connections(mut self, min: u32) -> Self;
    pub fn connection_timeout(mut self, timeout: Duration) -> Self;
    pub fn idle_timeout(mut self, timeout: Duration) -> Self;
    pub async fn build(self, db_path: &str) -> Result<SqliteStorage, StorageError>;
}
```

**Example:**
```rust
let storage = SqliteStorage::builder()
    .max_connections(32)
    .min_idle_connections(4)
    .connection_timeout(Duration::from_secs(30))
    .idle_timeout(Duration::from_secs(600))
    .build("production.db")
    .await?;
```

### `MemoryStorage`

High-performance in-memory storage for testing and development.

```rust
impl MemoryStorage {
    pub fn new() -> Self;
}
```

**Example:**
```rust
let storage = MemoryStorage::new();
```

### `WasmStorage` (WASM only)

IndexedDB-based storage for browser applications.

```rust
impl WasmStorage {
    pub async fn new() -> Result<Self, StorageError>;
}
```

## Query Builder

Type-safe SQL query construction with parameterized queries for security.

```rust
pub struct QueryBuilder<T> {
    // Internal fields
}

impl<T> QueryBuilder<T> {
    pub fn new(table: &str) -> Self;
    pub fn where_eq(self, field: &str, value: &str) -> Self;
    pub fn where_like(self, field: &str, pattern: &str) -> Self;
    pub fn where_gt(self, field: &str, value: i64) -> Self;
    pub fn order_by(self, field: &str, desc: bool) -> Self;
    pub fn limit(self, limit: usize) -> Self;
    pub fn offset(self, offset: usize) -> Self;
    pub fn build(&self) -> (String, Vec<String>);
}
```

### Security Features

The QueryBuilder uses parameterized queries with `?` placeholders to prevent SQL injection:

**Example:**
```rust
let (query, params) = QueryBuilder::<AgentModel>::new("agents")
    .where_eq("status", "active")
    .where_like("agent_type", "worker%")
    .order_by("created_at", true)
    .limit(50)
    .build();

// Generated SQL: "SELECT * FROM agents WHERE status = ? AND agent_type LIKE ? ORDER BY created_at DESC LIMIT 50"
// Parameters: ["active", "worker%"]
```

## Error Handling

### `StorageError`

Comprehensive error types for all storage operations.

```rust
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Database error: {0}")]
    Database(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Transaction error: {0}")]
    Transaction(String),
    
    #[error("Migration error: {0}")]
    Migration(String),
    
    #[error("Connection pool error: {0}")]
    Pool(String),
    
    #[error("Other error: {0}")]
    Other(String),
}
```

### Error Handling Patterns

#### Retry Logic

```rust
async fn robust_operation<S: Storage>(storage: &S) -> Result<(), StorageError> {
    let mut retries = 3;
    
    loop {
        match storage.store_agent(&agent).await {
            Ok(_) => return Ok(()),
            Err(StorageError::Database(msg)) if retries > 0 => {
                eprintln!("Database error, retrying: {}", msg);
                retries -= 1;
                tokio::time::sleep(Duration::from_millis(100)).await;
            },
            Err(e) => return Err(e),
        }
    }
}
```

#### Pattern Matching

```rust
match storage.get_agent("agent-123").await {
    Ok(Some(agent)) => println!("Found agent: {}", agent.name),
    Ok(None) => println!("Agent not found"),
    Err(StorageError::Database(msg)) => eprintln!("Database error: {}", msg),
    Err(StorageError::NotFound(id)) => eprintln!("Agent {} not found", id),
    Err(e) => eprintln!("Unexpected error: {}", e),
}
```

## Utility Functions

### `init_storage(path: Option<&str>)`

Platform-aware storage initialization.

```rust
pub async fn init_storage(path: Option<&str>) -> Result<Box<dyn Storage<Error = StorageError>>, StorageError>
```

**Example:**
```rust
// SQLite on native platforms
let storage = init_storage(Some("swarm.db")).await?;

// IndexedDB on WASM
let storage = init_storage(None).await?;
```

## Thread Safety

All storage implementations are thread-safe and can be shared across async tasks:

```rust
use std::sync::Arc;

let storage = Arc::new(SqliteStorage::new("swarm.db").await?);

// Clone and use in multiple tasks
let storage_clone = storage.clone();
tokio::spawn(async move {
    storage_clone.store_agent(&agent).await.unwrap();
});
```

## Performance Considerations

### Connection Pooling

SQLite storage uses R2D2 connection pooling:
- Default pool size: 16 connections
- Configurable min/max connections
- Automatic connection recycling

### Batch Operations

For high-throughput scenarios, use transactions for batch operations:

```rust
let mut tx = storage.begin_transaction().await?;

for task in tasks {
    storage.store_task(&task).await?;
}

tx.commit().await?;
```

### Indexing

The SQLite backend includes optimized indexes for common queries:
- Agent status queries
- Task priority ordering
- Event timestamp ranges
- Message read status

---

*This API reference covers all public APIs in ruv-swarm-persistence. For implementation examples, see the [Getting Started Guide](./getting-started.md).*