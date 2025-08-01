# API Reference - ruv-swarm-core

## Core Traits

### Agent Trait

The foundational trait that all agents must implement.

```rust
#[async_trait]
pub trait Agent: Send + Sync {
    type Input: Send;
    type Output: Send;
    type Error: fmt::Debug + Send;

    // Required methods
    async fn process(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    fn capabilities(&self) -> &[String];
    fn id(&self) -> &str;

    // Optional methods with defaults
    async fn start(&mut self) -> Result<()>;
    async fn shutdown(&mut self) -> Result<()>;
    async fn health_check(&self) -> Result<HealthStatus>;
    fn status(&self) -> AgentStatus;
    fn can_handle(&self, task: &Task) -> bool;
}
```

## Core Types

### DynamicAgent

Ready-to-use agent implementation for most scenarios.

```rust
impl DynamicAgent {
    pub fn new(id: impl Into<String>, capabilities: Vec<String>) -> Self;
    pub async fn start(&mut self) -> Result<()>;
    pub async fn shutdown(&mut self) -> Result<()>;
    pub fn can_handle(&self, task: &Task) -> bool;
    pub fn has_capability(&self, capability: &str) -> bool;
    pub fn status(&self) -> AgentStatus;
    pub fn set_status(&mut self, status: AgentStatus);
}
```

### Task

Represents work to be processed by agents.

```rust
impl Task {
    // Creation
    pub fn new(id: impl Into<TaskId>, task_type: impl Into<String>) -> Self;
    
    // Builder methods
    pub fn with_priority(mut self, priority: TaskPriority) -> Self;
    pub fn with_payload(mut self, payload: TaskPayload) -> Self;
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self;
    pub fn require_capability(mut self, capability: impl Into<String>) -> Self;
    pub fn add_dependency(mut self, dependency: TaskId) -> Self;
}

// Task payload types
pub enum TaskPayload {
    Binary(Vec<u8>),
    Json(String),
    Custom(Box<dyn CustomPayload>),
}

// Task priorities
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}
```

## Swarm Orchestrators

### Swarm (Single-threaded)

Simple, single-threaded swarm for basic use cases.

```rust
impl Swarm {
    pub fn new(config: SwarmConfig) -> Self;
    
    // Agent management
    pub fn register_agent(&mut self, agent: DynamicAgent) -> Result<()>;
    pub fn unregister_agent(&mut self, agent_id: &AgentId) -> Result<()>;
    pub fn get_agent(&self, agent_id: &AgentId) -> Option<&DynamicAgent>;
    pub fn get_agent_mut(&mut self, agent_id: &AgentId) -> Option<&mut DynamicAgent>;
    
    // Task management
    pub fn submit_task(&mut self, task: Task) -> Result<()>;
    pub async fn distribute_tasks(&mut self) -> Result<Vec<(TaskId, AgentId)>>;
    
    // Lifecycle
    pub async fn start_all_agents(&mut self) -> Result<()>;
    pub async fn shutdown_all_agents(&mut self) -> Result<()>;
    
    // Monitoring
    pub fn metrics(&self) -> SwarmMetrics;
    pub fn agent_statuses(&self) -> HashMap<AgentId, AgentStatus>;
    pub fn task_queue_size(&self) -> usize;
    pub fn assigned_tasks_count(&self) -> usize;
}
```

### AsyncSwarm (Multi-threaded)

Production-ready, thread-safe swarm with advanced features.

```rust
impl AsyncSwarm {
    pub fn new(config: AsyncSwarmConfig) -> Self;
    
    // Agent management (all async)
    pub async fn register_agent(&self, agent: DynamicAgent) -> Result<()>;
    pub async fn unregister_agent(&self, agent_id: &AgentId) -> Result<()>;
    pub async fn has_agent(&self, agent_id: &AgentId) -> bool;
    
    // Task management
    pub async fn submit_task(&self, task: Task) -> Result<()>;
    pub async fn distribute_tasks(&self) -> Result<Vec<(TaskId, AgentId)>>;
    pub async fn process_tasks_concurrently(&self, max_concurrent: usize) 
        -> Result<Vec<(TaskId, Result<()>)>>;
    
    // Lifecycle
    pub async fn start_all_agents(&self) -> Result<()>;
    pub async fn shutdown_all_agents(&self) -> Result<()>;
    
    // Monitoring & Health
    pub fn start_health_monitoring(&mut self) -> Result<()>;
    pub fn stop_health_monitoring(&mut self);
    pub async fn metrics(&self) -> AsyncSwarmMetrics;
    pub async fn agent_statuses(&self) -> HashMap<AgentId, AgentStatus>;
    pub async fn task_queue_size(&self) -> usize;
    pub async fn assigned_tasks_count(&self) -> usize;
}
```

## Configuration Types

### SwarmConfig

Configuration for basic Swarm.

```rust
pub struct SwarmConfig {
    pub topology_type: TopologyType,
    pub distribution_strategy: DistributionStrategy,
    pub max_agents: usize,
    pub enable_auto_scaling: bool,
    pub health_check_interval_ms: u64,
}
```

### AsyncSwarmConfig

Extended configuration for AsyncSwarm.

```rust
pub struct AsyncSwarmConfig {
    // Inherits all SwarmConfig fields, plus:
    pub max_concurrent_tasks_per_agent: usize,
    pub task_timeout_ms: u64,
}
```

## Enums & Constants

### TopologyType

```rust
pub enum TopologyType {
    Mesh,        // All agents connected
    Star,        // Hub-and-spoke with coordinator
    Pipeline,    // Linear chain of agents
    Hierarchical,// Tree structure
    Clustered,   // Groups with leaders
}
```

### DistributionStrategy

```rust
pub enum DistributionStrategy {
    RoundRobin,      // Rotate through agents
    LeastLoaded,     // Assign to least busy agent
    Random,          // Random assignment
    Priority,        // Based on task priority
    CapabilityBased, // Best capability match
}
```

### AgentStatus

```rust
pub enum AgentStatus {
    Idle,     // Ready for work
    Running,  // Active and processing
    Busy,     // Processing tasks
    Offline,  // Not available
    Error,    // In error state
}
```

### HealthStatus

```rust
pub enum HealthStatus {
    Healthy,    // Fully operational
    Degraded,   // Reduced performance
    Unhealthy,  // Significant issues
    Stopping,   // Shutting down
}
```

## Error Types

### SwarmError

Comprehensive error handling for all failure modes.

```rust
pub enum SwarmError {
    AgentNotFound { id: String },
    TaskExecutionFailed { reason: String },
    InvalidTopology { reason: String },
    ResourceExhausted { resource: String },
    CommunicationError { reason: String },
    SerializationError { reason: String },
    Timeout { duration_ms: u64 },
    CapabilityMismatch { agent_id: String, capability: String },
    StrategyError { reason: String },
    Custom(String),
}

impl SwarmError {
    pub fn is_retriable(&self) -> bool;
    pub fn custom(msg: impl Into<String>) -> Self;
}
```

## Metrics Types

### SwarmMetrics

Basic metrics for Swarm.

```rust
pub struct SwarmMetrics {
    pub total_agents: usize,
    pub active_agents: usize,
    pub queued_tasks: usize,
    pub assigned_tasks: usize,
    pub total_connections: usize,
}
```

### AsyncSwarmMetrics

Extended metrics for AsyncSwarm.

```rust
pub struct AsyncSwarmMetrics {
    // Inherits all SwarmMetrics fields, plus:
    pub avg_agent_load: f64,
}
```

## Utility Functions

### Result Type

```rust
pub type Result<T> = std::result::Result<T, SwarmError>;
```

### Type Aliases

```rust
pub type AgentId = String;
pub type TaskId = String;
pub type BoxedAgent<I, O, E> = Box<dyn Agent<Input = I, Output = O, Error = E>>;
```

## Usage Examples

### Creating a Custom Agent

```rust
use ruv_swarm_core::{Agent, Result};
use async_trait::async_trait;

struct MyAgent {
    id: String,
    capabilities: Vec<String>,
}

#[async_trait]
impl Agent for MyAgent {
    type Input = String;
    type Output = String;
    type Error = String;
    
    async fn process(&mut self, input: Self::Input) -> std::result::Result<Self::Output, Self::Error> {
        // Your processing logic here
        Ok(format!("Processed: {}", input))
    }
    
    fn capabilities(&self) -> &[String] {
        &self.capabilities
    }
    
    fn id(&self) -> &str {
        &self.id
    }
}
```

### Error Handling Pattern

```rust
match result {
    Ok(data) => println!("Success: {:?}", data),
    Err(SwarmError::AgentNotFound { id }) => {
        eprintln!("Agent {} not found", id);
    },
    Err(SwarmError::ResourceExhausted { resource }) => {
        eprintln!("Not enough {}", resource);
        // Maybe scale up or wait
    },
    Err(e) if e.is_retriable() => {
        eprintln!("Retrying after error: {}", e);
        // Implement retry logic
    },
    Err(e) => {
        eprintln!("Fatal error: {}", e);
        return Err(e);
    }
}
```

For complete examples, see the [Getting Started Guide](./getting-started.md).