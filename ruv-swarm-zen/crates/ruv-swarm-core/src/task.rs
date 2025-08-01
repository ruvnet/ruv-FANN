//! Task definitions and task distribution logic

#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};

use core::fmt;

/// Unique identifier for a task
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TaskId(pub String);

impl TaskId {
    /// Create a new task ID
    pub fn new(id: impl Into<String>) -> Self {
        TaskId(id.into())
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[derive(Default)]
pub enum TaskPriority {
    /// Low priority task - executed when no higher priority tasks are available
    Low = 0,
    /// Normal priority task - default priority level for most tasks
    #[default]
    Normal = 1,
    /// High priority task - prioritized over normal and low priority tasks
    High = 2,
    /// Critical priority task - highest priority, executed immediately
    Critical = 3,
}


/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is waiting to be assigned
    Pending,
    /// Task has been assigned to an agent
    Assigned,
    /// Task is currently being processed
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
    /// Task timed out
    TimedOut,
}

/// Task definition
#[derive(Debug, Clone)]
pub struct Task {
    /// Unique identifier for this task
    pub id: TaskId,
    /// Type of task (e.g., "compute", "io", "network")
    pub task_type: String,
    /// Priority level for task scheduling
    pub priority: TaskPriority,
    /// Task data payload
    pub payload: TaskPayload,
    /// List of capabilities required by agents to execute this task
    pub required_capabilities: Vec<String>,
    /// Optional timeout in milliseconds for task execution
    pub timeout_ms: Option<u64>,
    /// Current number of retry attempts
    pub retry_count: u32,
    /// Maximum number of retry attempts allowed
    pub max_retries: u32,
}

impl Task {
    /// Create a new task
    pub fn new(id: impl Into<String>, task_type: impl Into<String>) -> Self {
        Task {
            id: TaskId::new(id),
            task_type: task_type.into(),
            priority: TaskPriority::Normal,
            payload: TaskPayload::Empty,
            required_capabilities: Vec::new(),
            timeout_ms: None,
            retry_count: 0,
            max_retries: 3,
        }
    }

    /// Set task priority
    #[must_use]
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set task payload
    #[must_use]
    pub fn with_payload(mut self, payload: TaskPayload) -> Self {
        self.payload = payload;
        self
    }

    /// Add required capability
    #[must_use]
    pub fn require_capability(mut self, capability: impl Into<String>) -> Self {
        self.required_capabilities.push(capability.into());
        self
    }

    /// Set timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
    
    /// Check if task can be retried
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }
    
    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
}

/// Task payload variants
#[derive(Debug, Clone)]
pub enum TaskPayload {
    /// Empty payload
    Empty,
    /// Text data
    Text(String),
    /// Binary data
    Binary(Vec<u8>),
    /// JSON data (as string)
    Json(String),
    /// Custom data
    Custom(Box<dyn CustomPayload>),
}

/// Trait for custom task payloads
pub trait CustomPayload: Send + Sync + fmt::Debug {
    /// Clone the payload
    fn clone_box(&self) -> Box<dyn CustomPayload>;
}

impl Clone for Box<dyn CustomPayload> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Task result
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// The unique identifier of the task that was executed
    pub task_id: TaskId,
    /// Final status of the task execution
    pub status: TaskStatus,
    /// Optional output data produced by the task
    pub output: Option<TaskOutput>,
    /// Optional error message if the task failed
    pub error: Option<String>,
    /// Total execution time in milliseconds
    pub execution_time_ms: u64,
}

impl TaskResult {
    /// Create a successful task result
    pub fn success(output: impl Into<TaskOutput>) -> Self {
        TaskResult {
            task_id: TaskId::new(""),
            status: TaskStatus::Completed,
            output: Some(output.into()),
            error: None,
            execution_time_ms: 0,
        }
    }

    /// Create a failed task result
    pub fn failure(error: impl Into<String>) -> Self {
        TaskResult {
            task_id: TaskId::new(""),
            status: TaskStatus::Failed,
            output: None,
            error: Some(error.into()),
            execution_time_ms: 0,
        }
    }

    /// Set task ID
    #[must_use]
    pub fn with_task_id(mut self, task_id: TaskId) -> Self {
        self.task_id = task_id;
        self
    }

    /// Set execution time
    #[must_use]
    pub fn with_execution_time(mut self, time_ms: u64) -> Self {
        self.execution_time_ms = time_ms;
        self
    }
}

/// Task output variants
#[derive(Debug, Clone)]
pub enum TaskOutput {
    /// Text output
    Text(String),
    /// Binary output
    Binary(Vec<u8>),
    /// JSON output (as string)
    Json(String),
    /// No output
    None,
}

impl From<String> for TaskOutput {
    fn from(s: String) -> Self {
        TaskOutput::Text(s)
    }
}

impl From<&str> for TaskOutput {
    fn from(s: &str) -> Self {
        TaskOutput::Text(s.to_string())
    }
}

impl From<Vec<u8>> for TaskOutput {
    fn from(bytes: Vec<u8>) -> Self {
        TaskOutput::Binary(bytes)
    }
}

/// Task distribution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum DistributionStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least loaded agent first
    #[default]
    LeastLoaded,
    /// Random distribution
    Random,
    /// Priority-based distribution
    Priority,
    /// Capability-based distribution
    CapabilityBased,
}

