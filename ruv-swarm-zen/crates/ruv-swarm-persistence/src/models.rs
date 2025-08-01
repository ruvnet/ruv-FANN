//! Data models for persistence layer

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Agent model representing a swarm agent
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

impl AgentModel {
    pub fn new(name: String, agent_type: String, capabilities: Vec<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            agent_type,
            status: AgentStatus::Initializing,
            capabilities,
            metadata: HashMap::new(),
            heartbeat: now,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn update_heartbeat(&mut self) {
        self.heartbeat = Utc::now();
        self.updated_at = Utc::now();
    }

    pub fn set_status(&mut self, status: AgentStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }
}

/// Agent status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    Initializing,
    Active,
    Idle,
    Busy,
    Paused,
    Error,
    Shutdown,
}

impl std::fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status_str = match self {
            AgentStatus::Initializing => "initializing",
            AgentStatus::Active => "active",
            AgentStatus::Idle => "idle",
            AgentStatus::Busy => "busy",
            AgentStatus::Paused => "paused",
            AgentStatus::Error => "error",
            AgentStatus::Shutdown => "shutdown",
        };
        write!(f, "{}", status_str)
    }
}

/// Task model representing work items in the swarm
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

impl TaskModel {
    pub fn new(task_type: String, payload: serde_json::Value, priority: TaskPriority) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            task_type,
            priority,
            status: TaskStatus::Pending,
            assigned_to: None,
            payload,
            result: None,
            error: None,
            retry_count: 0,
            max_retries: 3,
            dependencies: Vec::new(),
            created_at: now,
            updated_at: now,
            started_at: None,
            completed_at: None,
        }
    }

    pub fn assign_to(&mut self, agent_id: &str) {
        self.assigned_to = Some(agent_id.to_string());
        self.status = TaskStatus::Assigned;
        self.updated_at = Utc::now();
    }

    pub fn start(&mut self) {
        self.status = TaskStatus::Running;
        self.started_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }

    pub fn complete(&mut self, result: serde_json::Value) {
        self.status = TaskStatus::Completed;
        self.result = Some(result);
        self.completed_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }

    pub fn fail(&mut self, error: String) {
        self.retry_count += 1;
        if self.retry_count >= self.max_retries {
            self.status = TaskStatus::Failed;
        } else {
            self.status = TaskStatus::Pending;
            self.assigned_to = None;
        }
        self.error = Some(error);
        self.updated_at = Utc::now();
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum TaskPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Task status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Assigned,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Event model for event sourcing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventModel {
    pub id: String,
    pub event_type: String,
    pub agent_id: Option<String>,
    pub task_id: Option<String>,
    pub payload: serde_json::Value,
    pub metadata: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
    pub sequence: u64,
}

impl EventModel {
    pub fn new(event_type: String, payload: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            event_type,
            agent_id: None,
            task_id: None,
            payload,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            sequence: 0, // Should be set by storage layer
        }
    }

    pub fn with_agent(mut self, agent_id: String) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    pub fn with_task(mut self, task_id: String) -> Self {
        self.task_id = Some(task_id);
        self
    }
}

/// Message model for inter-agent communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageModel {
    pub id: String,
    pub from_agent: String,
    pub to_agent: String,
    pub message_type: String,
    pub content: serde_json::Value,
    pub priority: MessagePriority,
    pub read: bool,
    pub created_at: DateTime<Utc>,
    pub read_at: Option<DateTime<Utc>>,
}

impl MessageModel {
    pub fn new(
        from_agent: String,
        to_agent: String,
        message_type: String,
        content: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            from_agent,
            to_agent,
            message_type,
            content,
            priority: MessagePriority::Normal,
            read: false,
            created_at: Utc::now(),
            read_at: None,
        }
    }

    pub fn with_priority(mut self, priority: MessagePriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn mark_read(&mut self) {
        self.read = true;
        self.read_at = Some(Utc::now());
    }
}

/// Message priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Urgent,
}

/// Metric model for performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricModel {
    pub id: String,
    pub metric_type: String,
    pub agent_id: Option<String>,
    pub value: f64,
    pub unit: String,
    pub tags: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

impl MetricModel {
    pub fn new(metric_type: String, value: f64, unit: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            metric_type,
            agent_id: None,
            value,
            unit,
            tags: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    pub fn with_agent(mut self, agent_id: String) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_model() {
        let mut agent = AgentModel::new(
            "test-agent".to_string(),
            "worker".to_string(),
            vec!["compute".to_string(), "storage".to_string()],
        );

        assert_eq!(agent.status, AgentStatus::Initializing);
        agent.set_status(AgentStatus::Active);
        assert_eq!(agent.status, AgentStatus::Active);
    }

    #[test]
    fn test_task_model() {
        let mut task = TaskModel::new(
            "process".to_string(),
            serde_json::json!({"data": "test"}),
            TaskPriority::High,
        );

        assert_eq!(task.status, TaskStatus::Pending);
        task.assign_to("agent-1");
        assert_eq!(task.status, TaskStatus::Assigned);
        assert_eq!(task.assigned_to, Some("agent-1".to_string()));
    }
}
