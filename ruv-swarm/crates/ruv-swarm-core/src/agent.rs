//! Core agent trait and related types

use async_trait::async_trait;
use core::fmt;
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};

use crate::error::Result;

/// Unique identifier for an agent
pub type AgentId = String;

/// Agent status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent is idle and ready for tasks
    Idle,
    /// Agent is running and ready for tasks
    Running,
    /// Agent is busy processing a task
    Busy,
    /// Agent is offline or unavailable
    Offline,
    /// Agent is in error state
    Error,
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent ID
    pub id: AgentId,
    /// Agent capabilities
    pub capabilities: Vec<String>,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Resource limits
    pub resource_limits: Option<ResourceRequirements>,
}

/// Core trait for all swarm agents
#[async_trait]
pub trait Agent: Send + Sync {
    /// Input type for this agent
    type Input: Send;

    /// Output type for this agent
    type Output: Send;

    /// Error type for this agent
    type Error: fmt::Debug + Send;

    /// Process an input and produce an output
    async fn process(
        &mut self,
        input: Self::Input,
    ) -> core::result::Result<Self::Output, Self::Error>;

    /// Get agent capabilities
    fn capabilities(&self) -> &[String];

    /// Get unique agent identifier
    fn id(&self) -> &str;

    /// Get agent metadata
    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::default()
    }

    /// Check if agent can handle a specific capability
    fn has_capability(&self, capability: &str) -> bool {
        self.capabilities().iter().any(|c| c == capability)
    }

    /// Agent health check
    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus::Healthy)
    }

    /// Get current agent status
    fn status(&self) -> AgentStatus {
        AgentStatus::Running
    }

    /// Check if agent can handle a specific task
    fn can_handle(&self, task: &crate::task::Task) -> bool {
        task.required_capabilities
            .iter()
            .all(|cap| self.has_capability(cap))
    }

    /// Lifecycle: Start the agent
    async fn start(&mut self) -> Result<()> {
        self.initialize().await
    }

    /// Lifecycle: Initialize the agent
    async fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Lifecycle: Shutdown the agent
    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Agent metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    /// Agent name
    pub name: String,

    /// Agent version
    pub version: String,

    /// Agent description
    pub description: String,

    /// Cognitive pattern
    pub cognitive_pattern: CognitivePattern,

    /// Resource requirements
    pub resources: ResourceRequirements,

    /// Performance metrics
    pub metrics: AgentMetrics,
}

impl Default for AgentMetadata {
    fn default() -> Self {
        AgentMetadata {
            name: "Unknown".to_string(),
            version: "0.0.0".to_string(),
            description: "No description".to_string(),
            cognitive_pattern: CognitivePattern::Convergent,
            resources: ResourceRequirements::default(),
            metrics: AgentMetrics::default(),
        }
    }
}

/// Cognitive patterns for agent behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitivePattern {
    /// Convergent thinking - focused, analytical
    Convergent,
    /// Divergent thinking - creative, exploratory
    Divergent,
    /// Lateral thinking - unconventional approaches
    Lateral,
    /// Systems thinking - holistic, interconnected
    Systems,
    /// Critical thinking - evaluative, questioning
    Critical,
    /// Abstract thinking - conceptual, theoretical
    Abstract,
}

impl CognitivePattern {
    /// Get all available patterns
    pub fn all() -> &'static [CognitivePattern] {
        &[
            CognitivePattern::Convergent,
            CognitivePattern::Divergent,
            CognitivePattern::Lateral,
            CognitivePattern::Systems,
            CognitivePattern::Critical,
            CognitivePattern::Abstract,
        ]
    }

    /// Get complementary pattern
    pub fn complement(&self) -> CognitivePattern {
        match self {
            CognitivePattern::Convergent => CognitivePattern::Divergent,
            CognitivePattern::Divergent => CognitivePattern::Convergent,
            CognitivePattern::Lateral => CognitivePattern::Systems,
            CognitivePattern::Systems => CognitivePattern::Lateral,
            CognitivePattern::Critical => CognitivePattern::Abstract,
            CognitivePattern::Abstract => CognitivePattern::Critical,
        }
    }
}

/// Agent health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Agent is healthy and ready
    Healthy,
    /// Agent is degraded but operational
    Degraded,
    /// Agent is unhealthy and should not receive tasks
    Unhealthy,
    /// Agent is shutting down
    Stopping,
}

/// Resource requirements for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum memory in MB
    pub min_memory_mb: u32,

    /// Maximum memory in MB
    pub max_memory_mb: u32,

    /// CPU cores required
    pub cpu_cores: f32,

    /// GPU required
    pub requires_gpu: bool,

    /// Network bandwidth in Mbps
    pub network_bandwidth_mbps: u32,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        ResourceRequirements {
            min_memory_mb: 128,
            max_memory_mb: 512,
            cpu_cores: 0.5,
            requires_gpu: false,
            network_bandwidth_mbps: 10,
        }
    }
}

/// Agent performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// Total tasks processed
    pub tasks_processed: u64,

    /// Tasks succeeded
    pub tasks_succeeded: u64,

    /// Tasks failed
    pub tasks_failed: u64,

    /// Average processing time in ms
    pub avg_processing_time_ms: f64,

    /// Current queue size
    pub queue_size: usize,

    /// Uptime in seconds
    #[cfg(feature = "std")]
    pub uptime_seconds: u64,
}

impl Default for AgentMetrics {
    fn default() -> Self {
        AgentMetrics {
            tasks_processed: 0,
            tasks_succeeded: 0,
            tasks_failed: 0,
            avg_processing_time_ms: 0.0,
            queue_size: 0,
            #[cfg(feature = "std")]
            uptime_seconds: 0,
        }
    }
}

/// Boxed agent type for dynamic dispatch
pub type BoxedAgent<I, O, E> = Box<dyn Agent<Input = I, Output = O, Error = E>>;

/// Type-erased agent trait for heterogeneous collections
#[async_trait]
pub trait ErasedAgent: Send + Sync {
    /// Get unique agent identifier
    fn id(&self) -> &str;

    /// Get agent capabilities
    fn capabilities(&self) -> &[String];

    /// Check if agent has a specific capability
    #[inline]
    fn has_capability(&self, capability: &str) -> bool {
        self.capabilities().iter().any(|c| c == capability)
    }

    /// Get current agent status
    fn status(&self) -> AgentStatus;

    /// Check if agent can handle a specific task
    fn can_handle(&self, task: &crate::task::Task) -> bool {
        task.required_capabilities
            .iter()
            .all(|cap| self.has_capability(cap))
    }

    /// Get agent metadata
    fn metadata(&self) -> AgentMetadata {
        AgentMetadata::default()
    }

    /// Agent health check
    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus::Healthy)
    }

    /// Lifecycle: Start the agent
    async fn start(&mut self) -> Result<()> {
        Ok(())
    }

    /// Lifecycle: Shutdown the agent
    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }

    /// Process a JSON value (type-erased)
    async fn process_json(&mut self, input: serde_json::Value) -> Result<serde_json::Value>;
}

/// Agent wrapper for type erasure
pub struct DynamicAgent {
    id: String,
    capabilities: Vec<String>,
    metadata: AgentMetadata,
    status: AgentStatus,
    processor: Box<dyn AgentProcessor>,
    #[cfg(feature = "std")]
    message_receiver: Option<tokio::sync::mpsc::Receiver<AgentMessage<serde_json::Value>>>,
    #[cfg(feature = "std")]
    communication_manager: Option<std::sync::Arc<crate::communication::SwarmCommunicationManager>>,
}

impl DynamicAgent {
    /// Create a new dynamic agent
    pub fn new(id: impl Into<String>, capabilities: Vec<String>) -> Self {
        DynamicAgent {
            id: id.into(),
            capabilities,
            metadata: AgentMetadata::default(),
            status: AgentStatus::Running,
            processor: Box::new(DefaultProcessor),
            #[cfg(feature = "std")]
            message_receiver: None,
            #[cfg(feature = "std")]
            communication_manager: None,
        }
    }

    /// Get agent ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get agent capabilities  
    pub fn capabilities(&self) -> &[String] {
        &self.capabilities
    }

    /// Get agent status
    pub fn status(&self) -> AgentStatus {
        self.status
    }

    /// Set agent status
    pub fn set_status(&mut self, status: AgentStatus) {
        self.status = status;
    }

    /// Check if agent can handle a task
    pub fn can_handle(&self, task: &crate::task::Task) -> bool {
        task.required_capabilities
            .iter()
            .all(|cap| self.capabilities.contains(cap))
    }

    /// Check if agent has capability
    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|c| c == capability)
    }

    /// Start the agent
    pub async fn start(&mut self) -> crate::error::Result<()> {
        self.status = AgentStatus::Running;
        Ok(())
    }

    /// Shutdown the agent
    pub async fn shutdown(&mut self) -> crate::error::Result<()> {
        self.status = AgentStatus::Offline;
        Ok(())
    }

    /// Set the message receiver for this agent
    #[cfg(feature = "std")]
    pub fn set_message_receiver(&mut self, receiver: tokio::sync::mpsc::Receiver<AgentMessage<serde_json::Value>>) {
        self.message_receiver = Some(receiver);
    }

    /// Set the communication manager for this agent
    #[cfg(feature = "std")]
    pub fn set_communication_manager(&mut self, manager: std::sync::Arc<crate::communication::SwarmCommunicationManager>) {
        self.communication_manager = Some(manager);
    }

    /// Receive a message (non-blocking)
    #[cfg(feature = "std")]
    pub fn try_receive_message(&mut self) -> Option<AgentMessage<serde_json::Value>> {
        if let Some(receiver) = &mut self.message_receiver {
            receiver.try_recv().ok()
        } else {
            None
        }
    }

    /// Receive a message (blocking)
    #[cfg(feature = "std")]
    pub async fn receive_message(&mut self) -> Option<AgentMessage<serde_json::Value>> {
        if let Some(receiver) = &mut self.message_receiver {
            receiver.recv().await
        } else {
            None
        }
    }

    /// Send a message to another agent
    #[cfg(feature = "std")]
    pub async fn send_message<T: serde::Serialize + Send + Sync + 'static>(
        &self,
        message: AgentMessage<T>,
    ) -> crate::error::Result<()> {
        if let Some(manager) = &self.communication_manager {
            manager.send_message(message).await
        } else {
            Err(crate::error::SwarmError::Custom("Communication manager not set for agent".to_string()))
        }
    }

    /// Update knowledge in the shared knowledge base
    #[cfg(feature = "std")]
    pub fn update_knowledge(&self, key: String, value: serde_json::Value) {
        if let Some(manager) = &self.communication_manager {
            manager.update_knowledge(key, value);
        }
    }

    /// Query knowledge from the shared knowledge base
    #[cfg(feature = "std")]
    pub fn query_knowledge(&self, query: &str) -> Vec<(String, crate::communication::KnowledgeEntry)> {
        if let Some(manager) = &self.communication_manager {
            manager.query_knowledge(query)
        } else {
            Vec::new()
        }
    }
}

/// Default processor for dynamic agents
struct DefaultProcessor;

#[async_trait]
impl AgentProcessor for DefaultProcessor {
    async fn process_dynamic(
        &mut self,
        input: serde_json::Value,
    ) -> crate::error::Result<serde_json::Value> {
        Ok(input)
    }
}

/// Internal trait for type-erased processing
#[async_trait]
trait AgentProcessor: Send + Sync {
    async fn process_dynamic(&mut self, input: serde_json::Value) -> Result<serde_json::Value>;
}

/// Agent communication message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage<T> {
    /// Source agent ID
    pub from: String,

    /// Target agent ID
    pub to: String,

    /// Message payload
    pub payload: T,

    /// Message type
    pub msg_type: MessageType,

    /// Correlation ID for request/response
    pub correlation_id: Option<String>,

    /// Semantic type of the information being communicated (e.g., "task_status", "intermediate_result", "environmental_data")
    pub info_type: Option<String>,

    /// Contextual metadata relevant to the message (e.g., task_id, sub_task_id, data_schema_version)
    pub context: Option<serde_json::Value>,

    /// Urgency or priority of the message
    pub urgency: Option<MessageUrgency>,
}

/// Types of agent messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Task assignment
    TaskAssignment,
    /// Task result
    TaskResult,
    /// Status update
    StatusUpdate,
    /// Coordination message
    Coordination,
    /// Error report
    Error,
    /// Agent requests specific information
    InformationRequest,
    /// Agent shares information proactively
    InformationShare,
    /// Agent queries a knowledge base
    Query,
    /// Response to a query or request
    Response,
}

/// Urgency levels for messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageUrgency {
    /// Low priority message
    Low,
    /// Medium priority message
    Medium,
    /// High priority message
    High,
    /// Critical priority message requiring immediate attention
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_pattern_complement() {
        assert_eq!(
            CognitivePattern::Convergent.complement(),
            CognitivePattern::Divergent
        );
        assert_eq!(
            CognitivePattern::Divergent.complement(),
            CognitivePattern::Convergent
        );
        assert_eq!(
            CognitivePattern::Lateral.complement(),
            CognitivePattern::Systems
        );
    }

    #[test]
    fn test_agent_metadata_default() {
        let metadata = AgentMetadata::default();
        assert_eq!(metadata.name, "Unknown");
        assert_eq!(metadata.cognitive_pattern, CognitivePattern::Convergent);
    }
}
