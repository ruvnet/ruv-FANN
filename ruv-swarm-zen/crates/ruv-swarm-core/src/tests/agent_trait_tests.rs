//! Tests for Agent trait implementations

use crate::agent::{Agent, AgentId, AgentStatus, AgentMetadata, HealthStatus};
use crate::error::{Result, SwarmError};
use crate::task::Task;
use async_trait::async_trait;

// Test implementation of Agent trait
struct TestAgent {
    id: String,
    capabilities: Vec<String>,
    status: AgentStatus,
}

impl TestAgent {
    fn new(id: impl Into<String>, capabilities: Vec<String>) -> Self {
        Self {
            id: id.into(),
            capabilities,
            status: AgentStatus::Idle,
        }
    }
}

#[async_trait]
impl Agent for TestAgent {
    type Input = String;
    type Output = String;
    type Error = SwarmError;
    
    async fn process(&mut self, input: Self::Input) -> core::result::Result<Self::Output, Self::Error> {
        self.status = AgentStatus::Busy;
        let result = Ok(format!("Processed: {input}"));
        self.status = AgentStatus::Idle;
        result
    }
    
    fn capabilities(&self) -> &[String] {
        &self.capabilities
    }
    
    fn id(&self) -> &str {
        &self.id
    }
    
    fn status(&self) -> AgentStatus {
        self.status
    }
}

#[tokio::test]
async fn test_agent_trait_implementation() {
    let mut agent = TestAgent::new("test-agent", vec!["process".to_string(), "analyze".to_string()]);
    
    // Test ID
    assert_eq!(agent.id(), "test-agent");
    
    // Test capabilities
    assert_eq!(agent.capabilities().len(), 2);
    assert!(agent.has_capability("process"));
    assert!(agent.has_capability("analyze"));
    assert!(!agent.has_capability("compute"));
    
    // Test processing
    let result = agent.process("input data".to_string()).await;
    assert_eq!(result.unwrap(), "Processed: input data");
    
    // Test status
    assert_eq!(agent.status(), AgentStatus::Idle);
}

#[tokio::test]
async fn test_agent_lifecycle() {
    let mut agent = TestAgent::new("lifecycle-agent", vec!["compute".to_string()]);
    
    // Test start
    let start_result = agent.start().await;
    assert!(start_result.is_ok());
    
    // Test health check
    let health = agent.health_check().await;
    assert_eq!(health.unwrap(), HealthStatus::Healthy);
    
    // Test shutdown
    let shutdown_result = agent.shutdown().await;
    assert!(shutdown_result.is_ok());
}

#[test]
fn test_agent_can_handle_task() {
    let agent = TestAgent::new("task-agent", vec!["compute".to_string(), "store".to_string()]);
    
    // Task with single required capability
    let task1 = Task::new("task-1", "compute")
        .require_capability("compute");
    assert!(agent.can_handle(&task1));
    
    // Task with multiple required capabilities
    let task2 = Task::new("task-2", "complex")
        .require_capability("compute")
        .require_capability("store");
    assert!(agent.can_handle(&task2));
    
    // Task with missing capability
    let task3 = Task::new("task-3", "ml")
        .require_capability("neural-net");
    assert!(!agent.can_handle(&task3));
    
    // Task with partially matching capabilities
    let task4 = Task::new("task-4", "hybrid")
        .require_capability("compute")
        .require_capability("neural-net");
    assert!(!agent.can_handle(&task4));
}

// Test agent with custom metadata
struct MetadataAgent {
    id: String,
    capabilities: Vec<String>,
    metadata: AgentMetadata,
}

#[async_trait]
impl Agent for MetadataAgent {
    type Input = Vec<u8>;
    type Output = Vec<u8>;
    type Error = String;
    
    async fn process(&mut self, input: Self::Input) -> core::result::Result<Self::Output, Self::Error> {
        // Simple echo processing
        Ok(input)
    }
    
    fn capabilities(&self) -> &[String] {
        &self.capabilities
    }
    
    fn id(&self) -> &str {
        &self.id
    }
    
    fn metadata(&self) -> AgentMetadata {
        self.metadata.clone()
    }
}

#[test]
fn test_agent_with_custom_metadata() {
    let metadata = AgentMetadata {
        name: "Advanced Agent".to_string(),
        version: "2.0.0".to_string(),
        description: "An agent with custom metadata".to_string(),
        cognitive_pattern: crate::agent::CognitivePattern::Divergent,
        resources: crate::agent::ResourceRequirements {
            min_memory_mb: 256,
            max_memory_mb: 1024,
            cpu_cores: 2.0,
            requires_gpu: true,
            network_bandwidth_mbps: 100,
        },
        metrics: crate::agent::AgentMetrics::default(),
    };
    
    let agent = MetadataAgent {
        id: "metadata-agent".to_string(),
        capabilities: vec!["advanced".to_string()],
        metadata: metadata.clone(),
    };
    
    // Test metadata retrieval
    let retrieved_metadata = agent.metadata();
    assert_eq!(retrieved_metadata.name, "Advanced Agent");
    assert_eq!(retrieved_metadata.version, "2.0.0");
    assert_eq!(retrieved_metadata.cognitive_pattern, crate::agent::CognitivePattern::Divergent);
    assert!(retrieved_metadata.resources.requires_gpu);
}

// Test agent with error handling
struct FallibleAgent {
    id: String,
    capabilities: Vec<String>,
    fail_count: u32,
    max_failures: u32,
}

#[async_trait]
impl Agent for FallibleAgent {
    type Input = String;
    type Output = String;
    type Error = SwarmError;
    
    async fn process(&mut self, input: Self::Input) -> core::result::Result<Self::Output, Self::Error> {
        if self.fail_count < self.max_failures {
            self.fail_count += 1;
            Err(SwarmError::TaskExecutionFailed {
                reason: format!("Simulated failure {}/{}", self.fail_count, self.max_failures),
            })
        } else {
            Ok(format!("Success after {} failures: {}", self.fail_count, input))
        }
    }
    
    fn capabilities(&self) -> &[String] {
        &self.capabilities
    }
    
    fn id(&self) -> &str {
        &self.id
    }
}

#[tokio::test]
async fn test_agent_error_handling() {
    let mut agent = FallibleAgent {
        id: "fallible-agent".to_string(),
        capabilities: vec!["retry".to_string()],
        fail_count: 0,
        max_failures: 2,
    };
    
    // First two calls should fail
    let result1 = agent.process("attempt 1".to_string()).await;
    assert!(result1.is_err());
    
    let result2 = agent.process("attempt 2".to_string()).await;
    assert!(result2.is_err());
    
    // Third call should succeed
    let result3 = agent.process("attempt 3".to_string()).await;
    assert!(result3.is_ok());
    assert_eq!(result3.unwrap(), "Success after 2 failures: attempt 3");
}

// Test BoxedAgent usage
#[tokio::test]
async fn test_boxed_agent() {
    use crate::agent::BoxedAgent;
    
    let agent = TestAgent::new("boxed-agent", vec!["compute".to_string()]);
    let boxed: BoxedAgent<String, String, SwarmError> = Box::new(agent);
    
    // We can't call methods directly on BoxedAgent, but we can store it
    // This test mainly verifies that the type alias works correctly
    let _stored: Vec<BoxedAgent<String, String, SwarmError>> = vec![boxed];
}