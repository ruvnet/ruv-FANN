//! Simple agent tests that match the actual API

use crate::agent::*;
use crate::task::{Task, TaskResult, TaskStatus};
use crate::error::Result;

#[cfg(test)]
use crate::MockAgent;

#[tokio::test]
async fn test_mock_agent_creation() {
    let agent = MockAgent::new("test-agent");
    assert_eq!(agent.id(), "test-agent");
    assert_eq!(agent.status(), AgentStatus::Idle);
}

#[tokio::test]
async fn test_mock_agent_start_shutdown() {
    let mut agent = MockAgent::new("test-agent");
    
    agent.start().await.unwrap();
    assert_eq!(agent.status(), AgentStatus::Running);
    
    agent.shutdown().await.unwrap();
    assert_eq!(agent.status(), AgentStatus::Offline);
}

#[tokio::test]
async fn test_mock_agent_capabilities() {
    let capabilities = vec![
        Capability::new("compute", "1.0"),
        Capability::new("analyze", "2.0"),
    ];
    
    let agent = MockAgent::new("test-agent")
        .with_capabilities(capabilities);
    
    assert_eq!(agent.capabilities().len(), 2);
    assert_eq!(agent.capabilities()[0].name, "compute");
    assert_eq!(agent.capabilities()[1].name, "analyze");
}

#[tokio::test]
async fn test_mock_agent_can_handle_task() {
    let agent = MockAgent::new("test-agent")
        .with_capabilities(vec![
            Capability::new("compute", "1.0"),
        ]);
    
    let task1 = Task::new("task-1", "compute")
        .require_capability("compute");
    assert!(agent.can_handle(&task1));
    
    let task2 = Task::new("task-2", "analyze")
        .require_capability("analyze");
    assert!(!agent.can_handle(&task2));
}

#[tokio::test]
async fn test_mock_agent_process_task() {
    let mut agent = MockAgent::new("test-agent")
        .with_process_result(Ok(TaskResult::success("Test output")));
    
    let task = Task::new("task-1", "compute");
    let result = agent.process(task).await.unwrap();
    
    assert_eq!(result.status, TaskStatus::Completed);
    assert!(result.output.is_some());
}

#[test]
fn test_capability_creation() {
    let cap = Capability::new("neural-processing", "2.0");
    assert_eq!(cap.name, "neural-processing");
    assert_eq!(cap.version, "2.0");
}

#[test]
fn test_capability_equality() {
    let cap1 = Capability::new("compute", "1.0");
    let cap2 = Capability::new("compute", "1.0");
    let cap3 = Capability::new("compute", "2.0");
    
    assert_eq!(cap1, cap2);
    assert_ne!(cap1, cap3);
}

#[test]
fn test_agent_status_values() {
    let statuses = [AgentStatus::Idle,
        AgentStatus::Running,
        AgentStatus::Busy,
        AgentStatus::Offline,
        AgentStatus::Error];
    
    // Verify all statuses are distinct
    for (i, status1) in statuses.iter().enumerate() {
        for (j, status2) in statuses.iter().enumerate() {
            if i == j {
                assert_eq!(status1, status2);
            } else {
                assert_ne!(status1, status2);
            }
        }
    }
}

#[test]
fn test_agent_metadata_default() {
    let metadata = AgentMetadata::default();
    assert_eq!(metadata.name, "Unknown");
    assert_eq!(metadata.version, "0.0.0");
    assert!(!metadata.description.is_empty());
}

#[test]
fn test_cognitive_patterns() {
    let patterns = CognitivePattern::all();
    assert!(patterns.len() >= 6);
    
    // Test complement relationships
    assert_eq!(CognitivePattern::Convergent.complement(), CognitivePattern::Divergent);
    assert_eq!(CognitivePattern::Divergent.complement(), CognitivePattern::Convergent);
}

#[test]
fn test_dynamic_agent_creation() {
    let agent = DynamicAgent::new("dynamic-agent", vec!["compute".to_string(), "analyze".to_string()]);
    assert_eq!(agent.id(), "dynamic-agent");
    assert_eq!(agent.capabilities(), &["compute", "analyze"]);
    assert_eq!(agent.status(), AgentStatus::Running);
}

#[test]
fn test_dynamic_agent_status_transitions() {
    let mut agent = DynamicAgent::new("dynamic-agent", vec!["compute".to_string()]);
    
    // Initial status
    assert_eq!(agent.status(), AgentStatus::Running);
    
    // Set different statuses
    agent.set_status(AgentStatus::Busy);
    assert_eq!(agent.status(), AgentStatus::Busy);
    
    agent.set_status(AgentStatus::Idle);
    assert_eq!(agent.status(), AgentStatus::Idle);
    
    agent.set_status(AgentStatus::Error);
    assert_eq!(agent.status(), AgentStatus::Error);
    
    agent.set_status(AgentStatus::Offline);
    assert_eq!(agent.status(), AgentStatus::Offline);
}

#[test]
fn test_dynamic_agent_capability_checking() {
    let agent = DynamicAgent::new("dynamic-agent", vec![
        "compute".to_string(),
        "analyze".to_string(),
        "neural-processing".to_string(),
    ]);
    
    assert!(agent.has_capability("compute"));
    assert!(agent.has_capability("analyze"));
    assert!(agent.has_capability("neural-processing"));
    assert!(!agent.has_capability("quantum-compute"));
}

#[test]
fn test_dynamic_agent_task_handling() {
    let agent = DynamicAgent::new("dynamic-agent", vec![
        "compute".to_string(),
        "analyze".to_string(),
    ]);
    
    // Task with matching capabilities
    let task1 = Task::new("task-1", "compute")
        .require_capability("compute");
    assert!(agent.can_handle(&task1));
    
    // Task with multiple required capabilities
    let task2 = Task::new("task-2", "complex")
        .require_capability("compute")
        .require_capability("analyze");
    assert!(agent.can_handle(&task2));
    
    // Task with missing capability
    let task3 = Task::new("task-3", "special")
        .require_capability("quantum-compute");
    assert!(!agent.can_handle(&task3));
    
    // Task with partially matching capabilities
    let task4 = Task::new("task-4", "partial")
        .require_capability("compute")
        .require_capability("quantum-compute");
    assert!(!agent.can_handle(&task4));
}

#[tokio::test]
async fn test_dynamic_agent_lifecycle() {
    let mut agent = DynamicAgent::new("dynamic-agent", vec!["compute".to_string()]);
    
    // Start agent
    let result = agent.start().await;
    assert!(result.is_ok());
    assert_eq!(agent.status(), AgentStatus::Running);
    
    // Shutdown agent
    let result = agent.shutdown().await;
    assert!(result.is_ok());
    assert_eq!(agent.status(), AgentStatus::Offline);
    
    // Can restart
    let result = agent.start().await;
    assert!(result.is_ok());
    assert_eq!(agent.status(), AgentStatus::Running);
}

#[test]
fn test_health_status_values() {
    let statuses = [HealthStatus::Healthy,
        HealthStatus::Degraded,
        HealthStatus::Unhealthy,
        HealthStatus::Stopping];
    
    // Verify all statuses are distinct
    for (i, status1) in statuses.iter().enumerate() {
        for (j, status2) in statuses.iter().enumerate() {
            if i == j {
                assert_eq!(status1, status2);
            } else {
                assert_ne!(status1, status2);
            }
        }
    }
}

#[test]
fn test_resource_requirements_default() {
    let resources = ResourceRequirements::default();
    assert_eq!(resources.min_memory_mb, 128);
    assert_eq!(resources.max_memory_mb, 512);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(resources.cpu_cores, 0.5);
    }
    assert!(!resources.requires_gpu);
    assert_eq!(resources.network_bandwidth_mbps, 10);
}

#[test]
fn test_resource_requirements_custom() {
    let resources = ResourceRequirements {
        min_memory_mb: 1024,
        max_memory_mb: 4096,
        cpu_cores: 2.0,
        requires_gpu: true,
        network_bandwidth_mbps: 100,
    };
    
    assert_eq!(resources.min_memory_mb, 1024);
    assert_eq!(resources.max_memory_mb, 4096);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(resources.cpu_cores, 2.0);
    }
    assert!(resources.requires_gpu);
    assert_eq!(resources.network_bandwidth_mbps, 100);
}

#[test]
fn test_agent_metrics_default() {
    let metrics = AgentMetrics::default();
    assert_eq!(metrics.tasks_processed, 0);
    assert_eq!(metrics.tasks_succeeded, 0);
    assert_eq!(metrics.tasks_failed, 0);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(metrics.avg_processing_time_ms, 0.0);
    }
    assert_eq!(metrics.queue_size, 0);
    #[cfg(feature = "std")]
    assert_eq!(metrics.uptime_seconds, 0);
}

#[test]
fn test_agent_metrics_tracking() {
    // Simulate processing tasks
    let metrics = AgentMetrics {
        tasks_processed: 100,
        tasks_succeeded: 95,
        tasks_failed: 5,
        avg_processing_time_ms: 25.5,
        queue_size: 10,
        ..Default::default()
    };
    
    assert_eq!(metrics.tasks_processed, 100);
    assert_eq!(metrics.tasks_succeeded, 95);
    assert_eq!(metrics.tasks_failed, 5);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(metrics.avg_processing_time_ms, 25.5);
    }
    assert_eq!(metrics.queue_size, 10);
    
    // Calculate success rate
    #[allow(clippy::cast_precision_loss)]
    let success_rate = metrics.tasks_succeeded as f64 / metrics.tasks_processed as f64;
    assert!((success_rate - 0.95).abs() < 0.001);
}

#[test]
fn test_agent_metadata_with_cognitive_pattern() {
    // Test different cognitive patterns
    let mut metadata = AgentMetadata {
        cognitive_pattern: CognitivePattern::Divergent,
        ..Default::default()
    };
    assert_eq!(metadata.cognitive_pattern, CognitivePattern::Divergent);
    
    metadata.cognitive_pattern = CognitivePattern::Systems;
    assert_eq!(metadata.cognitive_pattern, CognitivePattern::Systems);
    
    metadata.cognitive_pattern = CognitivePattern::Critical;
    assert_eq!(metadata.cognitive_pattern, CognitivePattern::Critical);
}

#[test]
fn test_cognitive_pattern_complement_all() {
    // Test all complement relationships
    assert_eq!(CognitivePattern::Convergent.complement(), CognitivePattern::Divergent);
    assert_eq!(CognitivePattern::Divergent.complement(), CognitivePattern::Convergent);
    assert_eq!(CognitivePattern::Lateral.complement(), CognitivePattern::Systems);
    assert_eq!(CognitivePattern::Systems.complement(), CognitivePattern::Lateral);
    assert_eq!(CognitivePattern::Critical.complement(), CognitivePattern::Abstract);
    assert_eq!(CognitivePattern::Abstract.complement(), CognitivePattern::Critical);
}

#[test]
fn test_capability_serialization() {
    let cap = Capability::new("neural-net", "3.0");
    
    // Serialize
    let json = serde_json::to_string(&cap).unwrap();
    
    // Deserialize
    let deserialized: Capability = serde_json::from_str(&json).unwrap();
    
    assert_eq!(cap, deserialized);
    assert_eq!(deserialized.name, "neural-net");
    assert_eq!(deserialized.version, "3.0");
}

#[test]
fn test_agent_status_serialization() {
    let statuses = vec![
        AgentStatus::Idle,
        AgentStatus::Running,
        AgentStatus::Busy,
        AgentStatus::Offline,
        AgentStatus::Error,
    ];
    
    for status in statuses {
        let json = serde_json::to_string(&status).unwrap();
        let deserialized: AgentStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, deserialized);
    }
}

#[test]
fn test_health_status_serialization() {
    let statuses = vec![
        HealthStatus::Healthy,
        HealthStatus::Degraded,
        HealthStatus::Unhealthy,
        HealthStatus::Stopping,
    ];
    
    for status in statuses {
        let json = serde_json::to_string(&status).unwrap();
        let deserialized: HealthStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, deserialized);
    }
}

#[test]
fn test_agent_config_creation() {
    let config = AgentConfig {
        id: "test-agent".to_string(),
        capabilities: vec!["compute".to_string(), "analyze".to_string()],
        max_concurrent_tasks: 10,
        resource_limits: Some(ResourceRequirements {
            min_memory_mb: 256,
            max_memory_mb: 1024,
            cpu_cores: 1.5,
            requires_gpu: false,
            network_bandwidth_mbps: 50,
        }),
    };
    
    assert_eq!(config.id, "test-agent");
    assert_eq!(config.capabilities.len(), 2);
    assert_eq!(config.max_concurrent_tasks, 10);
    assert!(config.resource_limits.is_some());
    
    let resources = config.resource_limits.unwrap();
    assert_eq!(resources.min_memory_mb, 256);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(resources.cpu_cores, 1.5);
    }
}

#[tokio::test]
async fn test_mock_agent_health_check() {
    let agent = MockAgent::new("test-agent");
    let health = agent.health_check().await.unwrap();
    assert_eq!(health, HealthStatus::Healthy);
}

#[tokio::test]
async fn test_mock_agent_process_error() {
    let mut agent = MockAgent::new("test-agent")
        .with_process_result(Err(crate::error::SwarmError::custom("Processing failed")));
    
    let task = Task::new("task-1", "compute");
    let result = agent.process(task).await;
    
    assert!(result.is_err());
}

#[tokio::test]
async fn test_mock_agent_metadata() {
    let agent = MockAgent::new("test-agent");
    let metadata = agent.metadata();
    
    assert_eq!(metadata.name, "Unknown");
    assert_eq!(metadata.version, "0.0.0");
    assert_eq!(metadata.cognitive_pattern, CognitivePattern::Convergent);
}

#[test]
fn test_agent_metadata_complete() {
    let metadata = AgentMetadata {
        name: "AdvancedAgent".to_string(),
        version: "2.1.0".to_string(),
        description: "Advanced processing agent".to_string(),
        cognitive_pattern: CognitivePattern::Systems,
        resources: ResourceRequirements {
            min_memory_mb: 512,
            max_memory_mb: 2048,
            cpu_cores: 2.0,
            requires_gpu: true,
            network_bandwidth_mbps: 100,
        },
        metrics: AgentMetrics {
            tasks_processed: 1000,
            tasks_succeeded: 950,
            tasks_failed: 50,
            avg_processing_time_ms: 15.5,
            queue_size: 5,
            #[cfg(feature = "std")]
            uptime_seconds: 3600,
        },
    };
    
    assert_eq!(metadata.name, "AdvancedAgent");
    assert_eq!(metadata.version, "2.1.0");
    assert_eq!(metadata.cognitive_pattern, CognitivePattern::Systems);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(metadata.resources.cpu_cores, 2.0);
    }
    assert_eq!(metadata.metrics.tasks_processed, 1000);
}