//! Simple error handling tests that match the actual API

use crate::error::*;

#[test]
fn test_swarm_error_variants() {
    // Test that all error variants can be created correctly
    let agent_not_found = SwarmError::AgentNotFound { 
        id: "test-agent".to_string() 
    };
    
    let task_execution_failed = SwarmError::TaskExecutionFailed { 
        reason: "test task failed".to_string() 
    };
    
    let invalid_topology = SwarmError::InvalidTopology { 
        reason: "Test topology error".to_string() 
    };
    
    let resource_exhausted = SwarmError::ResourceExhausted { 
        resource: "memory".to_string(),
    };
    
    let communication_error = SwarmError::CommunicationError { 
        reason: "Connection failed".to_string() 
    };
    
    let serialization_error = SwarmError::SerializationError { 
        reason: "Failed to serialize data".to_string() 
    };
    
    let timeout_error = SwarmError::Timeout { 
        duration_ms: 5000,
    };
    
    let capability_mismatch = SwarmError::CapabilityMismatch { 
        agent_id: "agent-1".to_string(),
        capability: "compute".to_string(),
    };
    
    let strategy_error = SwarmError::StrategyError { 
        reason: "Strategy failed".to_string() 
    };
    
    let custom_error = SwarmError::Custom("Custom error message".to_string());
    
    // Verify all errors are distinct
    let errors = vec![
        agent_not_found,
        task_execution_failed,
        invalid_topology,
        resource_exhausted,
        communication_error,
        serialization_error,
        timeout_error,
        capability_mismatch,
        strategy_error,
        custom_error,
    ];
    
    for (i, error1) in errors.iter().enumerate() {
        for (j, error2) in errors.iter().enumerate() {
            if i != j {
                // Using debug format for comparison since SwarmError implements PartialEq
                assert_ne!(error1, error2);
            }
        }
    }
}

#[test]
fn test_swarm_error_display() {
    let error = SwarmError::AgentNotFound { 
        id: "missing-agent".to_string() 
    };
    
    let error_string = error.to_string();
    assert!(error_string.contains("missing-agent"));
}

#[test]
fn test_swarm_error_retriable() {
    // Test which errors are retriable
    let retriable_error = SwarmError::CommunicationError { 
        reason: "Temporary network issue".to_string() 
    };
    assert!(retriable_error.is_retriable());
    
    let timeout_error = SwarmError::Timeout { duration_ms: 5000 };
    assert!(timeout_error.is_retriable());
    
    let resource_error = SwarmError::ResourceExhausted { 
        resource: "memory".to_string() 
    };
    assert!(resource_error.is_retriable());
    
    // Non-retriable errors
    let config_error = SwarmError::InvalidTopology { 
        reason: "Invalid configuration".to_string() 
    };
    assert!(!config_error.is_retriable());
}

#[test]
fn test_custom_error_creation() {
    let error = SwarmError::custom("Custom message");
    match error {
        SwarmError::Custom(msg) => assert_eq!(msg, "Custom message"),
        _ => panic!("Expected custom error"),
    }
}

#[test]
fn test_agent_error() {
    let swarm_error = SwarmError::AgentNotFound { 
        id: "test-agent".to_string() 
    };
    
    let agent_error = AgentError {
        agent_id: "test-agent".to_string(),
        error: swarm_error,
    };
    
    assert_eq!(agent_error.agent_id, "test-agent");
    assert!(format!("{agent_error}").contains("test-agent"));
}

// Test error propagation and handling in swarm operations
#[test]
fn test_error_propagation_in_swarm() {
    use crate::swarm::{Swarm, SwarmConfig};
    use crate::agent::DynamicAgent;
    
    let config = SwarmConfig {
        max_agents: 1, // Very low limit
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    // Register first agent - should succeed
    let agent1 = DynamicAgent::new("agent-1", vec!["compute".to_string()]);
    assert!(swarm.register_agent(agent1).is_ok());
    
    // Try to register second agent - should fail with ResourceExhausted
    let agent2 = DynamicAgent::new("agent-2", vec!["compute".to_string()]);
    match swarm.register_agent(agent2) {
        Err(SwarmError::ResourceExhausted { resource }) => {
            assert_eq!(resource, "agent slots");
        }
        _ => panic!("Expected ResourceExhausted error"),
    }
    
    // Try to unregister non-existent agent
    match swarm.unregister_agent(&"non-existent".to_string()) {
        Err(SwarmError::AgentNotFound { id }) => {
            assert_eq!(id, "non-existent");
        }
        _ => panic!("Expected AgentNotFound error"),
    }
}

// Test edge cases in topology operations
#[test]
fn test_topology_edge_case_errors() {
    use crate::topology::{Topology, TopologyType};
    
    let mut topology = Topology::new(TopologyType::Pipeline);
    
    // Test operations on empty topology
    assert!(topology.get_neighbors(&"non-existent".to_string()).is_none());
    assert!(!topology.are_connected(&"a".to_string(), &"b".to_string()));
    
    // Add and remove same connection multiple times
    topology.add_connection("a".to_string(), "b".to_string());
    topology.add_connection("a".to_string(), "b".to_string()); // Duplicate
    topology.remove_connection(&"a".to_string(), &"b".to_string());
    topology.remove_connection(&"a".to_string(), &"b".to_string()); // Already removed
    
    // Should handle gracefully
    assert!(!topology.are_connected(&"a".to_string(), &"b".to_string()));
}

// Test task error scenarios
#[test]
fn test_task_error_scenarios() {
    use crate::task::{Task, TaskStatus, TaskResult};
    
    // Create task with error result
    let task = Task::new("error-task", "compute");
    let error_result = TaskResult::failure("Task failed due to error");
    
    assert_eq!(error_result.status, TaskStatus::Failed);
    assert!(error_result.error.is_some());
    assert_eq!(error_result.error.unwrap(), "Task failed due to error");
}

// Test async swarm error handling
#[cfg(feature = "std")]
#[tokio::test]
async fn test_async_swarm_error_handling() {
    use crate::async_swarm::{AsyncSwarm, AsyncSwarmConfig};
    use crate::agent::DynamicAgent;
    
    let config = AsyncSwarmConfig {
        max_agents: 2,
        ..Default::default()
    };
    let swarm = AsyncSwarm::new(config);
    
    // Fill up the swarm
    for i in 0..2 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        assert!(swarm.register_agent(agent).await.is_ok());
    }
    
    // Try to exceed limit
    let agent = DynamicAgent::new("overflow", vec!["compute".to_string()]);
    match swarm.register_agent(agent).await {
        Err(SwarmError::ResourceExhausted { .. }) => {} // Expected
        _ => panic!("Expected ResourceExhausted error"),
    }
    
    // Try to unregister non-existent agent
    match swarm.unregister_agent(&"ghost".to_string()).await {
        Err(SwarmError::AgentNotFound { .. }) => {} // Expected
        _ => panic!("Expected AgentNotFound error"),
    }
}

// Test error recovery patterns
#[test]
fn test_error_recovery_patterns() {
    use crate::error::{Result, SwarmError};
    
    fn operation_that_might_fail(should_fail: bool) -> Result<String> {
        if should_fail {
            Err(SwarmError::CommunicationError {
                reason: "Network error".to_string(),
            })
        } else {
            Ok("Success".to_string())
        }
    }
    
    // Test retry logic with retriable error
    let mut attempts = 0;
    let max_attempts = 3;
    let mut result = Err(SwarmError::Custom("Initial".to_string()));
    
    while attempts < max_attempts {
        result = operation_that_might_fail(attempts < 2);
        attempts += 1;
        
        match &result {
            Err(e) if e.is_retriable() => {},
            _ => break,
        }
    }
    
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Success");
}

// Test error serialization
#[test]
fn test_error_serialization() {
    let errors = vec![
        SwarmError::AgentNotFound { id: "test".to_string() },
        SwarmError::TaskExecutionFailed { reason: "failed".to_string() },
        SwarmError::InvalidTopology { reason: "bad".to_string() },
        SwarmError::ResourceExhausted { resource: "cpu".to_string() },
        SwarmError::CommunicationError { reason: "net".to_string() },
        SwarmError::SerializationError { reason: "json".to_string() },
        SwarmError::Timeout { duration_ms: 1000 },
        SwarmError::CapabilityMismatch { 
            agent_id: "agent".to_string(), 
            capability: "cap".to_string() 
        },
        SwarmError::StrategyError { reason: "strat".to_string() },
        SwarmError::Custom("custom".to_string()),
    ];
    
    // Test that all errors can be converted to strings
    for error in errors {
        let error_string = error.to_string();
        assert!(!error_string.is_empty());
        
        // Test debug representation
        let debug_string = format!("{error:?}");
        assert!(!debug_string.is_empty());
    }
}

// Test concurrent error scenarios
#[cfg(feature = "std")]
#[tokio::test]
async fn test_concurrent_error_scenarios() {
    use crate::async_swarm::{AsyncSwarm, AsyncSwarmConfig};
    use crate::agent::DynamicAgent;
    use crate::task::Task;
    use futures::future::join_all;
    
    let config = AsyncSwarmConfig {
        max_agents: 5,
        ..Default::default()
    };
    let swarm = AsyncSwarm::new(config);
    
    // Register agents
    for i in 0..3 {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(agent).await.unwrap();
    }
    
    // Submit tasks concurrently
    let task_futures = (0..10).map(|i| {
        let task = Task::new(format!("task-{i}"), "compute")
            .require_capability("compute");
        swarm.submit_task(task)
    });
    
    let results = join_all(task_futures).await;
    
    // All should succeed
    for result in results {
        assert!(result.is_ok());
    }
    
    // Try concurrent agent registration that exceeds limit
    let agent_futures = (3..8).map(|i| {
        let agent = DynamicAgent::new(format!("agent-{i}"), vec!["compute".to_string()]);
        swarm.register_agent(agent)
    });
    
    let results = join_all(agent_futures).await;
    
    // Some should fail due to max agents limit
    let failures = results.iter().filter(|r| r.is_err()).count();
    assert!(failures >= 3); // At least 3 should fail (max_agents = 5, already have 3)
}

// Test error conditions in agent operations
#[tokio::test]
async fn test_agent_error_conditions() {
    use crate::agent::{DynamicAgent, AgentStatus};
    
    let mut agent = DynamicAgent::new("test-agent", vec!["compute".to_string()]);
    
    // Test status transitions
    agent.set_status(AgentStatus::Error);
    assert_eq!(agent.status(), AgentStatus::Error);
    
    // Agent in error state should still be able to transition
    agent.start().await.unwrap();
    assert_eq!(agent.status(), AgentStatus::Running);
    
    agent.shutdown().await.unwrap();
    assert_eq!(agent.status(), AgentStatus::Offline);
}

// Test boundary conditions
#[tokio::test]
async fn test_boundary_conditions() {
    use crate::swarm::{Swarm, SwarmConfig};
    use crate::agent::DynamicAgent;
    use crate::task::Task;
    
    // Test with zero max agents
    let config = SwarmConfig {
        max_agents: 0,
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    let agent = DynamicAgent::new("agent", vec!["compute".to_string()]);
    match swarm.register_agent(agent) {
        Err(SwarmError::ResourceExhausted { .. }) => {} // Expected
        _ => panic!("Should fail with zero max agents"),
    }
    
    // Test with empty capability requirements
    let mut swarm = Swarm::new(SwarmConfig::default());
    let agent = DynamicAgent::new("agent", vec![]);
    swarm.register_agent(agent).unwrap();
    
    let task = Task::new("task", "any")
        .require_capability("compute"); // Task requires a capability the agent doesn't have
    swarm.submit_task(task).unwrap();
    
    // Should not assign task to agent with no capabilities
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 0);
}

// Test error message quality
#[test]
fn test_error_message_quality() {
    let errors = vec![
        (
            SwarmError::AgentNotFound { id: "agent-123".to_string() },
            "agent-123"
        ),
        (
            SwarmError::TaskExecutionFailed { reason: "out of memory".to_string() },
            "out of memory"
        ),
        (
            SwarmError::InvalidTopology { reason: "cyclic dependency".to_string() },
            "cyclic dependency"
        ),
        (
            SwarmError::ResourceExhausted { resource: "thread pool".to_string() },
            "thread pool"
        ),
        (
            SwarmError::CommunicationError { reason: "connection refused".to_string() },
            "connection refused"
        ),
        (
            SwarmError::SerializationError { reason: "invalid JSON".to_string() },
            "invalid JSON"
        ),
        (
            SwarmError::Timeout { duration_ms: 5000 },
            "5000"
        ),
        (
            SwarmError::CapabilityMismatch { 
                agent_id: "worker-1".to_string(), 
                capability: "gpu-compute".to_string() 
            },
            "gpu-compute"
        ),
        (
            SwarmError::StrategyError { reason: "no viable path".to_string() },
            "no viable path"
        ),
        (
            SwarmError::Custom("detailed error information".to_string()),
            "detailed error information"
        ),
    ];
    
    // Verify error messages contain relevant information
    for (error, expected_content) in errors {
        let message = error.to_string();
        assert!(
            message.contains(expected_content),
            "Error message '{message}' should contain '{expected_content}'"
        );
    }
}